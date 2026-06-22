# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.





import editdistance
from .uyghur_bpe import uyghur_bpe

import torch
import numpy as np
from typing import List, Tuple, Optional

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


import re
import Levenshtein
def calculate_cer(reference, hypothesis):
    """计算字符错误率 (Character Error Rate)"""
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    # 移除多余空格，统一小写（如果适用）
    ref = reference.strip()
    hyp = hypothesis.strip()
    
    dist = Levenshtein.distance(ref, hyp)
    length = len(ref)
    
    return dist / length


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self._current_multimodal_features = {}


        self.bpe_tokenizer = uyghur_bpe


        from .reward import UyghurTokenLevelRewardComputer
        self.UyghurTokenLevelReward = UyghurTokenLevelRewardComputer(self.tokenizer, self.bpe_tokenizer)


        # from .llm_reward import UyghurTokenLevelRewardComputer
        # self.UyghurTokenLevelReward = UyghurTokenLevelRewardComputer(self.tokenizer)


    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        loss_p_meter = AverageMeter()  # <--- 新增：用于记录 policy loss
        loss_v_meter = AverageMeter()  # <--- 新增：用于记录 value loss
        ratio_meter = AverageMeter()
        clip_cov_meter = AverageMeter()  # <--- 新增：用于记录 clipfrac (clip coverage)
        kl_cov_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)
            # print("-------------batch-----------------")
            # print(batch)
            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            input_features, feature_attention_masks = [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch = {}
                mini_labels = {}
                for key, value in batch.items():
                    if key=='labels':
                        mini_labels[key] = value[idx : idx + self.config.mini_batch_size]
                        continue
                    if isinstance(value, torch.Tensor):
                        # 对 Tensor 进行切片，注意 input_features 第一维也是 batch size，所以可以直接切
                        mini_batch[key] = value[idx : idx + self.config.mini_batch_size]
                    else:
                        raise ValueError("not isinstance(value, torch.Tensor)")
                
                # mini_batch = {
                #     "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
                #     "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
                # }
                # print("-------------labels-----------------")
                # print(batch['labels'][idx : idx + self.config.mini_batch_size])
                # print(batch['input_features'][idx : idx + self.config.mini_batch_size].shape)
                self._current_multimodal_features = mini_batch
                mini_batch_queries, mini_batch_responses, mini_batch_input_features, mini_batch_feature_attention_masks  = self.get_inputs(mini_batch)
                # print("mini_batch_input_features")
                # print(mini_batch_input_features)
                # print("mini_batch_feature_attention_masks")
                # print(mini_batch_feature_attention_masks)
                self._current_multimodal_features = mini_batch
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses, mini_labels)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)
                input_features.extend(mini_batch_input_features)
                # print("----------------input_features.extend(mini_batch_input_features)-------------------")
                # print(input_features)
                feature_attention_masks.extend(mini_batch_feature_attention_masks)
            # print("---------before step----------------")
            # print("queries")
            # print(queries)
            # print("responses")
            # print(responses)
            # print("input_features")
            # print(input_features)
            # print("feature_attention_masks")
            # print(feature_attention_masks)
            # raise ValueError("not over")
            # Run PPO step
            self.model.train()
            stats = self.step(queries, responses, rewards, input_features = input_features, feature_attention_masks = feature_attention_masks)
            # print(stats)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            loss_p_meter.update(float(stats["ppo/loss/policy"]), n=len(rewards)) # <--- 新增：更新 policy loss
            loss_v_meter.update(float(stats["ppo/loss/value"]), n=len(rewards))  # <--- 新增：更新 value loss
            # reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))
            ratio_raw = stats.get("ppo/policy/ratio", 0.0)
            ratio_scalar = float(ratio_raw.mean()) if hasattr(ratio_raw, "mean") else float(ratio_raw)
            ratio_meter.update(ratio_scalar, n=len(rewards))
            clip_cov_meter.update(float(stats.get("ppo/policy/clipfrac", 0.0)), n=len(rewards))
            
            
            # 提取 approxkl 作为 kl_cov (你也可以换成 stats["objective/kl"] 如果你想看真实 KL)
            kl_cov_meter.update(float(stats.get("ppo/policy/approxkl", 0.0)), n=len(rewards))
            reward_meter.update(torch.stack([r.mean() for r in rewards]).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    loss_p=round(loss_p_meter.avg, 4), # <--- 新增：加入 logs 字典
                    loss_v=round(loss_v_meter.avg, 4),
                    ratio=round(ratio_meter.avg, 4),
                    clip_cov=round(clip_cov_meter.avg, 4), # <--- 新增：写入 logs
                    kl_cov=round(kl_cov_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                loss_p_meter.reset() # <--- 新增：重置 meter
                loss_v_meter.reset() # <--- 新增：重置 meter
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: dict[str, "torch.Tensor"]) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries."""
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            # print("batch", batch)
            
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                if k=="input_ids" or k=="attention_mask":
                    batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        input_feature = batch["input_features"].detach()
        feature_attention_mask = batch["feature_attention_mask"].detach().cpu()
        queries, responses = [], []
        input_features, feature_attention_masks = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right
            input_features.append(input_feature[i,:])
            feature_attention_masks.append(feature_attention_mask[i,:])
        return queries, responses, input_features, feature_attention_masks



    @torch.no_grad()
    def compute_custom_rewards(
        self,
        messages: List[str],
        target_str: List[str],
        queries,
        responses,
        hallucination_penalty: float = -0.5,
        max_length_ratio: float = 1.2,
        min_length_ratio: float = 0.8,
        penalty_weight: float = 1.0,
    ) -> List[torch.Tensor]:
        """
        为维吾尔语PPO训练计算token级别的自定义奖励。

        奖励由以下四部分组成：
        1. BPE子词奖励：基于TER的全局奖励，通过字符级对齐映射到每个LLM token
        2. Token级细粒度奖励：根据每个BPE subword是否与reference对齐，加权分配给LLM token
        3. OOV幻觉惩罚：对词表外词汇对应的LLM token施加负向惩罚
        4. 长度惩罚：对过长输出施加均匀的全局惩罚

        Args:
            messages:              PPO rollout后解码的模型输出字符串列表
            target_str:            Ground truth字符串列表
            hallucination_penalty: OOV词对应token的惩罚值（默认 -0.5）
            max_length_ratio:      超过此长度比时触发长度惩罚（默认 1.5）
            penalty_weight:        长度惩罚的强度系数（默认 1.0）

        Returns:
            List[torch.Tensor]，每个tensor shape为 (response_token_length,)
        """
        reward_list = []

        for message, target in zip(messages, target_str):

            # ── Step 1：获取LLM token数量及字符级span ──────────────────────────
            llm_token_spans = _get_llm_token_char_spans(message, self.tokenizer)
            response_length = len(llm_token_spans)

            if response_length == 0:
                # 空输出：返回单个零奖励（防止空tensor）
                token_rewards = torch.zeros(2)
                token_rewards[:] = hallucination_penalty * 2
                reward_list.append(token_rewards)
                continue

            # ── Step 2：BPE tokenize并计算TER ──────────────────────────────────
            hyp_bpe_ids = self.bpe_tokenizer.encode(message)
            ref_bpe_ids = self.bpe_tokenizer.encode(target)

            ter = _compute_ter(hyp_bpe_ids, ref_bpe_ids)
            bpe_reward = 1.0 - ter   # 全局BPE奖励，范围 [0, 1]

            # print("bpe_reward", bpe_reward)

            # ── Step 3：计算每个BPE subword的对齐正确性mask ────────────────────
            bpe_correct_mask = _compute_bpe_correct_mask(hyp_bpe_ids, ref_bpe_ids)

            # print("bpe_correct_mask", bpe_correct_mask)
            # bpe_correct_mask: List[float]，长度 = len(hyp_bpe_ids)
            # 值域：1.0（正确对齐）或 0.0（错误）

            # ── Step 4：BPE subword奖励 → LLM token 字符级对齐映射 ────────────
            token_rewards = _align_bpe_to_llm_tokens(
                text=message,
                bpe_correct_mask=bpe_correct_mask,
                bpe_tokenizer=self.bpe_tokenizer,
                tokenizer=self.tokenizer,
                bpe_reward=bpe_reward
            )

            # print("token_rewards", token_rewards)

            # 确保token_rewards长度与LLM token数量一致
            if len(token_rewards) < response_length:
                # 不足时用全局bpe_reward填充
                token_rewards.extend([bpe_reward] * (response_length - len(token_rewards)))
            elif len(token_rewards) > response_length:
                token_rewards = token_rewards[:response_length]

            token_rewards.append(0)

            token_rewards = np.array(token_rewards, dtype=np.float32)

            # ── Step 5：OOV幻觉惩罚 ──────────────────────────────────────────
            oov_penalties = _compute_oov_penalty(
                text=message,
                llm_token_spans=llm_token_spans,
                bpe_tokenizer=self.bpe_tokenizer,
                hallucination_penalty=hallucination_penalty
            )

            if len(oov_penalties) == response_length:
                oov_penalties.append(0)
                token_rewards += np.array(oov_penalties, dtype=np.float32)

            # ── Step 6：长度惩罚（全局均匀叠加） ────────────────────────────────
            length_ratio = len(message) / max(len(target), 1)

            if length_ratio > 1.5 or length_ratio<0.5:
                length_penalty = -penalty_weight * (length_ratio - max_length_ratio)
                # 分配在最后一个token上
                token_rewards[-1] += length_penalty
            elif length_ratio < min_length_ratio:
                # 过短惩罚：距离min_length_ratio越远惩罚越大
                length_penalty = -penalty_weight * (1.0 / length_ratio - 1.0 / min_length_ratio)
                token_rewards[-1] += length_penalty

            # ── Step 7：转为Tensor并收集结果 ────────────────────────────────────
            reward_tensor = torch.tensor(token_rewards, dtype=torch.float32)
            reward_list.append(reward_tensor)

            # print("message", message)
            # print("target", target)
            # print("queries", queries)
            # print("responses", responses)
            if len(responses[0])!=len(reward_list[0]):
                print("len(responses[0])!=len(reward_list[0])", len(responses[0]), len(reward_list[0]))
                llm_tokens = self.tokenizer(message, return_offsets_mapping=True, add_special_tokens=False)
                print("llm_tokens", llm_tokens, len(llm_tokens))

            # print("llm_token_spans", llm_token_spans)
            # print("reward_list", reward_list)

        return reward_list







    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        labels = None
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "api":
            
            token_ids = [r.tolist() for r in responses]
            
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

            label_ids = labels['labels'].detach().clone()
            # print(token_ids)
            # print(label_ids)
            
            # 2. 将 -100 替换为 pad_token_id (如果没有 pad_token_id 则设为 0)
            # 这样做是为了让 decode 函数能正常运行，decode 出来的结果里这些位置会变成 <pad> 或者空
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            label_ids[label_ids == -100] = pad_token_id
            # print(label_ids)
            target_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            target_str = [tar.strip() for tar in target_str]
            # print("mes:",messages)
            # print("tar:",target_str)

            # reward_list = self.compute_custom_rewards(messages, target_str, queries, responses)
            reward_list = self.UyghurTokenLevelReward.compute_custom_rewards(messages, target_str, queries, responses)
            # print("responses", responses)
            # print("reward_list", reward_list)

            return reward_list

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)
            # print("----------------------input_kwargs--------------------------")
            # print(input_kwargs['input_ids'])
            # print("----------------------query_batch--------------------------")
            # print(query_batch)
            # print("----------------------response_batch--------------------------")
            # print(response_batch)
            # print("----------------------all_logprobs--------------------------")
            # print(logits)
            # print(torch.argmax(logits, dim=-1))
            # preds = torch.argmax(logits, dim=-1)


            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])


                # print("response_batch[j]", response_batch[j])
                # print(self.tokenizer.batch_decode(response_batch[j].unsqueeze(0), skip_special_tokens=True))

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]
            # # print("--------------------masks--------------------------")
            # # print(masks)
            # # print("------- 按 Batch 提取预测结果 -------")
            
            # mask_bool = masks.bool()
            # for i in range(len(preds)):
            #     # 取出第 i 个样本中，mask 为 True 的那些 token
            #     valid_tokens = preds[i][mask_bool[i]] 
                
            #     print(f"Batch {i} 对应的 TokenIDs: {valid_tokens}")

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
