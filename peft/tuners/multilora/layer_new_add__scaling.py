# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import (
    dequantize_module_weight,
    gather_params_ctx,
    get_bnb_param_type,
    skip_init_on_device,
)
from peft.utils.other import transpose

from .config import MULTILoraConfig
from .dora import DoraConv2dLayer, DoraConv3dLayer, DoraEmbeddingLayer, DoraLinearLayer, _DoraConvNdLayer
from peft.tuners._buffer_dict import BufferDict

class MULTILoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "multilora_A", "multilora_B", "multilora_scaling", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.task_num = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.multilora_scaling = nn.ParameterDict({})
        self.multilora_A = nn.ModuleDict({})
        self.multilora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        raise NotImplementedError(f"MULTILoraLayer can not update_layer")

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        raise NotImplementedError(f"MULTILoraLayer can not reset_lora_parameters")

    def olora_init(self, adapter_name):
        raise NotImplementedError(f"olora_init not implemented")

    def pissa_init(self, adapter_name, init_lora_weights):
        raise NotImplementedError(f"pissa_init not implemented")

    def corda_init(self, adapter_name, init_lora_weights):
        raise NotImplementedError(f"corda_init not implemented")

    def loftq_init(self, adapter_name):
        raise NotImplementedError(f"loftq_init not implemented")

    def dora_init(self, adapter_name: str) -> None:
        raise NotImplementedError(f"dora_init not implemented")

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.multilora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.multilora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError(f"_mixed_batch_forward not implemented")


    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Whether to cast the dtype of the input to the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if (not self.cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class MULTILoraLinear(nn.Module, MULTILoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        task_num: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MULTILoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
            task_num=task_num,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
    
    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        task_num: int,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters


        self.task_num[adapter_name] = task_num

        self.multilora_scaling[adapter_name] = nn.Parameter(torch.zeros((task_num, self.out_features)))
        for num in range(task_num):

            mtllora_name = adapter_name+"_"+str(num)
            self.multilora_A[mtllora_name] = nn.Linear(self.in_features, r, bias=lora_bias)

            self.multilora_B[mtllora_name] = nn.Linear(r, self.out_features, bias=lora_bias)


        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            raise NotImplementedError(f"pissa_init not implemented")
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            raise NotImplementedError(f"corda_init not implemented")
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            raise NotImplementedError(f"olora_init not implemented")
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            raise NotImplementedError(f"loftq not implemented")
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            raise NotImplementedError(f"eva_init not implemented")
            nn.init.zeros_(self.multilora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights, task_num)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name, task_num)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)
    
    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, task_num, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        if device is None:
            base_layer = self.get_base_layer()
            if isinstance(base_layer, nn.MultiheadAttention):
                base_layer = base_layer.out_proj
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(base_layer, weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        meta = torch.device("meta")

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639


        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue

            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

        for num in range(task_num):
            adapter_layer = getattr(self, "multilora_A", None)
            multilora_A_name = adapter_name+"_"+str(num)
            if multilora_A_name not in adapter_layer:
                raise ValueError("multilora_A_name not in adapter_layer")
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[multilora_A_name] = adapter_layer[multilora_A_name].to(device, dtype=dtype)
            else:
                adapter_layer[multilora_A_name] = adapter_layer[multilora_A_name].to(device)


        for num in range(task_num):
            adapter_layer = getattr(self, "multilora_B", None)
            multilora_B_name = adapter_name+"_"+str(num)
            if multilora_B_name not in adapter_layer:
                raise ValueError("multilora_B_name not in adapter_layer")
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[multilora_B_name] = adapter_layer[multilora_B_name].to(device, dtype=dtype)
            else:
                adapter_layer[multilora_B_name] = adapter_layer[multilora_B_name].to(device)

        

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                layer.requires_grad_(False)

        for adapter_name in adapter_names:
            self.multilora_scaling[adapter_name].requires_grad_(True)
            for num in range(self.task_num[adapter_name]):
                mtllora_name = adapter_name+"_"+str(num)
                self.multilora_A[mtllora_name].requires_grad_(True)
                self.multilora_B[mtllora_name].requires_grad_(True)
            

        self._active_adapter = adapter_names


    def reset_lora_parameters(self, adapter_name, init_lora_weights, task_num):
        if init_lora_weights is False:
            return

        if adapter_name in self.multilora_scaling.keys():
            nn.init.zeros_(self.multilora_scaling[adapter_name])
            for num in range(task_num):
                nn.init.kaiming_uniform_(self.multilora_A[adapter_name+"_"+str(num)].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.multilora_B[adapter_name+"_"+str(num)].weight, a=math.sqrt(5))

            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.multilora_B[adapter_name].bias)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError(f"MTL Linear can not be merged")
        

    def unmerge(self) -> None:
        raise NotImplementedError(f"MTL Linear can not be unmerged")
        

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError(f"MTL Linear get_delta_weight not implemented")

    def forward(self, x: torch.Tensor, language_id: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:

        # print("MTLLinear language_id")
        # print(language_id)
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # language_id = kwargs.pop("language_id", None)
        # if not language_id:
        #     raise ValueError("language_id shouldn't be `None`")
        if len(language_id)>1:
            raise ValueError("language_id shouldn't > 1")
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError(f"adapter_names shouldn't be none _mixed_batch_forward not implemented")
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            raise ValueError("self.merged shouldn't be `True`")
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            find = False
            multilora_scaling_keys = self.multilora_scaling.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in multilora_scaling_keys:
                    continue
                find = True


                # print("multilora_scaling[active_adapter].requires_grad")
                # print(self.multilora_scaling[active_adapter].requires_grad)


                
                multilora_A_task = self.multilora_A[active_adapter+"_"+str(0)]


                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, multilora_A_task.weight.dtype)

                if not self.use_dora[active_adapter]:
                    for i in range(self.task_num[active_adapter]):
                        multilora_A_task = self.multilora_A[active_adapter+"_"+str(i)]
                        multilora_B_task = self.multilora_B[active_adapter+"_"+str(i)]
                        result += multilora_B_task(multilora_A_task(dropout(x))) * self.multilora_scaling[active_adapter][i, ...].unsqueeze(0).unsqueeze(0)*scaling

                else:
                    raise ValueError("self.use_dora[active_adapter] shouldn't be `True`")
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            if not find:
                raise ValueError("active_adapter not in multilora_A_keys")

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "multilora_." + rep


class MULTILoraEmbedding(nn.Module, MULTILoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        if lora_bias:
            # lora_bias=True is not supported (yet) for embedding layers, as they use nn.Parameter
            raise ValueError(f"lora_bias={lora_bias} is not supported for {self.__class__.__name__}.")

        super().__init__()
        MULTILoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            raise NotImplementedError(f"Embedding init_lora_weights not implemented")
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        raise NotImplementedError(f"Embedding dora_init not implemented")

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]

                if not self.use_dora[active_adapter]:
                    after_A = self._embed(x, embedding_A)
                    result = result + (after_A @ embedding_B) * scaling
                else:
                    mag_norm_scale, dora_result = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=embedding_A,
                        lora_B=embedding_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        embed_fn=self._embed,
                    )
                    result = mag_norm_scale * result + dora_result
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "mtllora." + rep


class _MULTILoraConvNd(nn.Module, MULTILoraLayer):
    # Lora implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MULTILoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.lora_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = conv_layer(r, self.out_features, out_kernel, out_stride, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def _get_dora_factor_view(self):
        return (-1,) + (1,) * (self._kernel_dim - 1)

    def dora_init(self, adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer_class = self._get_dora_layer_class()
        dora_layer = dora_layer_class(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _get_dora_layer_class(self) -> type[_DoraConvNdLayer]:
        # Subclasses should override this method to return the appropriate DoraLayer class
        raise NotImplementedError

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(*self._get_dora_factor_view()) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(*self._get_dora_factor_view()) * (
                            base_layer.weight.data + delta_weight
                        )
                        base_layer.weight.data = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(*self._get_dora_factor_view()) - delta_weight
                    weight.data = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            output_tensor = (
                self.conv_fn(
                    weight_A.transpose(0, 1),
                    weight_B,
                ).transpose(0, 1)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)

        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class MULTILoraConv2d(_MULTILoraConvNd):
    # Lora implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d

    def _get_dora_layer_class(self):
        return DoraConv2dLayer


class MULTILoraConv1d(_MULTILoraConvNd):
    # Lora implemented in a conv1d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 3:
            raise ValueError(f"Conv1d layer kernel must have 3 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv1d

    def _get_dora_layer_class(self):
        raise NotImplementedError


class MULTILoraConv3d(_MULTILoraConvNd):
    # Lora implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d

    def _get_dora_layer_class(self):
        return DoraConv3dLayer


class MULTILoraMultiheadAttention(nn.Module, MULTILoraLayer):
    """LoRA implemented in a multihead attention layer

    This is currently only implemented for the case of `_qkv_same_embed_dim = True`, i.e. query, key, and value having
    the same dimension.

    Note: LoRA is applied to both the in_proj (query/key/value) and out_proj. There is currently no way to specify only
    one of them. Don't try to apply LoRA to the out_proj of MultiheadAttention by targeting that layer specifically,
    since the forward method of that layer is not being used, hence the LoRA adapter would be ignored.

    This is a little bit hacky because of the way that MultiheadAttention is implemented in PyTorch: There are no
    `nn.Linear` layers which we can hook onto or, in case of output projection, `.forward` is not used. This
    implementation works around these problems by merging the weights before the forward call and unmerging them after
    the forward call.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        # TODO work with separate weights
        raise NotImplementedError(f"MTL MultiheadAttention not implemented")


        if not getattr(base_layer, "_qkv_same_embed_dim", True):
            # default for this value appears to be True:
            # https://github.com/pytorch/pytorch/blob/701ba5203fe68d55d655bd4d6c008be94cf34ea5/torch/nn/modules/activation.py#L1128-L1130
            raise ValueError(
                f"Only same embed for query/key/value is supported as of now for {self.__class__.__name__}."
            )
        if use_dora:
            # TODO: probably not so hard to implement
            raise ValueError(f"{self.__class__.__name__} does not support DoRA (yet), please set use_dora to False")

        super().__init__()
        MULTILoraLayer.__init__(self, base_layer, **kwargs)

        # Note: LoRA is applied to both in_proj and out_proj. There is currently no way to only specify one of them.
        if isinstance(base_layer.out_proj, nn.Linear):
            self.base_layer.out_proj = Linear(
                base_layer.out_proj,
                adapter_name,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                **kwargs,
            )
        else:
            raise ValueError(f"out_proj must be an instance of nn.Linear for {self.__class__.__name__}.")

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    @property
    def embed_dim(self) -> int:
        return self.get_base_layer().embed_dim

    @property
    def kdim(self) -> Optional[int]:
        return self.get_base_layer().kdim

    @property
    def vdim(self) -> Optional[int]:
        return self.get_base_layer().vdim

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return self.get_base_layer()._qkv_same_embed_dim

    @property
    def num_heads(self) -> int:
        return self.get_base_layer().num_heads

    @property
    def dropout(self) -> float:
        return self.get_base_layer().dropout

    @property
    def batch_first(self) -> bool:
        return self.get_base_layer().batch_first

    @property
    def head_dim(self) -> int:
        return self.get_base_layer().head_dim

    @property
    def in_proj_weight(self) -> nn.Parameter:
        return self.get_base_layer().in_proj_weight

    @property
    def in_proj_bias(self) -> nn.Parameter:
        return self.get_base_layer().in_proj_bias

    @property
    def out_proj(self) -> nn.Module:
        return self.get_base_layer().out_proj.get_base_layer()

    @property
    def bias_k(self) -> Optional[nn.Parameter]:
        return self.get_base_layer().bias_k

    @property
    def bias_v(self) -> Optional[nn.Parameter]:
        return self.get_base_layer().bias_v

    def merge_masks(self, *args, **kwargs) -> tuple[Optional[torch.Tensor], Optional[int]]:
        return self.get_base_layer().merge_masks(*args, **kwargs)

    @property
    def add_zero_attn(self) -> bool:
        return self.get_base_layer().add_zero_attn

    def update_layer(self, *args, **kwargs) -> None:
        super().update_layer(*args, **kwargs)
        # Note: LoRA is applied to both in_proj and out_proj. There is currently no way to only specify one of them.
        self.base_layer.out_proj.update_layer(*args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        # Implementation follows this:
        # https://github.com/Baijiong-Lin/LoRA-Torch/blob/4bfed6820b64fcf47064c30f30606a190a4f0d2e/loratorch/layers.py#L73-L79
        # Notably, instead of mutating the weight, we delete the original weight and replace it by the merged weight
        # TODO: work with separate weights
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # TODO: work with separate weights
                    # merging in_proj (nn.Parameter)
                    orig_weights_in = base_layer.in_proj_weight.data.detach().clone()
                    orig_weights_in += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights_in).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    # merging out_proj (subclass of nn.Linear)
                    orig_weights_out = base_layer.out_proj.weight.data.detach().clone()
                    orig_weights_out += base_layer.out_proj.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights_out).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    # unregister parameter implicitly and overwrite using merged weights; gradients are computed after
                    # forward and, thus, after unmerging (see forward()), therefore this is safe to do.
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = orig_weights_in

                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = orig_weights_out
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                else:
                    # merging in_proj (nn.Parameter)
                    # TODO: work with separate weights
                    weight_merged = base_layer.in_proj_weight.data.detach() + self.get_delta_weight(active_adapter)

                    # unregister parameter implicitly and overwrite using merged weights; gradients are computed after
                    # forward and, thus, after unmerging (see forward()), therefore this is safe to do.
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = weight_merged

                    # merging out_proj (subclass of nn.Linear)
                    weight_merged = base_layer.out_proj.weight.data.detach() + base_layer.out_proj.get_delta_weight(
                        active_adapter
                    )
                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = weight_merged
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        # TODO work with separate weights
        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                # Ensure that requires_grad=False for the base weights after unmerging. This may not matter since
                # requires_grad was False when the optimizer was initialized, but still let's try to be correct here.

                # in_proj
                old_weight = base_layer.in_proj_weight.data - self.get_delta_weight(active_adapter)
                del base_layer.in_proj_weight
                base_layer.register_parameter("in_proj_weight", nn.Parameter(old_weight, requires_grad=False))

                # out_proj
                old_weight = base_layer.out_proj.base_layer.weight.data - base_layer.out_proj.get_delta_weight(
                    active_adapter
                )
                del base_layer.out_proj.base_layer.weight
                base_layer.out_proj.base_layer.register_parameter(
                    "weight", nn.Parameter(old_weight, requires_grad=False)
                )

        self.get_base_layer().out_proj.unmerge()

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool, adapter_names: Optional[list[str]]
    ) -> nn.MultiheadAttention:
        """
        Merging and unloading of the MultiheadAttention module

        This requires an extra step for MultiheadAttention, which is why there is this special method instead of
        relying on the normal merge_and_unload code path.
        """
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        base_layer = self.get_base_layer()

        # extra steps: re-register weights, take care of out_proj layer
        # in_proj
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        base_layer.register_parameter("in_proj_weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

        # out_proj
        out_proj_layer = base_layer.out_proj.get_base_layer()
        weight = out_proj_layer.weight
        del out_proj_layer.weight
        out_proj_layer.register_parameter("weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

        base_layer.out_proj = out_proj_layer
        return base_layer

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _check_forward_args(self, x, *args, **kwargs):
        if "adapter_names" in kwargs:
            raise TypeError(f"lora.{self.__class__.__name__} does not support mixed adapter batches.")
        super()._check_forward_args(x, *args, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        self._check_forward_args(x, *args, **kwargs)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            out_proj = self.get_base_layer().out_proj
            if out_proj.active_adapters != self.active_adapters:
                # We have a case that in_proj and out_proj have diverging merged adapters. We cannot
                # really deal with this correctly, thus it's better to raise than possibly create a hard to debug mess
                cls_name = self.get_base_layer().__class__.__name__
                raise ValueError(
                    f"The out_proj layer of {cls_name} has merged layers but {cls_name} itself doesn't; please ensure "
                    "that either both or none have merged layers"
                )

            # Merge all adapters that are active for this module, i.e. the LoRA weights for in_proj and out_proj.
            # in_proj uses nn.Parameters, therefore, there is no forward method to be used and we have to explicitly
            # merge for the LoRA weights to have an effect:
            # https://github.com/pytorch/pytorch/blob/6ebb26d572d5fcdc6ac0d1297bdf8d1eb5d20722/torch/nn/modules/activation.py#L1020
            # For out_proj, we have an nn.Linear (or rather: NonDynamicallyQuantizableLinear), but its forward method
            # is not used:
            # https://github.com/pytorch/pytorch/blob/6ebb26d572d5fcdc6ac0d1297bdf8d1eb5d20722/torch/nn/modules/activation.py#L1267-L1271
            # Therefore, its LoRA weights also need to be merged to have an effect.
            active_adapters = [a for a in self.active_adapters if a in self.lora_A]
            try:
                self.merge(adapter_names=active_adapters)
                result = self.base_layer(x, *args, **kwargs)
            finally:
                # it's safe to call unmerge(), which unmerges all adapters, because we checked that not self.merged,
                # i.e. there is was no merged layer before
                self.unmerge()

        result = (result[0].to(previous_dtype), result[1].to(previous_dtype) if result[1] is not None else result[1])
        return result

    # The decorator is needed in case low_cpu_mem_usage=True is used, as we don't want the base layer weights to be
    # moved to meta device. This requires the use of PEFT's implementation of init_empty_weight instead of using the one
    # from accelerate.
    @skip_init_on_device
    def _restore_weights(self):
        # Restore the weights as registered parameters on the base layer.
        # This is necessary because the way that weights are merged/unmerged (which is necessary for forward to work
        # correctly), the Module "forgets" these attributes. Therefore, we need to call register_parameter explicitly.
        # We cannot call register_parameter for merging/unmerging because that cuts them off from the autograd graph.
        # Note that this is hacky, since we need to ensure that _restore_weights is called by each method that needs it.

        # in_proj
        # TODO work with separate weights
        base_layer = self.get_base_layer()
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        base_layer.register_parameter("in_proj_weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

        # out_proj
        base_layer = base_layer.out_proj.get_base_layer()
        weight = base_layer.weight
        del base_layer.weight
        base_layer.register_parameter("weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

    def state_dict(self, *args, **kwargs):
        self._restore_weights()
        return super().state_dict(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        # Note: no need to also implement modules(), as modules() calls named_modules() under the hood
        self._restore_weights()
        return super().named_modules(*args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_MULTILora_lora(
    target: torch.nn.Module,
    adapter_name: str,
    mtllora_config: MTLLoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(mtllora_config.loftq_config)
        new_module = MULTILoraEmbedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(mtllora_config.loftq_config)
        new_module = MULTILoraConv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        kwargs.update(mtllora_config.loftq_config)
        new_module = MULTILoraConv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Conv1d):
        kwargs.update(mtllora_config.loftq_config)
        new_module = MULTILoraConv1d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.MultiheadAttention):
        raise ValueError(f"target_base_layer can not be MultiheadAttention")
        kwargs.update(mtllora_config.loftq_config)
        new_module = MULTILoraMultiheadAttention(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = mtllora_config.fan_in_fan_out = False
        kwargs.update(mtllora_config.loftq_config)
        new_module = MULTILoraLinear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = mtllora_config.fan_in_fan_out = True
        kwargs.update(mtllora_config.loftq_config)
        new_module = MULTILoraLinear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
