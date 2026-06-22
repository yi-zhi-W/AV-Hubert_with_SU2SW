"""
维吾尔语PPO强化学习 Token级奖励计算器
=====================================
通过BPE子词对齐、OOV检测、长度惩罚计算token级reward。

核心设计：
- BPE(维吾尔语专用) → 字符级span → LLM tokenizer token级奖励
- response_ids与llm_ids的对齐：直接匹配 + 不匹配时按字符span合并处理
- 长度惩罚集中在EOS token，通过GAE向前自然衰减
"""

import torch
import numpy as np
import editdistance
from typing import List, Tuple, Optional


class UyghurTokenLevelRewardComputer:
    """
    维吾尔语PPO训练的Token级奖励计算器。

    Usage:
        computer = UyghurTokenLevelRewardComputer(tokenizer, bpe_tokenizer)
        rewards = computer.compute_custom_rewards(messages, targets, queries, responses)
    """

    def __init__(
        self,
        tokenizer,
        bpe_tokenizer,
        hallucination_penalty: float = -0.5,
        max_length_ratio: float = 1.1,
        min_length_ratio: float = 0.9,
        penalty_weight: float = 1.0,
    ):
        """
        Args:
            tokenizer:             HuggingFace LLM tokenizer
            bpe_tokenizer:         维吾尔语BPE tokenizer（Uyghur()实例）
            hallucination_penalty: OOV词对应token的惩罚值（默认-0.5）
            max_length_ratio:      过长惩罚阈值（默认1.2）
            min_length_ratio:      过短惩罚阈值（默认0.8）
            penalty_weight:        惩罚强度系数（默认1.0）
        """
        self.tokenizer = tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.hallucination_penalty = hallucination_penalty
        self.max_length_ratio = max_length_ratio
        self.min_length_ratio = min_length_ratio
        self.penalty_weight = penalty_weight

    # =========================================================================
    # 公开接口
    # =========================================================================

    @torch.no_grad()
    def compute_custom_rewards(
        self,
        messages: List[str],
        target_str: List[str],
        queries,
        responses,
    ) -> List[torch.Tensor]:
        """
        为维吾尔语PPO训练计算token级别的自定义奖励。

        奖励组成：
          1. BPE子词奖励：TER → bpe_reward，通过字符级对齐映射到每个content token
          2. OOV幻觉惩罚：encode→decode还原失败的词对应token施加负向惩罚
          3. 长度惩罚：集中在EOS token，过长线性惩罚，过短倒数惩罚
          4. EOS token：bpe_reward + length_penalty，汇总序列级质量信号

        Args:
            messages:   PPO rollout后解码的模型输出字符串列表
            target_str: Ground truth字符串列表
            queries:    PPO rollout的query token ids列表
            responses:  PPO rollout的response token ids列表（含EOS）

        Returns:
            List[torch.Tensor]，每个tensor shape=(response_token_length,)
        """
        reward_list = []

        for message, target, response in zip(messages, target_str, responses):

            # ── Step 1：统一strip，防止\n影响OOV检测 ─────────────────────────
            message = message.strip()
            target = target.strip()

            # ── Step 2：以rollout的response为长度基准 ────────────────────────
            eos_token_id = self.tokenizer.eos_token_id
            if len(response) > 0 and response[-1] == eos_token_id:
                content_ids = response[:-1]
                has_eos = True
            else:
                content_ids = response
                has_eos = False
            content_length = len(content_ids)
            total_length = len(response)

            # ── Step 3：空输出处理 ────────────────────────────────────────────
            if total_length == 0 or len(message) == 0:
                if total_length > 0:
                    # response有token但message为空：全部填充强惩罚
                    reward_list.append(
                        torch.full((total_length,), self.hallucination_penalty * 2)
                    )
                else:
                    # response也为空：这种情况不应出现，上游rollout有问题
                    raise AssertionError(
                        f"Both response and message are empty, message='{message}'"
                    )
                continue

            # ── Step 4：重新tokenize获取offset_mapping ───────────────────────
            # input_ids可能与content_ids不完全一致，但offset_mapping准确反映
            # 每个llm token在message字符串中的位置
            llm_encoding = self.tokenizer(
                message,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            llm_ids = llm_encoding["input_ids"]
            llm_spans = llm_encoding["offset_mapping"]  # List of (start, end)

            # ── Step 5：BPE tokenize并计算TER ────────────────────────────────
            hyp_bpe_ids = self.bpe_tokenizer.encode(message)
            ref_bpe_ids = self.bpe_tokenizer.encode(target)

            ter = self._compute_ter(hyp_bpe_ids, ref_bpe_ids)
            bpe_reward = 1.0 - ter   # 全局BPE奖励，范围[0, 1]

            # ── Step 6：BPE subword对齐正确性mask ────────────────────────────
            bpe_correct_mask = self._compute_bpe_correct_mask(
                hyp_bpe_ids, ref_bpe_ids
            )

            # ── Step 7：BPE奖励映射到llm_tokens粒度 ──────────────────────────
            token_rewards_by_llm = self._align_bpe_to_llm_tokens(
                text=message,
                bpe_correct_mask=bpe_correct_mask,
                llm_spans=llm_spans,
                bpe_reward=bpe_reward
            )

            # ── Step 8：OOV惩罚叠加到llm_tokens粒度 ──────────────────────────
            oov_penalties = self._compute_oov_penalty(
                text=message,
                llm_token_spans=llm_spans,
            )
            # oov_penalties由llm_token_spans逐个处理，长度必须与llm_spans一致
            assert len(oov_penalties) == len(token_rewards_by_llm), (
                f"oov_penalties({len(oov_penalties)}) != "
                f"token_rewards_by_llm({len(token_rewards_by_llm)}), "
                f"message='{message}'"
            )
            token_rewards_by_llm = [
                r + p for r, p in zip(token_rewards_by_llm, oov_penalties)
            ]

            # ── Step 9：llm_tokens奖励对齐到response_ids ─────────────────────
            # response_ids与llm_ids大部分相同，少数位置切分不同
            # 直接匹配的token直接取奖励，不匹配的按字符span合并处理
            token_rewards = self._align_llm_rewards_to_response_ids(
                token_rewards_by_llm=token_rewards_by_llm,
                llm_ids=llm_ids,
                llm_spans=llm_spans,
                content_ids=content_ids,
                bpe_reward=bpe_reward
            )

            token_rewards = np.array(token_rewards, dtype=np.float32)

            # ── Step 10：长度惩罚 ─────────────────────────────────────────────
            length_ratio = len(message) / max(len(target), 1)

            if length_ratio > self.max_length_ratio:
                length_penalty = -self.penalty_weight * (length_ratio - self.max_length_ratio)
            elif length_ratio < self.min_length_ratio:
                # 使用对称的线性惩罚，替代倒数公式，防止梯度爆炸
                length_penalty = -self.penalty_weight * (self.min_length_ratio - length_ratio)
            else:
                length_penalty = 0.0

            # ── Step 11：EOS token奖励 ────────────────────────────────────────
            # EOS承担序列级质量汇总：bpe_reward（整体翻译质量）+ length_penalty
            # 集中在EOS上，通过GAE向前自然衰减，符合因果逻辑
            eos_reward = float(bpe_reward) + length_penalty

            if has_eos:
                token_rewards = np.append(token_rewards, eos_reward)
            else:
                # 无EOS时，长度惩罚叠加到最后一个content token
                token_rewards[-1] += length_penalty

            # ── Step 12：最终长度校验 ─────────────────────────────────────────
            if len(token_rewards) != total_length:
                raise AssertionError(
                    f"reward length {len(token_rewards)} != "
                    f"response length {total_length}, "
                    f"message='{message}', has_eos={has_eos}, "
                    f"content_length={content_length}"
                )

            reward_list.append(
                torch.tensor(token_rewards, dtype=torch.float32)
            )

        return reward_list

    # =========================================================================
    # 私有方法
    # =========================================================================

    def _compute_ter(
        self,
        hyp_ids: List[int],
        ref_ids: List[int]
    ) -> float:
        """
        计算BPE序列的Token Error Rate。

        TER = edit_distance(hyp, ref) / max(len(hyp), len(ref))
        分母取max防止ref为空时除零，同时对过长hyp施加惩罚。
        """
        dist = editdistance.eval(hyp_ids, ref_ids)
        length = max(len(hyp_ids), len(ref_ids))
        ter = (dist / length) if length > 0 else 1.0
        return float(min(ter, 1.0))

    def _compute_bpe_correct_mask(
        self,
        hyp_ids: List[int],
        ref_ids: List[int]
    ) -> List[float]:
        """
        通过DP回溯，标记hyp中每个BPE token是否与ref对齐正确。
        匹配=1.0，替换/插入/删除=0.0。
        """
        n, m = len(hyp_ids), len(ref_ids)

        if n == 0:
            return []
        if m == 0:
            return [0.0] * n

        # 构建完整DP矩阵（回溯需要完整矩阵，不能只保留两行）
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if hyp_ids[i - 1] == ref_ids[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j - 1],  # 替换
                        dp[i - 1][j],      # 删除hyp[i-1]
                        dp[i][j - 1]       # 插入ref[j-1]
                    )

        # 回溯路径
        correct_mask = [0.0] * n
        i, j = n, m

        while i > 0 and j > 0:
            if hyp_ids[i - 1] == ref_ids[j - 1]:
                correct_mask[i - 1] = 0.2
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                correct_mask[i - 1] = -0.2   # 替换
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                correct_mask[i - 1] = -0.2   # hyp多余token
                i -= 1
            else:
                j -= 1                       # ref多余token，不影响hyp索引

        while i > 0:
            correct_mask[i - 1] = -0.2
            i -= 1

        return correct_mask

    def _get_bpe_char_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        贪心前缀匹配，获取每个BPE subword在原始字符串中的字符级span。
        """
        bpe_ids = self.bpe_tokenizer.encode(text)
        if not bpe_ids:
            return []

        subwords = [self.bpe_tokenizer.ind2char[idx] for idx in bpe_ids]
        spans = []
        cursor = 0

        for sw in subwords:
            start = text.find(sw, cursor)
            if start == -1:
                clean_sw = sw.lstrip("▁").lstrip("##")
                start = text.find(clean_sw, cursor)
                if start == -1:
                    spans.append((cursor, cursor))
                    continue
                sw = clean_sw

            end = start + len(sw)
            spans.append((start, end))
            cursor = end

        return spans

    def _align_bpe_to_llm_tokens(
        self,
        text: str,
        bpe_correct_mask: List[float],
        llm_spans: List[Tuple[int, int]],
        bpe_reward: float
    ) -> List[float]:
        """
        通过字符级span重叠，将BPE subword奖励加权映射到llm_tokens粒度。

        对每个llm token：
          - 找到与其字符span有重叠的所有BPE subword
          - 按重叠字符数加权平均各BPE subword的奖励值
          - 若无重叠，回退使用全局bpe_reward
        """
        bpe_spans = self._get_bpe_char_spans(text)

        if not llm_spans:
            return []

        # bpe_spans和bpe_correct_mask都来自同一text的bpe_tokenizer.encode结果
        # 长度必须严格一致，不一致说明上游逻辑有误
        assert len(bpe_spans) == len(bpe_correct_mask), (
            f"bpe_spans({len(bpe_spans)}) != bpe_correct_mask({len(bpe_correct_mask)}), "
            f"text='{text}'"
        )

        token_rewards = []

        for llm_start, llm_end in llm_spans:
            total_overlap = 0
            weighted_reward = 0.0

            for (bpe_start, bpe_end), bpe_val in zip(bpe_spans, bpe_correct_mask):
                overlap = max(
                    0, min(llm_end, bpe_end) - max(llm_start, bpe_start)
                )
                if overlap > 0:
                    weighted_reward += overlap * bpe_val
                    total_overlap += overlap

            if total_overlap > 0:
                token_rewards.append(weighted_reward / total_overlap)
            else:
                token_rewards.append(bpe_reward)

        return token_rewards

    def _compute_oov_penalty(
        self,
        text: str,
        llm_token_spans: List[Tuple[int, int]],
    ) -> List[float]:
        """
        检测OOV词汇，对对应llm token施加幻觉惩罚。
        OOV判断：词经BPE encode后decode无法还原原词。
        """
        words = text.split(" ")
        word_spans = []
        cursor = 0

        for word in words:
            if not word:
                cursor += 1
                continue

            start = text.find(word, cursor)
            if start == -1:
                cursor += len(word)
                continue
            end = start + len(word)

            try:
                encoded_ids = self.bpe_tokenizer.encode(word)
                if encoded_ids:
                    decoded = "".join(
                        self.bpe_tokenizer.ind2char.get(i, "")
                        for i in encoded_ids
                    ).lstrip("▁").lstrip("##")
                    is_oov = (decoded != word)
                else:
                    is_oov = True
            except Exception:
                is_oov = False

            word_spans.append((start, end, is_oov))
            cursor = end

        penalties = [0.0] * len(llm_token_spans)

        for tok_idx, (tok_start, tok_end) in enumerate(llm_token_spans):
            for (word_start, word_end, is_oov) in word_spans:
                if not is_oov:
                    continue
                overlap = max(
                    0, min(tok_end, word_end) - max(tok_start, word_start)
                )
                if overlap > 0:
                    penalties[tok_idx] = self.hallucination_penalty
                    break

        return penalties

    def _align_llm_rewards_to_response_ids(
        self,
        token_rewards_by_llm: List[float],
        llm_ids: List[int],
        llm_spans: List[Tuple[int, int]],
        content_ids,
        bpe_reward: float
    ) -> List[float]:
        """
        将基于llm_tokens计算的奖励对齐到rollout的content_ids上。

        核心策略：
          - content_ids与llm_ids逐位比较
          - 相同的token：直接取对应奖励（O(1)，无需字符计算）
          - 不同的token：找到content_ids和llm_ids中对应同一字符区间的token组，
                         对content_ids中的每个token，按其覆盖的llm span加权平均

        对齐算法：
          双指针同步推进，检测分歧区间，通过累计字符长度确定分歧结束位置。

        Args:
            token_rewards_by_llm: 与llm_ids对应的奖励列表
            llm_ids:              重新tokenize得到的token id列表
            llm_spans:            llm_ids的offset_mapping
            content_ids:          rollout的content token ids（不含EOS）
            bpe_reward:           无法对齐时的fallback

        Returns:
            与content_ids等长的奖励列表
        """
        content_ids_list = content_ids.tolist() \
            if hasattr(content_ids, 'tolist') else list(content_ids)

        n_resp = len(content_ids_list)
        n_llm = len(llm_ids)

        response_rewards = []
        i = 0  # content_ids指针
        j = 0  # llm_ids指针

        while i < n_resp:
            # ── 情况1：两者当前token相同，直接对应 ──────────────────────────
            if j < n_llm and content_ids_list[i] == llm_ids[j]:
                response_rewards.append(token_rewards_by_llm[j])
                i += 1
                j += 1
                continue

            # ── 情况2：不匹配，找到分歧区间的边界 ───────────────────────────
            # 向前扫描，找到content_ids和llm_ids重新同步的位置
            # 判断标准：两者累计decode字符数相等时，说明覆盖了相同的字符区间
            #
            # 例：content: [14062, 17273, 18530] → "бар" + "ай" + "" → "барай"
            #     llm:     [128986, 127705]       → "бар" + "ай"      → "барай"
            #     字符数累计相等时停止扫描

            # 收集分歧区间内的content token索引
            resp_chunk = [i]
            llm_chunk = []

            if j < n_llm:
                llm_chunk = [j]

            # 累计字符长度（用于同步判断）
            def decode_len(tid):
                s = self.tokenizer.decode([tid], skip_special_tokens=True)
                return len(s)

            resp_char_len = decode_len(content_ids_list[i])
            llm_char_len = decode_len(llm_ids[j]) if j < n_llm else 0

            # 双指针扩展，直到两侧累计字符长度相等（找到同步点）
            max_scan = 8  # 最多向前扫描8个token，防止无限循环
            scan = 0

            while resp_char_len != llm_char_len and scan < max_scan:
                if resp_char_len < llm_char_len and i + len(resp_chunk) < n_resp:
                    # content侧字符不足，向前扩展content
                    next_i = i + len(resp_chunk)
                    resp_chunk.append(next_i)
                    resp_char_len += decode_len(content_ids_list[next_i])
                elif llm_char_len < resp_char_len and j + len(llm_chunk) < n_llm:
                    # llm侧字符不足，向前扩展llm
                    next_j = j + len(llm_chunk)
                    llm_chunk.append(next_j)
                    llm_char_len += decode_len(llm_ids[next_j])
                else:
                    # 无法继续扩展（到达边界），退出
                    break
                scan += 1

            # ── 将llm_chunk的奖励分配给resp_chunk中的每个token ────────────
            if llm_chunk:
                # 收集llm_chunk覆盖的字符span
                chunk_llm_spans = [llm_spans[jj] for jj in llm_chunk]
                chunk_llm_rewards = [token_rewards_by_llm[jj] for jj in llm_chunk]

                # 对resp_chunk中每个token，通过字符位置确定其覆盖的llm span
                # 用累计decode字符长度重建resp token的字符span
                resp_cursor = llm_spans[llm_chunk[0]][0] \
                    if llm_chunk and llm_chunk[0] < len(llm_spans) else 0

                for ri in resp_chunk:
                    rid = content_ids_list[ri]
                    tok_str = self.tokenizer.decode(
                        [rid], skip_special_tokens=True
                    )
                    tok_len = len(tok_str)
                    tok_start = resp_cursor
                    tok_end = resp_cursor + tok_len
                    resp_cursor = tok_end

                    # 在llm_chunk的span中找重叠，加权平均奖励
                    total_overlap = 0
                    weighted = 0.0
                    for (s, e), r in zip(chunk_llm_spans, chunk_llm_rewards):
                        ov = max(0, min(tok_end, e) - max(tok_start, s))
                        if ov > 0:
                            weighted += ov * r
                            total_overlap += ov

                    if total_overlap > 0:
                        response_rewards.append(weighted / total_overlap)
                    else:
                        # 无重叠（如空字符串token）：取llm_chunk奖励均值
                        response_rewards.append(
                            float(np.mean(chunk_llm_rewards))
                        )
            else:
                # llm_ids已耗尽，剩余content token用bpe_reward填充
                for _ in resp_chunk:
                    response_rewards.append(bpe_reward)

            # 推进双指针越过已处理的chunk
            i += len(resp_chunk)
            j += len(llm_chunk)

        # 若llm_ids还有剩余（content_ids先耗尽），不影响结果

        # 双指针对齐后长度必须严格等于content_length
        # 不一致说明分歧区间处理逻辑有误
        assert len(response_rewards) == n_resp, (
            f"response_rewards({len(response_rewards)}) != "
            f"content_length({n_resp})"
        )

        return response_rewards