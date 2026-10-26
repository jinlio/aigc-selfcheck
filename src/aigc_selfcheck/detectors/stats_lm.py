"""路线 C：统计特征路（可解释性的来源）。

特征 1 困惑度：本地小因果 LM（Qwen3.5-2B-Base）逐窗计算条件 NLL，
带上文（窗口起点之前的正文，见 contexts.py）算更稳；AI 文本「过于流畅」，
PPL 系统性偏低，比人写更顺得越多分越高（注意：此处按该直觉方向实现，见 constants 注）。
特征 2 突发性是文档级指标，由 fuse 层用各窗 PPL 计算，不在本类内。
特征 3 中文风格计量向量来自 features.window_features，按人写分布 z 归一化。

C 路总输出：p_C = PPL_WEIGHT_IN_C·p_ppl + (1−…)·p_style
（ALGORITHM 未定合成方式，这里取 PPL 为主、风格为辅，meta 分别透出。）

GPU：批量前向 + 混合精度（默认 bf16，与校准缓存的 checkpoint 原生精度一致；
CPU 上 from_pretrained 同样是 bf16）。批大小按 token 预算自适应，控制 logits 显存（B×L×V）。
"""

from __future__ import annotations

import hashlib
import math
import os

from ..constants import (
    PPL_CALIBRATION,
    PPL_CONTEXT_TOKENS,
    PPL_LOG_MEAN,
    PPL_LOG_STD,
    PPL_WEIGHT_IN_C,
    STYLE_BIAS,
    STYLE_STATS,
)
from ..features import cjk_count, window_features
from ..preprocess import split_sentences
from .base import Detector, DetectorResult, resolve_device

MIN_CJK_FOR_PPL = 10  # 少于该汉字数的窗口 PPL 不可靠，退化为纯风格分
# 单批 token 总量上限（logits 约 0.5 GB @bf16）；显存紧时可用环境变量调小
_BATCH_TOKEN_BUDGET = int(os.environ.get("AIGC_BATCH_TOKEN_BUDGET", "1600"))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def style_score(features: dict[str, float]) -> tuple[float, dict[str, float]]:
    """风格特征 → p_style = σ(w·z + b)，同时返回各特征 z 值供解释层使用。"""
    total = STYLE_BIAS
    z_values: dict[str, float] = {}
    for name, stats in STYLE_STATS.items():
        if name not in features:
            continue
        std = stats["std"] or 1.0
        z = stats["direction"] * (features[name] - stats["mean"]) / std
        z_values[name] = z
        total += stats["weight"] * z
    return _sigmoid(total), z_values


class StatsDetector(Detector):
    path_key = "c"
    uses_context = True
    # v6 起（2026-10-26，train-only 选型）：Qwen3.5-2B-Base 文本塔经
    # AutoModelForCausalLM 加载（multimodal checkpoint 的 qwen3_5_text）。
    # 换模型必须同步重导 constants.PPL_LOG_MEAN/STD。
    model_id = "Qwen/Qwen3.5-2B-Base"
    # 默认基座的 PPL 校准常量；换 model_id 时在 __init__ 里按 PPL_CALIBRATION 覆盖。
    ppl_mean = PPL_LOG_MEAN
    ppl_std = PPL_LOG_STD

    @property
    def cache_id(self) -> str:
        # p_C 依赖校准常量，常量一变分数就变；把指纹并进缓存键，
        # 否则改 constants.py 后旧缓存分仍被命中（v0.1 曾因此用旧常量分评测）。
        digest = hashlib.sha256(
            repr(
                (self.ppl_mean, self.ppl_std, PPL_WEIGHT_IN_C, STYLE_BIAS, sorted(STYLE_STATS.items()))
            ).encode()
        ).hexdigest()[:12]
        return f"{self.model_id}@{digest}"

    def __init__(
        self,
        device: str = "auto",
        model_id: str | None = None,
        max_batch: int = 8,
        dtype: str | None = None,
    ):
        super().__init__(device)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = resolve_device(device)
        if model_id:  # 参考模型可换（P0/P1 实验与 CPU 回退用）；默认是校准时的 2B
            self.model_id = model_id
        self.ppl_mean, self.ppl_std = PPL_CALIBRATION.get(
            self.model_id, (PPL_LOG_MEAN, PPL_LOG_STD)
        )
        self.max_batch = max_batch
        self.token_budget = _BATCH_TOKEN_BUDGET
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=False
        )
        self.dtype = self._pick_dtype(dtype)
        kwargs = {"dtype": self.dtype} if self.dtype is not None else {}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=False, **kwargs
            )
        except TypeError:  # 旧版 transformers 只认 torch_dtype
            kwargs = {"torch_dtype": self.dtype} if self.dtype is not None else {}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=False, **kwargs
            )
        self.model = self.model.to(self.device).eval()

    def _pick_dtype(self, requested: str | None):
        """精度策略：未指定时按设备取——CUDA 用 bf16（与校准缓存口径一致、显存减半），
        MPS 用 fp16，CPU 用 checkpoint 原生精度（None → from_pretrained 默认）。"""
        torch = self._torch
        if requested == "fp32":
            return torch.float32
        if requested == "bf16":
            return torch.bfloat16
        if requested == "fp16":
            return torch.float16
        if self.device.startswith("cuda"):
            # 校准缓存是 checkpoint 原生 bf16（CPU 上 from_pretrained 的默认精度），
            # CUDA 默认也走 bf16 以对齐校准口径；显存也减半。
            return torch.bfloat16
        if self.device == "mps":
            return torch.float16
        return None  # CPU 交给 from_pretrained 默认精度（不强制 fp32）

    def _batch_log_ppl(self, texts: list[str], contexts: list[str]) -> list[float]:
        """批量条件 NLL：只对窗口 token 取均值，上下文只提供条件。

        与逐窗实现逐位同式（logits[p] 预测 p+1；首个窗口 token 无预测者）。
        """
        torch = self._torch
        import torch.nn.functional as F

        encs = []
        for text, context in zip(texts, contexts, strict=True):
            ctx_ids = (
                self.tokenizer.encode(context, add_special_tokens=False)[-PPL_CONTEXT_TOKENS:]
                if context
                else []
            )
            win_ids = self.tokenizer.encode(text, add_special_tokens=False)
            encs.append((ctx_ids, win_ids))

        out = [float("nan")] * len(texts)
        order = [i for i, (_ctx, win) in enumerate(encs) if win]
        pos = 0
        while pos < len(order):
            batch: list[int] = []
            tokens = 0
            while pos < len(order) and len(batch) < self.max_batch:
                i = order[pos]
                length = len(encs[i][0]) + len(encs[i][1])
                if batch and tokens + length > self.token_budget:
                    break
                batch.append(i)
                tokens += length
                pos += 1
            seqs = [encs[i][0] + encs[i][1] for i in batch]
            lmax = max(len(seq) for seq in seqs)
            ids = torch.zeros((len(seqs), lmax), dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for row, seq in enumerate(seqs):
                ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                mask[row, : len(seq)] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                for row, i in enumerate(batch):
                    ctx_len, win_len = len(encs[i][0]), len(encs[i][1])
                    start = max(ctx_len, 1)  # 首个待预测 token 的位置
                    if win_len - (start - ctx_len) <= 0:  # 仅 1 token 的窗口无预测位置
                        continue
                    lp = logits[row, start - 1 : len(seqs[row]) - 1].float()
                    tgt = ids[row, start : len(seqs[row])]
                    out[i] = float(F.cross_entropy(lp, tgt))
            del logits
        return out

    def detect_batch(self, texts: list[str], contexts: list[str] | None = None) -> list[DetectorResult]:
        if contexts is None:
            contexts = [""] * len(texts)
        log_ppls = [float("nan")] * len(texts)
        need = [i for i, text in enumerate(texts) if cjk_count(text) >= MIN_CJK_FOR_PPL]
        if need:
            vals = self._batch_log_ppl([texts[i] for i in need], [contexts[i] for i in need])
            for i, value in zip(need, vals, strict=True):
                log_ppls[i] = value

        results = []
        for text, log_ppl in zip(texts, log_ppls, strict=True):
            has_ppl = log_ppl == log_ppl  # NaN 检查
            if has_ppl:
                ppl = math.exp(log_ppl)
                p_ppl = _sigmoid((self.ppl_mean - log_ppl) / self.ppl_std)
            else:
                ppl, p_ppl = float("nan"), 0.5

            sentences = [s.text.strip() for s in split_sentences(text)]
            features = window_features(text, sentences)
            p_style, z_values = style_score(features)

            score = PPL_WEIGHT_IN_C * p_ppl + (1 - PPL_WEIGHT_IN_C) * p_style
            results.append(
                DetectorResult(
                    score=score,
                    meta={
                        "model": self.model_id,
                        "device": self.device,
                        "p_ppl": p_ppl,
                        "p_style": p_style,
                        "log_ppl": log_ppl,
                        "ppl": ppl,
                        "features": features,
                        "z_values": z_values,
                        "ppl_degraded": not has_ppl,
                    },
                )
            )
        return results

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        """用同一基座模型生成建议；只给高风险窗口少量调用，避免拖慢主报告。"""
        torch = self._torch
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        enc = {key: value.to(self.device) for key, value in enc.items()}
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            try:
                # 基座模型容易续写“检测器提示/改写说明”；遇到空行就停，省时也省清洗
                output = self.model.generate(
                    **enc,
                    **kwargs,
                    stop_strings=["\n\n", "检测器提示", "改写说明"],
                    tokenizer=self.tokenizer,
                )
            except (TypeError, ValueError):
                output = self.model.generate(**enc, **kwargs)
        generated = output[:, enc["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def detect(self, text: str, context: str = "") -> DetectorResult:
        return self.detect_batch([text], [context])[0]
