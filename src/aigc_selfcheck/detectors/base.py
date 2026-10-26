"""检测器统一接口与公共工具（可插拔的检测器集成层）。

所有检测路实现同一协议：detect(text, context) -> DetectorResult。
score 语义统一为「AI 生成概率」∈ [0,1]，融合层不感知路的内部实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectorResult:
    score: float  # AI 概率 ∈ [0,1]
    meta: dict = field(default_factory=dict)  # 路级附加信息，供报告详情与调试


class Detector(ABC):
    """检测路基类。重型依赖（torch/transformers）在子类 __init__ 内 lazy import。"""

    path_key: str = ""  # a | b | c
    model_id: str = ""
    uses_context: bool = False  # 分数依赖前窗上下文的路（C 路）置 True；缓存键随上下文变化

    def __init__(self, device: str = "auto"):
        self.device_arg = device

    @property
    def cache_id(self) -> str:
        """分数缓存的模型标识。分数依赖外部常量的路（如 C 路的校准常量）
        必须覆盖此属性把常量指纹并进来——否则常量更新后旧缓存分仍会被命中。"""
        return self.model_id

    @abstractmethod
    def detect(self, text: str, context: str = "") -> DetectorResult: ...

    def detect_batch(self, texts: list[str], contexts: list[str] | None = None) -> list[DetectorResult]:
        """批量打分默认实现：逐条调用 detect。子类覆盖为真批处理（GPU 上快数倍）。"""
        if contexts is None:
            contexts = [""] * len(texts)
        return [self.detect(text, context=context) for text, context in zip(texts, contexts, strict=True)]

    def close(self) -> None:  # noqa: B027 —— 可选的清理钩子，不需要每路都实现
        """子类按需覆盖的资源清理钩子，默认无操作。"""


def resolve_device(device: str = "auto") -> str:
    """torch 设备解析：cuda > mps > cpu；显式指定时不做探测。"""
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def softmax(scores: np.ndarray) -> np.ndarray:
    """一维向量或二维（逐行）softmax。"""
    if scores.ndim == 2:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def ai_label_index(id2label: dict) -> int:
    """从模型 config 的 id2label 里找「AI 类」的索引。

    已知标签形如 human/AI、LABEL_0/LABEL_1。启发式找不到时按约定取 1，
    smoke.py 会用已知 AI 文本实测极性做二次确认（M0 退出标准）。
    """
    for idx, label in (id2label or {}).items():
        low = str(label).lower()
        if any(token in low for token in ("ai", "machine", "generat", "synthetic", "fake", "llm")):
            return int(idx)
    return 1


def softmax_ai_score(logits: np.ndarray, ai_index: int) -> float:
    """logits → AI 类概率。"""
    return float(softmax(logits)[ai_index])


def chunk_long_text(tokenizer, text: str, max_length: int = 512, half: int = 256) -> list[str]:
    """超长文本按前后各 half token 切成两段（ALGORITHM §2.1：扫到一角就算暴露，取 max）。

    返回值保证至少一段。切分用 decode→re-encode，让每段自带特殊符号。
    """
    enc = tokenizer(text, truncation=True, max_length=max_length)
    ids = enc["input_ids"]
    if len(ids) <= max_length:
        return [text]
    head = tokenizer.decode(ids[1 : 1 + half], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-1 - half : -1], skip_special_tokens=True)
    return [head, tail]
