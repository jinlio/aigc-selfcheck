"""窗口融合、句级热力图与文档级稳健聚合（ALGORITHM §3）。

- 融合 p_fuse = Σ α_i·p_i（α 取 constants.ALPHA，对实际可用的路重归一化，
  支持 --detectors b,c 的纯 CPU 形态）；δ = 三路两两最大差，δ > 0.5 标
  「⚠ 三路分歧」不做硬判定。
- 句级风险 = 包含该句的所有窗口融合分均值（天然平滑单窗噪声）。
- 文档级拒绝全文平均：median（基调）+ P90（最像 AI 的十分位）→ 区间带。
- 疑似段落区间：阈值游程，连续 ≥3 窗 > 0.6 合并。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .constants import (
    ALPHA,
    BAND_ROBUST_HIGH,
    BAND_ROBUST_LOW,
    BAND_TAIL_HIGH,
    BAND_TAIL_LOW,
    BOOST_B,
    BOOST_C,
    BOOST_FLOOR,
    DISAGREE_DELTA,
    HEATMAP_HIGH,
    HEATMAP_LOW,
    HOT_RUN_LENGTH,
    HOT_RUN_THRESHOLD,
)


@dataclass
class FusedWindow:
    window_key: str  # "{block_index}:{window_index}"
    block_index: int
    sentence_ids: list[int]
    text: str
    scores: dict[str, float]  # {"a":…, "b":…, "c":…}，只含实际可用的路
    p_fuse: float
    delta: float
    metas: dict = field(default_factory=dict)  # 路键 → 该路 meta
    meta_prob: float | None = None  # 实验性增强召回的元分类器概率
    enhanced: bool = False  # 是否被增强召回额外标记
    reason_items: list[dict] = field(default_factory=list)  # 白话原因 + 修改建议
    suggestions: list[str] = field(default_factory=list)
    review_note: dict | None = None  # 结构性误伤提示：这段可能不该为了降分而改写
    llm_advice: dict | None = None  # 可选：基座模型针对本段给出的具体建议

    @property
    def disputed(self) -> bool:
        return self.delta > DISAGREE_DELTA

    @property
    def heat(self) -> str:
        if self.p_fuse < HEATMAP_LOW:
            return "low"
        if self.p_fuse > HEATMAP_HIGH:
            return "high"
        return "mid"


def fuse_scores(scores: dict[str, float]) -> tuple[float, float]:
    """可用路的加权重归一化融合 + 两两最大差。单路时 δ=0。

    v11 起：b 路非常确定（≥BOOST_B）且 c 路至少中等确定（≥BOOST_C）时，
    融合分至少抬到 BOOST_FLOOR。规则只在 train 上选定，用于补回 b 路高置信
    但被 c 拉低的文学/口语 AI 窗口；人写段高与跨语域探针不变。
    """
    if not scores:
        return 0.0, 0.0
    active = {k: ALPHA[k] for k in scores if k in ALPHA}
    total_weight = sum(active.values())
    p_fuse = sum(scores[k] * active[k] for k in active) / total_weight
    if scores.get("b", 0.0) >= BOOST_B and scores.get("c", 0.0) >= BOOST_C:
        p_fuse = max(p_fuse, BOOST_FLOOR)
    values = list(scores.values())
    delta = max(values) - min(values) if len(values) >= 2 else 0.0
    return p_fuse, delta


def sentence_risks(sentences_count: int, fused: list[FusedWindow]) -> list[float]:
    """句级风险：包含该句的所有窗口融合分均值；无窗口覆盖的句子取相邻窗口值。"""
    buckets: list[list[float]] = [[] for _ in range(sentences_count)]
    for window in fused:
        for sid in window.sentence_ids:
            if 0 <= sid < sentences_count:
                buckets[sid].append(window.p_fuse)
    risks = [statistics.fmean(b) if b else float("nan") for b in buckets]
    # 补洞：未被任何窗口覆盖（理论上不会发生）或 NaN 的句子，取最近的有限值
    last = 0.0
    for i, r in enumerate(risks):
        if r == r:  # 非 NaN
            last = r
        else:
            risks[i] = last
    return risks


def composite_score(median: float, p90: float, band: str) -> int:
    """展示用 0–100 综合风险分；与区间带一致：低 0–34、中 35–64、高 65–100。"""
    if band == "低":
        pos = max(median / BAND_ROBUST_LOW, p90 / BAND_TAIL_LOW)
        return min(34, max(0, int(round(34 * pos))))
    if band == "中":
        pos = max(median / BAND_ROBUST_HIGH, p90 / BAND_TAIL_HIGH)
        return min(64, max(35, 35 + int(round(29 * pos))))
    pos = max(
        (median - BAND_ROBUST_HIGH) / (1.0 - BAND_ROBUST_HIGH),
        (p90 - BAND_TAIL_HIGH) / (1.0 - BAND_TAIL_HIGH),
        0.0,
    )
    return min(100, max(65, 65 + int(round(35 * pos))))


def doc_level(fused: list[FusedWindow]) -> dict:
    """稳健聚合：median + P90 → 区间带；窗口数不足时降级为单窗直报。"""
    values = sorted(w.p_fuse for w in fused)
    if not values:
        return {"median": 0.0, "p90": 0.0, "score": 0, "band": "低", "n_windows": 0}
    median = statistics.median(values)
    p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    if median < BAND_ROBUST_LOW and p90 < BAND_TAIL_LOW:
        band = "低"
    elif median >= BAND_ROBUST_HIGH or p90 >= BAND_TAIL_HIGH:
        band = "高"
    else:
        band = "中"
    return {
        "median": median,
        "p90": p90,
        "score": composite_score(median, p90, band),
        "band": band,
        "n_windows": len(values),
    }


@dataclass
class HotInterval:
    block_index: int
    sentence_ids: list[int]
    mean_score: float
    window_keys: list[str]


def hot_intervals(fused: list[FusedWindow]) -> list[HotInterval]:
    """阈值游程变点：连续 ≥HOT_RUN_LENGTH 窗 p_fuse > HOT_RUN_THRESHOLD 合并为疑似区间。

    窗口序列按 (block_index, 窗口序) 排列；跨 block 的游程照算
    （相邻段落同热本身就是证据），但 block 边界记录在案供报告标注。
    """
    ordered = sorted(fused, key=lambda w: (w.block_index, w.window_key))
    runs: list[list[FusedWindow]] = []
    current: list[FusedWindow] = []
    for window in ordered:
        if window.p_fuse > HOT_RUN_THRESHOLD:
            current.append(window)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    intervals = []
    for run in runs:
        if len(run) < HOT_RUN_LENGTH:
            continue
        intervals.append(
            HotInterval(
                block_index=run[0].block_index,
                sentence_ids=sorted({sid for w in run for sid in w.sentence_ids}),
                mean_score=statistics.fmean(w.p_fuse for w in run),
                window_keys=[w.window_key for w in run],
            )
        )
    return intervals


def burstiness_from_windows(window_ppls: list[float | None]) -> float:
    """文档级突发性。窗口 PPL 缺失（None/NaN/未启用 C 路）时返回 NaN 由上层忽略。"""
    clean: list[float] = []
    for value in window_ppls:
        if value is None:
            continue
        try:
            ppl = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(ppl) and ppl > 0:
            clean.append(ppl)
    if len(clean) < 2:
        return float("nan")
    mean = statistics.fmean(clean)
    return statistics.pstdev(clean) / mean
