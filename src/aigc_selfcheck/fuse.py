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
    BAND_SETS,
    BOOST_B,
    BOOST_C,
    BOOST_FLOOR,
    DEFAULT_BAND_SET,
    DISAGREE_DELTA,
    HEATMAP_HIGH,
    HEATMAP_LOW,
    HOT_RUN_LENGTH,
    HOT_RUN_THRESHOLD,
    MTLD_B_HI,
    MTLD_C_LO,
    MTLD_FLOOR,
    MTLD_HI,
    MTLD_LO,
    BandSet,
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
    # 被下限规则抬过分的窗口记下规则名（b_lane_boost / mtld_arbitration）。
    # 这类窗口的分数是个门槛值、不是测量值，报告里必须标出来——
    # 云璟论文 333 窗里有 44 窗被 b 路增强钉在正好 0.800。
    floor_rule: str | None = None

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


def boost_floor_rule(scores: dict[str, float], p_fuse: float) -> str | None:
    """判断融合分是否被 v11 的 b 路下限规则抬过；抬过就返回规则名。

    报告需要区分「测出来的 0.80」和「抬上来的 0.80」。两者在热力图里长得
    一模一样，但后者只是「b 路很确定、c 路至少中等」这条规则的产物，
    不能当成精确定量读。
    """
    if (
        scores.get("b", 0.0) >= BOOST_B
        and scores.get("c", 0.0) >= BOOST_C
        and abs(p_fuse - BOOST_FLOOR) < 1e-12
    ):
        return "b_lane_boost"
    return None


def fuse_scores(scores: dict[str, float]) -> tuple[float, float]:
    """可用路的加权重归一化融合 + 两两最大差。单路时 δ=0。

    v11 起：b 路非常确定（≥BOOST_B）且 c 路至少中等确定（≥BOOST_C）时，
    融合分至少抬到 BOOST_FLOOR。规则只在 train 上选定，用于补回 b 路高置信
    但被 c 拉低的文学/口语 AI 窗口；人写段高与跨语域探针不变。

    注意：本函数只拿到三路分数，无法施加 v12 的 mtld 分歧裁决
    （`apply_mtld_arbitration`），因为那需要窗口特征。产品路径在
    `pipeline` 里对 `fused` 列表调用后者；离线评测走 `cv_evaluate.fused`。
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


def mtld_arbitration(
    p_fuse: float,
    scores: dict[str, float],
    features: dict[str, float] | None,
) -> float:
    """v12：b 高 / c 低的分歧区用 char_mtld 裁决（只抬分、不压分）。

    命中条件 `b≥MTLD_B_HI`、`c<MTLD_C_LO`、`MTLD_LO≤char_mtld≤MTLD_HI`
    时把融合分抬到至少 MTLD_FLOOR。上界挡住文言/古白话（mtld 常 >160）。
    缺特征（旧缓存/异常路径）时原样返回，行为与 v11 一致。
    """
    if not features:
        return p_fuse
    if scores.get("b", 0.0) < MTLD_B_HI or scores.get("c", 0.0) >= MTLD_C_LO:
        return p_fuse
    mtld = features.get("char_mtld")
    if mtld is None or mtld != mtld:
        return p_fuse
    if MTLD_LO <= float(mtld) <= MTLD_HI:
        return max(p_fuse, MTLD_FLOOR)
    return p_fuse


def apply_mtld_arbitration(fused: list[FusedWindow]) -> int:
    """就地施加 mtld 分歧裁决；返回被抬分的窗口数。"""
    changed = 0
    for window in fused:
        meta_c = window.metas.get("c") or {}
        new_value = mtld_arbitration(window.p_fuse, window.scores, meta_c.get("features"))
        if new_value > window.p_fuse:
            window.p_fuse = new_value
            changed += 1
    return changed


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


def composite_score(median: float, p90: float, band: str, band_set: BandSet | None = None) -> int:
    """展示用 0–100 综合风险分；与区间带一致：低 0–34、中 35–64、高 65–100。

    每段的 `pos` 都在**本段自己的区间**里归一化：低段从 0 到 1 映射到 0–34，
    中段从 (robust_low, tail_low) 到 (robust_high, tail_high) 映射到 35–64，
    高段从高档门槛到 1.0 映射到 65–100。所以跨带边界是连续的（34→35），
    不会再出现 v0.2.1 里 34 直接跳到 57 的断崖——那个断崖让中带实际只有
    55–64 十个格子，改写收益在分数上完全看不见。
    """
    bs = band_set or BAND_SETS[DEFAULT_BAND_SET]
    if band == "低":
        span_m = bs.robust_low or 1.0
        span_p = bs.tail_low or 1.0
        pos = max(median / span_m, p90 / span_p, 0.0)
        return min(34, max(0, int(round(34 * pos))))
    if band in ("中", "样本不足"):
        # 「样本不足」不判带，但分数仍要有个位置：按中段算，报告里单独标注。
        span_m = bs.robust_high - bs.robust_low
        span_p = bs.tail_high - bs.tail_low
        pos_m = (median - bs.robust_low) / span_m if span_m > 0 else 0.0
        pos_p = (p90 - bs.tail_low) / span_p if span_p > 0 else 0.0
        pos = min(1.0, max(0.0, max(pos_m, pos_p)))
        return min(64, max(35, 35 + int(round(29 * pos))))
    pos = max(
        (median - bs.robust_high) / (1.0 - bs.robust_high) if bs.robust_high < 1 else 1.0,
        (p90 - bs.tail_high) / (1.0 - bs.tail_high) if bs.tail_high < 1 else 1.0,
        0.0,
    )
    return min(100, max(65, 65 + int(round(35 * pos))))


def band_distance(values: list[float], band_set: BandSet, target: str = "低") -> dict:
    """离目标带还差多少窗——比 0–100 分更能指导「还值不值得改」。

    「低」带要求 median<robust_low 且 p90<tail_low。median 过半即可、p90 要
    九成窗口过关，所以两个条件是「需要多少个窗口低于阈值」，取更紧的那个。
    """
    n = len(values)
    if n == 0:
        return {"target": target, "reachable": False, "n_windows": 0}
    if target == "低":
        need_median = n // 2 + 1
        need_p90 = n - int(round(0.1 * n)) + 1
        threshold_median, threshold_p90 = band_set.robust_low, band_set.tail_low
    elif target == "中":
        # 出低带即进中带：只要 median 或 p90 有一个过线。
        need_median = n // 2 + 1
        need_p90 = n - int(round(0.1 * n)) + 1
        threshold_median, threshold_p90 = band_set.robust_low, band_set.tail_low
    else:
        return {"target": target, "reachable": False, "n_windows": n}
    have_median = sum(1 for v in values if v < threshold_median)
    have_p90 = sum(1 for v in values if v < threshold_p90)
    return {
        "target": target,
        "band_set": band_set.name,
        "n_windows": n,
        "median": {"threshold": threshold_median, "need": need_median, "have": have_median, "gap": max(0, need_median - have_median)},
        "p90": {"threshold": threshold_p90, "need": need_p90, "have": have_p90, "gap": max(0, need_p90 - have_p90)},
        # 两个条件必须同时满足，缺口取大的
        "gap": max(max(0, need_median - have_median), max(0, need_p90 - have_p90)),
        "reachable": max(max(0, need_median - have_median), max(0, need_p90 - have_p90)) == 0,
    }


def doc_level(
    fused: list[FusedWindow], band_set: BandSet | None = None
) -> dict:
    """稳健聚合：median + P90 → 区间带；窗口数不足时降级为「样本不足」。"""
    bs = band_set or BAND_SETS[DEFAULT_BAND_SET]
    values = sorted(w.p_fuse for w in fused)
    if not values:
        return {
            "median": 0.0,
            "p90": 0.0,
            "score": 0,
            "band": "低",
            "n_windows": 0,
            "band_set": bs.name,
        }
    median = statistics.median(values)
    p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    band = bs.band_or_insufficient(median, p90, len(values))
    return {
        "median": median,
        "p90": p90,
        "score": composite_score(median, p90, band, bs),
        "band": band,
        "n_windows": len(values),
        "band_set": bs.name,
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
