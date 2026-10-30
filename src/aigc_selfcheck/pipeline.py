"""分析管线：读取 → 结构过滤 → 分句滑窗 → 三路打分（带缓存）→ 融合 → 报告数据。

对 cli 与测试共用。检测器以参数注入，便于 mock（黄金测试不下载模型）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import meta as meta_layer
from .advice import review_note_for
from .cache import ScoreCache
from .constants import BAND_SETS, DEFAULT_BAND_SET, MTLD_FLOOR, BandSet
from .contexts import preceding_context
from .detectors import Detector
from .docio import read_document
from .explain import (
    dedupe_reason_items,
    reason_items_for,
    sentence_reason_items,
    suggestions_for,
)
from .fuse import (
    FusedWindow,
    apply_mtld_arbitration,
    band_distance,
    boost_floor_rule,
    burstiness_from_windows,
    doc_level,
    fuse_scores,
    hot_intervals,
    sentence_risks,
)
from .llm_advice import generate_for_windows
from .plan import revision_plan
from .preprocess import classify_block, detectable, prepare_block
from .sections import build_sections

MIN_WINDOW_CJK = 6  # 汉字过少的窗口（如纯编号行）三路全跳，记中性分


def _band_summary(values: list[FusedWindow], band_set: BandSet) -> dict:
    """一套口径下的档位摘要，供报告并列显示另一套口径。"""
    stats = doc_level(values, band_set)
    return {
        "label": band_set.label,
        "band": stats["band"],
        "score": stats["score"],
        "median": stats["median"],
        "p90": stats["p90"],
        "thresholds": {
            "robust_low": band_set.robust_low,
            "tail_low": band_set.tail_low,
            "robust_high": band_set.robust_high,
            "tail_high": band_set.tail_high,
        },
    }


def match_excluded_sections(sections: list[dict], patterns: list[str]) -> set[str]:
    """把 `--exclude-heading` 的模式展开成实际命中的章节标题。

    用子串匹配而不是正则：用户的直觉是「排除所有含『外文翻译』的标题」，
    正则元字符（括号、点）在真实标题里到处都是，容易写出意料之外的匹配。
    """
    if not patterns:
        return set()
    hits: set[str] = set()
    for section in sections:
        title = section.get("title") or ""
        if any(pattern in title for pattern in patterns if pattern):
            hits.add(title)
    return hits


@dataclass
class SentenceView:
    """block 内句子的展示与定位信息。"""

    global_id: int
    text: str
    start: int
    end: int
    risk: float = 0.0
    disputed: bool = False
    enhanced: bool = False
    reasons: list[str] = field(default_factory=list)
    reason_items: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    section: str = ""
    review_note: dict | None = None
    llm_advice: dict | None = None


@dataclass
class BlockView:
    index: int
    kind: str
    text: str
    detectable: bool
    note: str = ""  # 非检测块的解释，如「参考文献：已跳过」
    sentences: list[SentenceView] = field(default_factory=list)


@dataclass
class AnalysisResult:
    source: str
    blocks: list[BlockView]
    fused: list[FusedWindow]
    doc: dict
    burstiness: float
    hot: list
    duration_s: float
    detector_ids: dict[str, str]  # path_key → model_id
    warnings: list[str] = field(default_factory=list)
    enhanced: bool = False
    meta_info: dict = field(default_factory=dict)
    sections: list[dict] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    llm_advice_info: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def top_sentences(self, n: int = 20) -> list[dict]:
        items = []
        for block in self.blocks:
            for sent in block.sentences:
                if sent.text.strip():
                    items.append(
                        {
                            "block": block.index,
                            "risk": sent.risk,
                            "text": sent.text.strip(),
                            "reasons": sent.reasons,
                            "reason_items": sent.reason_items,
                            "suggestions": sent.suggestions,
                            "enhanced": sent.enhanced,
                            "section": sent.section,
                            "review_note": sent.review_note,
                            "llm_advice": sent.llm_advice,
                        }
                    )
        items.sort(key=lambda item: item["risk"], reverse=True)
        return items[:n]


def _json_safe(value):
    """NaN/Inf → None，保证缓存与内联 JSON 合法。"""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


SKIP_NOTES = {
    "heading": "标题：不参与检测",
    "toc": "目录条目：不参与检测",
    "metadata": "文档信息：不参与检测",
    "non_zh": "非中文段落（如英文摘要）：中文检测器不适用，已跳过",
    "table": "表格：结构化内容，已跳过",
    "list": "列表条目：碎片化结构，已跳过",
    "code": "代码块：已跳过",
    "reference": "参考文献：格式化文本，已跳过",
    "formula": "公式/符号密集段：已跳过",
    "caption": "图表注：已跳过",
    "quote": "直接引文：引用他人文风，已跳过",
}


def analyze_document(
    path: str | Path,
    detectors: list[Detector],
    cache: ScoreCache | None = None,
    enhanced: bool = False,
    profile: dict | None = None,
    llm_advice: bool = False,
    band_set: BandSet | None = None,
    exclude_headings: list[str] | None = None,
) -> AnalysisResult:
    """跑完整条管线。

    `band_set` 决定用哪套档位阈值（默认学术口径，见 constants.BAND_SETS）。
    `exclude_headings` 里的标题及其下属内容不计入**文档级**档位，但仍出热力图
    与章节地图——外文翻译/附录这类「不是作者原创正文」的部分，混进全文会
    把正文水位整个抬高（云璟论文实测附录占 61% 窗口）。
    """
    bs = band_set or BAND_SETS[DEFAULT_BAND_SET]
    timings: dict[str, float] = {}
    started = time.time()
    blocks = [classify_block(b) for b in read_document(path)]
    timings["read"] = round(time.time() - started, 2)

    # ── 分句与滑窗（全局句子编号跨 block 递增）──────────────────────────
    block_views: list[BlockView] = []
    global_sentences: list[SentenceView] = []  # 仅可检测块的句子，按文档顺序
    window_jobs: list[tuple[int, int, str, str]] = []  # (block_index, window_index, text, context)

    for block in blocks:
        if not detectable(block):
            block_views.append(
                BlockView(index=block.index, kind=block.kind, text=block.text, detectable=False, note=SKIP_NOTES.get(block.kind, "已跳过"))
            )
            continue
        sentences, windows = prepare_block(block)
        view = BlockView(index=block.index, kind=block.kind, text=block.text, detectable=True)
        block_first_global = len(global_sentences)
        for local_id, sent in enumerate(sentences):
            sview = SentenceView(
                global_id=block_first_global + local_id,
                text=sent.text,
                start=sent.start,
                end=sent.end,
            )
            view.sentences.append(sview)
            global_sentences.append(sview)
        block_views.append(view)
        for win_index, win in enumerate(windows):
            context = preceding_context(block.text, win.start)
            window_jobs.append((block.index, win_index, win.text, context))

    # 窗口句子 id 映射到全局句子（按 block 顺序重放 prepare_block 的分句结果）
    window_sentence_ids: dict[tuple[int, int], list[int]] = {}
    cursor = 0
    for block in blocks:
        if not detectable(block):
            continue
        sentences, windows = prepare_block(block)
        for win_index, win in enumerate(windows):
            window_sentence_ids[(block.index, win_index)] = [
                cursor + local for local in win.sentence_ids
            ]
        cursor += len(sentences)

    # ── 三路打分（带缓存；按路批量前向，GPU 上一次处理整篇窗口）──
    scores_by_job: list[dict] = [{} for _ in window_jobs]
    metas_by_job: list[dict] = [{} for _ in window_jobs]
    for detector in detectors:
        pending: list[int] = []
        for i, (_block, _win, text, context) in enumerate(window_jobs):
            cache_context = context if detector.uses_context else ""
            cached = (
                cache.get(detector.path_key, detector.cache_id, text, context=cache_context)
                if cache
                else None
            )
            if cached is not None and "score" in cached:
                scores_by_job[i][detector.path_key] = float(cached["score"])
                metas_by_job[i][detector.path_key] = cached.get("meta") or {}
            else:
                pending.append(i)
        if pending:
            results = detector.detect_batch(
                [window_jobs[i][2] for i in pending],
                [window_jobs[i][3] for i in pending],
            )
            for i, result in zip(pending, results, strict=True):
                scores_by_job[i][detector.path_key] = result.score
                metas_by_job[i][detector.path_key] = result.meta
                if cache:
                    cache.put(
                        detector.path_key,
                        detector.cache_id,
                        window_jobs[i][2],
                        _json_safe({"score": result.score, "meta": result.meta}),
                        context=window_jobs[i][3] if detector.uses_context else "",
                    )

    fused: list[FusedWindow] = []
    for i, (block_index, win_index, text, _context) in enumerate(window_jobs):
        scores = scores_by_job[i]
        p_fuse, delta = fuse_scores(scores)
        fused.append(
            FusedWindow(
                window_key=f"{block_index}:{win_index}",
                block_index=block_index,
                sentence_ids=window_sentence_ids.get((block_index, win_index), []),
                text=text,
                scores=scores,
                p_fuse=p_fuse,
                delta=delta,
                metas=metas_by_job[i],
                floor_rule=boost_floor_rule(scores, p_fuse),
            )
        )
    timings["detect"] = round(time.time() - started - sum(timings.values()), 2)

    # ── v12：b 高 / c 低的分歧区用 char_mtld 裁决（只抬分）────────────
    # 必须在生成理由之前施加，理由里的 risk 才与实际展示分一致。
    if apply_mtld_arbitration(fused):
        for window in fused:
            if window.floor_rule is None and abs(window.p_fuse - MTLD_FLOOR) < 1e-12:
                window.floor_rule = "mtld_arbitration"
    timings["fuse"] = round(time.time() - started - sum(timings.values()), 2)

    # ── 白话解释与修改建议（窗口与句级共用，避免重复计算）────────────
    for window in fused:
        items = dedupe_reason_items(
            reason_items_for(
                window.metas.get("c") or {},
                text=window.text,
                risk=window.p_fuse,
            )
        )
        window.reason_items = items
        window.suggestions = suggestions_for(items)

    # ── 实验性增强召回（默认关闭；不改变 p_fuse / 文档带）────────────
    warnings: list[str] = []
    meta_info: dict = {}
    enhanced_active = False
    if enhanced:
        try:
            meta_info = meta_layer.apply_enhanced(fused)
            enhanced_active = True
        except Exception as exc:  # noqa: BLE001 —— 增强模式失败不应阻断主报告
            warnings.append(f"增强召回已跳过：{type(exc).__name__}: {exc}")

    # ── 句级风险与解释理由 ────────────────────────────────────────────
    risks = sentence_risks(len(global_sentences), fused)
    for sview in global_sentences:
        sview.risk = risks[sview.global_id]
    # 每句取覆盖它的融合分最高的窗口，复用该窗已生成的白话理由与建议
    best_window: dict[int, FusedWindow] = {}
    for window in fused:
        for sid in window.sentence_ids:
            if sid not in best_window or window.p_fuse > best_window[sid].p_fuse:
                best_window[sid] = window
    enhanced_sids = {sid for w in fused if w.enhanced for sid in w.sentence_ids}
    for sview in global_sentences:
        window = best_window.get(sview.global_id)
        if window is None:
            continue
        sview.disputed = window.disputed
        sview.enhanced = sview.global_id in enhanced_sids
        items = sentence_reason_items(
            window.metas.get("c") or {},
            sview.text,
            sview.risk,
        )
        sview.reason_items = items
        sview.suggestions = suggestions_for(items)
        sview.reasons = [item["why"] for item in items]

    # ── 章节聚合与结构性误伤提示 ──────────────────────────────────────
    sections, section_for_block = build_sections(block_views, fused, bs)
    for window in fused:
        window.review_note = review_note_for(
            window, section_for_block.get(window.block_index, "")
        )
    for sview in global_sentences:
        window = best_window.get(sview.global_id)
        if window is None:
            continue
        sview.section = section_for_block.get(window.block_index, "")
        sview.review_note = window.review_note

    # ── 可选：大模型具体建议（只给最高风险的少量窗口）──────────────────
    llm_info: dict = {}
    if llm_advice:
        advice, llm_info = generate_for_windows(fused, detectors)
        for window in fused:
            window.llm_advice = advice.get(window.window_key)
        # 同一窗口只把改写示例挂到风险最高的那句，避免清单里整段建议重复出现
        representative: dict[str, SentenceView] = {}
        for sview in global_sentences:
            window = best_window.get(sview.global_id)
            if window is None or not window.llm_advice:
                continue
            current = representative.get(window.window_key)
            if current is None or sview.risk > current.risk:
                representative[window.window_key] = sview
        for sview in global_sentences:
            window = best_window.get(sview.global_id)
            if window is None:
                continue
            sview.llm_advice = (
                window.llm_advice if representative.get(window.window_key) is sview else None
            )
        if not llm_info.get("enabled"):
            warnings.append(f"大模型建议已跳过：{llm_info.get('reason', '不可用')}")

    # ── 文档级聚合与修改优先级 ────────────────────────────────────────
    window_ppls = [
        (w.metas.get("c") or {}).get("ppl", float("nan")) if "c" in w.metas else float("nan")
        for w in fused
    ]

    # 分部分统计：exclude_headings 命中的标题及其下属内容不计入文档级档位。
    # 典型用法是排除「外文翻译」「附录」「致谢」——这些不是作者原创正文，
    # 却常常占掉一半以上窗口，把正文水位整个淹没。
    excluded_titles = match_excluded_sections(sections, exclude_headings or [])
    kept = [w for w in fused if section_for_block.get(w.block_index) not in excluded_titles]
    doc = doc_level(kept, bs)
    partial = None
    if excluded_titles:
        partial = {
            "excluded_titles": sorted(excluded_titles),
            "n_windows_excluded": len(fused) - len(kept),
        }
    doc["band_distance"] = band_distance([w.p_fuse for w in kept], bs, target="低")
    # 档位边界要能在报告里原样写出来：用户看到「中」时必须知道「中」是怎么划的
    doc["band_thresholds"] = {
        "robust_low": bs.robust_low,
        "tail_low": bs.tail_low,
        "robust_high": bs.robust_high,
        "tail_high": bs.tail_high,
        "min_windows": bs.min_windows,
    }
    # 另一套口径只作对照，不覆盖主口径——用户要能同时看到「学术:低 / 通用:中」
    doc["alternates"] = {
        name: _band_summary(kept, other)
        for name, other in BAND_SETS.items()
        if name != bs.name
    }
    doc["partial"] = partial

    plan = [
        item
        for item in revision_plan(fused, doc, section_for_block, limit=8, band_set=bs)
        if item["cleared_windows"] > 0 or item["max_risk"] >= 0.75
    ][:5]
    timings["aggregate"] = round(time.time() - started - sum(timings.values()), 2)
    result = AnalysisResult(
        source=str(path),
        blocks=block_views,
        fused=fused,
        doc=doc,
        burstiness=burstiness_from_windows(window_ppls),
        hot=hot_intervals(fused),
        duration_s=time.time() - started,
        detector_ids={d.path_key: d.model_id for d in detectors},
        enhanced=enhanced_active,
        meta_info=meta_info,
        sections=sections,
        plan=plan,
        profile=profile or {},
        llm_advice_info=llm_info,
        timings=timings,
    )
    result.warnings.extend(warnings)

    # 汉字过少的窗口三路打分意义有限，若整篇没有可用窗口要给出警告
    if not fused:
        result.warnings.append("没有产出任何检测窗口：文档可能全是结构内容（表格/参考文献/公式）。")
    return result
