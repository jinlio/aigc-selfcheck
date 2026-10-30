"""修改优先级：用「移除某一段后，有多少窗从超标回到达标」估计贡献。

旧实现按文档总分差排序（`impact = base_score - score_without`）。文档分是
0–100 的压缩刻度，单段改动通常只动 0–1 分，于是 `impact` 常年是
`[1, 0, 0, 0, 0]`——五个候选里四个显示「改了一点用没有」，排序退化成
按 `max_risk` 排队。这里换成**窗口计数**：删掉这一段，有多少窗从
「超过档位门槛」回到「不超标」。计数是离散的、可加的，不会因为刻度压缩
而丢失区分度，也直接对应报告要给用户的「还需 N 窗」。
"""

from __future__ import annotations

from dataclasses import replace

from .constants import BAND_SETS, DEFAULT_BAND_SET, BandSet
from .fuse import FusedWindow, doc_level


def _unmet(values: list[float], band_set: BandSet) -> int:
    """判定「超标」的窗数：达到低档门槛（`robust_low`）及以上的窗口数。

    这是 `band_distance` 的口径：要进低带，需要 median < robust_low，
    等价于「低于门槛的窗数过半」。所以「清掉一个超标窗」就直接推进这个缺口。
    """
    return sum(1 for value in values if value >= band_set.robust_low)


def revision_plan(
    fused: list[FusedWindow],
    base_doc: dict,
    section_for_block: dict[int, str],
    limit: int = 5,
    band_set: BandSet | None = None,
) -> list[dict]:
    """返回按影响排序的修改候选；impact 是单段敏感性估计，不是分数承诺。"""
    bs = band_set or BAND_SETS[DEFAULT_BAND_SET]
    base_score = int(base_doc.get("score", 0))
    base_unmet = _unmet([window.p_fuse for window in fused], bs)
    items: list[dict] = []
    for block in sorted({window.block_index for window in fused}):
        block_windows = [window for window in fused if window.block_index == block]
        if not block_windows:
            continue
        reduced = [
            replace(window, p_fuse=0.0) if window.block_index == block else window
            for window in fused
        ]
        without = doc_level(reduced, bs)
        cleared = base_unmet - _unmet([window.p_fuse for window in reduced], bs)
        top = max(block_windows, key=lambda window: window.p_fuse)
        items.append(
            {
                "block": block,
                "section": section_for_block.get(block, ""),
                "impact": max(0, base_score - int(without.get("score", 0))),
                "cleared_windows": cleared,
                "block_windows": len(block_windows),
                "score_without": int(without.get("score", 0)),
                "band_without": without.get("band", ""),
                "n_windows": len(block_windows),
                "max_risk": top.p_fuse,
                "text": top.text,
                "reason_items": top.reason_items,
                "suggestions": top.suggestions,
                "review_note": top.review_note,
                "llm_advice": top.llm_advice,
            }
        )

    # 先按「清掉的超标窗数」，再按段内最高风险——分数只作为最后的平局裁决
    items.sort(
        key=lambda item: (item["cleared_windows"], item["max_risk"], item["impact"]),
        reverse=True,
    )
    return items[:limit]
