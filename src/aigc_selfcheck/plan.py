"""修改优先级：用「移除某一段后文档分如何变化」估计每段的贡献。"""

from __future__ import annotations

from dataclasses import replace

from .fuse import FusedWindow, doc_level


def revision_plan(
    fused: list[FusedWindow],
    base_doc: dict,
    section_for_block: dict[int, str],
    limit: int = 5,
) -> list[dict]:
    """返回按影响排序的修改候选；impact 是单段敏感性估计，不是分数承诺。"""
    base_score = int(base_doc.get("score", 0))
    items: list[dict] = []
    for block in sorted({window.block_index for window in fused}):
        block_windows = [window for window in fused if window.block_index == block]
        if not block_windows:
            continue
        reduced = [
            replace(window, p_fuse=0.0) if window.block_index == block else window
            for window in fused
        ]
        without = doc_level(reduced)
        top = max(block_windows, key=lambda window: window.p_fuse)
        items.append(
            {
                "block": block,
                "section": section_for_block.get(block, ""),
                "impact": max(0, base_score - int(without.get("score", 0))),
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

    items.sort(key=lambda item: (item["impact"], item["max_risk"]), reverse=True)
    return items[:limit]
