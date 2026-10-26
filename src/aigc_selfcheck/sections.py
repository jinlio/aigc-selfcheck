"""章节级风险聚合：把窗口映射到标题，输出每章的 median/P90/综合分。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .features import cjk_count
from .fuse import FusedWindow, doc_level

if TYPE_CHECKING:
    from .pipeline import BlockView

_METHOD_SECTION = re.compile(
    r"(方法|实验|结果|讨论|结论|method|experiment|result|discussion|conclusion)",
    re.IGNORECASE,
)


def build_sections(
    blocks: list[BlockView], fused: list[FusedWindow]
) -> tuple[list[dict], dict[int, str]]:
    """返回 (章节摘要列表, block_index → 章节标题)。"""
    section_for_block: dict[int, str] = {}
    ordered: list[dict] = []
    current = {"title": "文首", "start_block": blocks[0].index if blocks else 0, "blocks": []}

    for block in blocks:
        if block.kind == "heading":
            if current["blocks"]:
                ordered.append(current)
            current = {
                "title": block.text.strip() or "未命名章节",
                "start_block": block.index,
                "blocks": [],
            }
            continue
        current["blocks"].append(block)
        section_for_block[block.index] = current["title"]
    if current["blocks"]:
        ordered.append(current)

    windows_by_section: dict[str, list[FusedWindow]] = {}
    for window in fused:
        title = section_for_block.get(window.block_index)
        if title is not None:
            windows_by_section.setdefault(title, []).append(window)

    sections: list[dict] = []
    for item in ordered:
        windows = windows_by_section.get(item["title"], [])
        if not windows:
            continue
        doc = doc_level(windows)
        method_like = bool(_METHOD_SECTION.search(item["title"]))
        if method_like and doc["band"] == "高":
            hint = "方法/结果/结论段易误伤：先人工确认，不要为了降分删改实验细节或关键结论。"
        elif doc["band"] == "高":
            hint = "优先复核这一章。"
        elif doc["band"] == "中":
            hint = "部分窗口接近红线。"
        else:
            hint = ""
        sections.append(
            {
                "title": item["title"],
                "start_block": item["start_block"],
                "end_block": item["blocks"][-1].index,
                "n_windows": doc["n_windows"],
                "median": doc["median"],
                "p90": doc["p90"],
                "score": doc["score"],
                "band": doc["band"],
                "n_cjk": sum(cjk_count(block.text) for block in item["blocks"] if block.detectable),
                "method_like": method_like,
                "hint": hint,
            }
        )
    return sections, section_for_block
