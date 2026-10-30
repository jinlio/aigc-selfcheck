"""章节级风险聚合：把窗口映射到标题，输出每章的 median/P90/综合分。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .constants import BAND_SETS, DEFAULT_BAND_SET, BandSet
from .features import cjk_count
from .fuse import FusedWindow, doc_level

if TYPE_CHECKING:
    from .pipeline import BlockView

_METHOD_SECTION = re.compile(
    r"(方法|实验|结果|讨论|结论|method|experiment|result|discussion|conclusion)",
    re.IGNORECASE,
)


def build_sections(
    blocks: list[BlockView],
    fused: list[FusedWindow],
    band_set: BandSet | None = None,
) -> tuple[list[dict], dict[int, str]]:
    """返回 (章节摘要列表, block_index → 章节标题)。"""
    bs = band_set or BAND_SETS[DEFAULT_BAND_SET]
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
        doc = doc_level(windows, bs)
        # 章节天然比整篇小：把文档级的 min_windows（8）直接套过来，会让多数学
        # 章节变成「样本不足」，章节地图就废了。这里分两级——不足 3 窗确实没
        # 法判（P90 就是最大值本身）；3–7 窗给档位但标「参考」，让地图仍然可
        # 用来定位，同时不假装它有整篇级别的把握。
        provisional = False
        if doc["band"] == "样本不足" and doc["n_windows"] >= 3:
            doc = {**doc, "band": bs.band(doc["median"], doc["p90"])}
            provisional = True
        method_like = bool(_METHOD_SECTION.search(item["title"]))
        if doc["band"] == "样本不足":
            hint = f"窗口数不足（{doc['n_windows']} 个），不判区间带；请按逐句热力图人工复核。"
        elif provisional and method_like and doc["band"] == "高":
            hint = f"仅 {doc['n_windows']} 窗，档位仅供参考；方法/结果段易误伤，先人工确认。"
        elif provisional:
            hint = f"仅 {doc['n_windows']} 窗，档位仅供参考。"
        elif method_like and doc["band"] == "高":
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
                "band_set": doc.get("band_set", bs.name),
                "band_provisional": provisional,
                "n_cjk": sum(cjk_count(block.text) for block in item["blocks"] if block.detectable),
                "method_like": method_like,
                "hint": hint,
            }
        )
    return sections, section_for_block
