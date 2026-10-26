"""结构性误伤提示：告诉用户哪些段落不适合为了降分而改写。"""

from __future__ import annotations

import re

from .features import cjk_count, symbol_density
from .fuse import FusedWindow

_METHOD_SECTION = re.compile(
    r"(方法|实验|结果|讨论|结论|method|experiment|result|discussion|conclusion)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\s*=\s*\S")


def review_note_for(window: FusedWindow, section_title: str = "") -> dict | None:
    """返回 {level, label, why}；没有结构性风险时返回 None。"""
    text = window.text.strip()
    if not text:
        return None

    if _ASSIGNMENT.match(text) or symbol_density(text) > 0.30:
        return {
            "level": "leave",
            "label": "公式/符号密集",
            "why": "这类内容容易因符号和固定写法被误判；不建议为了降分改写公式，补充解释或实验条件即可。",
        }
    if cjk_count(text) < 60 and window.p_fuse > 0.65:
        return {
            "level": "short",
            "label": "文本偏短",
            "why": "这个窗口太短，统计特征不稳定；请结合上下文判断，不要只凭这一段分数改写。",
        }
    if window.delta > 0.55:
        return {
            "level": "review",
            "label": "三路分歧大",
            "why": "三个检测器意见不一致，这一段更适合人工判断；不要只看融合分。",
        }
    if _METHOD_SECTION.search(section_title) and window.p_fuse > 0.65:
        return {
            "level": "review",
            "label": "方法/结果/结论段易误伤",
            "why": "这类段落常有固定句式和术语，检测器容易误判；先人工确认，不要为了降分删改实验细节或关键结论。",
        }
    return None
