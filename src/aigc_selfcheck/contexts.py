"""窗口上下文：产品路径与校准路径共用同一口径。

C 路的 PPL 是条件 PPL——同一窗口带上「前文」会比孤立打分更稳。
上下文必须取**窗口起点之前**的正文，而不是「前一个窗口的尾部」：
滑窗步长是 1 句，相邻窗口天然重叠，用前窗尾部会把当前窗口自身
（甚至整窗）喂给模型，log-PPL 被系统性压低，分数失去意义。

约定：
- 上下文长度固定为 `CONTEXT_TAIL_CHARS`（与 pipeline 一致）；
- 每个块/段落的第一个窗口上下文为空串；
- 校准行优先读取落盘的 `context` 字段；旧版校准集没有该字段时，
  按 `(para_sha, window_index - 1)` 回退到前窗尾部（仅兼容旧数据）。
"""

from __future__ import annotations

CONTEXT_TAIL_CHARS = 100


def preceding_context(text: str, start: int) -> str:
    """取 `text` 中窗口起点 `start` 之前的最后 `CONTEXT_TAIL_CHARS` 个字符。"""
    return text[max(0, start - CONTEXT_TAIL_CHARS) : start]


def _window_index(unit: dict) -> int:
    try:
        return int(str(unit.get("uid", "")).rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def window_contexts(units: list[dict]) -> list[str]:
    """为窗口级单元重建上下文（与 `pipeline` 的窗口起点口径一致）。

    输入行需含 `uid`、`para_sha`、`text`，新数据另含 `context`；输出与
    输入等长。找不到前文（段落首窗、索引断档、缺字段）时返回空串。
    """
    if units and all("context" in unit for unit in units):
        return [str(unit.get("context") or "") for unit in units]
    by_key = {
        (unit.get("para_sha"), _window_index(unit)): unit.get("text", "")
        for unit in units
    }
    contexts: list[str] = []
    for unit in units:
        previous = by_key.get((unit.get("para_sha"), _window_index(unit) - 1))
        contexts.append(previous[-CONTEXT_TAIL_CHARS:] if previous else "")
    return contexts
