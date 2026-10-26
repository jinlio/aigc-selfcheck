r"""单文件 HTML 渲染器。

架构模式（参照 agent-xray 的做法，代码全新实现）：数据以 JSON 内联进
<script type="application/json">，CSS/JS/ECharts 全部内联，无模板引擎、
无前端构建链、完全离线可看。payload 里不允许出现 "</"（转义为 "<\/"）。
"""

from __future__ import annotations

import json
from pathlib import Path

_ASSETS = Path(__file__).parent
_MAX_DATA_BYTES = 50 * 1024 * 1024  # payload 守卫：超过则说明文档过大，提示分块


def _load(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def _json_payload(data: dict) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # payload 位于 <script type="application/json"> 内，"</" 会提前终止标签；
    # "<\/" 是合法 JSON 转义。
    return text.replace("</", "<\\/")


def render_report(payload: dict, out_path: str | Path) -> Path:
    data_text = _json_payload(payload)
    if len(data_text.encode("utf-8")) > _MAX_DATA_BYTES:
        raise ValueError(
            "报告数据超过 50MB：文档过大，请拆分后分块检测。"
        )
    html = (
        _load("template.html")
        .replace("__TITLE__", _esc(payload.get("title") or "AIGC 自检报告"))
        .replace("__CSS__", _load("style.css"))
        .replace("__ECHARTS__", _load("echarts.min.js"))
        .replace("__APP__", _load("app.js"))
        .replace("__DATA__", data_text)
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
