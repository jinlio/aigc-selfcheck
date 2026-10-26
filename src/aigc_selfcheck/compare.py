"""版本对比：同一篇论文改稿前后，分数与章节风险如何变化。

对比按章节标题匹配；标题改动会被视为新增/删除，而不是强行对齐。
输出是可独立打开的静态 HTML，不依赖任何外部资源。
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from .features import cjk_count
from .pipeline import AnalysisResult

_RE_NUMBERING = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百]+[章节部分]|[0-9]+(?:\.[0-9]+)*|[一二三四五六七八九十]+、)\s*"
)


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", "", _RE_NUMBERING.sub("", title or "")).lower()


def _summary(result: AnalysisResult) -> dict:
    doc = result.doc
    return {
        "score": int(doc.get("score", 0)),
        "band": doc.get("band", ""),
        "median": doc.get("median", 0.0),
        "p90": doc.get("p90", 0.0),
        "n_windows": int(doc.get("n_windows", 0)),
        "n_sentences": sum(len(block.sentences) for block in result.blocks),
        "n_cjk": sum(cjk_count(block.text) for block in result.blocks if block.detectable),
    }


def _status(delta: int | None) -> str:
    if delta is None:
        return "新增/删除"
    if delta <= -5:
        return "改善"
    if delta >= 5:
        return "变差"
    return "基本持平"


def compare_results(old: AnalysisResult, new: AnalysisResult) -> dict:
    """返回可直接 JSON 序列化的版本对比。"""
    old_summary = _summary(old)
    new_summary = _summary(new)
    old_map = {_normalize_title(section["title"]): section for section in old.sections}
    new_map = {_normalize_title(section["title"]): section for section in new.sections}

    keys: list[str] = []
    for section in new.sections:
        key = _normalize_title(section["title"])
        if key not in keys:
            keys.append(key)
    for section in old.sections:
        key = _normalize_title(section["title"])
        if key not in keys:
            keys.append(key)

    sections = []
    for key in keys:
        old_section = old_map.get(key)
        new_section = new_map.get(key)
        if old_section and new_section:
            delta = int(new_section["score"]) - int(old_section["score"])
            status = _status(delta)
        else:
            delta = None
            status = "新增" if new_section else "删除"
        title = (new_section or old_section or {}).get("title", "未命名章节")
        sections.append(
            {
                "title": title,
                "old_score": old_section["score"] if old_section else None,
                "new_score": new_section["score"] if new_section else None,
                "delta": delta,
                "status": status,
                "old_band": old_section["band"] if old_section else "",
                "new_band": new_section["band"] if new_section else "",
                "old_median": old_section["median"] if old_section else None,
                "new_median": new_section["median"] if new_section else None,
                "old_p90": old_section["p90"] if old_section else None,
                "new_p90": new_section["p90"] if new_section else None,
                "old_windows": old_section["n_windows"] if old_section else 0,
                "new_windows": new_section["n_windows"] if new_section else 0,
            }
        )

    changed = [item for item in sections if item["delta"] is not None]
    changed.sort(key=lambda item: abs(item["delta"]), reverse=True)
    score_delta = new_summary["score"] - old_summary["score"]
    return {
        "schema": "aigc-selfcheck/compare@v1",
        "verdict": _status(score_delta),
        "score_delta": score_delta,
        "band_changed": old_summary["band"] != new_summary["band"],
        "old": old_summary,
        "new": new_summary,
        "sections": sections,
        "top_changes": changed[:5],
        "warnings": list(dict.fromkeys(old.warnings + new.warnings)),
        "note": (
            "章节按标题匹配；标题改动会被视为新增/删除。分数是本地启发式估计，"
            "不是官方检测结果；改稿后请重新运行，不要只追求分数下降。"
        ),
    }


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _band_class(band: str) -> str:
    return "low" if band == "低" else "high" if band == "高" else "mid"


def _status_class(status: str) -> str:
    return "good" if status == "改善" else "bad" if status == "变差" else "same"


def render_compare(comparison: dict, out_path: str | Path) -> Path:
    old = comparison["old"]
    new = comparison["new"]
    verdict = comparison["verdict"]
    verdict_class = _status_class(verdict)
    rows = []
    for item in comparison["sections"]:
        delta = item["delta"]
        delta_text = "—" if delta is None else f"{delta:+d}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['title'])}</td>"
            f"<td class='num'>{_fmt(item['old_score'])}</td>"
            f"<td class='num'>{_fmt(item['new_score'])}</td>"
            f"<td class='num {_status_class(item['status'])}'>{delta_text}</td>"
            f"<td>{html.escape(item['old_band'] or '—')}</td>"
            f"<td>{html.escape(item['new_band'] or '—')}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            "</tr>"
        )
    warnings = "".join(
        f"<li>{html.escape(warning)}</li>" for warning in comparison.get("warnings", [])
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>版本对比 — aigc-selfcheck</title>
<style>
:root {{ --bg:#0d1117; --soft:#161b22; --border:#2d333b; --fg:#e6edf3; --dim:#8a93a5;
  --green:#3fb950; --yellow:#d29922; --red:#f85149; --accent:#58a6ff; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.7 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  max-width:980px; margin:0 auto; padding:20px 24px 60px; }}
h1 {{ font-size:20px; margin:0 0 10px; }}
.verdict {{ border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin:10px 0; background:var(--soft); }}
.verdict.good {{ border-color:rgba(63,185,80,.55); background:rgba(63,185,80,.07); }}
.verdict.bad {{ border-color:rgba(248,81,73,.55); background:rgba(248,81,73,.07); }}
.verdict.same {{ border-color:rgba(210,153,34,.55); background:rgba(210,153,34,.07); }}
.verdict b {{ font-size:18px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:12px 0; }}
.card {{ flex:1 1 240px; border:1px solid var(--border); border-radius:10px; padding:12px 16px; background:var(--soft); }}
.card .label {{ color:var(--dim); font-size:12px; }}
.card .score {{ font:700 38px/1.1 ui-monospace,Menlo,monospace; margin:4px 0; }}
.card .score.low {{ color:var(--green); }} .card .score.mid {{ color:var(--yellow); }} .card .score.high {{ color:var(--red); }}
.card .meta {{ color:var(--dim); font-size:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; margin-top:10px; }}
th,td {{ border-bottom:1px solid var(--border); padding:7px 9px; text-align:left; vertical-align:top; }}
th {{ color:var(--dim); font-weight:500; }}
.num {{ text-align:right; font-family:ui-monospace,Menlo,monospace; }}
.good {{ color:var(--green); }} .bad {{ color:var(--red); }} .same {{ color:var(--yellow); }}
.note {{ color:var(--dim); font-size:12px; margin-top:14px; }}
ul {{ color:var(--dim); font-size:12px; }}
</style>
</head>
<body>
<h1>版本对比 · 改稿前后</h1>
<div class="verdict {verdict_class}">
  <b>{html.escape(verdict)}</b>
  <span>综合分 {old['score']} → {new['score']}（{comparison['score_delta']:+d}）</span>
  <div class="meta">区间带 {html.escape(old['band'])} → {html.escape(new['band'])}
    {'（区间带发生变化）' if comparison['band_changed'] else '（区间带未变）'}</div>
</div>
<div class="cards">
  <div class="card"><div class="label">改稿前</div>
    <div class="score {_band_class(old['band'])}">{old['score']}</div>
    <div class="meta">区间带 {html.escape(old['band'])} · median {_fmt(old['median'])} · P90 {_fmt(old['p90'])}
      · {old['n_windows']} 窗 / {old['n_cjk']} 字</div></div>
  <div class="card"><div class="label">改稿后</div>
    <div class="score {_band_class(new['band'])}">{new['score']}</div>
    <div class="meta">区间带 {html.escape(new['band'])} · median {_fmt(new['median'])} · P90 {_fmt(new['p90'])}
      · {new['n_windows']} 窗 / {new['n_cjk']} 字</div></div>
</div>
<table>
<thead><tr><th>章节</th><th>旧分</th><th>新分</th><th>变化</th><th>旧带</th><th>新带</th><th>状态</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="7">没有可对比的章节。</td></tr>'}</tbody>
</table>
<p class="note">{html.escape(comparison['note'])}</p>
{('<div class="note"><b>警告</b><ul>' + warnings + '</ul></div>') if warnings else ''}
<p class="note">本地分析，文本未离开本机 · aigc-selfcheck</p>
</body>
</html>
"""
    path = Path(out_path)
    path.write_text(document, encoding="utf-8")
    return path
