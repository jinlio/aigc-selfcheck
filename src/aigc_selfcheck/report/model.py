"""报告数据模型：AnalysisResult → 可 JSON 序列化的 payload。

payload 是单文件 HTML 的唯一数据源，内联进 <script type="application/json">。
结构约定：blocks 按文档顺序渲染热力图；windows 承载三路明细与 PPL 曲线数据。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .. import __version__
from ..features import cjk_count
from ..pipeline import AnalysisResult

DISCLAIMER = (
    "本报告是本地启发式风险估计，与知网、朱雀等任何官方 AIGC 检测系统无关，"
    "不能预测或保证官方检测结果；请把分数和标记当作修改线索，不要当作学术不端认定。"
    "所有分析均在本机完成，文本不会上传。"
)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_calibration_summary() -> dict:
    """读 `data/calibration_summary.json`——报告里的校准数字唯一来源。

    以前这里写死过一版 94.9% / 2.6% / 11.1%（v8 时代的探针结果），部署常量
    升到 v12 后没人同步，报告就一直在展示一套自己没在跑的成绩。改成读数
    以后，`tests/test_calibration_summary.py` 会比对这个文件与探针产物，
    数字漂了测试就红。
    """
    try:
        raw = (_DATA_DIR / "calibration_summary.json").read_text(encoding="utf-8")
    except OSError:  # noqa: BLE001 —— 缺文件时退化为空，报告仍能出
        return {"deployed_constants": "?", "notes": []}
    return json.loads(raw)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


SUMMARY = load_calibration_summary()


def _calibration_block() -> dict:
    """把产物摘要排版成报告里展示的字符串。"""
    general = SUMMARY.get("general", {})
    academic = SUMMARY.get("academic_default", {})
    return {
        "version": SUMMARY.get("deployed_constants", "?"),
        "general": {
            "label": general.get("label", ""),
            "auc": f"{general.get('auc', 0):.3f}",
            "human_fp": _pct(general.get("human_fp")),
            "recall": _pct(general.get("recall")),
        },
        "academic_default": {
            "label": academic.get("label", ""),
            "ai_window_recall": _pct(academic.get("ai_window_recall")),
            "human_window_fp": _pct(academic.get("human_window_fp")),
            "human_doc_high": _pct(academic.get("human_doc_high")),
            "ai_doc_high": _pct(academic.get("ai_doc_high")),
        },
        "notes": SUMMARY.get("notes", []),
    }


CALIBRATION = _calibration_block()

PROVISIONAL_NOTE = (
    f"当前为 v{__version__} 工具版本（{SUMMARY.get('deployed_constants', '?')} 校准常量）："
    "分数与区间带来自本地冻结校准集，不是知网/朱雀等官方检测结果，也不能预测它们的判定。"
)

# 窗口 meta 只透出报告需要的键，避免 payload 膨胀
_META_KEYS = ("p_ppl", "p_style", "log_ppl", "ppl", "device", "truncated", "chunks", "model")


def _evidence(result: AnalysisResult) -> dict:
    n_windows = len(result.fused)
    n_cjk = sum(cjk_count(block.text) for block in result.blocks if block.detectable)
    sufficient = n_windows >= 3 and n_cjk >= 300
    if sufficient:
        note = f"证据充足：{n_windows} 个可检测窗口 / {n_cjk} 个汉字。"
    else:
        note = (
            f"证据不足：仅 {n_windows} 个可检测窗口 / {n_cjk} 个汉字。"
            "短文本的困惑度与风格分不稳定，区间带仅供参考；"
            "建议至少 300 字或 3 个窗口后再判断。"
        )
    return {
        "sufficient": sufficient,
        "n_windows": n_windows,
        "n_cjk": n_cjk,
        "note": note,
    }


def _suggestions(result: AnalysisResult) -> list[str]:
    ranked = sorted(result.fused, key=lambda window: window.p_fuse, reverse=True)
    found: list[str] = []

    def add(text: str) -> None:
        if text not in found:
            found.append(text)

    for window in ranked[:5]:
        if window.p_fuse < 0.5:
            continue
        for text in window.suggestions:
            add(text)
        if len(found) >= 5:
            break
    if not found:
        if result.doc.get("band") == "低":
            add("未发现明显高风险窗口；仍建议通读高亮句，确认没有大段直接粘贴的 AI 文本。")
        else:
            add(
                "模型给出较高风险，但没有单一显著风格特征；建议补充只有作者能提供的细节"
                "（数据、过程、限制），并通读全文。"
            )
    final = "修改后重新检测；不要为了降分堆砌口语词或故意写病句——那会损害论文质量，也不保证通过官方检测。"
    core = [text for text in found if text != final][:5]
    core.append(final)
    return core


def _guidance(result: AnalysisResult) -> dict:
    band = result.doc.get("band", "低")
    if band == "高":
        headline = "整体风险偏高：至少一个稳健指标进入高带；请逐段复核，不要直接提交。"
    elif band == "中":
        headline = "整体风险中等：部分窗口接近或越过红线，建议优先复核高亮段落。"
    elif band == "样本不足":
        headline = "可检测窗口太少，不够给出档位判断；请只看逐句热力图，不要下结论。"
    else:
        headline = "整体风险偏低：没有窗口越过 0.75 红线；仍建议通读高亮句。"
    distance = result.doc.get("band_distance") or {}
    band_set_note = (
        "低/中/高是按**写法可预测性**分的档，不是「是否用了 AI」的判定。"
        "学术语体在通用阈值下天然偏中偏高，所以默认用学术口径；"
        "口径里每一档的边界都在报告的「区间带」卡片里写明。"
        "换口径用 --band-set academic|general，档位只提供按语域校准过的预设。"
    )
    steps = [
        "先看区间带、口径与证据是否充足；短文本不要下结论。",
        "看「章节地图」定位高风险章节，再看「修改计划」判断先改哪一段。",
        "按热力图从红到黄逐句复核，点句子查看白话原因、修改建议和技术细节。",
        "遇到「公式/符号密集」「方法/结果段易误伤」提示时先人工确认，不要为了降分删改实验细节。",
        "修改后重新运行本工具；官方检测结果仍以知网等平台为准。",
    ]
    if distance and not distance.get("reachable") and band != "低":
        steps.insert(
            1,
            f"离「低」带还差 {distance.get('gap')} 个窗：需要 "
            f"{distance['median']['need']} 个窗低于 {distance['median']['threshold']}"
            f"（现 {distance['median']['have']}）且 "
            f"{distance['p90']['need']} 个窗低于 {distance['p90']['threshold']}"
            f"（现 {distance['p90']['have']}）。",
        )
    return {
        "evidence": _evidence(result),
        "intro": (
            "分数高只说明这段写法容易被检测器误判，不代表你用了 AI。"
            "先看“为什么被标”，再按“怎么改”调整；不要为了降分故意写病句。"
        ),
        "headline": headline,
        "band_set_note": band_set_note,
        "steps": steps,
        "suggestions": _suggestions(result),
    }


def _safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def build_payload(result: AnalysisResult, title: str | None = None) -> dict:
    enhanced_windows = [w for w in result.fused if w.enhanced]
    enhanced_sids = {sid for w in enhanced_windows for sid in w.sentence_ids}
    windows = []
    for window in result.fused:
        metas = {}
        for key, meta in window.metas.items():
            metas[key] = {k: _safe(v) for k, v in (meta or {}).items() if k in _META_KEYS}
        windows.append(
            {
                "key": window.window_key,
                "block": window.block_index,
                "sentences": window.sentence_ids,
                "risk": window.p_fuse,
                "delta": window.delta,
                "disputed": window.disputed,
                "heat": window.heat,
                "scores": window.scores,
                "metas": metas,
                "reasons": [item["why"] for item in window.reason_items],
                "reason_items": window.reason_items,
                "suggestions": window.suggestions,
                "review_note": window.review_note,
                "llm_advice": window.llm_advice,
                "text": window.text,
                "meta_prob": _safe(window.meta_prob),
                "enhanced": window.enhanced,
                "floor_rule": window.floor_rule,
            }
        )

    blocks = []
    for block in result.blocks:
        blocks.append(
            {
                "index": block.index,
                "kind": block.kind,
                "detectable": block.detectable,
                "note": block.note,
                "text": block.text,
                "sentences": [
                    {
                        "text": s.text,
                        "risk": s.risk,
                        "disputed": s.disputed,
                        "enhanced": s.global_id in enhanced_sids,
                        "reasons": s.reasons,
                        "reason_items": s.reason_items,
                        "suggestions": s.suggestions,
                        "section": s.section,
                        "review_note": s.review_note,
                        "llm_advice": s.llm_advice,
                        "gid": s.global_id,
                    }
                    for s in block.sentences
                ],
            }
        )

    ppls = []
    for window in result.fused:
        meta = window.metas.get("c") or {}
        ppl = meta.get("ppl")
        if ppl is not None and math.isfinite(ppl) and ppl > 0:
            ppls.append({"key": window.window_key, "ppl": ppl})

    hot = [
        {
            "block": item.block_index,
            "sentences": item.sentence_ids,
            "mean": item.mean_score,
            "n_windows": len(item.window_keys),
        }
        for item in result.hot
    ]

    enhanced = {
        "enabled": result.enhanced,
        **{k: _safe(v) for k, v in result.meta_info.items() if k != "enabled"},
        "windows": [
            {
                "key": w.window_key,
                "block": w.block_index,
                "risk": w.p_fuse,
                "meta_prob": _safe(w.meta_prob),
                "text": w.text,
            }
            for w in enhanced_windows
        ],
        "note": (
            "实验性增强召回：除固定融合 >0.75 外，再把融合分 >0.55 且元分类器"
            "高于阈值的灰度窗口标为增强命中。它会提高召回，但可能增加误报，"
            "请把增强标记当作复核线索，不要当作结论。"
        ),
    }

    return {
        "schema": "aigc-selfcheck/report@v1",
        "tool_version": __version__,
        "title": title or f"{result.source} — AIGC 自检报告",
        "source": result.source,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(result.duration_s, 2),
        "disclaimer": DISCLAIMER,
        "provisional": PROVISIONAL_NOTE,
        "calibration": CALIBRATION,
        "profile": result.profile,
        "detectors": result.detector_ids,
        "warnings": result.warnings,
        "guidance": _guidance(result),
        "enhanced": enhanced,
        "doc": {
            **result.doc,
            "burstiness": _safe(result.burstiness),
            "n_sentences": sum(len(b.sentences) for b in result.blocks),
            "n_blocks": len(result.blocks),
        },
        "blocks": blocks,
        "windows": windows,
        "sections": result.sections,
        "plan": result.plan,
        "llm_advice": result.llm_advice_info,
        "timings": result.timings,
        "ppl_curve": ppls,
        "hot_intervals": hot,
        "top_sentences": result.top_sentences(20),
    }
