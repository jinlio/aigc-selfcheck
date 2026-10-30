"""校准摘要守卫：报告头引用的数字必须与探针产物一致。

背景：报告里曾长期写 v8 时代的 94.9% / 2.6% / 11.1%，而产物已更新到
93.2% / 1.8% / 17.8%。数字写死在两处就一定会漂——所以摘要由产物生成，
这里再加一道测试，产物一变就红。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "src" / "aigc_selfcheck" / "data" / "calibration_summary.json"
SOURCES = {
    "academic_human": ROOT / "calibration" / "academic" / "report.json",
    "academic_ai": ROOT / "calibration" / "ai_academic_lite" / "report.json",
}


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _report(key: str) -> dict:
    path = SOURCES[key]
    if not path.exists():
        pytest.skip(f"探针产物缺失（本地数据归档未就位）：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_declares_its_sources(summary: dict) -> None:
    assert summary["sources"] == {
        "academic_human": "calibration/academic/report.json",
        "academic_ai": "calibration/ai_academic_lite/report.json",
    }


def test_academic_window_numbers_match_probes(summary: dict) -> None:
    """窗口级数字必须逐位对齐两个探针的 overall 块。"""
    block = summary["academic_default"]
    human = _report("academic_human")["overall"]
    ai = _report("academic_ai")["overall"]

    assert block["n_human_windows"] == human["n_windows"]
    assert block["n_ai_windows"] == ai["n_windows"]
    assert block["ai_window_recall"] == pytest.approx(ai["window_recall"])
    assert block["human_window_fp"] == pytest.approx(human["window_positive_rate"])
    assert block["ai_window_p50"] == pytest.approx(ai["window_p50"])
    assert block["ai_window_p90"] == pytest.approx(ai["window_p90"])
    assert block["human_window_p50"] == pytest.approx(human["window_p50"])
    assert block["human_window_p90"] == pytest.approx(human["window_p90"])


def test_academic_doc_high_rates_match_probes(summary: dict) -> None:
    """文档级「高」带比例依赖阈值，阈值改了这个测试会红——这是有意的。"""
    block = summary["academic_default"]
    assert block["human_doc_high"] == pytest.approx(_report("academic_human")["overall"]["doc_band_high"])
    assert block["ai_doc_high"] == pytest.approx(_report("academic_ai")["overall"]["doc_band_high"])


def test_summary_is_marked_report_only(summary: dict) -> None:
    """摘要只能来自 report-only 探针，不能混入训练集统计。"""
    assert "report-only" in json.dumps(summary["notes"], ensure_ascii=False) or any(
        "探针" in note for note in summary["notes"]
    )


def test_deployed_constants_matches_code(summary: dict) -> None:
    from aigc_selfcheck.constants import CALIBRATION_VERSION

    assert summary["deployed_constants"] == CALIBRATION_VERSION
