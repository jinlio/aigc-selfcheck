"""版本对比与对比报告测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_selfcheck.cli import main
from aigc_selfcheck.compare import compare_results, render_compare
from aigc_selfcheck.detectors import build_detectors
from aigc_selfcheck.pipeline import analyze_document

OLD_TEXT = (
    "# 1 引言\n\n"
    "本研究首先回顾了已有工作，其次分析了现有方法的局限性。"
    "与此同时，现有检测器在跨领域文本上的表现并不稳定，综上所述，本文提出一个新的框架。\n\n"
    "# 2 方法\n\n"
    "本文首先把文档切分为句子，其次用滑动窗口聚合三路检测器的输出。"
    "为了降低噪声，本文对窗口分数取中位数与九十分位数。\n"
)
NEW_TEXT = (
    "# 1 引言\n\n"
    "我折腾了一晚上，把之前的方法重新捋了一遍。说实话，跨领域的误判比想象中多，"
    "所以这次换了个思路，先看几个具体的错例，再决定怎么改。\n\n"
    "# 2 方法\n\n"
    "这次没有一上来就写公式，而是先把文档切成句子，再按段落试几组窗口大小。"
    "试下来发现，取中位数比直接平均稳一些，至少不会被一个特别高的窗口带跑。\n"
)


@pytest.fixture()
def mock_detectors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    return build_detectors("a,b,c", device="cpu")


def _analyze(path: Path, detectors) -> object:
    return analyze_document(path, detectors, cache=None)


def test_compare_results_reports_improvement(tmp_path: Path, mock_detectors) -> None:
    old_path = tmp_path / "old.md"
    new_path = tmp_path / "new.md"
    old_path.write_text(OLD_TEXT, encoding="utf-8")
    new_path.write_text(NEW_TEXT, encoding="utf-8")

    old_result = _analyze(old_path, mock_detectors)
    new_result = _analyze(new_path, mock_detectors)
    comparison = compare_results(old_result, new_result)

    assert comparison["schema"] == "aigc-selfcheck/compare@v1"
    assert comparison["score_delta"] <= 0
    assert comparison["verdict"] in ("改善", "基本持平")
    assert len(comparison["sections"]) == 2
    assert all(item["old_score"] is not None and item["new_score"] is not None for item in comparison["sections"])
    assert comparison["top_changes"]


def test_render_compare_is_self_contained(tmp_path: Path, mock_detectors) -> None:
    old_path = tmp_path / "old.md"
    new_path = tmp_path / "new.md"
    old_path.write_text(OLD_TEXT, encoding="utf-8")
    new_path.write_text(NEW_TEXT, encoding="utf-8")
    comparison = compare_results(
        _analyze(old_path, mock_detectors), _analyze(new_path, mock_detectors)
    )

    out = tmp_path / "compare.html"
    render_compare(comparison, out)
    html = out.read_text(encoding="utf-8")

    assert "版本对比" in html
    assert "1 引言" in html
    assert "http://" not in html and "https://" not in html


def test_cli_compare_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    old_path = tmp_path / "old.md"
    new_path = tmp_path / "new.md"
    out = tmp_path / "compare.html"
    old_path.write_text(OLD_TEXT, encoding="utf-8")
    new_path.write_text(NEW_TEXT, encoding="utf-8")

    rc = main(
        [
            str(new_path),
            "--compare-with",
            str(old_path),
            "--out",
            str(out),
            "--no-cache",
            "--quiet",
        ]
    )

    assert rc == 0
    assert out.exists()
    assert "版本对比" in out.read_text(encoding="utf-8")
