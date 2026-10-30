"""CLI 端到端测试（mock 模式）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aigc_selfcheck.cli import main
from conftest import GOLDEN_TEXT


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text(GOLDEN_TEXT, encoding="utf-8")
    return path


def test_cli_json(monkeypatch, sample_txt: Path, capsys):
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main([str(sample_txt), "--json", "--no-cache", "--quiet"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "aigc-selfcheck/report@v1"
    assert payload["doc"]["band"] in ("低", "中", "高", "样本不足")
    assert payload["doc"]["band_set"] == "academic"


def test_cli_band_set_override(monkeypatch, sample_txt: Path, capsys):
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main([str(sample_txt), "--json", "--no-cache", "--quiet", "--band-set", "general"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["doc"]["band_set"] == "general"
    assert "academic" in payload["doc"]["alternates"]


def test_cli_rejects_custom_thresholds(monkeypatch, sample_txt: Path, capsys):
    """档位只接受预设名，不接受自由阈值。"""
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    with pytest.raises(SystemExit) as exc:
        main([str(sample_txt), "--quiet", "--band-set", "lo=0.50,tlo=0.62,hi=0.72,thi=0.86"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_band_thresholds_are_reported(monkeypatch, sample_txt: Path, capsys):
    """报告要能原样写出档位边界——用户看到「中」时得知道「中」怎么划的。"""
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main([str(sample_txt), "--json", "--no-cache", "--quiet"])
    assert rc == 0
    thresholds = json.loads(capsys.readouterr().out)["doc"]["band_thresholds"]
    assert thresholds == {
        "robust_low": 0.58,
        "tail_low": 0.68,
        "robust_high": 0.76,
        "tail_high": 0.88,
        "min_windows": 8,
    }


def test_cli_exclude_heading(monkeypatch, tmp_path: Path, capsys):
    """排除章节后，文档级窗口数减少、报告里写明排除了什么。"""
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    path = tmp_path / "doc.md"
    path.write_text(
        "# 正文\n\n" + GOLDEN_TEXT + "\n\n# 外文翻译\n\n" + GOLDEN_TEXT + "\n",
        encoding="utf-8",
    )
    rc = main([str(path), "--json", "--no-cache", "--quiet"])
    assert rc == 0
    full = json.loads(capsys.readouterr().out)
    rc = main([str(path), "--json", "--no-cache", "--quiet", "--exclude-heading", "外文翻译"])
    assert rc == 0
    partial = json.loads(capsys.readouterr().out)
    assert partial["doc"]["n_windows"] < full["doc"]["n_windows"]
    assert partial["doc"]["partial"]["excluded_titles"] == ["外文翻译"]
    assert partial["doc"]["partial"]["n_windows_excluded"] > 0


def test_cli_html_report(monkeypatch, sample_txt: Path, tmp_path: Path, capsys):
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    out = tmp_path / "custom.html"
    rc = main([str(sample_txt), "--out", str(out), "--no-cache", "--quiet"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == str(out)
    assert out.exists()


def test_cli_missing_file():
    rc = main(["/nonexistent/file.docx", "--quiet"])
    assert rc == 2
