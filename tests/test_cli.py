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
    assert payload["doc"]["band"] in ("低", "中", "高")


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
