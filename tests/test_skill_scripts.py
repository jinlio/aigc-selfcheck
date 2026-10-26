"""Agent Skill 脚本的端到端冒烟测试（mock 检测器，不下载模型）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "aigc-selfcheck" / "scripts"


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["AIGC_SELFCHECK_MOCK"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
    )


@pytest.fixture()
def sample_doc(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text(
        "# 1 引言\n\n"
        "本研究首先回顾了已有工作，其次分析了现有方法的局限性。"
        "与此同时，现有检测器在跨领域文本上的表现并不稳定，综上所述，本文提出一个新的框架。\n",
        encoding="utf-8",
    )
    return path


def test_check_script_outputs_json(sample_doc: Path) -> None:
    result = _run("check.py", str(sample_doc), "--json", "--no-cache", "--quiet")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "aigc-selfcheck/report@v1"
    assert "sections" in payload and "plan" in payload


def test_check_script_accepts_profile_and_llm_flag(sample_doc: Path) -> None:
    result = _run(
        "check.py",
        str(sample_doc),
        "--profile",
        "lite",
        "--llm-advice",
        "--json",
        "--no-cache",
        "--quiet",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["profile"]["name"] == "lite"
    assert payload["llm_advice"]["enabled"] is False


def test_explain_script_is_readable(sample_doc: Path) -> None:
    result = _run("explain.py", str(sample_doc), "--top", "1", "--detectors", "b,c")
    assert result.returncode == 0, result.stderr
    assert "综合风险分" in result.stdout
    assert "为什么" in result.stdout


def test_plan_script_is_readable(sample_doc: Path) -> None:
    result = _run("plan.py", str(sample_doc), "--detectors", "b,c")
    assert result.returncode == 0, result.stderr
    assert "修改计划" in result.stdout


def test_compare_script_is_readable(tmp_path: Path, sample_doc: Path) -> None:
    old = tmp_path / "old.md"
    old.write_text(
        "# 1 引言\n\n我折腾了一晚上，把之前的方法重新捋了一遍。说实话，跨领域的误判比想象中多。\n",
        encoding="utf-8",
    )
    out = tmp_path / "compare.html"
    result = _run(
        "compare.py",
        str(sample_doc),
        "--old",
        str(old),
        "--out",
        str(out),
        "--detectors",
        "b,c",
    )
    assert result.returncode == 0, result.stderr
    assert "版本对比" in result.stdout
    assert out.exists()


def test_skill_rejects_url() -> None:
    result = _run("check.py", "https://example.com/paper.docx", "--quiet")
    assert result.returncode != 0
    assert "必须是本地文件" in result.stderr
