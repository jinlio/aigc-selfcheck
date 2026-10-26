"""运行档位解析与 CLI 集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aigc_selfcheck.cli import main
from aigc_selfcheck.profiles import PROFILES, resolve_profile
from conftest import GOLDEN_TEXT


def test_lite_profile_uses_cpu_and_small_model() -> None:
    profile = resolve_profile("lite")
    assert profile["detectors"] == "b,c"
    assert profile["device"] == "cpu"
    assert profile["c_model"] == "Qwen/Qwen2.5-0.5B"
    assert profile["label"] == "轻量"
    assert profile["llm_advice"] is False


def test_balanced_profile_uses_two_lanes() -> None:
    profile = resolve_profile("balanced")
    assert profile["detectors"] == "b,c"
    assert profile["c_model"] is None
    assert profile["device"] == "auto"
    assert profile["llm_advice"] is True


def test_max_profile_uses_all_lanes() -> None:
    profile = resolve_profile("max")
    assert profile["detectors"] == "a,b,c"
    assert profile["label"] == "上限"
    assert profile["llm_advice"] is True


def test_explicit_llm_flag_overrides_profile() -> None:
    assert resolve_profile("balanced", llm_advice=False)["llm_advice"] is False
    assert resolve_profile("lite", llm_advice=True)["llm_advice"] is True


def test_explicit_parameters_override_profile() -> None:
    profile = resolve_profile("lite", detectors="a,b,c", device="cuda", c_model="custom/model")
    assert profile["detectors"] == "a,b,c"
    assert profile["device"] == "cuda"
    assert profile["c_model"] == "custom/model"


def test_no_profile_keeps_legacy_defaults() -> None:
    profile = resolve_profile(None)
    assert profile["name"] == "custom"
    assert profile["detectors"] == "a,b,c"
    assert profile["device"] == "auto"


def test_all_profiles_are_documented() -> None:
    assert set(PROFILES) == {"lite", "balanced", "max"}
    assert all(profile.description for profile in PROFILES.values())


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text(GOLDEN_TEXT, encoding="utf-8")
    return path


def test_cli_lite_profile(monkeypatch, sample_txt: Path, capsys) -> None:
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main(
        [str(sample_txt), "--profile", "lite", "--json", "--no-cache", "--quiet"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["name"] == "lite"
    assert payload["profile"]["detectors"] == "b,c"
    assert set(payload["detectors"]) == {"b", "c"}


def test_cli_explicit_detectors_override_profile(monkeypatch, sample_txt: Path, capsys) -> None:
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main(
        [
            str(sample_txt),
            "--profile",
            "lite",
            "--detectors",
            "a,b,c",
            "--json",
            "--no-cache",
            "--quiet",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["detectors"] == "a,b,c"
    assert set(payload["detectors"]) == {"a", "b", "c"}


def test_cli_llm_advice_falls_back_with_mock(monkeypatch, sample_txt: Path, capsys) -> None:
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main(
        [str(sample_txt), "--profile", "balanced", "--llm-advice", "--json", "--no-cache", "--quiet"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["llm_advice"]["enabled"] is False
    assert any("大模型建议已跳过" in warning for warning in payload["warnings"])


def test_cli_balanced_on_cpu_skips_llm_by_default(monkeypatch, sample_txt: Path, capsys) -> None:
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main(
        [
            str(sample_txt),
            "--profile",
            "balanced",
            "--device",
            "cpu",
            "--json",
            "--no-cache",
            "--quiet",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["llm_advice"] is False
    assert payload["llm_advice"] == {}


def test_cli_no_llm_advice_disables_profile_default(monkeypatch, sample_txt: Path, capsys) -> None:
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    rc = main(
        [
            str(sample_txt),
            "--profile",
            "balanced",
            "--no-llm-advice",
            "--json",
            "--no-cache",
            "--quiet",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["llm_advice"] == {}
