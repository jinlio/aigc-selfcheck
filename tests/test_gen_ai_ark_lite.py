"""豆包 Lite 批量生成脚本的纯函数测试（不联网）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gen_ai_ark_lite as ark  # noqa: E402


def test_plan_is_five_prompts_times_loops() -> None:
    tasks = ark.build_tasks(100)
    assert len(tasks) == 500
    assert [task["prompt_idx"] for task in tasks[:5]] == [1, 2, 3, 4, 5]
    assert tasks[0]["loop"] == 1
    assert tasks[-1]["loop"] == 100
    assert {task["prompt_idx"] for task in tasks} == {1, 2, 3, 4, 5}


def test_academic_plan_is_seven_sections_times_loops() -> None:
    tasks = ark.build_tasks(20, pack="academic")
    assert len(tasks) == 140
    assert {task["prompt_idx"] for task in tasks} == set(range(1, 8))
    assert tasks[0]["style"] == "学术论文-摘要"
    assert tasks[0]["topic"] == "摘要"
    assert tasks[0]["prefix"] == "arkacad"
    assert "深度学习在医学影像诊断中的应用" in tasks[0]["prompt"]
    assert all("{" not in task["prompt"] for task in tasks)


def test_clean_content_strips_fence_and_word_count() -> None:
    raw = "```markdown\n这是一段正文。\n```\n（全文 245 字）"
    assert ark.clean_content(raw) == "这是一段正文。"


def test_body_ok_requires_length_and_terminal_punctuation() -> None:
    assert ark.body_ok("字" * 100 + "。")
    assert ark.body_ok("字" * 50 + "。" + "字" * 50 + "。")
    assert not ark.body_ok("字" * 49 + "。" + "字" * 49 + "。")
    assert not ark.body_ok("字" * 100)


def test_validate_base_rejects_http_and_foreign_host() -> None:
    with pytest.raises(ValueError, match="https"):
        ark.validate_base_url("http://ark.cn-beijing.volces.com/api/coding/v3")
    with pytest.raises(ValueError, match="host"):
        ark.validate_base_url("https://evil.example.com/api/coding/v3")


def test_validate_base_rejects_private_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ark.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="非公网"):
        ark.validate_base_url(ark.DEFAULT_BASE_URL)


def test_validate_base_accepts_public_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ark.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("180.184.47.154", 443))],
    )
    assert ark.validate_base_url(ark.DEFAULT_BASE_URL) == ark.DEFAULT_BASE_URL


def test_sample_path_and_provenance() -> None:
    assert ark.sample_path(1, 1, "数据库索引") == "texts/arklite-001-01-数据库索引.txt"
    assert (
        ark.sample_path(1, 1, "摘要", prefix="arkacad")
        == "texts/arkacad-001-01-摘要.txt"
    )
    header = ark.provenance(
        ark.DEFAULT_MODEL, "2026-10-25", "技术说明", 1, 1, 1.0, 0.95, 800
    )
    assert "模型: doubao-seed-2-1-lite-260915" in header
    assert "轮次: 001" in header
    assert "思考: 关闭" in header
