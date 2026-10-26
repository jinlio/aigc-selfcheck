"""章节地图、结构性误伤提示与修改优先级的单元测试。"""

from __future__ import annotations

from aigc_selfcheck.advice import review_note_for
from aigc_selfcheck.fuse import FusedWindow, doc_level
from aigc_selfcheck.pipeline import BlockView
from aigc_selfcheck.plan import revision_plan
from aigc_selfcheck.sections import build_sections

LONG_TEXT = "这是一段用于测试的中文正文，用来模拟方法部分的连续叙述。" * 4


def _window(block: int, p_fuse: float, text: str = LONG_TEXT, delta: float = 0.1) -> FusedWindow:
    return FusedWindow(
        window_key=f"{block}:0",
        block_index=block,
        sentence_ids=[block],
        text=text,
        scores={"c": p_fuse},
        p_fuse=p_fuse,
        delta=delta,
    )


def _block(index: int, kind: str, text: str = LONG_TEXT, detectable: bool = True) -> BlockView:
    return BlockView(index=index, kind=kind, text=text, detectable=detectable)


class TestReviewNote:
    def test_formula_is_leave_alone(self) -> None:
        note = review_note_for(_window(0, 0.8, text="S = w1 * S_w + w2 * S_k"), "3 方法")
        assert note is not None
        assert note["level"] == "leave"
        assert "公式" in note["label"]

    def test_short_text_is_unstable(self) -> None:
        note = review_note_for(_window(0, 0.8, text="这是一段很短的文字。"), "1 引言")
        assert note is not None
        assert note["level"] == "short"

    def test_large_disagreement_needs_review(self) -> None:
        note = review_note_for(_window(0, 0.8, delta=0.8), "1 引言")
        assert note is not None
        assert note["label"] == "三路分歧大"

    def test_method_section_high_risk_needs_review(self) -> None:
        note = review_note_for(_window(0, 0.8, delta=0.1), "3 实验结果")
        assert note is not None
        assert note["label"] == "方法/结果/结论段易误伤"

    def test_low_risk_prose_has_no_note(self) -> None:
        assert review_note_for(_window(0, 0.3, delta=0.1), "1 引言") is None


class TestSections:
    def test_sections_follow_headings_and_aggregate(self) -> None:
        blocks = [
            _block(0, "heading", "1 引言", detectable=False),
            _block(1, "paragraph"),
            _block(2, "heading", "2 方法", detectable=False),
            _block(3, "paragraph"),
        ]
        fused = [_window(1, 0.2), _window(3, 0.8)]
        sections, section_for_block = build_sections(blocks, fused)

        assert [s["title"] for s in sections] == ["1 引言", "2 方法"]
        assert section_for_block == {1: "1 引言", 3: "2 方法"}
        assert sections[0]["band"] == "低"
        assert sections[1]["band"] == "高"
        assert sections[1]["method_like"] is True
        assert sections[1]["n_windows"] == 1
        assert "方法/结果/结论段易误伤" in sections[1]["hint"]

    def test_no_heading_uses_leading_section(self) -> None:
        blocks = [_block(0, "paragraph")]
        sections, section_for_block = build_sections(blocks, [_window(0, 0.3)])
        assert sections[0]["title"] == "文首"
        assert section_for_block == {0: "文首"}


class TestRevisionPlan:
    def test_high_risk_block_tops_plan(self) -> None:
        fused = [_window(0, 0.2), _window(1, 0.2), _window(2, 0.9)]
        base = doc_level(fused)
        plan = revision_plan(fused, base, {0: "引言", 1: "引言", 2: "结果"})

        assert plan
        assert plan[0]["block"] == 2
        assert plan[0]["section"] == "结果"
        assert plan[0]["impact"] == base["score"] - plan[0]["score_without"]
        assert plan[0]["score_without"] < base["score"]
        assert plan[0]["band_without"] == "低"

    def test_empty_document_has_empty_plan(self) -> None:
        assert revision_plan([], doc_level([]), {}) == []
