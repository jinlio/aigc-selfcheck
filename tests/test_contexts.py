"""窗口上下文重建、上下文感知缓存键与管线集成。"""

from __future__ import annotations

from pathlib import Path

from aigc_selfcheck.cache import ScoreCache
from aigc_selfcheck.contexts import CONTEXT_TAIL_CHARS, preceding_context, window_contexts
from aigc_selfcheck.detectors.base import Detector, DetectorResult
from aigc_selfcheck.pipeline import analyze_document


class _RecordingDetector(Detector):
    path_key = "c"
    model_id = "recording/c"
    uses_context = True

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def detect(self, text: str, context: str = "") -> DetectorResult:
        return DetectorResult(0.5, {})

    def detect_batch(
        self, texts: list[str], contexts: list[str] | None = None
    ) -> list[DetectorResult]:
        self.calls.append((list(texts), list(contexts or [""] * len(texts))))
        return [DetectorResult(0.5, {}) for _ in texts]


def _unit(para: str, index: int, text: str) -> dict:
    return {
        "uid": f"human-{para[:12]}-{index}",
        "para_sha": para,
        "text": text,
    }


class TestWindowContexts:
    def test_preceding_context_never_includes_window(self):
        text = "前文" * 100 + "窗口正文"
        start = len(text) - len("窗口正文")
        assert preceding_context(text, start) == text[start - CONTEXT_TAIL_CHARS : start]
        assert "窗口正文" not in preceding_context(text, start)
        assert preceding_context(text, 0) == ""

    def test_stored_context_wins(self):
        units = [
            {"uid": "human-x-0", "para_sha": "x", "text": "甲", "context": ""},
            {"uid": "human-x-1", "para_sha": "x", "text": "乙", "context": "真正的窗口前文"},
        ]
        assert window_contexts(units) == ["", "真正的窗口前文"]

    def test_previous_window_tail(self):
        first = "甲" * (CONTEXT_TAIL_CHARS + 20)
        second = "乙" * 10
        third = "丙" * 5
        units = [
            _unit("p1" * 32, 0, first),
            _unit("p1" * 32, 1, second),
            _unit("p1" * 32, 2, third),
        ]
        contexts = window_contexts(units)
        assert contexts[0] == ""
        assert contexts[1] == first[-CONTEXT_TAIL_CHARS:]
        assert contexts[2] == second

    def test_paragraph_boundary_resets(self):
        units = [
            _unit("a" * 64, 0, "第一段首窗"),
            _unit("a" * 64, 1, "第一段次窗"),
            _unit("b" * 64, 0, "第二段首窗"),
        ]
        contexts = window_contexts(units)
        assert contexts == ["", "第一段首窗", ""]

    def test_gap_does_not_borrow_older_window(self):
        units = [
            _unit("c" * 64, 0, "窗口零"),
            _unit("c" * 64, 2, "窗口二"),
        ]
        assert window_contexts(units) == ["", ""]

    def test_missing_fields_are_safe(self):
        units = [{"uid": "bad", "text": "x"}, {"uid": "human-x-0", "text": "y"}]
        assert window_contexts(units) == ["", ""]


class TestContextAwareCache:
    def test_empty_context_keeps_legacy_key(self):
        assert ScoreCache.key("c", "model", "text") == ScoreCache.key(
            "c", "model", "text", context=""
        )

    def test_context_changes_key(self):
        base = ScoreCache.key("c", "model", "text")
        assert ScoreCache.key("c", "model", "text", context="上文") != base

    def test_get_put_isolated_by_context(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        cache = ScoreCache(path=path)
        cache.put("c", "model", "text", {"score": 0.1})
        cache.put("c", "model", "text", {"score": 0.9}, context="上文")
        cache.flush()

        reloaded = ScoreCache(path=path)
        assert reloaded.get("c", "model", "text")["score"] == 0.1
        assert reloaded.get("c", "model", "text", context="上文")["score"] == 0.9
        assert reloaded.get("c", "model", "text", context="别的上文") is None


class TestPipelineContext:
    def test_pipeline_uses_preceding_text_not_previous_window(self, tmp_path: Path):
        first = "第一句先说明研究背景与问题边界，这句话写得足够长以避免被合并。"
        second = "第二句介绍数据来源与实验设置，并说明样本筛选、清洗与标注的具体流程。"
        third = "第三句总结主要结论与局限，并给出后续工作的方向与可验证的改进建议。"
        doc = tmp_path / "doc.txt"
        doc.write_text(first + second + third, encoding="utf-8")

        detector = _RecordingDetector()
        analyze_document(doc, [detector])

        texts, contexts = detector.calls[0]
        assert len(texts) == 3  # 短段落：3 个后缀窗口
        assert contexts[0] == ""
        assert contexts[1] == first
        # 错误实现会取「前一个重叠窗口的尾部」，这里必须是窗口起点之前的正文
        assert contexts[2] == first + second
