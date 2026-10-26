"""管线端到端测试：mock 检测器跑通 读取→过滤→窗口→打分→融合→渲染。

黄金样本：一段 AI 风格长文 + 一段口语化人写风格 + 参考文献结构块。
CI 上不下载任何模型（AIGC_SELFCHECK_MOCK=1）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aigc_selfcheck.cache import ScoreCache
from aigc_selfcheck.detectors import build_detectors
from aigc_selfcheck.pipeline import analyze_document
from aigc_selfcheck.report import render_report
from aigc_selfcheck.report.model import build_payload


class TestPipeline:
    def test_blocks_classified(self, mock_result):
        kinds = [b.kind for b in mock_result.blocks]
        assert "paragraph" in kinds
        assert "reference" in kinds
        ref_block = next(b for b in mock_result.blocks if b.kind == "reference")
        assert not ref_block.detectable
        assert ref_block.note

    def test_windows_fused(self, mock_result):
        assert mock_result.fused
        for window in mock_result.fused:
            assert set(window.scores) == {"a", "b", "c"}
            assert 0.0 <= window.p_fuse <= 1.0
            assert window.delta == max(window.scores.values()) - min(window.scores.values())

    def test_ai_style_scores_higher(self, mock_result):
        """mock 规则下，含骨架词的段落窗口得分应显著高于口语段。"""
        by_block = {}
        for window in mock_result.fused:
            by_block.setdefault(window.block_index, []).append(window.p_fuse)
        blocks = [b for b in mock_result.blocks if b.detectable]
        ai_block = blocks[0].index
        human_block = blocks[1].index
        assert max(by_block[ai_block]) > max(by_block[human_block])

    def test_sentence_risks_in_range(self, mock_result):
        for block in mock_result.blocks:
            for sent in block.sentences:
                assert 0.0 <= sent.risk <= 1.0

    def test_doc_level(self, mock_result):
        assert mock_result.doc["band"] in ("低", "中", "高")
        assert mock_result.doc["n_windows"] == len(mock_result.fused)
        assert 0.0 <= mock_result.doc["median"] <= 1.0

    def test_reasons_generated_for_high_risk(self, mock_result):
        """高风险句应有白话原因和修改建议；技术指标只放在 detail。"""
        with_reasons = [s for b in mock_result.blocks for s in b.sentences if s.reasons]
        assert with_reasons
        assert all("σ" not in reason for s in with_reasons for reason in s.reasons)
        assert all(s.reason_items for s in with_reasons)
        assert all(
            item["label"] and item["why"] and item["suggestion"] and item["detail"]
            for s in with_reasons
            for item in s.reason_items
        )
        assert all(s.suggestions for s in with_reasons)

    def test_top_sentences_sorted(self, mock_result):
        top = mock_result.top_sentences(5)
        risks = [item["risk"] for item in top]
        assert risks == sorted(risks, reverse=True)


class TestPayloadAndRender:
    def test_payload_structure(self, mock_result):
        payload = build_payload(mock_result, title="测试报告")
        assert payload["schema"] == "aigc-selfcheck/report@v1"
        assert payload["disclaimer"]
        assert payload["doc"]["n_windows"] == len(payload["windows"])
        assert all("sentences" in w for w in payload["windows"])
        assert all("reason_items" in w and "suggestions" in w for w in payload["windows"])
        assert payload["blocks"][0]["detectable"] is True
        assert isinstance(payload["guidance"]["evidence"]["sufficient"], bool)
        assert payload["guidance"]["intro"]
        assert payload["guidance"]["suggestions"]
        assert any("不要为了降分" in item for item in payload["guidance"]["suggestions"])
        assert payload["calibration"]["version"] == "v8"
        assert isinstance(payload["sections"], list)
        assert isinstance(payload["plan"], list)
        assert all("review_note" in w for w in payload["windows"])
        sentences = [s for block in payload["blocks"] for s in block["sentences"]]
        assert all("reason_items" in s and "suggestions" in s for s in sentences)
        assert all("section" in s and "review_note" in s for s in sentences)
        assert "llm_advice" in payload
        assert all("llm_advice" in w for w in payload["windows"])
        assert all("llm_advice" in s for s in sentences)

    def test_long_text_reports_sufficient_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
        long_text = (
            "本研究首先回顾了近年来中文文本生成技术的演进历程，其次分析了主流检测方法的局限性，"
            "并在此基础上提出了一个多路融合的检测框架。"
        ) * 8
        path = tmp_path / "long.txt"
        path.write_text(long_text, encoding="utf-8")
        detectors = build_detectors("a,b,c", device="cpu")
        result = analyze_document(path, detectors, cache=None)
        payload = build_payload(result)
        assert payload["guidance"]["evidence"]["sufficient"] is True

    def test_short_text_reports_insufficient_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
        short = tmp_path / "short.txt"
        short.write_text("这是一段很短的文字，只有一个句子。", encoding="utf-8")
        detectors = build_detectors("a,b,c", device="cpu")
        result = analyze_document(short, detectors, cache=None)
        payload = build_payload(result)
        assert payload["guidance"]["evidence"]["sufficient"] is False
        assert "证据不足" in payload["guidance"]["evidence"]["note"]

    def test_render_report(self, mock_result, tmp_path: Path):
        payload = build_payload(mock_result, title="测试报告")
        out = tmp_path / "report.html"
        render_report(payload, out)
        html = out.read_text(encoding="utf-8")
        assert "__DATA__" not in html  # 占位符全部替换
        assert "__CSS__" not in html and "__APP__" not in html
        # 内联 JSON 可解析且不含 "</" 提前终止
        start = html.index('<script type="application/json" id="selfcheck-data">')
        end = html.index("</script>", start)
        data = json.loads(html[start + len('<script type="application/json" id="selfcheck-data">') : end])
        assert data["title"] == "测试报告"

    def test_payload_golden_snapshot(self, mock_result):
        """确定性 mock + 固定输入 → payload 除易变字段外逐字节一致。"""
        payload = build_payload(mock_result, title="测试报告")
        for volatile in ("generated_at", "duration_s", "source"):
            payload.pop(volatile)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        golden_path = Path(__file__).parent / "golden" / "payload_snapshot.json"
        if not golden_path.exists():
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(canonical, encoding="utf-8")
            pytest.skip("黄金快照首次生成")
        assert canonical == golden_path.read_text(encoding="utf-8")


class TestCache:
    def test_cache_roundtrip(self, golden_txt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")  # 与 CI/其余用例对齐，不依赖本机模型
        cache_path = tmp_path / "scores.jsonl"
        detectors = build_detectors("a,b,c", device="cpu")
        cache = ScoreCache(path=cache_path)
        result1 = analyze_document(golden_txt, detectors, cache)
        cache.flush()
        assert cache_path.exists()
        result2 = analyze_document(golden_txt, detectors, ScoreCache(path=cache_path))
        scores1 = [round(w.p_fuse, 9) for w in result1.fused]
        scores2 = [round(w.p_fuse, 9) for w in result2.fused]
        assert scores1 == scores2


class TestEnhancedMode:
    def test_enhanced_sets_meta_probability_and_payload(
        self, golden_txt: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
        detectors = build_detectors("a,b,c", device="cpu")
        result = analyze_document(golden_txt, detectors, cache=None, enhanced=True)
        assert result.enhanced is True
        assert result.meta_info["enabled"] is True
        assert all(w.meta_prob is not None for w in result.fused)
        assert all(0.0 <= w.meta_prob <= 1.0 for w in result.fused)
        payload = build_payload(result)
        assert payload["enhanced"]["enabled"] is True
        assert payload["enhanced"]["n_flagged"] == sum(w.enhanced for w in result.fused)
        assert len(payload["enhanced"]["windows"]) == payload["enhanced"]["n_flagged"]
        assert all("meta_prob" in w for w in payload["windows"])

    def test_enhanced_falls_back_without_three_lanes(
        self, golden_txt: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
        detectors = build_detectors("b,c", device="cpu")
        result = analyze_document(golden_txt, detectors, cache=None, enhanced=True)
        assert result.enhanced is False
        assert any("增强召回已跳过" in warning for warning in result.warnings)
        assert all(w.meta_prob is None for w in result.fused)


class TestLlmAdviceFallback:
    def test_mock_detector_falls_back_to_rules(
        self, golden_txt: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
        detectors = build_detectors("b,c", device="cpu")
        result = analyze_document(golden_txt, detectors, cache=None, llm_advice=True)
        assert result.llm_advice_info["enabled"] is False
        assert any("大模型建议已跳过" in warning for warning in result.warnings)
        assert all(w.llm_advice is None for w in result.fused)
        payload = build_payload(result)
        assert payload["llm_advice"]["enabled"] is False
        assert payload["windows"][0]["llm_advice"] is None
