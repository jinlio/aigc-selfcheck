"""可选大模型建议层：prompt、安全清洗、失败兜底。"""

from __future__ import annotations

from aigc_selfcheck.fuse import FusedWindow
from aigc_selfcheck.llm_advice import build_prompt, clean_advice, generate_for_windows

SOURCE = "随着人工智能技术的不断发展，其在教育领域的应用日益广泛。"


def make_window(risk: float = 0.9, text: str = SOURCE, key: str = "0:0") -> FusedWindow:
    return FusedWindow(
        window_key=key,
        block_index=0,
        sentence_ids=[0],
        text=text,
        scores={"b": risk, "c": risk},
        p_fuse=risk,
        delta=0.0,
        reason_items=[
            {
                "label": "模板化开头",
                "why": "开头像套话。",
                "suggestion": "换成具体场景。",
                "detail": "模板开头",
                "evidence": "随着…不断发展",
            }
        ],
    )


class FakeGenerator:
    model_id = "fake/base"

    def __init__(self, outputs):
        self.outputs = outputs

    def generate_batch(self, prompts, max_new_tokens=80):
        return self.outputs


class TestPrompt:
    def test_prompt_contains_source_and_evidence(self):
        prompt = build_prompt(make_window())
        assert SOURCE in prompt
        assert "模板化开头" in prompt
        assert "随着…不断发展" in prompt
        assert "不改变原意" in prompt


class TestCleanAdvice:
    def test_accepts_related_rewrite(self):
        text = clean_advice("改写：人工智能进入课堂的速度很快，但真正落到教学环节的案例仍然不多。", SOURCE)
        assert text
        assert text.startswith("人工智能")

    def test_rejects_new_numbers(self):
        assert clean_advice("人工智能在教育中的应用扩大了30%，效果显著。", SOURCE) is None

    def test_rejects_verbatim_copy(self):
        assert clean_advice(SOURCE, SOURCE) is None

    def test_rejects_new_claim_terms(self):
        output = "人工智能在教育中的应用日益广泛，能够有效提升教学的准确性和鲁棒性。"
        assert clean_advice(output, SOURCE) is None

    def test_allows_claim_terms_present_in_source(self):
        source = "本文方法在鲁棒性和准确性上均有提升。"
        output = "本文方法在鲁棒性和准确性上都有提升。"
        assert clean_advice(output, source)

    def test_rejects_unrelated_text(self):
        assert clean_advice("今天天气很好，我们一起去公园散步。", SOURCE) is None

    def test_rejects_meta_or_refusal(self):
        assert clean_advice("抱歉，我无法完成这个请求，请提供更多信息。", SOURCE) is None
        assert clean_advice("这段文字检测器认为有问题，建议修改。", SOURCE) is None


class TestGenerate:
    def test_returns_advice_for_high_risk_window(self):
        advice, info = generate_for_windows(
            [make_window()],
            [FakeGenerator(["人工智能进入课堂的速度很快，但真正落到教学环节的案例仍然不多。"])],
        )
        assert info["enabled"] is True
        assert info["n_generated"] == 1
        assert advice["0:0"]["kind"] == "rewrite"
        assert advice["0:0"]["experimental"] is True

    def test_skips_low_risk_windows(self):
        advice, info = generate_for_windows([make_window(risk=0.3)], [FakeGenerator([])])
        assert advice == {}
        assert info["enabled"] is True
        assert info["n_generated"] == 0
        assert "0.55" in info["reason"]

    def test_no_generator_falls_back(self):
        advice, info = generate_for_windows([make_window()], [object()])
        assert advice == {}
        assert info["enabled"] is False
        assert "规则建议" in info["reason"]

    def test_invalid_output_falls_back(self):
        advice, info = generate_for_windows(
            [make_window()],
            [FakeGenerator(["今天天气很好，我们一起去公园散步。"])],
        )
        assert advice == {}
        assert info["enabled"] is True
        assert info["n_generated"] == 0
        assert "安全校验" in info["reason"]

    def test_generation_error_does_not_raise(self):
        class Broken:
            model_id = "broken"

            def generate_batch(self, prompts, max_new_tokens=80):
                raise RuntimeError("out of memory")

        advice, info = generate_for_windows([make_window()], [Broken()])
        assert advice == {}
        assert info["enabled"] is False
        assert "生成失败" in info["reason"]
