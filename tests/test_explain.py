"""解释层测试：白话原因、修改建议、技术细节分层。"""

from aigc_selfcheck.explain import (
    dedupe_reason_items,
    reason_items_for,
    reasons_for,
    sentence_reason_items,
    suggestions_for,
)


class TestReasonItems:
    def test_ppl_reason_is_plain_and_actionable(self):
        items = reason_items_for({"p_ppl": 0.9, "ppl": 12.0, "log_ppl": 2.5})
        assert items[0]["feature"] == "ppl"
        assert "语言模型" in items[0]["why"]
        assert "数据" in items[0]["suggestion"]
        assert "PPL" in items[0]["detail"]
        assert "σ" not in items[0]["why"]

    def test_style_reason_has_plain_why_and_tech_detail(self):
        items = reason_items_for(
            {"z_values": {"connective_density": 2.1}, "features": {"connective_density": 12.4}}
        )
        assert items[0]["label"] == "连接词偏多"
        assert "首先" in items[0]["why"]
        assert "删掉" in items[0]["suggestion"]
        assert "σ" not in items[0]["why"]
        assert "σ" in items[0]["detail"]

    def test_de_density_suggestion_matches_low_density_direction(self):
        items = reason_items_for({"z_values": {"de_density": 2.0}})
        assert "少" in items[0]["why"]
        assert "偏高" not in items[0]["suggestion"]
        assert "补充" in items[0]["suggestion"]

    def test_char_mtld_suggestion_matches_high_diversity_direction(self):
        items = reason_items_for({"z_values": {"char_mtld": 2.0}})
        assert "同义替换" in items[0]["suggestion"] or "术语" in items[0]["suggestion"]
        assert "偏低" not in items[0]["suggestion"]

    def test_top_k_and_empty(self):
        meta = {
            "z_values": {
                "connective_density": 2.1,
                "punct_diversity": 2.0,
                "sent_initial_rate": 1.9,
                "de_density": 1.8,
            }
        }
        assert len(reason_items_for(meta, top_k=2)) == 2
        assert reason_items_for({}) == []

    def test_reasons_for_returns_plain_strings(self):
        reasons = reasons_for({"z_values": {"connective_density": 2.1}})
        assert reasons
        assert all("σ" not in reason for reason in reasons)

    def test_suggestions_are_deduplicated(self):
        items = reason_items_for(
            {
                "z_values": {
                    "connective_density": 2.1,
                    "sent_initial_rate": 1.9,
                }
            }
        )
        suggestions = suggestions_for(items)
        assert suggestions
        assert len(suggestions) == len(set(suggestions))


class TestTextEvidence:
    def test_template_opening_quotes_source(self):
        text = "随着人工智能技术的不断发展，其在教育领域的应用日益广泛。"
        items = reason_items_for({"p_ppl": 0.9}, text=text, risk=0.8)
        labels = [item["label"] for item in items]
        assert "模板化开头" in labels
        template = next(item for item in items if item["label"] == "模板化开头")
        assert template["evidence"] == "随着"
        assert "随着" in template["suggestion"]

    def test_empty_summary_and_vague_claim(self):
        text = "实验取得了显著效果。综上所述，本文的方法具有重要的现实意义。"
        items = reason_items_for({"p_ppl": 0.9}, text=text, risk=0.8)
        labels = [item["label"] for item in items]
        assert "总结句空泛" in labels
        assert "结论空泛、缺具体信息" in labels
        assert all(item.get("evidence") for item in items if item["feature"].startswith("text:"))

    def test_repeated_openings_are_detected(self):
        text = "因此，我们改进了方法。因此，实验更稳定。因此，结论更可靠。"
        items = reason_items_for({}, text=text, risk=0.8)
        repeated = next(item for item in items if item["label"] == "句首重复")
        assert "3 句" in repeated["why"]
        assert "因此" in repeated["evidence"]

    def test_long_sentence_is_detected(self):
        text = (
            "本研究在三个公开数据集上分别比较了基线方法、消融版本和完整模型的表现，"
            "并在不同随机种子下重复了五次实验以检查结果的稳定性，"
            "同时对每个模块的贡献进行了单独分析，最后给出了误差来源的讨论。"
        )
        items = reason_items_for({}, text=text, risk=0.8)
        assert any(item["label"] == "单句过长" for item in items)

    def test_low_risk_skips_text_evidence(self):
        text = "随着人工智能技术的不断发展，其在教育领域的应用日益广泛。"
        items = reason_items_for({"z_values": {"connective_density": 2.1}}, text=text, risk=0.2)
        assert all(not item["feature"].startswith("text:") for item in items)


class TestSentenceEvidence:
    """句级理由不能引用同窗口其它句子的词句。"""

    def test_evidence_stays_in_sentence(self):
        text = "本研究首先回顾了相关研究，其次分析了现有方法的局限。"
        items = sentence_reason_items(
            {"z_values": {"connective_density": 2.1}, "p_ppl": 0.9},
            text,
            0.8,
        )
        assert items
        assert all("综上所述" not in item.get("evidence", "") for item in items)
        assert all("综上所述" not in item.get("why", "") for item in items)
        connective = next(item for item in items if item["feature"] == "connective_density")
        assert "首先、其次" in connective["evidence"]

    def test_same_evidence_is_not_repeated(self):
        text = "综上所述，本文的方法具有重要的现实意义。"
        items = sentence_reason_items(
            {"z_values": {"connective_density": 2.1}, "p_ppl": 0.9},
            text,
            0.8,
        )
        evidences = [item.get("evidence") for item in items]
        assert evidences.count("综上所述") == 1

    def test_sentence_without_local_evidence_explains_window_context(self):
        text = "该模块在实现上采用了两阶段策略。"
        items = sentence_reason_items(
            {"z_values": {"connective_density": 2.1}},
            text,
            0.8,
        )
        assert len(items) == 1
        assert items[0]["feature"] == "window_context"
        assert "所在段落" in items[0]["label"] + items[0]["why"]

    def test_low_risk_sentence_without_local_evidence_is_empty(self):
        items = sentence_reason_items(
            {"z_values": {"connective_density": 2.1}},
            "该模块采用两阶段策略。",
            0.2,
        )
        assert items == []

    def test_low_risk_sentence_skips_window_statistics(self):
        items = sentence_reason_items(
            {"z_values": {"char_mtld": 2.0}},
            "该模块采用两阶段策略。",
            0.2,
        )
        assert items == []

    def test_cross_sentence_feature_is_skipped(self):
        items = sentence_reason_items(
            {"z_values": {"sent_initial_rate": 2.1}},
            "首先，我们收集数据。",
            0.8,
        )
        assert all(item["feature"] != "sent_initial_rate" for item in items)

    def test_overlapping_connectives_are_not_double_counted(self):
        text = "与此同时，检测器在跨领域文本上的表现明显下降。"
        items = reason_items_for(
            {"z_values": {"connective_density": 2.1}},
            text=text,
            risk=0.8,
        )
        connective = next(item for item in items if item["feature"] == "connective_density")
        terms = connective["evidence"].split("、")
        assert "与此同时" in terms
        assert "同时" not in terms

    def test_openings_use_whole_connectives(self):
        text = "首先，收集数据。其次，清洗数据。因此，结果更稳定。"
        items = reason_items_for(
            {"z_values": {"sent_initial_rate": 2.0}},
            text=text,
            risk=0.8,
        )
        item = next(item for item in items if item["feature"] == "sent_initial_rate")
        assert "首先" in item["evidence"]
        assert "首先，收" not in item["evidence"]

    def test_single_connective_opening_is_not_called_multi(self):
        text = "综上所述，本文方法有效。未来的工作会继续验证。"
        items = reason_items_for(
            {"z_values": {"sent_initial_rate": 2.0}},
            text=text,
            risk=0.8,
        )
        item = next(item for item in items if item["feature"] == "sent_initial_rate")
        assert item["label"] == "句首用连接词"
        assert item["evidence"] == "综上所述"

    def test_connective_openings_exclude_subject_prefixes(self):
        text = "实验结果表明方法有效。首先，我们收集数据。其次，我们清洗数据。"
        items = reason_items_for(
            {"z_values": {"sent_initial_rate": 2.0}},
            text=text,
            risk=0.8,
        )
        item = next(item for item in items if item["feature"] == "sent_initial_rate")
        assert "实验" not in item["evidence"]
        assert "首先、其次" in item["evidence"]


class TestReasonDedupe:
    def test_same_evidence_is_kept_once(self):
        items = reason_items_for(
            {"z_values": {"connective_density": 2.1}},
            text="综上所述，本文方法有效。",
            risk=0.8,
        )
        deduped = dedupe_reason_items(items)
        evidences = [item.get("evidence") for item in deduped]
        assert evidences.count("综上所述") == 1
