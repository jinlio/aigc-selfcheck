"""features 单元测试：风格计量特征与突发性。"""

from aigc_selfcheck.constants import STYLE_STATS
from aigc_selfcheck.features import (
    burstiness,
    char_mtld,
    cjk_count,
    colloquial_density,
    connective_density,
    de_density,
    mtld,
    period_comma_ratio,
    punct_density,
    punct_diversity,
    sentence_initial_rate,
    sentence_length_variance,
    symbol_density,
    window_features,
)
from aigc_selfcheck.fuse import burstiness_from_windows


class TestCounts:
    def test_cjk_count(self):
        assert cjk_count("中文 abc 123，。！") == 2

    def test_symbol_density(self):
        assert symbol_density("a=b+c") == 0.4
        assert symbol_density("纯中文文本") == 0.0

    def test_de_density(self):
        # 5 个汉字里 2 个「的」→ 400/千字
        assert abs(de_density("我的你的他") - 2 / 5 * 1000) < 1e-6


class TestConnectives:
    def test_word_hits(self):
        text = "首先，我们回顾背景。其次，我们提出方法。综上所述，本文有效。"
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        assert abs(connective_density(text) - 3 / cjk * 1000) < 1e-6

    def test_pattern_hits(self):
        text = "该方法不仅提升了精度，而且降低了成本，效果显著非常好啊。"
        assert connective_density(text) > 0

    def test_none(self):
        assert connective_density("今天天气很好我们去公园散步吧朋友。") == 0


class TestStyle:
    def test_punct_diversity(self):
        assert punct_diversity("有破折号——有省略号……还有分号；以及叹号！") == 4
        assert punct_diversity("只有句号。") == 0

    def test_punct_density(self):
        # 13 个汉字 2 个标点 → 2/13 千字密度
        text = "今天天气很好，我们去公园散步。"
        assert abs(punct_density(text) - 2 / 13 * 1000) < 1e-6

    def test_period_comma_ratio(self):
        # 两逗号一句号 → 0.5；无逗号取上限 10
        assert abs(period_comma_ratio("天气很好，我们散步，顺便吃饭。") - 0.5) < 1e-6
        assert period_comma_ratio("没有逗号只有句号。") == 10.0

    def test_sentence_initial_rate(self):
        sentences = ["因此我们认为该方法可行。", "实验结果支持这一结论。", "此外它开销很小。"]
        assert abs(sentence_initial_rate(sentences) - 2 / 3) < 1e-6
        assert sentence_initial_rate([]) == 0.0

    def test_colloquial(self):
        assert colloquial_density("其实这件事吧，反而挺有意思的呢，朋友。") > 0

    def test_length_variance(self):
        assert sentence_length_variance([10, 10, 10]) == 0.0
        assert sentence_length_variance([5, 15]) > sentence_length_variance([9, 11])

    def test_char_mtld_repetitive_low(self):
        assert char_mtld("的的的的的的的的的") < char_mtld("今天我去图书馆看了一本很有趣的书")

    def test_mtld_empty(self):
        assert mtld([]) == 0.0

    def test_mtld_diverse_high(self):
        diverse_text = (
            "清晨的图书馆里，阳光斜照在木桌上，几个学生正在低声讨论课题。"
            "我翻开一本旧杂志，看到一篇关于方言保护的长文，作者走访了十几座小城，"
            "记录了大量鲜活的口语材料，论证扎实又充满温度，读完让人久久回味。"
        )
        diverse = mtld([ch for ch in diverse_text if "\u4e00" <= ch <= "\u9fff"])
        repetitive = mtld(["的"] * 40)
        assert diverse > repetitive


class TestBurstiness:
    def test_constant_is_zero(self):
        assert burstiness([10.0, 10.0, 10.0]) == 0.0

    def test_varied_positive(self):
        assert burstiness([5.0, 50.0, 8.0, 60.0]) > 0.1

    def test_needs_two(self):
        assert burstiness([10.0]) == 0.0


class TestWindowBurstiness:
    def test_none_values_are_ignored(self):
        assert burstiness_from_windows([None, 10.0, 20.0]) > 0

    def test_all_missing_is_nan(self):
        value = burstiness_from_windows([None, float("nan"), 0.0])
        assert value != value


class TestWindowFeatures:
    def test_keys_present(self):
        text = "首先要有足够的汉字数量，这句话的长度必须超过合并阈值才行啊。"
        sentences = [text]
        feats = window_features(text, sentences)
        for key in (
            "connective_density",
            "de_density",
            "colloquial_density",
            "punct_diversity",
            "punct_density",
            "period_comma_ratio",
            "sentence_length_variance",
            "char_mtld",
            "sent_initial_rate",
            "cjk_count",
        ):
            assert key in feats

    def test_style_stats_drift_guard(self):
        """STYLE_STATS 的评分键必须都在 window_features 输出中。

        v3.2 教训：sent_initial_rate 曾只在拟合脚本里计算而未进生产特征管线，
        导致 constants 里的权重静默失效（style_score 跳过缺失键）。此测试防复发。
        """
        text = "因此我们继续推进这项工作。同时也要注意风险的存在。"
        feats = window_features(text, ["因此我们继续推进这项工作。", "同时也要注意风险的存在。"])
        missing = [name for name in STYLE_STATS if name not in feats]
        assert not missing, f"STYLE_STATS 引用了 window_features 未产出的特征: {missing}"
