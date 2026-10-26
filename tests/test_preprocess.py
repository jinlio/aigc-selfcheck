"""preprocess 单元测试：分句、短句合并、滑窗与结构过滤。"""

from aigc_selfcheck.docio import Block
from aigc_selfcheck.preprocess import (
    classify_block,
    detectable,
    make_windows,
    merge_short_sentences,
    prepare_block,
    split_sentences,
)


def _sents(text):
    return [s.text for s in split_sentences(text)]


class TestSplitSentences:
    def test_basic(self):
        text = "这是第一句。这是第二句！这是第三句？还有一句；"
        assert _sents(text) == ["这是第一句。", "这是第二句！", "这是第三句？", "还有一句；"]

    def test_closers_swallowed(self):
        text = "他说：“实验成功了。”然后继续。"
        assert _sents(text) == ["他说：“实验成功了。”", "然后继续。"]

    def test_trailing_without_delimiter(self):
        text = "第一句。结尾没有分隔符"
        assert _sents(text) == ["第一句。", "结尾没有分隔符"]

    def test_decimal_untouched(self):
        # ASCII 句点不是分隔符，小数不受影响
        text = "误差为 3.5 个百分点。结论成立。"
        assert _sents(text) == ["误差为 3.5 个百分点。", "结论成立。"]

    def test_empty(self):
        assert _sents("") == []


class TestMergeShort:
    def test_merge_until_long_enough(self):
        # 累计合并语义：缓冲不足 30 字就一直向后吞，达标即整块输出
        second = "第二句需要写得足够长一些这样两句加在一起就能超过三十个字符了。"  # 31 字
        third = "这一句足够长可以直接达标了不需要和前面的短句合并在一起。"  # 28 字
        sentences = split_sentences("短句一。" + second + third)
        merged = merge_short_sentences(sentences)
        assert [s.text for s in merged] == ["短句一。" + second, third]

    def test_cumulative_merge_across_three(self):
        third = "这一句足够长可以直接达标了不需要和前面的短句合并在一起。"  # 28 字
        sentences = split_sentences("短句一。短句二。" + third)
        merged = merge_short_sentences(sentences)
        # 4 + 4 = 8 仍不足，吞入第三句后 36 ≥ 30 → 三句合一块输出
        assert len(merged) == 1
        assert merged[0].text == "短句一。短句二。" + third

    def test_tail_kept_even_if_short(self):
        first = "这是一句长度超过三十个字符的句子可以直接输出不需要合并的没问题呀。"  # 32 字
        sentences = split_sentences(first + "短尾。")
        merged = merge_short_sentences(sentences)
        assert [s.text for s in merged] == [first, "短尾。"]

    def test_offsets_cover_original(self):
        text = "短句一。短句二。这一句足够长可以直接达标了不用合并。"
        merged = merge_short_sentences(split_sentences(text))
        for s in merged:
            assert text[s.start : s.end] == s.text


class TestWindows:
    def test_stride_and_size(self):
        sent = "一二三四五六七八九十。" * 3  # 单句 33 字，不触发合并
        sentences = merge_short_sentences(split_sentences(sent * 6))
        windows = make_windows(sentences)
        assert len(windows) == len(sentences) - 4
        assert windows[0].sentence_ids == [0, 1, 2, 3, 4]
        assert windows[1].sentence_ids == [1, 2, 3, 4, 5]

    def test_char_cap(self):
        from aigc_selfcheck.preprocess import Sentence

        long_sent = "这" * 130 + "。"  # 131 字
        sentences = [
            Sentence(long_sent, 0, 131),
            Sentence(long_sent, 131, 262),
        ]
        windows = make_windows(sentences, max_sents=5, max_chars=250)
        # 两句合计 262 字超过 250 上限：第二个句被拒，窗口只装一句
        assert len(windows[0].sentence_ids) == 1

    def test_fewer_than_five(self):
        sentences = merge_short_sentences(split_sentences("只有一句但是这句话肯定超过三十个字符的长度要求了呀。"))
        windows = make_windows(sentences)
        assert len(windows) == 1
        assert windows[0].sentence_ids == [0]


def _block(text, kind="paragraph"):
    return Block(kind=kind, text=text, index=0)


class TestClassifyBlock:
    def test_reference_bracket_num(self):
        assert classify_block(_block("[1] 张三. 论文标题[J]. 期刊, 2020.")).kind == "reference"

    def test_reference_gbt(self):
        assert classify_block(_block("李四. 某研究[M]. 北京: 出版社, 2019.")).kind == "reference"

    def test_formula_latex(self):
        assert classify_block(_block("损失函数为 $\\frac{1}{n}\\sum_i L(y_i, \\hat y_i)$")).kind == "formula"

    def test_formula_symbol_density(self):
        text = "a=b+c*d/e^f | x<y>=z {} $$ && __ ~~ " + "短"
        assert classify_block(_block(text)).kind == "formula"

    def test_formula_assignment_line(self):
        assert classify_block(_block("S = w1 * S_sens + w2 * S_know + w3 * S_time")).kind == "formula"
        assert classify_block(_block("R_sens = sum(match_count_i) / (N_records * N_fields)")).kind == "formula"

    def test_prose_with_equals_is_not_formula(self):
        assert classify_block(_block("The result is x = y.")).kind == "paragraph"
        assert classify_block(_block("实验结果表明，准确率 = 0.92，召回率 = 0.88。")).kind == "paragraph"

    def test_caption(self):
        assert classify_block(_block("图 1 系统总体架构示意")).kind == "caption"
        assert classify_block(_block("表 2 各组实验对比结果")).kind == "caption"

    def test_quote_dominant(self):
        quote = "“" + "这是一段很长的直接引文。" * 8 + "”"
        assert classify_block(_block(quote)).kind == "quote"

    def test_normal_paragraph_untouched(self):
        text = "本文提出了一种面向中文文本的检测方法，并在多个数据集上验证了有效性。"
        assert classify_block(_block(text)).kind == "paragraph"

    def test_table_not_reclassified(self):
        b = _block("方法 | 指标\nA | 0.9", kind="table")
        assert classify_block(b).kind == "table"

    def test_detectable_only_paragraph(self):
        assert detectable(_block("正文段落。"))
        assert not detectable(_block("1 绪论", kind="heading"))
        assert not detectable(_block("[1] 文献[J].", kind="reference"))


class TestPrepareBlock:
    def test_offsets_and_windows(self):
        text = "第一段的第一句内容比较长一些用来过合并的门槛线。" * 2 + "结尾一句。"
        block = _block(text)
        sentences, windows = prepare_block(block)
        assert windows
        for s in sentences:
            assert text[s.start : s.end] == s.text
        for w in windows:
            assert text[w.start : w.end] == w.text
