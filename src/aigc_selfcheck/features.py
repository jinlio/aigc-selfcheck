"""中文风格计量特征（统计特征路的可解释部分）。

对应 ALGORITHM.md §2.3 特征 3。全部特征返回原始值（多为密度），
归一化与合成打分在 stats_lm / constants 中处理——先把可解释性做出来，再谈学参数。

词汇多样性用字级 MTLD（不引入分词器依赖；对中文而言字级多样性
与词级高度相关，作为代理指标够用）。
"""

from __future__ import annotations

import re
import statistics

# AI 高频连接词/骨架词（ALGORITHM.md §2.3 表格）
AI_CONNECTIVES = [
    "首先",
    "其次",
    "再者",
    "总而言之",
    "综上所述",
    "值得注意的是",
    "与此同时",
    "总的来说",
]
AI_CONNECTIVE_PATTERNS = [
    r"不仅[^。！？；]{0,20}而且",
    r"一方面[^。！？；]{0,40}另一方面",
]

# 口语痕迹：学术 AI 文本几乎为 0，但手写论文也低——使用时降权
COLLOQUIAL_MARKERS = ["其实", "挺", "蛮", "吧", "呢", "反而"]

# 数学/公式符号集合（公式块识别用）
MATH_SYMBOLS = set("Σ∫≤≥∈≠≈±×÷∑∏√∞∂∇=<>+-*/^|\\{}$&~")

DASH = "——"
ELLIPSIS = "……"

# 全量中文标点（标点密度用；分母为汉字数）
PUNCT_CHARS = set("，。！？；：、""''（）——…·《》〔〕【】")


def cjk_count(text: str) -> int:
    """汉字计数（不含标点、空白、拉丁字符），作为所有密度的分母。"""
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def symbol_density(text: str) -> float:
    """数学符号密度。> 0.30 是公式块的识别阈值之一。"""
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in MATH_SYMBOLS) / len(text)


def connective_density(text: str) -> float:
    """AI 高频连接词密度：每千字出现次数。"""
    cjk = max(1, cjk_count(text))
    hits = sum(text.count(w) for w in AI_CONNECTIVES)
    hits += sum(len(re.findall(p, text)) for p in AI_CONNECTIVE_PATTERNS)
    return hits / cjk * 1000


def de_density(text: str) -> float:
    """「的」字密度：AI 中文显著偏高。"""
    cjk = max(1, cjk_count(text))
    return text.count("的") / cjk * 1000


def colloquial_density(text: str) -> float:
    """口语痕迹密度。使用时降权（见 constants）。"""
    cjk = max(1, cjk_count(text))
    return sum(text.count(w) for w in COLLOQUIAL_MARKERS) / cjk * 1000


def punct_diversity(text: str) -> int:
    """标点多样性：破折号/省略号/分号/叹号/问号/冒号出现种类数。人写更杂。"""
    kinds = 0
    for mark in (DASH, ELLIPSIS, "；", "！", "？", "："):
        if mark in text:
            kinds += 1
    return kinds


def punct_density(text: str) -> float:
    """全量标点密度：每千字标点数。新一代生成器长句少停顿，密度偏低（v3.2）。"""
    cjk = max(1, cjk_count(text))
    return sum(1 for ch in text if ch in PUNCT_CHARS) / cjk * 1000


def period_comma_ratio(text: str) -> float:
    """句读节奏：句号数 / 逗号数。生成器爱用长从句串联，比值偏低；无逗号时取上限 10。"""
    commas = text.count("，")
    return min(text.count("。") / commas, 10.0) if commas else 10.0


# 句首连接词（v3.2 节奏特征：生成器爱用「因此/此外/同时」开句）
SENT_INITIALS = (
    "因此", "然而", "此外", "同时", "另外", "总之", "综上", "值得注意",
    "总体而言", "换句话说", "具体而言", "例如", "这意味着", "更重要的是",
)


def sentence_initial_rate(sentences: list[str]) -> float:
    """以连接词开头的句子占比。"""
    if not sentences:
        return 0.0
    hits = sum(1 for s in sentences if s.startswith(SENT_INITIALS))
    return hits / len(sentences)


def sentence_length_variance(lengths: list[int]) -> float:
    """窗口内句长方差。AI 句长均匀，方差偏低。"""
    if len(lengths) < 2:
        return 0.0
    return statistics.pvariance(lengths)


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """标准 MTLD（Measure of Textual Lexical Diversity）。

    TTR 衰减到 threshold 记为一个 factor，正反两个方向计算后取均值；
    不足一个 factor 的余项按 (1 - TTR) / (1 - threshold) 折算。
    """
    def _run_factor_count(seq: list[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        token_count = 0
        for tok in seq:
            token_count += 1
            types.add(tok)
            if len(types) / token_count <= threshold:
                factors += 1
                types = set()
                token_count = 0
        if token_count > 0:
            ttr = len(types) / token_count
            factors += (1 - ttr) / (1 - threshold) if ttr < 1 else 0.0
        return factors

    if not tokens:
        return 0.0
    forward = _run_factor_count(tokens)
    backward = _run_factor_count(tokens[::-1])
    denom = forward + backward
    # 从未触发阈值的完美多样性序列：多样性取最大（序列长度）
    return len(tokens) / denom if denom > 0 else float(len(tokens))


def char_mtld(text: str) -> float:
    """字级 MTLD（去标点空白后的汉字序列）。"""
    chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    return mtld(chars)


def window_features(text: str, sentences: list[str]) -> dict[str, float]:
    """一个窗口的全部风格特征（原始值）。

    sentences 为窗口内句子文本列表（句长方差等在内部推导）。评分特征集由
    constants.STYLE_STATS 决定（v3.2 起为 train 拟合的 LR 权重）；未入选的
    特征（如 sentence_length_variance）仍计算，供报告面板展示。
    tests.test_features 有 drift-guard：STYLE_STATS 的键必须都在本输出中。
    """
    lengths = [len(s) for s in sentences]
    return {
        "connective_density": connective_density(text),
        "de_density": de_density(text),
        "colloquial_density": colloquial_density(text),
        "punct_diversity": float(punct_diversity(text)),
        "punct_density": punct_density(text),
        "period_comma_ratio": period_comma_ratio(text),
        "sentence_length_variance": sentence_length_variance(lengths),
        "char_mtld": char_mtld(text),
        "sent_initial_rate": sentence_initial_rate(sentences),
        "cjk_count": float(cjk_count(text)),
    }


def burstiness(ppl_values: list[float]) -> float:
    """突发性 B = std(PPL) / mean(PPL)。人写句间方差大，B 偏高；仅文档级使用。"""
    if len(ppl_values) < 2:
        return 0.0
    mean = statistics.fmean(ppl_values)
    if mean <= 0:
        return 0.0
    return statistics.pstdev(ppl_values) / mean
