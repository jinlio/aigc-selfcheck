"""结构过滤、中文分句、短句合并与滑动窗口（ALGORITHM.md §1）。

- 结构过滤：参考文献/公式/图注/代码/直接引文必须排除后再送检测，
  否则统计特征全被污染；被过滤 block 保留位置，报告灰显。
- 分句：分割符 。！？； + 后引号，保留分隔符；小数点等 ASCII 句点不受影响。
- 短句合并：句长 < 30 字的连续短句向后合并，直到 ≥ 30 字
  （PPL 在 <20 token 上方差爆炸，分类器也需要上下文）。
- 窗口：连续 5 句、~250 字上限先到为准，步长 1 句；窗口不跨 block
  （AI 辅助常是段落级混入，段落是天然语义边界）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .docio import Block
from .features import symbol_density

SENT_DELIMITERS = "。！？；"
CLOSERS = "”』」！）】"
MIN_SENT_CHARS = 30
WINDOW_MAX_SENTS = 5
WINDOW_MAX_CHARS = 250

_RE_REF_START = re.compile(r"^\[\d+\]")
_RE_GBT_MARKER = re.compile(r"\[[JMDNCP]\]")
_RE_CAPTION = re.compile(r"^(图|表|附图|附表|Fig\.?|Table)\s*\d+")
_RE_LONG_QUOTE = re.compile(r"“([^”]{50,})”")
_RE_LATEX_TRACE = re.compile(r"\\(frac|begin|sum|int|left|right|alpha|beta|gamma|delta|mu|sigma)")
_RE_ASSIGNMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\s*=\s*\S")


def _looks_like_formula(text: str) -> bool:
    """识别「S = w1 * ...」这类无中文的赋值式公式行。"""
    stripped = text.strip()
    if len(stripped) > 120:
        return False
    if sum("\u4e00" <= ch <= "\u9fff" for ch in stripped) > 2:
        return False
    return bool(_RE_ASSIGNMENT.match(stripped))


@dataclass
class Sentence:
    """block 内的一个句子（可能是短句合并的结果），char 偏移相对 block 文本。"""

    text: str
    start: int
    end: int


@dataclass
class Window:
    """一个检测窗口：连续若干句的拼接文本，带块内 char 偏移。"""

    index: int
    sentence_ids: list[int]
    text: str
    start: int
    end: int


def classify_block(block: Block) -> Block:
    """细化 block 的结构语义。只对读入时的 paragraph/heading 做甄别，其余原样返回。"""
    if block.kind not in ("paragraph", "heading"):
        return block
    text = block.text
    kind = block.kind

    if (
        _RE_REF_START.match(text)
        or _RE_GBT_MARKER.search(text)
        or "doi.org" in text.lower()
        or "doi:" in text.lower()
        or (len(text) < 30 and "参考文献" in text)
    ):
        kind = "reference"
    elif (
        _RE_LATEX_TRACE.search(text)
        or "$$" in text
        or symbol_density(text) > 0.30
        or _looks_like_formula(text)
    ):
        kind = "formula"
    elif _RE_CAPTION.match(text) and len(text) < 80:
        kind = "caption"
    else:
        covered = sum(m.end() - m.start() for m in _RE_LONG_QUOTE.finditer(text))
        if covered > len(text) * 0.5:
            kind = "quote"
    return Block(
        kind=kind,
        text=text,
        index=block.index,
        source=block.source,
        meta={**block.meta, "original_kind": block.kind},
    )


def detectable(block: Block) -> bool:
    """只有普通段落参与检测；标题/表格/代码/参考文献/公式/图注/引文一律排除。"""
    return block.kind == "paragraph"


def split_sentences(text: str) -> list[Sentence]:
    """按 。！？； 分句，保留分隔符并吞并紧随的后引号/括号。"""
    sentences: list[Sentence] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in SENT_DELIMITERS:
            j = i + 1
            while j < n and text[j] in CLOSERS:
                j += 1
            piece = text[start:j]
            if piece.strip():
                sentences.append(Sentence(text=piece, start=start, end=j))
            start = j
            i = j
        else:
            i += 1
    if start < n and text[start:].strip():
        sentences.append(Sentence(text=text[start:], start=start, end=n))
    return sentences


def merge_short_sentences(sentences: list[Sentence], min_chars: int = MIN_SENT_CHARS) -> list[Sentence]:
    """短句向后合并直到 ≥ min_chars；文末残余不足也输出（不再有后句可并）。"""
    merged: list[Sentence] = []
    buf: Sentence | None = None
    for sent in sentences:
        if buf is None:
            buf = Sentence(text=sent.text, start=sent.start, end=sent.end)
        else:
            buf = Sentence(text=buf.text + sent.text, start=buf.start, end=sent.end)
        if len(buf.text.strip()) >= min_chars:
            merged.append(buf)
            buf = None
    if buf is not None:
        merged.append(buf)
    return merged


def make_windows(
    sentences: list[Sentence],
    max_sents: int = WINDOW_MAX_SENTS,
    max_chars: int = WINDOW_MAX_CHARS,
) -> list[Window]:
    """滑动窗口（ALGORITHM §1.3：连续 5 句 / ~250 字上限先到为准，步长 1 句）。

    只在能凑满 max_sents 句时开窗（M = N − max_sents + 1 个整窗），
    尾部残窗丢弃——残窗过短会放大分类器方差；句子总数不足 max_sents
    的短段落特例产出单个窗口，保证小段落仍可检测。
    """
    windows: list[Window] = []
    short_paragraph = len(sentences) < max_sents
    for start in range(len(sentences)):
        picked: list[int] = []
        total = 0
        for k in range(start, min(start + max_sents, len(sentences))):
            length = len(sentences[k].text)
            if picked and total + length > max_chars:
                break
            picked.append(k)
            total += length
        if not picked:
            continue
        if not short_paragraph and start + max_sents > len(sentences):
            break  # 尾部不足整窗：之后的起点只会更短，全部丢弃
        text = "".join(sentences[k].text for k in picked)
        windows.append(
            Window(
                index=len(windows),
                sentence_ids=picked,
                text=text,
                start=sentences[picked[0]].start,
                end=sentences[picked[-1]].end,
            )
        )
    return windows


def prepare_block(block: Block) -> tuple[list[Sentence], list[Window]]:
    """对单个可检测 block 执行分句 → 短句合并 → 滑窗。"""
    sentences = merge_short_sentences(split_sentences(block.text))
    return sentences, make_windows(sentences)
