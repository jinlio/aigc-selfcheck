"""文档读取：docx/pdf/txt/md → 有序 Block 流。

Block 是检测与报告的最外层定位单位：所有检测结果都落在 block 上，
被结构过滤排除的 block 保留原文顺序（index），供报告灰显标注「已跳过」。

docx 解析用 python-docx，保留段落/表格/标题结构；PDF 用 pypdf 提取文本，
按编号/章节标题启发式切分，扫描件与加密件给出明确错误；txt/md 按空行分段，
Markdown 围栏代码块整块识别为 code。中文纯文本的硬换行按续行拼接。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_RE_PDF_HEADING = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百]+[章节部分]"
    r"|[0-9]+(?:\.[0-9]+){0,3}\s+\S"
    r"|[一二三四五六七八九十]+、\S"
    r"|摘\s*要|Abstract|关键词|Key\s*words|参考文献|References|致谢|附录"
    r"|引言|绪论|相关工作|研究方法|实验|结果|讨论|结论"
    r")",
    re.IGNORECASE,
)

# docx 里大量论文用 Normal 样式手打标题：目录条目是 toc 样式，正文标题靠
# 大纲级别/字号/加粗/居中区分，封面字段是「标签＋空白＋值」。样式名只认
# Heading/Title 会把这些全部漏成正文。
_RE_DOCX_HEADING = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百]+[章节部分]"
    r"|\d{1,2}(?:\.\d{1,2}){0,3}[\.、]?\s+\S"
    r"|[（(][一二三四五六七八九十]+[）)]"
    r"|[一二三四五六七八九十]+、\S"
    r"|摘\s*要|Abstract|关键词|Key\s*words|参考文献|References|致谢|附录|目\s*录"
    r"|引言|绪论|相关工作|研究方法|实验|结果|讨论|结论"
    r")",
    re.IGNORECASE,
)
_RE_TOC_STYLE = re.compile(r"(?:toc|目录)\s*\d*", re.IGNORECASE)
_RE_TOC_LINE = re.compile(r"(?:\t|\.{3,}|…{2,})\s*\d{1,4}$")
# 封面字段是「标签＋空白＋值」；标签本身常被空格拆开（题    目）。用白名单
# 而不是通用「汉字＋空白」模式，避免把「摘  要」这类真标题误判成元数据。
_RE_COVER_FIELD = re.compile(
    r"^(?:题\s*目|学\s*院|专\s*业|学\s*号|学生姓名|姓\s*名|指导教师|起讫日期"
    r"|工作地点|班\s*级|院\s*系|学\s*校|日\s*期|作\s*者)[\u3000\s]+",
)
_RE_BRACKET_FIELD = re.compile(r"^【[^】]{1,10}】")


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _is_pdf_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return False
    if any(mark in stripped for mark in "。！？；"):
        return False
    return bool(_RE_PDF_HEADING.match(stripped))


def _join_pdf_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if not text:
            text = line
        elif _is_cjk(text[-1]) and _is_cjk(line[0]):
            text += line
        else:
            text += " " + line
    return text.strip()


def _pdf_text_to_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        joined = _join_pdf_lines(buffer)
        buffer.clear()
        if joined:
            blocks.append(Block(kind="paragraph", text=joined, index=len(blocks), source="pdf"))

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if _is_pdf_heading(line):
            flush()
            blocks.append(Block(kind="heading", text=line, index=len(blocks), source="pdf"))
            continue
        buffer.append(line)
        if line.endswith(("。", "！", "？", "；")):
            flush()
    flush()
    return blocks


def read_pdf(path: str | Path) -> list[Block]:
    """读取文本型 PDF；扫描件/加密件给出明确错误。"""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, ValueError) as exc:
        raise ValueError(f"PDF 读取失败：{exc}") from exc
    if reader.is_encrypted:
        raise ValueError("PDF 已加密，无法读取；请先解除密码保护，或提供 .docx/.txt/.md。")

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 —— 单页解析失败不应阻断整篇
            pages.append("")
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "PDF 没有可提取文本（可能是扫描件）；请先 OCR，或提供 .docx/.txt/.md。"
        )
    return _pdf_text_to_blocks(text)


def read_document(path: str | Path) -> list[Block]:
    """按扩展名分派读取。不支持的格式抛 ValueError。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return read_plaintext(path)
    raise ValueError(f"不支持的文件格式：{suffix}（支持 .docx/.pdf/.txt/.md）")


@dataclass
class Block:
    """一个结构块。kind 是检测前的结构语义，preprocess 阶段会进一步细化。"""

    kind: str  # heading | paragraph | table | code（读入时）；reference | formula | caption | quote（过滤时细化）
    text: str
    index: int
    source: str = ""  # 来源线索，如 docx 段落样式名
    meta: dict = field(default_factory=dict)


def _docx_outline_level(paragraph) -> int | None:
    """段落大纲级别（0–8）；未设置返回 None。"""
    try:
        pPr = paragraph._p.pPr
    except Exception:  # noqa: BLE001 —— 损坏的 XML 按无级别处理
        return None
    if pPr is None or pPr.outlineLvl is None:
        return None
    try:
        return int(pPr.outlineLvl.val)
    except (TypeError, ValueError):
        return None


def _docx_font_size(paragraph) -> float | None:
    """段落内非空 run 的最大字号（pt）；未显式设置返回 None。"""
    sizes = [run.font.size.pt for run in paragraph.runs if run.text.strip() and run.font.size]
    return max(sizes) if sizes else None


def _docx_all_bold(paragraph) -> bool:
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(bool(run.bold) for run in runs)


def _docx_paragraph_kind(paragraph, style: str) -> str:
    """按样式名 + 结构信号判定段落语义：heading / toc / metadata / paragraph。

    只看样式名会漏掉真实论文的常见排版：目录条目用 toc 样式，正文各级标题
    用 Normal 手打（靠大纲级别/字号/加粗区分），封面字段是「标签＋空白＋值」。
    """
    text = paragraph.text.strip()
    style_l = style.lower()

    if style.startswith(("Heading", "标题", "Title", "Subtitle")):
        return "heading"
    if style_l.startswith(("author", "date")):
        return "metadata"
    if _RE_TOC_STYLE.fullmatch(style) or style_l.startswith("toc"):
        return "toc"

    # 目录条目常被复制成正文样式：带制表符/点引线且以页码结尾
    if len(text) < 80 and _RE_TOC_LINE.search(text):
        return "toc"

    # 封面字段先于标题判定：「题　　目　xxx」既有大字号也可能带 outline
    if len(text) <= 40 and (_RE_COVER_FIELD.match(text) or _RE_BRACKET_FIELD.match(text)):
        return "metadata"

    outline = _docx_outline_level(paragraph)
    if outline is not None and outline <= 8:
        return "heading"

    if len(text) <= 40 and not re.search(r"[。！？；，、：]", text):
        if _RE_DOCX_HEADING.match(text):
            return "heading"
        size = _docx_font_size(paragraph)
        if size and size >= 16:
            return "heading"
        if _docx_all_bold(paragraph) and size and size >= 14:
            return "heading"

    return "paragraph"


def read_docx(path: str | Path) -> list[Block]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    blocks: list[Block] = []
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = (item.style.name if item.style is not None else "") or ""
            blocks.append(
                Block(
                    kind=_docx_paragraph_kind(item, style),
                    text=text,
                    index=len(blocks),
                    source=style,
                )
            )
        elif isinstance(item, Table):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in item.rows]
            blocks.append(Block(kind="table", text="\n".join(rows), index=len(blocks), source="table"))
    return blocks


def read_plaintext(path: str | Path) -> list[Block]:
    raw = Path(path).read_text(encoding="utf-8")
    blocks: list[Block] = []
    para_lines: list[str] = []
    fence_lines: list[str] = []
    in_fence = False

    def flush_para() -> None:
        if para_lines:
            text = "".join(para_lines).strip()
            if text:
                blocks.append(Block(kind="paragraph", text=text, index=len(blocks)))
            para_lines.clear()

    def flush_fence() -> None:
        if fence_lines:
            text = "\n".join(fence_lines).strip()
            if text:
                blocks.append(Block(kind="code", text=text, index=len(blocks)))
            fence_lines.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            if in_fence:
                flush_fence()
            else:
                flush_para()
            in_fence = not in_fence
            continue
        if in_fence:
            fence_lines.append(line)
            continue
        if not stripped:
            flush_para()
            continue
        if stripped.startswith("#"):
            flush_para()
            heading = stripped.lstrip("#").strip()
            if heading:
                blocks.append(Block(kind="heading", text=heading, index=len(blocks)))
            continue
        para_lines.append(stripped)
    flush_fence()
    flush_para()
    return blocks
