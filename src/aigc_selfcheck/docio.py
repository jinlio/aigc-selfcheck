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
            is_heading = style.startswith(("Heading", "标题", "Title", "Subtitle"))
            is_metadata = style.lower().startswith(("author", "date"))
            blocks.append(
                Block(
                    kind="heading" if is_heading else "metadata" if is_metadata else "paragraph",
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
