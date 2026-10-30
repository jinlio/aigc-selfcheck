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
    r"|符号说明|符号表|术语表|缩略语表|变量说明"
    r")",
    re.IGNORECASE,
)
# 自定义论文样式常把级别写在样式名里（FY H1 / Thesis H2 / 标题3），
# 但 outlineLvl 未必设置，段落级属性也读不到。
_RE_STYLE_HEADING = re.compile(r"(?:^|[^a-z])(?:h|heading|标题)\s*[1-9](?![0-9])", re.IGNORECASE)
# 「外文翻译：」「译文正文：」这类纯标签行，不是正文。用白名单而不是通用
# 「短语＋冒号」，避免把「具体如下：」这类正文引导句也跳掉。
_RE_LABEL_LINE = re.compile(
    r"^(?:外文翻译|译文正文|原文正文|原文|译文|文献综述|案例分析|数据来源)[：:]$"
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
    # 目录行：点引线/制表符接页码，或纯「标题 .... 7」
    if _RE_TOC_LINE.search(stripped):
        return True
    if _RE_DOCX_HEADING.match(stripped) and not re.search(r"[，、：]", stripped):
        return True
    # 「致  谢」这类两端留空的短标题
    if re.fullmatch(r"[\u4e00-\u9fff]{1,3}[\u3000 ]{2,}[\u4e00-\u9fff]{1,3}", stripped):
        return True
    return bool(_RE_PDF_HEADING.match(stripped))


def _is_pdf_cover_field(line: str) -> bool:
    """PDF 抽出的封面字段行（题　　目　…、指导教师　…、【关键词】…）。"""
    stripped = line.strip()
    return len(stripped) <= 40 and bool(
        _RE_COVER_FIELD.match(stripped) or _RE_BRACKET_FIELD.match(stripped)
    )


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


def _demote_leading_short_blocks(blocks: list[Block], max_scan: int = 3) -> list[Block]:
    """把文档开头连续的短段落降为 metadata（封面首行常无任何标记）。

    只在整篇块数足够（≥10）时生效，最多看前 3 块，且只处理 ≤20 字、不含
    句末标点的段落——正常正文的首段很少同时满足这三条。
    """
    if len(blocks) < 10:
        return blocks
    for block in blocks[:max_scan]:
        if block.kind != "paragraph":
            break
        text = block.text.strip()
        if len(text) > 20 or re.search(r"[。！？；]", text):
            break
        block.kind = "metadata"
    return blocks


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
        if _is_pdf_cover_field(line):
            flush()
            blocks.append(Block(kind="metadata", text=line, index=len(blocks), source="pdf"))
            continue
        if _is_pdf_heading(line):
            flush()
            kind = "toc" if _RE_TOC_LINE.search(line) else "heading"
            blocks.append(Block(kind=kind, text=line, index=len(blocks), source="pdf"))
            continue
        buffer.append(line)
        if line.endswith(("。", "！", "？", "；")):
            flush()
    flush()
    return _demote_leading_short_blocks(blocks)


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


def _style_outline_level(paragraph) -> int | None:
    """沿样式继承链读大纲级别。

    大量中文论文用自定义样式（`FY H1`、`标题 1`、`ThesisH2` …）承载标题级别，
    outlineLvl 定义在 style 上而不是段落 pPr 上。只看段落属性会把整份文档的
    标题漏成正文。
    """
    try:
        style = paragraph.style
    except Exception:  # noqa: BLE001 —— 缺样式定义按无级别处理
        return None
    seen: set[int] = set()
    while style is not None and id(style) not in seen:
        seen.add(id(style))
        try:
            pPr = style.element.pPr
        except Exception:  # noqa: BLE001
            return None
        if pPr is not None and pPr.outlineLvl is not None:
            try:
                return int(pPr.outlineLvl.val)
            except (TypeError, ValueError):
                return None
        try:
            style = style.base_style
        except Exception:  # noqa: BLE001
            return None
    return None


def _docx_outline_level(paragraph) -> int | None:
    """段落大纲级别（0–8）：先看段落自身，再回落到样式继承链。"""
    try:
        pPr = paragraph._p.pPr
    except Exception:  # noqa: BLE001 —— 损坏的 XML 按无级别处理
        return None
    if pPr is not None and pPr.outlineLvl is not None:
        try:
            return int(pPr.outlineLvl.val)
        except (TypeError, ValueError):
            return None
    return _style_outline_level(paragraph)


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
    # 论文模板常自带 FY H1 / Thesis H2 这类样式名，级别写在名字里
    if len(text) <= 40 and _RE_STYLE_HEADING.search(style) and style_l not in {"h1", "h2", "h3"}:
        return "heading"
    if style_l.startswith(("author", "date")):
        return "metadata"
    if _RE_TOC_STYLE.fullmatch(style) or style_l.startswith("toc"):
        return "toc"

    # 目录条目常被复制成正文样式：带制表符/点引线且以页码结尾
    if len(text) < 80 and _RE_TOC_LINE.search(text):
        return "toc"

    # 「外文翻译：」「译文正文：」这类纯标签行
    if len(text) <= 12 and _RE_LABEL_LINE.match(text):
        return "metadata"

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


_RE_MD_LIST = re.compile(r"^(?:[-*+]\s|\[[ xX]\]\s)")
# 「1. 研究背景」这类有序编号更像章节标题，只有明显带列表语义（缩进/多项/
# 长内容）才当列表；这里用「序号 + 空格 + 短标题且无句读」排除。
_RE_MD_ORDERED = re.compile(r"^\d+[.)]\s")
_RE_MD_TABLE = re.compile(r"^\s*\|.*\|\s*$")


def _plaintext_structural_kind(line: str) -> str | None:
    """纯文本行里可识别的结构块（无 Markdown 标记时的兜底）。

    返回 heading / toc / metadata；普通正文返回 None。只在行很短时才判定，
    避免把正常长句误杀。
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return None
    if _RE_TOC_LINE.search(stripped):
        return "toc"
    if _RE_COVER_FIELD.match(stripped) or _RE_BRACKET_FIELD.match(stripped):
        return "metadata"
    if re.search(r"[。！？；，、：]", stripped):
        return None
    if _RE_DOCX_HEADING.match(stripped):
        return "heading"
    # 「致  谢」这类两端留空的两字标题：中间有多个空格、两侧各 ≤3 个汉字
    if re.fullmatch(r"[\u4e00-\u9fff]{1,3}[\u3000 ]{2,}[\u4e00-\u9fff]{1,3}", stripped):
        return "heading"
    return None


def _is_md_list_item(line: str) -> bool:
    """Markdown 列表项判定。

    「1. 研究背景」这类短标题用有序编号，但它是章节标题不是列表项；
    列表项通常有缩进、含句读，或是链接/勾选框。
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _RE_MD_LIST.match(stripped):
        return True
    if _RE_MD_ORDERED.match(stripped):
        # 缩进的有序项一律算列表；顶格的短编号行（无句读、≤20 字）当标题
        indented = line[: len(line) - len(line.lstrip())] != ""
        body = _RE_MD_ORDERED.sub("", stripped)
        if indented or len(body) > 20 or re.search(r"[。！？；，、：]", body):
            return True
        return False
    return False


def read_plaintext(path: str | Path) -> list[Block]:
    raw = Path(path).read_text(encoding="utf-8")
    blocks: list[Block] = []
    para_lines: list[str] = []
    fence_lines: list[str] = []
    table_lines: list[str] = []
    list_lines: list[str] = []
    in_fence = False
    in_table = False
    in_list = False

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

    def flush_table() -> None:
        if table_lines:
            text = "\n".join(table_lines).strip()
            if text:
                blocks.append(Block(kind="table", text=text, index=len(blocks), source="markdown"))
            table_lines.clear()

    def flush_list() -> None:
        if list_lines:
            text = "\n".join(list_lines).strip()
            if text:
                # 列表项是碎片化结构，不做文风检测；但保留位置供报告灰显
                blocks.append(Block(kind="list", text=text, index=len(blocks), source="markdown"))
            list_lines.clear()

    def flush_all() -> None:
        flush_para()
        flush_list()
        flush_table()

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            if in_fence:
                flush_fence()
            else:
                flush_all()
            in_fence = not in_fence
            continue
        if in_fence:
            fence_lines.append(line)
            continue

        # Markdown 表格：连续以 | 开头/结尾的行
        if _RE_MD_TABLE.match(line):
            if not in_table:
                flush_para()
                flush_list()
                in_table = True
            table_lines.append(stripped)
            continue
        if in_table:
            flush_table()
            in_table = False

        # Markdown 列表：连续列表项聚成一块，不并入正文
        if _is_md_list_item(line):
            if not in_list:
                flush_para()
                in_list = True
            list_lines.append(stripped)
            continue
        if in_list:
            flush_list()
            in_list = False

        if not stripped:
            flush_para()
            continue
        if stripped.startswith("#"):
            flush_para()
            heading = stripped.lstrip("#").strip()
            if heading:
                blocks.append(Block(kind="heading", text=heading, index=len(blocks)))
            continue
        # 无标记纯文本里的标题/目录/封面行：单独成块，不拼进正文
        structural = _plaintext_structural_kind(stripped)
        if structural is not None:
            flush_para()
            blocks.append(Block(kind=structural, text=stripped, index=len(blocks), source="plaintext"))
            continue
        para_lines.append(stripped)
    flush_fence()
    flush_all()
    return _demote_leading_short_blocks(blocks)
