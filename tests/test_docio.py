from pathlib import Path

import pytest
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from aigc_selfcheck.docio import read_document, read_docx, read_pdf


def _make_pdf(path: Path, lines: list[str]) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path))
    pdf.setFont("STSong-Light", 12)
    y = 760
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 24
    pdf.save()


def test_title_and_author_styles_are_skipped(tmp_path: Path) -> None:
    doc = Document()
    doc.styles.add_style("Author", WD_STYLE_TYPE.PARAGRAPH)
    doc.add_paragraph("论文标题", style="Title")
    doc.add_paragraph("张三", style="Author")
    doc.add_paragraph("这是正文段落。", style="Normal")
    path = tmp_path / "sample.docx"
    doc.save(path)

    blocks = read_docx(path)

    assert blocks[0].kind == "heading"
    assert blocks[0].text == "论文标题"
    assert blocks[1].kind == "metadata"
    assert blocks[1].text == "张三"
    assert blocks[2].kind == "paragraph"


def _docx_with_paragraphs(tmp_path: Path, items: list[tuple[str, str]]) -> Path:
    doc = Document()
    existing = {style.name for style in doc.styles}
    for text, style in items:
        if style not in existing:
            doc.styles.add_style(style, WD_STYLE_TYPE.PARAGRAPH)
            existing.add(style)
        doc.add_paragraph(text, style=style)
    path = tmp_path / "handbuilt.docx"
    doc.save(path)
    return path


def test_manual_normal_headings_are_skipped(tmp_path: Path) -> None:
    """真实论文常用 Normal 样式手打标题（目录 toc 样式、标题靠编号/字号）。"""
    path = _docx_with_paragraphs(
        tmp_path,
        [
            ("目  录", "Normal"),
            ("1 总论\t10", "toc 1"),
            ("1.1 项目概况\t10", "toc 2"),
            ("1 总论", "Normal"),
            ("1.1 项目概况", "Normal"),
            ("项目名称为武汉市汉阳区四新“云璟”住宅小区建设项目，拟建于武汉市汉阳区四新片区。", "Normal"),
        ],
    )
    blocks = read_docx(path)

    assert [b.kind for b in blocks] == ["heading", "toc", "toc", "heading", "heading", "paragraph"]


def test_cover_fields_are_metadata(tmp_path: Path) -> None:
    path = _docx_with_paragraphs(
        tmp_path,
        [
            ("题    目\u3000某小区建设项目可行性研究", "Normal"),
            ("学    院\u3000经济与管理学院", "Normal"),
            ("【关键词】\u3000房地产项目\u3000可行性研究", "Normal"),
            ("摘  要", "Normal"),
            ("这是摘要正文段落，长度足够说明它应当被当作正文处理。", "Normal"),
        ],
    )
    blocks = read_docx(path)

    assert [b.kind for b in blocks] == ["metadata", "metadata", "metadata", "heading", "paragraph"]


def test_english_paragraph_is_not_chinese_detectable(tmp_path: Path) -> None:
    """英文摘要不适用中文检测器，应被 classify_block 标为 non_zh。"""
    from aigc_selfcheck.preprocess import classify_block, detectable

    path = _docx_with_paragraphs(
        tmp_path,
        [("As the real estate industry enters a period of deep adjustment, investment decisions become harder.", "Normal")],
    )
    block = classify_block(read_docx(path)[0])

    assert block.kind == "non_zh"
    assert not detectable(block)


def test_body_short_paragraph_without_number_is_detectable(tmp_path: Path) -> None:
    """无编号、无句读的正文短句不能被误判成标题。"""
    path = _docx_with_paragraphs(
        tmp_path,
        [("项目总投资约二十九亿元", "Normal"), ("项目建成后将显著改善区域居住条件并带动就业增长。", "Normal")],
    )
    blocks = read_docx(path)

    assert blocks[0].kind == "paragraph"
    assert blocks[1].kind == "paragraph"


def test_pdf_heading_and_paragraph(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    _make_pdf(
        path,
        [
            "1 引言",
            "这是第一段正文，用来测试 PDF 文本提取。",
            "S = w1 * S_w + w2 * S_k",
        ],
    )

    blocks = read_pdf(path)

    assert blocks[0].kind == "heading"
    assert blocks[0].text == "1 引言"
    assert blocks[1].kind == "paragraph"
    assert "这是第一段正文" in blocks[1].text
    assert blocks[2].kind == "paragraph"
    assert "S = w1" in blocks[2].text


def test_read_document_dispatches_pdf(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    _make_pdf(path, ["摘 要", "本文提出一种方法。"])

    blocks = read_document(path)

    assert [block.kind for block in blocks] == ["heading", "paragraph"]


def test_pdf_without_text_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.save()

    with pytest.raises(ValueError, match="没有可提取文本"):
        read_pdf(path)
