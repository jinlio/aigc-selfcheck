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
