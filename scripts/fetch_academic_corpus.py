"""抓取 2020 年中文开放获取论文，构建「人写学术语域」探针集。

为什么单独建这一层：现有 calibration 的人写语料以博客、科普、随笔为主，
学术论文的句长、连接词、术语密度和引用习惯都不同；检测器在通用语域上的
指标不能直接外推到「论文上传知网前自查」这个产品场景。

数据选择（保真关）：
  1. 时间截点：只收 2020 年（≤2021-12-31）发表的论文，早于 LLM 普及期；
  2. 来源：HansPub 开放获取中文期刊，页面与 PDF 均可公开访问；
  3. 作者：每篇论文最多取 3 段，且尽量跨「引言 / 方法 / 结果与讨论 / 结论」，
     避免单篇论文主导作者风格；
  4. provenance：每段记录论文 URL、PDF URL、标题、作者、期刊、DOI、章节、
     sha256 与抓取时间；原文只落在 calibration/academic/raw/（.gitignore 排除）。

网络安全：复用 fetch_human_corpus 的 https + 主机白名单 + DNS 公网 IP 校验 +
逐跳重定向校验；本脚本只把 HansPub 两个主机加入局部白名单，不扩大全局白名单。

用法：
    PYTHONPATH=src D:/anaconda/envs/learning/python.exe scripts/fetch_academic_corpus.py
    ... --only 39378,35515 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_human_corpus as fhc  # noqa: E402

DATA = ROOT / "calibration" / "academic"
RAW = DATA / "raw"
TEXTS = DATA / "human_texts"
MANIFEST = DATA / "human.jsonl"
README = DATA / "README.md"

ALLOWED_HOSTS = {"www.hanspub.org", "pdf.hanspub.org"}
PAGE = "https://www.hanspub.org/journal/paperinformation?paperid={pid}"
DATE_CUTOFF_YEAR = 2021
MIN_CJK, MAX_CJK = 100, 400
MIN_SENTS = 2
MAX_PER_PAPER = 3
SLEEP_SECONDS = 1.0

# 2020 年 HansPub 论文，覆盖计算机、数学、工程、管理、经济、心理、教育、
# 文学与情报等学科。只保留已核验可访问、PDF 可抽取中文正文的条目。
SEEDS = [
    {"paperid": 35515, "discipline": "计算机"},
    {"paperid": 39378, "discipline": "应用数学/医学"},
    {"paperid": 38972, "discipline": "教育/心理测量"},
    {"paperid": 34199, "discipline": "计算机"},
    {"paperid": 38602, "discipline": "化学工程/安全"},
    {"paperid": 37193, "discipline": "管理/运筹"},
    {"paperid": 35791, "discipline": "数学/文学"},
    {"paperid": 36075, "discipline": "计算机/公共管理"},
    {"paperid": 35285, "discipline": "应用数学"},
    {"paperid": 35373, "discipline": "应用数学/工程"},
    {"paperid": 39002, "discipline": "经济/生态"},
    {"paperid": 38837, "discipline": "心理学"},
    {"paperid": 38236, "discipline": "教育学"},
    {"paperid": 36238, "discipline": "情报/统计"},
    {"paperid": 35247, "discipline": "旅游/管理"},
]

_HEADING_RE = re.compile(
    r"^(?:"
    r"摘\s*要|abstract|关键词|keywords|"
    r"引\s*言|前\s*言|绪\s*论|"
    r"相关工作|研究现状|文献综述|"
    r"研究方法|方法|模型|算法|实验方法|"
    r"结果(?:与讨论)?|实验结果|实验(?:与)?分析|分析|讨\s*论|"
    r"结\s*论|结束语|总结|"
    r"致\s*谢|参考文献|references|"
    r"\d+(?:\.\d+)*\.\s*[^\s。；！？\d]"
    r")",
    re.I,
)
_STOP_RE = re.compile(r"^(?:参考文献|references|致\s*谢|基金项目|作者简介)\b", re.I)
_SKIP_LINE_RE = re.compile(
    r"^(?:"
    r"DOI:|https?://|www\.|"
    r"Published Online|文章引用|Copyright|Open Access|Hans Publishers|"
    r"收稿日期|录用日期|发布日期|基金项目|作者简介|通信作者|E-mail|"
    r"Figure\s*\d|Table\s*\d|图\s*\d|表\s*\d|"
    r"Advances in|应用.*进展|计算机科学与应用|心理学进展|"
    r"服务科学和管理|创意教育研究|统计与应用|世界经济研究|"
    r"Creative Education|Pure Mathematics|World Economic Research|"
    r"Operations Research|Service Science|Statistics and Application"
    r")",
    re.I,
)
_INLINE_HEADING_RE = re.compile(
    r"(?:^|.*?[。；！？])\s*"
    r"(\d+(?:\.\d+)*\.\s*[^\s。；！？\d][^。；！？]{0,40})$"
)
_KNOWN_HEADING_RE = re.compile(
    r"^(?:"
    r"摘\s*要|abstract|关键词|keywords|"
    r"引\s*言|前\s*言|绪\s*论|"
    r"相关工作|研究现状|文献综述|"
    r"研究方法|实验结果|实验(?:与)?分析|结果(?:与讨论)?|"
    r"讨\s*论|结\s*论|结束语|总结|"
    r"致\s*谢|参考文献|references"
    r")$",
    re.I,
)
_PAGE_NUM_RE = re.compile(r"^\d{2,4}$")
_JOURNAL_LINE_RE = re.compile(
    r"^\d{4},\s*\d+\(\d+\)|^\d{4}\s*年\s*\d+\s*月|^[A-Z][A-Za-z ]+,\s*2020"
)
_MATH_RE = re.compile(r"[=<>±×÷∑∫√≤≥∈∀∃]")
_SENT_RE = re.compile(r"[。！？!?]")


def cjk_count(text: str) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text)


def _meta(html: str, name: str) -> str:
    match = re.search(rf'<meta\s+name="{re.escape(name)}"\s+content="([^"]*)"', html, re.I)
    return match.group(1).strip() if match else ""


def _pdf_url(html: str) -> str:
    match = re.search(r'//pdf\.hanspub\.org/[^"\']+\.pdf', html, re.I)
    return f"https:{match.group(0)}" if match else ""


def _clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = _clean_line(line)
        if not line:
            continue
        if not out:
            out = line
        elif re.search(r"[\u4e00-\u9fff]$", out) or re.match(r"^[\u4e00-\u9fff]", line):
            out += line
        else:
            out += " " + line
    return out


def _normalize_section(heading: str) -> str:
    low = heading.lower()
    if "摘要" in heading or "abstract" in low:
        return "摘要"
    if any(k in heading for k in ("引言", "前言", "绪论")):
        return "引言"
    if any(k in heading for k in ("相关工作", "研究现状", "文献综述")):
        return "相关工作"
    if any(
        k in heading
        for k in ("方法", "模型", "算法", "设计", "回归", "原理", "框架", "结构", "建模", "训练")
    ):
        return "方法"
    if any(k in heading for k in ("结果", "实验", "分析", "讨论")):
        return "结果与讨论"
    if any(k in heading for k in ("结论", "结束语", "总结")):
        return "结论"
    return "其他"


def _body_ok(text: str, min_cjk: int, max_cjk: int) -> bool:
    n = cjk_count(text)
    if not (min_cjk <= n <= max_cjk):
        return False
    if len(_SENT_RE.findall(text)) < MIN_SENTS:
        return False
    if n / max(len(text), 1) < 0.5:
        return False
    if len(_MATH_RE.findall(text)) > 4:
        return False
    if re.search(r"(?:参考文献|DOI:|https?://|www\.)", text, re.I):
        return False
    # PDF 公式/分页截断：段末没有句末标点，且停在拉丁字母或数字上
    if not _SENT_RE.search(text[-1:]) and re.search(r"[A-Za-z0-9]$", text):
        return False
    return True


def _extract_paragraphs(pdf_bytes: bytes) -> list[tuple[str, str]]:
    try:
        import pymupdf
    except ImportError:  # 旧版 PyMuPDF
        import fitz as pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    paragraphs: list[tuple[str, str]] = []
    section = "其他"
    top_section = "其他"
    current: list[str] = []
    base_x: float | None = None
    stopped = False

    def apply_heading(heading: str) -> None:
        """一级标题定章节；二级标题（4.1/2.3）沿用所属一级章节。"""
        nonlocal section, top_section
        if re.match(r"^\d+\.\d+", heading):
            section = top_section
            return
        section = _normalize_section(heading)
        top_section = section

    def flush() -> None:
        nonlocal current
        text = _join_lines(current)
        if text:
            paragraphs.append((section, text))
        current = []

    for page in doc:
        height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                text = _clean_line(text)
                if not text:
                    continue
                y0 = line["bbox"][1]
                x0 = line["bbox"][0]
                # 页眉/页脚；保留正文区域（HansPub 页脚期刊名/页码在 y≈742）
                if y0 < 80 or y0 > height - 55:
                    continue
                if _PAGE_NUM_RE.match(text) or _JOURNAL_LINE_RE.match(text):
                    continue
                if _SKIP_LINE_RE.match(text):
                    continue
                if _STOP_RE.match(text):
                    stopped = True
                    break
                # 标题偶尔会和上一段末句挤在同一行（PDF 抽取常见）：
                # “……效果。 2. ε-支持向量机回归原理”。先切出标题，再按标题换段。
                inline = _INLINE_HEADING_RE.search(text)
                if inline and len(inline.group(1)) <= 48:
                    body = text[: inline.start(1)].strip()
                    if body and cjk_count(body) >= 4:
                        if base_x is not None and x0 > base_x + 12 and current:
                            flush()
                        current.append(body)
                    flush()
                    apply_heading(inline.group(1))
                    continue
                size = max(
                    (span.get("size", 10.0) for span in line.get("spans", [])),
                    default=10.0,
                )
                numeric_heading = bool(re.match(r"^\d+(?:\.\d+)*\.", text))
                known_heading = bool(_KNOWN_HEADING_RE.match(text))
                is_heading = (
                    bool(_HEADING_RE.match(text))
                    and len(text) <= 48
                    and (numeric_heading or known_heading or size >= 10.8)
                )
                if is_heading and cjk_count(text) + len(re.findall(r"[A-Za-z]", text)) >= 2:
                    flush()
                    apply_heading(text)
                    continue
                # 纯公式/表格碎片：汉字太少且无句末标点
                if cjk_count(text) < 4 and not _SENT_RE.search(text):
                    continue
                if base_x is None and cjk_count(text) >= 6:
                    base_x = x0
                if base_x is not None and x0 > base_x + 12 and current:
                    flush()
                current.append(text)
            if stopped:
                break
        if stopped:
            break
    flush()
    return paragraphs


def _select(paras: list[tuple[str, str]], min_cjk: int, max_cjk: int) -> list[tuple[str, str]]:
    """按章节优先级选段：先引言，再方法，再结果/结论，最后摘要/其他。"""
    cleaned: list[tuple[str, str]] = []
    seen: set[str] = set()
    for section, text in paras:
        text = re.sub(r"\s+", " ", text).strip()
        if not _body_ok(text, min_cjk, max_cjk):
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((section, text))

    by_section: dict[str, list[str]] = {}
    for section, text in cleaned:
        by_section.setdefault(section, []).append(text)

    picked: list[tuple[str, str]] = []
    for section in ("引言", "方法", "结果与讨论", "结论", "相关工作", "摘要", "其他"):
        if len(picked) >= MAX_PER_PAPER:
            break
        candidates = by_section.get(section, [])
        if not candidates:
            continue
        # 同章节取最长的一段（通常信息量最大），但不超过 1 段/章节
        text = max(candidates, key=len)
        picked.append((section, text))
    # 章节不足时，从剩余段落补齐
    if len(picked) < 2:
        for section, text in sorted(cleaned, key=lambda item: len(item[1]), reverse=True):
            if len(picked) >= 2:
                break
            if any(text == existing for _, existing in picked):
                continue
            picked.append((section, text))
    return picked


def _write_sample(meta: dict, section: str, text: str, index: int) -> dict:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sample_id = f"acad-{meta['paperid']}-{index:02d}"
    provenance = (
        "<!--\n"
        "来源: 学术论文\n"
        "平台: HansPub\n"
        f"标题: {meta['title']}\n"
        f"作者: {meta['authors']}\n"
        f"期刊: {meta['journal']}\n"
        f"年份: {meta['year']}\n"
        f"章节: {section}\n"
        f"论文URL: {meta['url']}\n"
        f"PDF: {meta['pdf_url']}\n"
        f"DOI: {meta['doi']}\n"
        f"抓取日期: {meta['fetched_at']}\n"
        f"文体: 学术论文-{section}\n"
        f"提示词: {index:02d}\n"
        "-->\n\n"
    )
    (TEXTS / f"{sample_id}.txt").write_text(provenance + text + "\n", encoding="utf-8")
    return {
        "id": sample_id,
        "text": text,
        "source": "hanspub",
        "discipline": meta["discipline"],
        "url": meta["url"],
        "pdf_url": meta["pdf_url"],
        "title": meta["title"],
        "authors": meta["authors"],
        "journal": meta["journal"],
        "year": meta["year"],
        "doi": meta["doi"],
        "section": section,
        "sha256": digest,
        "fetched_at": meta["fetched_at"],
    }


def fetch_one(
    seed: dict, refresh: bool, dry_run: bool, min_cjk: int, max_cjk: int
) -> list[dict]:
    pid = seed["paperid"]
    url = PAGE.format(pid=pid)
    page_path = RAW / f"{pid}.html"
    pdf_path = RAW / f"{pid}.pdf"
    if refresh or not page_path.exists():
        html = fhc.safe_fetch(url, allowed_hosts=ALLOWED_HOSTS)
        page_path.write_text(html, encoding="utf-8")
        time.sleep(SLEEP_SECONDS)
    else:
        html = page_path.read_text(encoding="utf-8")

    meta = {
        "paperid": pid,
        "discipline": seed["discipline"],
        "url": url,
        "title": _meta(html, "citation_title"),
        "authors": _meta(html, "citation_authors"),
        "journal": _meta(html, "citation_journal_title"),
        "year": _meta(html, "citation_year"),
        "doi": _meta(html, "citation_doi"),
        "pdf_url": _pdf_url(html),
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    if not meta["title"] or not meta["pdf_url"]:
        raise fhc.FetchError(f"paperid={pid} 缺标题或 PDF 链接")
    try:
        year = int(meta["year"])
    except ValueError as exc:
        raise fhc.FetchError(f"paperid={pid} 缺发表年份") from exc
    if year > DATE_CUTOFF_YEAR:
        raise fhc.FetchError(f"paperid={pid} 年份 {year} 超过截点")

    if refresh or not pdf_path.exists():
        pdf_path.write_bytes(
            fhc.safe_fetch_bytes(meta["pdf_url"], allowed_hosts=ALLOWED_HOSTS)
        )
        time.sleep(SLEEP_SECONDS)
    paras = _extract_paragraphs(pdf_path.read_bytes())
    (RAW / f"{pid}.txt").write_text(
        "\n\n".join(f"[{section}]\n{text}" for section, text in paras),
        encoding="utf-8",
    )
    picked = _select(paras, min_cjk, max_cjk)
    if dry_run:
        print(
            f"{pid} {meta['title'][:36]} | 抽到 {len(paras)} 段，合格 "
            f"{sum(_body_ok(t, min_cjk, max_cjk) for _, t in paras)} 段，选 {len(picked)} 段"
        )
        for section, text in picked:
            print(f"  [{section}] {cjk_count(text)}字 {text[:80]}")
        return []
    return [_write_sample(meta, section, text, i) for i, (section, text) in enumerate(picked, 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="逗号分隔的 paperid，调试用")
    parser.add_argument("--refresh", action="store_true", help="忽略本地缓存重新下载")
    parser.add_argument("--dry-run", action="store_true", help="只打印选段，不写文件")
    parser.add_argument("--min-cjk", type=int, default=MIN_CJK)
    parser.add_argument("--max-cjk", type=int, default=MAX_CJK)
    args = parser.parse_args()

    only = {int(x) for x in args.only.split(",") if x.strip()}
    seeds = [s for s in SEEDS if not only or s["paperid"] in only]

    RAW.mkdir(parents=True, exist_ok=True)
    TEXTS.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        for old in TEXTS.glob("acad-*.txt"):
            old.unlink()

    rows: list[dict] = []
    failures: list[str] = []
    for seed in seeds:
        try:
            rows.extend(
                fetch_one(seed, args.refresh, args.dry_run, args.min_cjk, args.max_cjk)
            )
        except Exception as exc:  # noqa: BLE001 —— 单篇失败不拖垮整批，记录后继续
            failures.append(f"{seed['paperid']}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {seed['paperid']}: {exc}", file=sys.stderr)

    if args.dry_run:
        print(f"\n共 {len(rows)} 段（dry-run 不写盘）；失败 {len(failures)} 篇")
        return 0

    # 全局去重
    unique: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        unique.append(row)
    MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique),
        encoding="utf-8",
    )
    README.write_text(_render_readme(unique, failures), encoding="utf-8")
    print(f"\n写入 {len(unique)} 段 → {MANIFEST}")
    print(f"原文 → {TEXTS}")
    print(f"失败 {len(failures)} 篇")
    return 0 if not failures else 1


def _render_readme(rows: list[dict], failures: list[str]) -> str:
    by_discipline: dict[str, int] = {}
    by_section: dict[str, int] = {}
    for row in rows:
        by_discipline[row["discipline"]] = by_discipline.get(row["discipline"], 0) + 1
        by_section[row["section"]] = by_section.get(row["section"], 0) + 1
    disciplines = "、".join(f"{k} {v}" for k, v in sorted(by_discipline.items()))
    sections = "、".join(f"{k} {v}" for k, v in sorted(by_section.items()))
    return f"""# 人写学术语域探针（HansPub 2020）

## 这是什么

现有校准集的人写侧以博客、科普、随笔为主。产品定位是「论文上传知网等
AIGC 检测前自查」，所以需要单独测学术论文语域：检测器会不会把正常论文
段落误判成 AI，以及 AI 写的论文章节能不能被抓住。

本目录是 **report-only 探针**：不进入 `calibration.jsonl`，不参与切分、
μ/σ、α 权重或 HOT 阈值调整。要进训练集，必须另走 v9 切分流程。

## 数据

- 来源：HansPub 开放获取中文期刊（`www.hanspub.org` / `pdf.hanspub.org`）。
- 时间：{DATE_CUTOFF_YEAR} 年及以前，全部为 2020 年发表，早于 LLM 普及期。
- 选段：每篇最多 {MAX_PER_PAPER} 段，优先「引言 / 方法 / 结果与讨论 / 结论」，
  单段 {MIN_CJK}–{MAX_CJK} 汉字、至少 {MIN_SENTS} 个句末标点。
- 去重：按正文 sha256 全局去重。
- 规模：{len(rows)} 段，覆盖 {disciplines}；章节分布：{sections}。
- provenance：`human.jsonl` 逐段记录论文 URL、PDF、标题、作者、期刊、DOI、
  章节、sha256 与抓取日期；原文在 `human_texts/`，PDF/HTML 缓存在 `raw/`
  （`raw/` 已在 `.gitignore` 中排除）。

## 复现

```bash
PYTHONPATH=src D:/anaconda/envs/learning/python.exe scripts/fetch_academic_corpus.py
```

脚本强制 https、主机白名单、DNS 公网 IP 校验、逐跳重定向校验；单篇失败
不会中断整批。`--only 39378,35515 --dry-run` 可先看选段效果。

## 已知局限

- 只覆盖 HansPub 一个开放获取平台，期刊/排版/写作风格有平台偏差；
- 2020 年论文可排除 LLM 生成，但不能排除作者使用翻译/润色工具；
- 选段是规则抽取，公式密集的「方法」段可能被过滤，样本更偏叙述性段落；
- 不能把本探针的误伤率直接当成知网误伤率——知网有自己的模型和阈值。

## 失败记录

{chr(10).join(f"- {line}" for line in failures) if failures else "- 无"}
"""


if __name__ == "__main__":
    raise SystemExit(main())
