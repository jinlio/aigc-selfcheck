"""构建跨语域人写探针集（report-only，永不进入校准/切分/调参）。

语域缺口：校准语料集中在科技/学术相邻语域。本探针从 Project Gutenberg
中文公版书取四个「目标语域之外」的语域，每个语域固定 25 段：

    modern_literary  现代白话小说/散文诗（鲁迅《彷徨》《阿Q正传》《野草》）
    modern_essay     现代白话游记/散文（朱自清《欧游杂记》、鲁迅《朝花夕拾》）
    late_qing_fiction 晚清白话小说（《老残游记》《孽海花》）
    classical_prose  文言散文/文论（《古文观止》《中国小说史略》）

段落筛选与校准语料同口径：100–400 汉字、至少 3 句。抽样用固定种子 +
按书内位置均匀铺开，避免只取开篇。产出 `calibration/probe/human.jsonl`；
它不写 manifest、不参与 build_calibration、不用于任何常量/阈值决策。

网络访问复用 `fetch_human_corpus.safe_fetch`：https + 主机白名单 +
私网/环回 IP 拦截 + 手动逐跳重定向。Gutenberg 的 `ebooks/*.txt.utf-8`
会 302 到 http，安全策略拒绝，故直接用 https 的 cache 文本地址。

用法：
    PYTHONPATH=src python scripts/build_probe.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_human_corpus import (  # noqa: E402
    cjk_count,
    gutenberg_paragraphs,
    safe_fetch,
    split_sentences,
)

OUT = ROOT / "calibration" / "probe" / "human.jsonl"
QUOTA_PER_REGISTER = 25
SEED = 20261026

# 每个语域至少两本书，避免单一作者的文风主导该语域。
BOOKS: dict[str, list[tuple[str, str]]] = {
    "modern_literary": [
        ("24042", "彷徨（鲁迅）"),
        ("25332", "阿Q正传（鲁迅）"),
        ("25242", "野草（鲁迅）"),
    ],
    "modern_essay": [
        ("24110", "欧游杂记（朱自清）"),
        ("25271", "朝花夕拾（鲁迅）"),
    ],
    "late_qing_fiction": [
        ("25124", "老残游记（刘鹗）"),
        ("25128", "孽海花（曾朴）"),
    ],
    "classical_prose": [
        ("25225", "古文观止"),
        ("25559", "中国小说史略（鲁迅）"),
    ],
}


def _usable(paragraphs: list[str]) -> list[str]:
    seen: set[str] = set()
    kept = []
    for para in paragraphs:
        text = para.strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if 100 <= cjk_count(text) <= 400 and len(split_sentences(text)) >= 3:
            kept.append(text)
    return kept


def _spread(items: list[str], count: int) -> list[str]:
    """按列表位置均匀取 count 段（确定性，不偏向开篇）。"""
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]]
    indices = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[i] for i in indices]


def main() -> int:
    rng = random.Random(SEED)  # noqa: S311 —— 固定种子保证探针抽样可复现，非加密用途
    rows: list[dict] = []
    for register, books in BOOKS.items():
        per_book: dict[str, list[str]] = {}
        for gid, title in books:
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            print(f"抓取 {title}（{gid}）…", flush=True)
            try:
                raw = safe_fetch(url)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {title}: {exc}")
                continue
            paras = _usable(gutenberg_paragraphs(raw))
            print(f"  可用段落 {len(paras)}")
            if paras:
                per_book[title] = paras

        if not per_book:
            print(f"  [warn] 语域 {register} 无可用文本")
            continue

        picked: list[tuple[str, str]] = []
        share = max(1, QUOTA_PER_REGISTER // len(per_book))
        for title, paras in per_book.items():
            picked.extend((title, text) for text in _spread(paras, share))
        pool = [
            (title, text)
            for title, paras in per_book.items()
            for text in paras
            if (title, text) not in picked
        ]
        rng.shuffle(pool)
        picked.extend(pool[: max(0, QUOTA_PER_REGISTER - len(picked))])
        picked = picked[:QUOTA_PER_REGISTER]

        for index, (title, text) in enumerate(picked, start=1):
            gid = next(gid for gid, book_title in books if book_title == title)
            rows.append(
                {
                    "id": f"probe-{register}-{index:02d}",
                    "register": register,
                    "book": title,
                    "gutenberg_id": gid,
                    "source_url": f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "cjk": cjk_count(text),
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "text": text,
                }
            )
        print(f"  {register}: {len(picked)} 段")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"写入 {OUT}（{len(rows)} 段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
