"""组装校准集：段级 80/20 冻结切分（ALGORITHM §4 + topic-plans 纪律）。

输入：
    calibration/manifest.jsonl   人写段落（fetch_human_corpus.py 产出）
    calibration/ai/*.txt         AI 生成段落（每文件一段）
    calibration/polished/*.txt   人写段落经 AI 润色（每文件一段，provenance 在文件头注释行）
输出（冻结，禁止手改）：
    calibration/out/calibration.jsonl   窗口级单元：{uid, class, split, text, context, para_sha, author}
    calibration/out/splits.md5          冻结指纹：切分内容 sha256，改动即失效

纪律：seed=42；按「段落」切分（同段窗口不跨 split）；人写每作者上限 5 段；
layer=classic（公版书年代层）不参与主切分，单独归档。

v8 起每个窗口同时落盘 `context`（窗口起点之前的正文尾部 100 字，见
`contexts.preceding_context`），产品路径与校准路径共用同一口径；`--keep-split`
只补写 context、沿用现有切分，用于口径迁移（不重新抽 test）。

v4 起的切分协议（评测纪律）：
1. 上一版 calibration.jsonl 中 split=test 的段落已被评测消耗，**强制并入 train**，
   永不再进 test；
2. 新 test 只从「从未出现在上一版语料」的段落中抽取（新段落从未被打过分）；
   不足额时（如 polished 无新段）才从上一版 train 段补足——train 段虽参与过
   常量拟合，但从未以 test 身份被查看，泄漏风险仅限拟合分布层面，如实记录；
3. 抽取按作者（人写）/生成器（AI）**整组**轮转：v7 起以整组为留出单位——
   被选入 test 的作者/生成器，其全部段落都进 test；同一作者/生成器不再跨
   train/test（v6 的按段落抽取会让 6/8 位「新作者」在两侧各有段落，使 test
   只是「新段落」口径而非「未见作者」口径）。整组抽取可能让 test 略超 20%
   配额，以组为单位是刻意取舍。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_selfcheck.contexts import preceding_context  # noqa: E402
from aigc_selfcheck.docio import Block  # noqa: E402
from aigc_selfcheck.preprocess import prepare_block  # noqa: E402

ROOT = Path("calibration")
OUT = ROOT / "out" / "calibration.jsonl"
FINGERPRINT = ROOT / "out" / "splits.md5"
PER_AUTHOR_CAP = 5
SEED = 42
TRAIN_RATIO = 0.8


def pick_groups(pool: list[dict], n: int, key, rng: random.Random) -> list[dict]:
    """按组整组抽取 test 段落（v7 起）。

    组 = 作者（人写）/生成器（AI）；组内全部段落同进同出，保证 test 组不在
    train 出现。软上限 1.2×配额：优先选能装进上限的小组，避免单一大组把
    test 撑到配额数倍；若所有组都超上限，退而取最小的一组。
    """
    if not pool:
        return []
    groups = sorted({key(p) for p in pool})
    rng.shuffle(groups)
    members = {g: [p for p in pool if key(p) == g] for g in groups}
    cap = max(n, int(n * 1.2) + 1)
    start = next(
        (g for g in groups if len(members[g]) <= cap),
        min(groups, key=lambda g: len(members[g])),
    )
    chosen = list(members[start])
    chosen_groups = {start}
    if len(chosen) >= n:
        return chosen
    for group in groups:
        if group in chosen_groups:
            continue
        if len(chosen) + len(members[group]) <= cap:
            chosen.extend(members[group])
            chosen_groups.add(group)
            if len(chosen) >= n:
                return chosen
    left = [g for g in groups if g not in chosen_groups]
    if left:
        chosen.extend(members[min(left, key=lambda g: len(members[g]))])
    return chosen


def load_human() -> list[dict]:
    if not (ROOT / "manifest.jsonl").exists():
        return []
    by_author: dict[str, int] = {}
    items: list[dict] = []
    for line in (ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("precheck_rejected") or record.get("review_rejected") or record.get("layer") != "modern":
            continue
        author = record.get("author", "unknown")
        if by_author.get(author, 0) >= PER_AUTHOR_CAP:
            continue
        # manifest 入库不含正文（版权约束），正文在 raw/human/<sha16>.txt
        raw = ROOT / "raw" / "human" / f"{record['sha256'][:16]}.txt"
        if not raw.exists():
            print(f"  [skip] 正文缺失：{author} {record['sha256'][:16]}")
            continue
        record["text"] = raw.read_text(encoding="utf-8").strip()
        by_author[author] = by_author.get(author, 0) + 1
        items.append(record)
    return items


def load_class_dir(directory: Path) -> list[dict]:
    items: list[dict] = []
    if not directory.exists():
        return items
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        provenance = ""
        if text.startswith("<!--"):
            end = text.index("-->")
            provenance = text[4:end].strip()
            text = text[end + 3 :].strip()
        if text:
            items.append(
                {
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                    "author": provenance or path.stem,
                    "url": str(path),
                }
            )
    return items


def paragraph_windows(text: str) -> list[tuple[str, str]]:
    """段落 → (窗口文本, 窗口起点之前的上下文)，与真实管线同一套预处理。"""
    _sentences, windows = prepare_block(Block(kind="paragraph", text=text, index=0))
    return [(w.text, preceding_context(text, w.start)) for w in windows]


def gen_key(author: str) -> str:
    """AI 分层分组：模型可辨按模型（API 批），否则按平台（与 cv_evaluate.generator_of 一致）。"""
    m = re.search(r"模型:\s*([^\|]+)", author)
    if m and "未记录" not in m.group(1):
        return m.group(1).strip()
    m = re.search(r"平台:\s*([^\|]+)", author)
    return m.group(1).strip() if m else "unknown"


def previous_shas() -> tuple[set[str], set[str]]:
    """读上一版 calibration.jsonl：返回 (全部段落 sha, 上一版 test 段落 sha)。"""
    if not OUT.exists():
        return set(), set()
    all_shas: set[str] = set()
    test_shas: set[str] = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            unit = json.loads(line)
            all_shas.add(unit["para_sha"])
            if unit["split"] == "test":
                test_shas.add(unit["para_sha"])
    return all_shas, test_shas


def self_unit(cls: str, split: str, index: int, text: str, context: str, para: dict) -> dict:
    return {
        "uid": f"{cls}-{para['sha256'][:12]}-{index}",
        "class": cls,
        "split": split,
        "text": text,
        "context": context,
        "para_sha": para["sha256"],
        "author": para.get("author", "unknown"),
        "url": para.get("url", ""),
    }


def main() -> int:
    keep_split = "--keep-split" in sys.argv
    classes = {
        "human": load_human(),
        "ai": load_class_dir(ROOT / "ai"),
        "polished": load_class_dir(ROOT / "polished"),
    }
    for name, items in classes.items():
        print(f"{name}: {len(items)} 段")

    split_by_sha: dict[str, str] = {}
    old_windows: dict[tuple[str, int], str] = {}
    if keep_split:
        if not OUT.exists():
            raise SystemExit("--keep-split 需要已有 calibration.jsonl")
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            unit = json.loads(line)
            split_by_sha[unit["para_sha"]] = unit["split"]
            old_windows[(unit["para_sha"], int(unit["uid"].rsplit("-", 1)[1]))] = unit["text"]
        print(f"--keep-split：沿用现有切分（{len(split_by_sha)} 段），只补写 context 字段")
    else:
        old_all, old_test = previous_shas()
        if old_all:
            print(f"上一版段落 {len(old_all)}（其中已消耗 test 段 {len(old_test)}）→ 强制并入 train")

    units: list[dict] = []
    for cls, items in classes.items():
        if keep_split:
            for para in items:
                split = split_by_sha.get(para["sha256"])
                if split is None:
                    raise SystemExit(f"--keep-split：段落不在现有切分中：{para['sha256'][:12]}")
                for index, (text, context) in enumerate(paragraph_windows(para["text"])):
                    old_text = old_windows.get((para["sha256"], index))
                    if old_text is not None and old_text != text:
                        raise SystemExit(f"--keep-split：窗口文本与现有版本不一致：{para['sha256'][:12]}#{index}")
                    units.append(self_unit(cls, split, index, text, context, para))
            continue

        class_rng = random.Random(SEED)  # noqa: S311 —— 冻结切分要求可复现
        n_test = max(1, round(len(items) * (1 - TRAIN_RATIO)))
        fresh = [p for p in items if p["sha256"] not in old_all]
        old_train = [p for p in items if p["sha256"] in old_all and p["sha256"] not in old_test]
        # 上一版 test 段已被评测消耗：强制 train（fresh+old_train 之外即 old_test）。
        # 全新段优先；v7 起以「整组」为留出单位：选中某作者/生成器后，其全部段落
        # 进 test，剩余组整组进 train，保证 test 组不在 train 出现。
        # 只从「组内全为全新段」的干净组里抽——若某组还有旧 train/旧 test 成员，
        # 抽它的新段会让同组跨两侧，直接排除（这些新段随组进 train）。
        key = (lambda p: str(p.get("author", "?"))) if cls != "ai" else (lambda p: gen_key(str(p.get("author", ""))))
        fresh_shas = {p["sha256"] for p in fresh}
        fresh_groups = {key(p) for p in fresh}
        nonfresh_groups = {key(p) for p in items if p["sha256"] not in fresh_shas}
        clean_fresh = [p for p in fresh if key(p) not in nonfresh_groups]
        # 回填也只用「整组都只属于旧 train」的组：含旧 test 或含全新段的组
        # 一旦回填就会跨两侧。
        old_test_groups = {key(p) for p in items if p["sha256"] in old_test}
        clean_old_train = [
            p for p in old_train if key(p) not in fresh_groups and key(p) not in old_test_groups
        ]

        picked = pick_groups(clean_fresh, n_test, key, class_rng)
        if len(picked) < n_test:
            picked += pick_groups(clean_old_train, n_test - len(picked), key, class_rng)
        if len(picked) < n_test:
            print(f"  [warn] {cls}: 可选池不足以填满 test 配额（{len(picked)}/{n_test}）")
        test_shas = {p["sha256"] for p in picked}
        test_groups = {key(p) for p in picked}
        train_groups = {key(p) for p in items if p["sha256"] not in test_shas}
        straddling = test_groups & train_groups
        n_old_test_in_train = sum(1 for p in items if p["sha256"] in old_test)
        print(f"  {cls}: test {len(test_shas)} 段（全新 {sum(1 for p in picked if p['sha256'] not in old_all)}"
              f"/旧train {sum(1 for p in picked if p['sha256'] in old_all)}，{len(test_groups)} 组）；"
              f"train {len(items) - len(test_shas)} 段（含上一版 test 强制并入 {n_old_test_in_train}）")
        if straddling:
            print(f"  [warn] {cls}: {len(straddling)} 组跨 train/test：{sorted(straddling)[:5]}")
        for para in items:
            split = "test" if para["sha256"] in test_shas else "train"
            for index, (text, context) in enumerate(paragraph_windows(para["text"])):
                units.append(self_unit(cls, split, index, text, context, para))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n" for u in units)
    OUT.write_text(payload, encoding="utf-8")
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    FINGERPRINT.write_text(fingerprint + "\n", encoding="utf-8")

    counts: dict[tuple[str, str], int] = {}
    for u in units:
        counts[(u["class"], u["split"])] = counts.get((u["class"], u["split"]), 0) + 1
    for (cls, split), count in sorted(counts.items()):
        print(f"  {cls}/{split}: {count} 窗口")
    print(f"冻结指纹 sha256: {fingerprint}（写入 {FINGERPRINT}，评测前必须核对）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
