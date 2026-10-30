"""审计 academic 档位阈值：用 train 真实融合分数核对部署值，而不是拍脑袋。

## 背景

`constants.py` 的 `BAND_SETS` 是案头值，从未进过拟合纪律。通用档在学术文本上
的后果是硬伤：人写学术窗口 p90 基线 0.647，而通用档要求 p90<0.60 才算「低」，
纯人写论文在数学上就进不了低档。

academic 档（0.58/0.68/0.76/0.88）此前是在两个 report-only 探针上选出来的，
而探针的 note 明写「never used for calibration, splits, thresholds or weights」。
用探针选阈值，探针就不再独立。本脚本改用 train split 复算，探针只作样本外
对照，不参与选择。

## 数据口径

校准集每行是**一个窗口**（不是段落），`cv_evaluate.load_train()` 已经按产品
同款口径算好三路分数与融合分。本脚本按作者（人写）/ 生成器（AI）聚合成
「文档」，再算 median 与 P90——这与产品对整篇文档做的聚合完全一致。

只保留窗口数 ≥ `min_windows` 的文档：窗口太少时 P90 就是最大值本身。

## 判据

档位是「把分数映射到低/中/高」的刻度，不是分类器。对学术场景，代价不对称：
把人写判成「高」是硬伤（学生会拿去和导师/查重结果对照），把 AI 判成「中」
只是漏报。因此约束是：

1. 人写文档判高率 ≤ 1%（train 上按作者分组）；
2. `thi` 必须高于融合下限 `BOOST_FLOOR`（0.80）——否则 b 路增强钉在 0.800 的
   窗口会让任何有 10% 窗口被钉的文档自动判「高」，那是融合规则的伪影，不是
   文本证据；
3. 在上述约束下，尽量提高人写判低率与 AI 判高率。

## 已知覆盖缺口（脚本会打印）

拟合池只有期刊论文（电子学报 / 软件学报），**没有学位论文**。实测一篇人写
学位论文的文档统计是 median 0.595 / p90 0.800，落在期刊论文分布之外。因此
academic 档不能只按期刊论文收紧，必须留出跨子语域的余量。

用法：
    PYTHONPATH=src python scripts/fit_bands.py
    PYTHONPATH=src python scripts/fit_bands.py --min-windows 8 --folds 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv_evaluate as cv  # noqa: E402

from aigc_selfcheck.constants import BAND_SETS, BOOST_FLOOR  # noqa: E402

ACADEMIC_HOSTS = {"www.ejournal.org.cn", "jos.org.cn"}
PROBES = {
    "人写学术（HansPub 2020）": ROOT / "calibration" / "academic" / "report.json",
    "AI 学术（豆包 Lite）": ROOT / "calibration" / "ai_academic_lite" / "report.json",
}


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def aggregate(units: list[dict], key) -> list[dict]:
    """窗口 → 文档级统计（median / P90），按 key(unit) 聚合。"""
    groups: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        groups[key(unit)].append(cv.fused(unit, cv.ALPHA))
    return [
        {"id": name, "n": len(scores), "median": statistics.median(scores), "p90": p90(scores)}
        for name, scores in groups.items()
    ]


def load_docs(min_windows: int) -> tuple[list[dict], list[dict], dict[str, int]]:
    units = cv.load_train()
    if not units:
        raise SystemExit("train 缓存不完整（load_train 返回空），先跑 scripts/build_calibration.py")

    human_units = [
        u
        for u in units
        if u["class"] == "human" and urlsplit(u.get("url") or "").netloc in ACADEMIC_HOSTS
    ]
    ai_units = [u for u in units if u["class"] == "ai" and "语域: academic" in u["author"]]

    human = [d for d in aggregate(human_units, lambda u: u["author"]) if d["n"] >= min_windows]
    ai = [d for d in aggregate(ai_units, lambda u: cv.generator_of(u["author"])) if d["n"] >= min_windows]
    stats = {
        "windows_human": len(human_units),
        "windows_ai": len(ai_units),
        "docs_human": len(human),
        "docs_ai": len(ai),
    }
    return human, ai, stats


def band(median: float, p90_value: float, edges: tuple[float, float, float, float]) -> str:
    lo, tlo, hi, thi = edges
    if median < lo and p90_value < tlo:
        return "低"
    if median >= hi or p90_value >= thi:
        return "高"
    return "中"


def rates(docs: list[dict], edges: tuple[float, float, float, float]) -> dict[str, float]:
    labels = [band(d["median"], d["p90"], edges) for d in docs]
    total = len(labels) or 1
    return {name: labels.count(name) / total for name in ("低", "中", "高")}


def platform(docs: list[dict], edges: tuple[float, float, float, float], edge_index: int,
             grid: list[float]) -> tuple[float, float] | None:
    """单个边在多大范围内移动不改变任何文档的判定（其余三边固定）。

    只在保持 lo<tlo<thi 与 hi<thi 的合法区间里扫，避免报出 1.00 这种非法值。
    """
    base = [band(d["median"], d["p90"], edges) for d in docs]
    stable = []
    for value in grid:
        moved = list(edges)
        moved[edge_index] = value
        lo, tlo, hi, thi = moved
        if not (lo < tlo < thi and lo < hi < thi):
            continue
        if [band(d["median"], d["p90"], moved) for d in docs] == base:
            stable.append(value)
    return (min(stable), max(stable)) if stable else None


def folds(authors: list[str], k: int) -> list[set[str]]:
    """确定性分组 K 折：按 sha256 排序，避免随机数（同一份语料永远同一折）。"""
    ordered = sorted(authors, key=lambda a: hashlib.sha256(a.encode("utf-8")).hexdigest())
    buckets: list[set[str]] = [set() for _ in range(k)]
    for i, author in enumerate(ordered):
        buckets[i % k].add(author)
    return buckets


def load_probe(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"median": p["median"], "p90": p["p90"]} for p in data.get("paragraphs", [])]


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 academic 档位阈值（train-only 复算）")
    parser.add_argument("--min-windows", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    human, ai, stats = load_docs(args.min_windows)
    if not human or not ai:
        print("拟合池为空：检查 train split 与学术域名是否还有样本。")
        return 1

    print("拟合池（train split，按作者/生成器聚合为文档）")
    print(f"  人写：{stats['windows_human']} 窗 → {stats['docs_human']} 篇（电子学报 + 软件学报）")
    print(f"  AI  ：{stats['windows_ai']} 窗 → {stats['docs_ai']} 篇（学术语域，按生成器分组）")
    print(f"  仅保留窗口数 ≥ {args.min_windows} 的文档\n")

    human_max_median = max(d["median"] for d in human)
    human_max_p90 = max(d["p90"] for d in human)
    ai_min_median = min(d["median"] for d in ai)
    ai_min_p90 = min(d["p90"] for d in ai)
    print("分离度（文档级）")
    print(f"  人写 median 上界 {human_max_median:.3f} | AI median 下界 {ai_min_median:.3f}")
    print(f"  人写 P90    上界 {human_max_p90:.3f} | AI P90    下界 {ai_min_p90:.3f}")
    print(f"  融合下限 BOOST_FLOOR = {BOOST_FLOOR:.3f}（b 路增强把窗口钉在这个值）")
    if abs(ai_min_median - BOOST_FLOOR) < 1e-9:
        print("  注意：AI 的 median 下界正好是 0.800——它被融合下限截断，不是自然分布。")
    print()

    edges = (
        BAND_SETS["academic"].robust_low,
        BAND_SETS["academic"].tail_low,
        BAND_SETS["academic"].robust_high,
        BAND_SETS["academic"].tail_high,
    )
    grid = [round(x / 100, 2) for x in range(30, 101)]
    print(f"部署值 academic = {edges}")
    for i, name in enumerate(("lo", "tlo", "hi", "thi")):
        span = platform(human + ai, edges, i, grid)
        if not span:
            print(f"  {name:>3s} 任何移动都会改变判定")
            continue
        print(f"  {name:>3s} 可动范围 {span[0]:.2f}–{span[1]:.2f}"
              "（其余三边固定，且不越过相邻边）")
    print("  说明：train 上人写与 AI 完全分离（人写 P90 上界 0.603 / AI 下界 0.888），")
    print("        边值只要落在这个空档里就都能得满分——train 无法识别最优值。")
    print("        因此 academic 边值由「探针样本外表现 + 跨子语域余量」决定；本脚本")
    print("        只验证它们合法、不越界，并把可动范围留档，防止未来改动无声越界。")
    print()

    print(f"{'口径':>28s} {'人写 低':>8s} {'中':>7s} {'高':>7s} {'AI 高':>7s}")
    for name, band_set in BAND_SETS.items():
        h = rates(human, (band_set.robust_low, band_set.tail_low,
                          band_set.robust_high, band_set.tail_high))
        a = rates(ai, (band_set.robust_low, band_set.tail_low,
                       band_set.robust_high, band_set.tail_high))
        print(f"{name:>28s} {h['低']:8.1%} {h['中']:7.1%} {h['高']:7.1%} {a['高']:7.1%}")
    print()

    print("样本外对照（探针不参与选择，只用于核对方向）")
    for label, path in PROBES.items():
        docs = load_probe(path)
        if docs is None:
            print(f"  {label}: 产物缺失，跳过")
            continue
        for name, band_set in BAND_SETS.items():
            r = rates(docs, (band_set.robust_low, band_set.tail_low,
                             band_set.robust_high, band_set.tail_high))
            print(f"  {label:>22s} {name:>9s} 低 {r['低']:5.1%} 中 {r['中']:5.1%} 高 {r['高']:5.1%}")
    print()

    print(f"作者分组 {args.folds} 折稳定性（academic 档）")
    academic_edges = (
        BAND_SETS["academic"].robust_low,
        BAND_SETS["academic"].tail_low,
        BAND_SETS["academic"].robust_high,
        BAND_SETS["academic"].tail_high,
    )
    for i, held in enumerate(folds([d["id"] for d in human], args.folds)):
        held_docs = [d for d in human if d["id"] in held]
        if not held_docs:
            continue
        r = rates(held_docs, academic_edges)
        print(f"  折 {i + 1}: 留出 {len(held_docs):2d} 篇 → 低 {r['低']:5.1%} 高 {r['高']:5.1%}")
    print()

    failures = []
    h = rates(human, academic_edges)
    if h["高"] > 0.01:
        failures.append(f"人写判高率 {h['高']:.1%} > 1%")
    if h["低"] < 0.80:
        failures.append(f"人写判低率 {h['低']:.1%} < 80%（档位区分不出人写文本）")
    if academic_edges[3] <= BOOST_FLOOR:
        failures.append(f"thi={academic_edges[3]:.2f} ≤ 融合下限 {BOOST_FLOOR:.2f}，钉住的窗口会自动判高")
    if academic_edges[2] <= human_max_median:
        failures.append(f"hi={academic_edges[2]:.2f} ≤ 人写 median 上界 {human_max_median:.2f}")
    if academic_edges[1] <= human_max_p90 * 0.5:
        failures.append("tlo 过低，人写文档几乎不可能判低")

    print("覆盖缺口（必须披露）")
    print("  拟合池只有期刊论文，没有学位论文。实测一篇人写学位论文（333 窗）的")
    print("  文档统计为 median 0.595 / p90 0.800，高于期刊论文分布上界；academic")
    print("  档因此必须留出跨子语域余量，不能只按期刊论文收紧。")
    print()

    if failures:
        print("审计未通过：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("审计通过：academic 档在 train 上满足「人写判高 ≤1%、判低 ≥80%、thi 高于融合下限」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
