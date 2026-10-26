"""为 C 路替代基座导出独立 PPL 常量（train-only，report-only）。

用途：0.5B CPU 回退。PPL 项的绝对值随基座模型变化，μ/σ 必须按模型重导；
风格特征与参考模型无关，STYLE_STATS/STYLE_BIAS/PPL_WEIGHT_IN_C 沿用部署值。

本脚本只读 calibration.jsonl 的 train split，不做 test 评测、不改常量文件：
    1. 用指定模型对 train 窗口带上下文重算 log_ppl；
    2. 按 v8 的 μ75 规则从人写窗口导出 μ/σ；
    3. 用当前 a/b 缓存分与重算 c 分给出 train AUC/误伤/召回；
    4. 复核 LOGO-作者（折内重导 μ/σ）与 LOGO-生成器（固定常量）。

用法：
    PYTHONPATH=src python scripts/fit_c_model.py --model=Qwen/Qwen2.5-0.5B --device=cuda
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_selfcheck.cache import ScoreCache  # noqa: E402
from aigc_selfcheck.constants import (  # noqa: E402
    ALPHA,
    HEATMAP_HIGH,
    PPL_WEIGHT_IN_C,
)
from aigc_selfcheck.contexts import window_contexts  # noqa: E402
from aigc_selfcheck.detectors.stats_lm import StatsDetector  # noqa: E402

CALIBRATION = ROOT / "calibration" / "out" / "calibration.jsonl"
CACHE = ROOT / "calibration" / "out" / "score-cache.jsonl"
HOT = HEATMAP_HIGH
A_CID = "yuchuantian/AIGC_detector_zhv2"
B_CID = "mokawa3018/cmj-chinese-aigc-text-detector"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def fused(c: float, a: float, b: float) -> float:
    active = {k: v for k, v in ALPHA.items() if v}
    values = {"a": a, "b": b, "c": c}
    return sum(values[k] * w for k, w in active.items()) / sum(active.values())


def c_score(log_ppl: float | None, p_style: float, mu: float, sigma: float) -> float:
    has_ppl = log_ppl is not None and math.isfinite(log_ppl)
    p_ppl = 0.5 if not has_ppl else sigmoid((mu - log_ppl) / sigma)
    return PPL_WEIGHT_IN_C * p_ppl + (1 - PPL_WEIGHT_IN_C) * p_style


def generator_of(author: str) -> str:
    m = re.search(r"模型:\s*([^\|]+)", author)
    if m and "未记录" not in m.group(1):
        return m.group(1).strip()
    m = re.search(r"平台:\s*([^\|]+)", author)
    return m.group(1).strip() if m else "unknown"


def main() -> int:
    model = "Qwen/Qwen2.5-0.5B"
    device = "auto"
    for arg in sys.argv:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
        elif arg.startswith("--device="):
            device = arg.split("=", 1)[1]

    rows = [
        json.loads(line)
        for line in CALIBRATION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contexts = window_contexts(rows)
    cache = ScoreCache(path=CACHE)
    units = []
    for i, row in enumerate(rows):
        if row["split"] != "train":
            continue
        a = cache.get("a", A_CID, row["text"])
        b = cache.get("b", B_CID, row["text"])
        if not a or not b or "score" not in a or "score" not in b:
            print(f"  [miss] a/b 缓存缺失：{row['uid']}")
            return 1
        units.append(
            {
                **row,
                "context": contexts[i],
                "a": float(a["score"]),
                "b": float(b["score"]),
            }
        )

    print(f"模型 {model}；train 单元 {len(units)}，开始带上下文重算 log_ppl …", flush=True)
    detector = StatsDetector(device=device, model_id=model)
    print(f"  设备 {detector.device}，批量上限 {detector.max_batch}", flush=True)
    results = detector.detect_batch(
        [u["text"] for u in units], [u["context"] for u in units]
    )
    for unit, result in zip(units, results, strict=True):
        log_ppl = result.meta.get("log_ppl")
        unit["log_ppl"] = float(log_ppl) if log_ppl is not None else None
        unit["p_style"] = float(result.meta["p_style"])
    detector.close()

    humans = [u for u in units if u["class"] == "human"]
    ais = [u for u in units if u["class"] == "ai"]
    lps = sorted(
        u["log_ppl"] for u in humans if u["log_ppl"] is not None and math.isfinite(u["log_ppl"])
    )
    mu75 = lps[int(len(lps) * 0.75)]
    mu_med = statistics.median(lps)
    sigma = (sum((x - statistics.fmean(lps)) ** 2 for x in lps) / len(lps)) ** 0.5 or 1.0
    print(f"\n人写 log_ppl：n={len(lps)} median={mu_med:.3f} μ75={mu75:.3f} σ={sigma:.3f}")

    for label, mu in (("μ75", mu75), ("median", mu_med)):
        c_h = [c_score(u["log_ppl"], u["p_style"], mu, sigma) for u in humans]
        c_a = [c_score(u["log_ppl"], u["p_style"], mu, sigma) for u in ais]
        f_h = [fused(c, u["a"], u["b"]) for c, u in zip(c_h, humans, strict=True)]
        f_a = [fused(c, u["a"], u["b"]) for c, u in zip(c_a, ais, strict=True)]
        fp = sum(v > HOT for v in f_h) / len(f_h)
        recall = sum(v > HOT for v in f_a) / len(f_a)
        print(
            f"{label:6s} μ={mu:.2f} σ={sigma:.2f} | c AUC {auc(c_a, c_h):.3f} "
            f"fused AUC {auc(f_a, f_h):.3f} | 人写误伤 {fp:.1%}（{sum(v > HOT for v in f_h)}/{len(f_h)}）"
            f" 召回 {recall:.1%}（{sum(v > HOT for v in f_a)}/{len(f_a)}）"
        )

    # 以下复核固定用 μ75（与 v8 部署规则一致）
    for u in units:
        u["c"] = c_score(u["log_ppl"], u["p_style"], mu75, sigma)
        u["fused"] = fused(u["c"], u["a"], u["b"])

    print("\n== LOGO-生成器（固定 μ75/σ，仅分组看召回/AUC）==")
    for gen in sorted({generator_of(u["author"]) for u in ais}):
        held = [u for u in ais if generator_of(u["author"]) == gen]
        neg_c = [u["c"] for u in humans]
        neg_f = [u["fused"] for u in humans]
        print(
            f"{gen:28s} n={len(held):3d} cAUC {auc([u['c'] for u in held], neg_c):.3f} "
            f"fusedAUC {auc([u['fused'] for u in held], neg_f):.3f} "
            f"recall {sum(u['fused'] > HOT for u in held) / len(held):.1%}"
        )

    print("\n== LOGO-作者（折内按 μ75 重导 μ/σ；风格常量固定）==")
    fprs = []
    for author in sorted({u["author"] for u in humans}):
        fit_lps = sorted(
            u["log_ppl"]
            for u in humans
            if u["author"] != author and u["log_ppl"] is not None and math.isfinite(u["log_ppl"])
        )
        fold_mu = fit_lps[int(len(fit_lps) * 0.75)]
        fold_sigma = (
            sum((x - statistics.fmean(fit_lps)) ** 2 for x in fit_lps) / len(fit_lps)
        ) ** 0.5 or 1.0
        held = [u for u in humans if u["author"] == author]
        values = [
            fused(c_score(u["log_ppl"], u["p_style"], fold_mu, fold_sigma), u["a"], u["b"])
            for u in held
        ]
        fpr = sum(v > HOT for v in values) / len(values)
        fprs.append(fpr)
        print(f"{author:22s} n={len(held):3d} μ={fold_mu:.2f} σ={fold_sigma:.2f} 误伤 {fpr:.0%}")
    print(f"  未见作者误伤: mean={statistics.fmean(fprs):.1%} max={max(fprs):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
