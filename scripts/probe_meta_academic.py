"""把通用 train 上拟合的元分类器拿到学术探针上做外部复核（report-only）。

这不是训练：元分类器只在 `calibration.jsonl` 的 train 上拟合；学术探针
（HansPub 人写 + 豆包 Lite 学术章节）只用于观察跨语域表现。报告当前固定
融合与「灰度区元分类器增强」在学术场景下的误伤/召回，不参与任何常量或
阈值调整。

用法：
    PYTHONPATH=src D:/anaconda/envs/learning/python.exe scripts/probe_meta_academic.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import cv_evaluate as cv  # noqa: E402
import fit_meta as fm  # noqa: E402
import numpy as np  # noqa: E402
import probe_ark_lite as probe  # noqa: E402

from aigc_selfcheck.cache import ScoreCache  # noqa: E402
from aigc_selfcheck.constants import HEATMAP_HIGH  # noqa: E402
from aigc_selfcheck.detectors.stats_lm import StatsDetector  # noqa: E402

HOT = HEATMAP_HIGH
HUMAN_DIR = ROOT / "calibration" / "academic"
AI_DIR = ROOT / "calibration" / "ai_academic_lite"


def _cache_ids() -> dict[str, str]:
    sd = StatsDetector.__new__(StatsDetector)
    return {
        "a": "yuchuantian/AIGC_detector_zhv2",
        "b": "mokawa3018/cmj-chinese-aigc-text-detector",
        "c": sd.cache_id,
    }


def _probe_units(data_dir: Path, texts_subdir: str, kind: str) -> list[dict]:
    rows = probe.load_paragraphs(data_dir, texts_subdir)
    cids = _cache_ids()
    cache = ScoreCache(path=data_dir / "score-cache.jsonl")
    units = []
    for index, row in enumerate(rows):
        for _index, text, context in probe.windows_for(index, row["text"]):
            scores, meta_c = {}, {}
            for lane in ("a", "b", "c"):
                hit = cache.get(
                    lane,
                    cids[lane],
                    text,
                    context=context if lane == "c" else "",
                )
                if not hit or "score" not in hit:
                    raise SystemExit(f"缓存缺失：{data_dir.name} {row['id']} {lane}")
                scores[lane] = float(hit["score"])
                if lane == "c":
                    meta_c = dict(hit.get("meta") or {})
            units.append(
                {
                    "text": text,
                    "scores": scores,
                    "meta_c": meta_c,
                    "class": kind,
                    "para_sha": row["id"],
                    "author": row["id"],
                    "style": row["style"],
                }
            )
    cv._augment_features(units)
    return units


def _predict(units: list[dict], model, train_humans: list[dict]) -> np.ndarray:
    c_values, p_styles, p_ppls = fm._fold_scores(units, train_humans)
    x = fm._matrix(units, c_values, p_styles, p_ppls)
    return model.predict_proba(x)[:, 1]


def _fused(units: list[dict]) -> np.ndarray:
    return np.array(
        [cv.fused(u, cv.ALPHA, u["scores"]["c"]) for u in units]
    )


def _rates(units: list[dict], scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores > threshold))


def _para_rates(units: list[dict], scores: np.ndarray, threshold: float) -> float:
    by_para: dict[str, list[float]] = {}
    for unit, score in zip(units, scores, strict=True):
        by_para.setdefault(unit["para_sha"], []).append(float(score))
    medians = [statistics.median(values) for values in by_para.values()]
    return float(np.mean(np.array(medians) > threshold))


def main() -> int:
    train_units = cv.load_train()
    if not train_units:
        return 1
    cv._augment_features(train_units)
    train_humans = [u for u in train_units if u["class"] == "human"]
    train_c, train_ps, train_pp = fm._fold_scores(train_units, train_humans)
    x_train = fm._matrix(train_units, train_c, train_ps, train_pp)
    y_train = np.array([1.0 if u["class"] == "ai" else 0.0 for u in train_units])
    model = fm._model()
    model.fit(x_train, y_train)
    human_train_p = model.predict_proba(x_train[y_train == 0])[:, 1]
    thresholds = {q: float(np.quantile(human_train_p, q)) for q in fm.QS}

    print("== 学术探针外部复核（元分类器只见过通用 train）==")
    for label, data_dir, subdir, kind in (
        ("人写学术（HansPub 2020）", HUMAN_DIR, "human_texts", "human"),
        ("AI 学术（豆包 Lite 7 章节）", AI_DIR, "texts", "ai"),
    ):
        units = _probe_units(data_dir, subdir, kind)
        meta = _predict(units, model, train_humans)
        fused = _fused(units)
        print(f"\n-- {label}：{len(units)} 窗 / {len({u['para_sha'] for u in units})} 段 --")
        print(
            f"  固定融合@HOT：窗 {_rates(units, fused, HOT):.1%} / "
            f"段中位 {_para_rates(units, fused, HOT):.1%}"
        )
        for q in (0.95, 0.99, 0.995):
            thr = thresholds[q]
            meta_win = _rates(units, meta, thr)
            meta_para = _para_rates(units, meta, thr)
            gray_win = float(
                np.mean(
                    (fused > HOT) | ((fused > 0.50) & (meta > thr))
                )
            )
            gray_para = _gray_para_rates(units, fused, meta, thr, gate=0.50)
            print(
                f"  q={q:.3f} thr={thr:.3f} | meta 窗 {meta_win:.1%} 段 {meta_para:.1%} | "
                f"灰度增强 窗 {gray_win:.1%} 段 {gray_para:.1%}"
            )
        print("  灰度门对比（窗/段）：")
        for q in (0.99, 0.995):
            for gate in (0.50, 0.55, 0.60):
                gray_win = float(
                    np.mean((fused > HOT) | ((fused > gate) & (meta > thresholds[q])))
                )
                gray_para = _gray_para_rates(
                    units, fused, meta, thresholds[q], gate=gate
                )
                print(
                    f"    q={q:.3f} gate={gate:.2f} | 窗 {gray_win:6.1%} 段 {gray_para:6.1%}"
                )
        # 按章节拆分固定融合与灰度增强（q=0.99）
        thr = thresholds[0.99]
        print("  分章节（固定融合 / 灰度增强 q=0.99）：")
        for style in sorted({u["style"] for u in units}):
            subset = [u for u in units if u["style"] == style]
            idx = [i for i, u in enumerate(units) if u["style"] == style]
            f = fused[idx]
            m = meta[idx]
            gray = (f > HOT) | ((f > 0.50) & (m > thr))
            print(
                f"    {style:16s} n={len(subset):3d} 融合 {np.mean(f > HOT):6.1%} "
                f"灰度 {np.mean(gray):6.1%}"
            )

    print("\n说明：学术探针不参与训练；本报告只用于判断元分类器是否值得进入下一版候选。")
    return 0


def _gray_para_rates(
    units: list[dict],
    fused: np.ndarray,
    meta: np.ndarray,
    thr: float,
    gate: float = 0.50,
) -> float:
    by_para: dict[str, list[tuple[float, float]]] = {}
    for unit, f, m in zip(units, fused, meta, strict=True):
        by_para.setdefault(unit["para_sha"], []).append((float(f), float(m)))
    flags = []
    for values in by_para.values():
        f_med = statistics.median(v[0] for v in values)
        m_med = statistics.median(v[1] for v in values)
        flags.append(f_med > HOT or (f_med > gate and m_med > thr))
    return float(np.mean(flags))


if __name__ == "__main__":
    raise SystemExit(main())
