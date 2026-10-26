"""窗口级元分类器：在三路分数 + 风格特征上做 train-only LOGO 评估。

动机：现有融合是固定 α 的线性加权，且 c 路把 PPL 与风格分固定成 0.6/0.4。
如果「哪一路更可信」随文本长度、句长节奏、连接词密度变化，一个轻量
元分类器可以在不换基座模型的前提下改善精确率/召回率。

协议（只读 train，test 不参与）：
  - 人写按作者留一（LOGO-author），AI 按生成器留一（LOGO-generator）；
  - 每个折内只用训练折拟合标准化 + L2 逻辑回归；
  - 人写折内重拟合 c 路 μ/σ/风格常量，避免把留出作者的校准信息带进特征；
  - 阈值按训练折人写概率的 95/99 分位选，再看留出组误伤/召回；
  - 同时给出当前固定融合（α=0.4/0.6，HOT=0.75）的折内基线。

输出是实验报告，不改 constants.py、不改产品路径。若 LOGO 证据支持，
下一版再决定是否把元分类器做成可选融合层。

用法：
    PYTHONPATH=src D:/anaconda/envs/learning/python.exe scripts/fit_meta.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import cv_evaluate as cv  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from aigc_selfcheck.constants import (  # noqa: E402
    ALPHA,
    HEATMAP_HIGH,
    PPL_WEIGHT_IN_C,
    STYLE_BIAS,
)
from aigc_selfcheck.features import cjk_count  # noqa: E402
from aigc_selfcheck.preprocess import split_sentences  # noqa: E402

HOT = HEATMAP_HIGH
QS = (0.90, 0.95, 0.975, 0.99, 0.995, 0.999)

FEATURES = [
    "score_a",
    "score_b",
    "score_c",
    "p_style",
    "p_ppl",
    "log_ppl",
    "fused",
    "cjk",
    "n_sents",
    "mean_sent_len",
    "max_sent_len",
    "connective_density",
    "de_density",
    "colloquial_density",
    "punct_diversity",
    "punct_density",
    "period_comma_ratio",
    "sentence_length_variance",
    "char_mtld",
    "sent_initial_rate",
    "sent_len_step",
    "enum_density",
    "b_minus_c",
    "c_minus_a",
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _feature_row(unit: dict, c_value: float, p_style: float, p_ppl: float) -> dict:
    feats = unit["meta_c"]["features"]
    text = unit["text"]
    sentences = [s.text.strip() for s in split_sentences(text) if s.text.strip()]
    lengths = [len(s) for s in sentences] or [0]
    log_ppl = unit["meta_c"].get("log_ppl")
    row = {
        "score_a": unit["scores"]["a"],
        "score_b": unit["scores"]["b"],
        "score_c": c_value,
        "p_style": p_style,
        "p_ppl": p_ppl,
        "log_ppl": np.nan if log_ppl is None else float(log_ppl),
        "fused": unit["scores"]["b"] * ALPHA["b"] / (ALPHA["b"] + ALPHA["c"])
        + c_value * ALPHA["c"] / (ALPHA["b"] + ALPHA["c"]),
        "cjk": cjk_count(text),
        "n_sents": len(sentences),
        "mean_sent_len": statistics.fmean(lengths),
        "max_sent_len": max(lengths),
        "b_minus_c": unit["scores"]["b"] - c_value,
        "c_minus_a": c_value - unit["scores"]["a"],
    }
    for name in (
        "connective_density",
        "de_density",
        "colloquial_density",
        "punct_diversity",
        "punct_density",
        "period_comma_ratio",
        "sentence_length_variance",
        "char_mtld",
        "sent_initial_rate",
        "sent_len_step",
        "enum_density",
    ):
        row[name] = float(feats.get(name, np.nan))
    return row


def _matrix(units: list[dict], c_values: list[float], p_styles: list[float], p_ppls: list[float]):
    rows = [
        _feature_row(u, c, ps, pp)
        for u, c, ps, pp in zip(units, c_values, p_styles, p_ppls, strict=True)
    ]
    return np.array([[row[name] for name in FEATURES] for row in rows], dtype=float)


def _model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000, random_state=0),
    )


def _fold_scores(units: list[dict], fit_humans: list[dict]) -> tuple[list[float], list[float], list[float]]:
    mu, sigma, stats = cv.fit_constants(fit_humans)
    c_values, p_styles, p_ppls = [], [], []
    for unit in units:
        log_ppl = unit["meta_c"].get("log_ppl")
        p_ppl = 0.5 if log_ppl is None else _sigmoid((mu - float(log_ppl)) / sigma)
        p_style = cv.style_p(unit["meta_c"]["features"], stats, STYLE_BIAS)
        c_values.append(PPL_WEIGHT_IN_C * p_ppl + (1 - PPL_WEIGHT_IN_C) * p_style)
        p_styles.append(p_style)
        p_ppls.append(p_ppl)
    return c_values, p_styles, p_ppls


def _fit_and_predict(
    fit_units: list[dict], held_units: list[dict], fit_humans: list[dict]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    fit_c, fit_ps, fit_pp = _fold_scores(fit_units, fit_humans)
    held_c, held_ps, held_pp = _fold_scores(held_units, fit_humans)
    x_fit = _matrix(fit_units, fit_c, fit_ps, fit_pp)
    y_fit = np.array([1.0 if u["class"] == "ai" else 0.0 for u in fit_units])
    x_held = _matrix(held_units, held_c, held_ps, held_pp)
    model = _model()
    model.fit(x_fit, y_fit)
    p_fit = model.predict_proba(x_fit)[:, 1]
    p_held = model.predict_proba(x_held)[:, 1]
    human_fit = p_fit[y_fit == 0]
    return p_held, human_fit, float(np.quantile(human_fit, 0.95)), float(np.quantile(human_fit, 0.99))


def _fused_with_c(unit: dict, c_value: float) -> float:
    return cv.fused(unit, ALPHA, c_value)


def _paragraph_medians(units: list[dict], scores: np.ndarray) -> np.ndarray:
    by_para: dict[str, list[float]] = {}
    for unit, score in zip(units, scores, strict=True):
        by_para.setdefault(unit["para_sha"], []).append(float(score))
    return np.array([statistics.median(values) for values in by_para.values()])


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.3f}"


def _flag(rule: str, row: dict, thr: float) -> bool:
    if rule == "meta":
        return row["meta"] > thr
    if rule == "meta_or_fused":
        return row["fused"] > HOT or row["meta"] > thr
    if rule == "gray50":
        return row["fused"] > HOT or (row["fused"] > 0.50 and row["meta"] > thr)
    if rule == "gray55":
        return row["fused"] > HOT or (row["fused"] > 0.55 and row["meta"] > thr)
    if rule == "gray57":
        return row["fused"] > HOT or (row["fused"] > 0.57 and row["meta"] > thr)
    if rule == "gray60":
        return row["fused"] > HOT or (row["fused"] > 0.60 and row["meta"] > thr)
    if rule == "meta_and_b":
        return row["fused"] > HOT or (row["meta"] > thr and row["b"] > 0.80)
    if rule == "meta_and_c":
        return row["fused"] > HOT or (row["meta"] > thr and row["c"] > 0.70)
    if rule == "gray50_b":
        return row["fused"] > HOT or (
            row["fused"] > 0.50 and row["meta"] > 0.50 and row["b"] > 0.80
        )
    raise ValueError(rule)


RULES = (
    "meta",
    "meta_or_fused",
    "gray50",
    "gray55",
    "gray57",
    "gray60",
    "meta_and_b",
    "meta_and_c",
    "gray50_b",
)


def _window_rates(
    details: list[dict],
    fold_thresholds: dict[tuple[str, str], dict[float, float]],
    rule: str,
    q: float,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    fpr_by_fold, rec_by_fold = {}, {}
    for fold, rows in _group_by_fold(details, "human").items():
        thr = fold_thresholds[fold][q]
        fpr_by_fold[fold] = statistics.fmean(_flag(rule, r, thr) for r in rows)
    for fold, rows in _group_by_fold(details, "ai").items():
        thr = fold_thresholds[fold][q]
        rec_by_fold[fold] = statistics.fmean(_flag(rule, r, thr) for r in rows)
    return fpr_by_fold, rec_by_fold


def _para_rates(
    details: list[dict],
    fold_thresholds: dict[tuple[str, str], dict[float, float]],
    rule: str,
    q: float,
    kind: str,
) -> dict[tuple[str, str], float]:
    by_fold: dict[tuple[str, str], float] = {}
    for fold, rows in _group_by_fold(details, kind).items():
        thr = fold_thresholds[fold][q]
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["para_sha"], []).append(row)
        flags = []
        for items in grouped.values():
            med = {
                key: statistics.median(item[key] for item in items)
                for key in ("meta", "fused", "b", "c")
            }
            flags.append(_flag(rule, med, thr))
        by_fold[fold] = statistics.fmean(flags)
    return by_fold


def _sweep(details: list[dict], fold_thresholds: dict[tuple[str, str], dict[float, float]]) -> None:
    print("\n== 阈值/规则扫描（窗口级；人写 FPR / AI 召回，均值/最差）==")
    print(f"{'规则':16s} {'q':>5s} {'FPR均值':>8s} {'FPR最差':>8s} {'召回均值':>8s} {'召回最差':>8s}")
    for rule in RULES:
        for q in (0.95, 0.99, 0.995):
            fpr_by_fold, rec_by_fold = _window_rates(details, fold_thresholds, rule, q)
            fpr_mean = statistics.fmean(fpr_by_fold.values())
            fpr_max = max(fpr_by_fold.values())
            rec_mean = statistics.fmean(rec_by_fold.values())
            rec_min = min(rec_by_fold.values())
            print(
                f"{rule:16s} {q:5.3f} {fpr_mean:8.1%} {fpr_max:8.1%} "
                f"{rec_mean:8.1%} {rec_min:8.1%}"
            )

    print("\n== 候选规则（FPR 均值≤2%、最差≤15%、召回均值≥90%；再看段落中位）==")
    found = False
    for rule in RULES:
        for q in (0.95, 0.99, 0.995):
            fpr_by_fold, rec_by_fold = _window_rates(details, fold_thresholds, rule, q)
            fpr_mean = statistics.fmean(fpr_by_fold.values())
            fpr_max = max(fpr_by_fold.values())
            rec_mean = statistics.fmean(rec_by_fold.values())
            rec_min = min(rec_by_fold.values())
            if not (fpr_mean <= 0.02 and fpr_max <= 0.15 and rec_mean >= 0.90):
                continue
            found = True
            para_fpr = _para_rates(details, fold_thresholds, rule, q, "human")
            para_rec = _para_rates(details, fold_thresholds, rule, q, "ai")
            print(
                f"  {rule:16s} q={q:.3f} | 窗口 FPR {fpr_mean:.1%}/{fpr_max:.1%} "
                f"召回 {rec_mean:.1%}/{rec_min:.1%} | 段落 FPR "
                f"{statistics.fmean(para_fpr.values()):.1%}/{max(para_fpr.values()):.1%} "
                f"召回 {statistics.fmean(para_rec.values()):.1%}/{min(para_rec.values()):.1%}"
            )
    if not found:
        print("  无规则同时满足上述门槛。")


def _group_by_fold(details: list[dict], kind: str) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in details:
        if row["kind"] == kind:
            grouped.setdefault(row["fold"], []).append(row)
    return grouped


def main() -> int:
    units = cv.load_train()
    if not units:
        return 1
    cv._augment_features(units)
    humans = [u for u in units if u["class"] == "human"]
    ais = [u for u in units if u["class"] == "ai"]
    print(f"train：human {len(humans)} 窗 / ai {len(ais)} 窗 / {len({u['author'] for u in humans})} 作者 / "
          f"{len({cv.generator_of(u['author']) for u in ais})} 生成器")
    print(f"特征 {len(FEATURES)} 个；模型 L2-LR(C=0.3)；阈值取训练折人写概率 95/99 分位\n")

    # ── LOGO-作者：人写误伤 ──
    author_rows = []
    details: list[dict] = []
    fold_thresholds: dict[tuple[str, str], dict[float, float]] = {}
    for author in sorted({u["author"] for u in humans}):
        fit_humans = [u for u in humans if u["author"] != author]
        held = [u for u in humans if u["author"] == author]
        fit_units = fit_humans + ais
        p_held, human_fit, thr95, thr99 = _fit_and_predict(fit_units, held, fit_humans)
        fold_key = ("author", author)
        fold_thresholds[fold_key] = {q: float(np.quantile(human_fit, q)) for q in QS}
        # 当前融合基线：同折 c 常量
        c_held, _ps, _pp = _fold_scores(held, fit_humans)
        fused_held = np.array([_fused_with_c(u, c) for u, c in zip(held, c_held, strict=True)])
        para_meta = _paragraph_medians(held, p_held)
        para_fused = _paragraph_medians(held, fused_held)
        details.extend(
            {
                "kind": "human",
                "fold": fold_key,
                "para_sha": u["para_sha"],
                "meta": float(p),
                "fused": float(f),
                "b": float(u["scores"]["b"]),
                "c": float(c),
            }
            for u, p, f, c in zip(held, p_held, fused_held, c_held, strict=True)
        )
        author_rows.append(
            {
                "author": author,
                "n": len(held),
                "fpr95": float(np.mean(p_held > thr95)),
                "fpr99": float(np.mean(p_held > thr99)),
                "fpr50": float(np.mean(p_held > 0.5)),
                "fpr_fused": float(np.mean(fused_held > HOT)),
                "para_fpr95": float(np.mean(para_meta > thr95)),
                "para_fpr99": float(np.mean(para_meta > thr99)),
                "para_fpr_fused": float(np.mean(para_fused > HOT)),
            }
        )
    print("== LOGO-作者：元分类器 vs 固定融合（留出作者误伤）==")
    print(f"{'作者':24s} {'n':>3s} {'meta@95':>8s} {'meta@99':>8s} {'meta@.5':>8s} {'融合@HOT':>9s}")
    for row in author_rows:
        print(
            f"{row['author']:24s} {row['n']:3d} {row['fpr95']:8.1%} {row['fpr99']:8.1%} "
            f"{row['fpr50']:8.1%} {row['fpr_fused']:9.1%}"
        )
    print(
        f"  mean: meta@95 {statistics.fmean(r['fpr95'] for r in author_rows):.1%} / "
        f"meta@99 {statistics.fmean(r['fpr99'] for r in author_rows):.1%} / "
        f"融合 {statistics.fmean(r['fpr_fused'] for r in author_rows):.1%}；"
        f"max: meta@95 {max(r['fpr95'] for r in author_rows):.1%} / "
        f"meta@99 {max(r['fpr99'] for r in author_rows):.1%} / "
        f"融合 {max(r['fpr_fused'] for r in author_rows):.1%}"
    )

    # ── LOGO-生成器：AI 召回 ──
    gen_rows = []
    for gen in sorted({cv.generator_of(u["author"]) for u in ais}):
        fit_ais = [u for u in ais if cv.generator_of(u["author"]) != gen]
        held = [u for u in ais if cv.generator_of(u["author"]) == gen]
        fit_units = humans + fit_ais
        p_held, human_fit, thr95, thr99 = _fit_and_predict(fit_units, held, humans)
        fold_key = ("gen", gen)
        fold_thresholds[fold_key] = {q: float(np.quantile(human_fit, q)) for q in QS}
        c_held, _ps, _pp = _fold_scores(held, humans)
        fused_held = np.array([_fused_with_c(u, c) for u, c in zip(held, c_held, strict=True)])
        para_meta = _paragraph_medians(held, p_held)
        para_fused = _paragraph_medians(held, fused_held)
        details.extend(
            {
                "kind": "ai",
                "fold": fold_key,
                "para_sha": u["para_sha"],
                "meta": float(p),
                "fused": float(f),
                "b": float(u["scores"]["b"]),
                "c": float(c),
            }
            for u, p, f, c in zip(held, p_held, fused_held, c_held, strict=True)
        )
        gen_rows.append(
            {
                "gen": gen,
                "n": len(held),
                "rec95": float(np.mean(p_held > thr95)),
                "rec99": float(np.mean(p_held > thr99)),
                "rec50": float(np.mean(p_held > 0.5)),
                "rec_fused": float(np.mean(fused_held > HOT)),
                "para_rec95": float(np.mean(para_meta > thr95)),
                "para_rec99": float(np.mean(para_meta > thr99)),
                "para_rec_fused": float(np.mean(para_fused > HOT)),
            }
        )
    print("\n== LOGO-生成器：元分类器 vs 固定融合（留出生成器召回）==")
    print(f"{'生成器':26s} {'n':>3s} {'meta@95':>8s} {'meta@99':>8s} {'meta@.5':>8s} {'融合@HOT':>9s}")
    for row in gen_rows:
        print(
            f"{row['gen']:26s} {row['n']:3d} {row['rec95']:8.1%} {row['rec99']:8.1%} "
            f"{row['rec50']:8.1%} {row['rec_fused']:9.1%}"
        )
    print(
        f"  mean: meta@95 {statistics.fmean(r['rec95'] for r in gen_rows):.1%} / "
        f"meta@99 {statistics.fmean(r['rec99'] for r in gen_rows):.1%} / "
        f"融合 {statistics.fmean(r['rec_fused'] for r in gen_rows):.1%}；"
        f"min: meta@95 {min(r['rec95'] for r in gen_rows):.1%} / "
        f"meta@99 {min(r['rec99'] for r in gen_rows):.1%} / "
        f"融合 {min(r['rec_fused'] for r in gen_rows):.1%}"
    )

    # ── 段落级（同一段的多窗取中位）──
    print("\n== 段落级（同段多窗取中位；LOGO 折内概率）==")
    print(
        f"  人写误伤: meta@95 {statistics.fmean(r['para_fpr95'] for r in author_rows):.1%} / "
        f"meta@99 {statistics.fmean(r['para_fpr99'] for r in author_rows):.1%} / "
        f"融合 {statistics.fmean(r['para_fpr_fused'] for r in author_rows):.1%}"
    )
    print(
        f"  AI 召回:  meta@95 {statistics.fmean(r['para_rec95'] for r in gen_rows):.1%} / "
        f"meta@99 {statistics.fmean(r['para_rec99'] for r in gen_rows):.1%} / "
        f"融合 {statistics.fmean(r['para_rec_fused'] for r in gen_rows):.1%}"
    )

    _sweep(details, fold_thresholds)

    print("\n结论口径：只有 meta@95 的 LOGO 误伤不高于融合、且 LOGO 召回均值/最小值")
    print("同时高于融合，才值得进入下一版候选；否则维持现融合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
