"""实验性增强召回：灰度区元分类器（meta v1）。

默认产品路径不调用本模块。`--enhanced` 打开后，除固定融合 `p_fuse > HOT`
之外，再把「`p_fuse > 0.55` 且元分类器概率 > 阈值」的窗口标为增强命中。
它提高召回，但会增加误报：学术探针上人写窗口误报 3.5%（默认融合 2.6%），
通用 LOGO-作者 FPR 均值 1.7% / 最差 11.1%。因此默认关闭，报告中必须标注
「实验性」。

训练/推理口径：工件里的 `meta_c` 常量来自 `cv.fit_constants(train_humans)`，
不是产品 c 路的部署常量（μ=4.24/σ=0.68）。元分类器的 c 特征必须用工件
常量重算，才能与训练时一致；固定融合仍用产品 c 分。
"""

from __future__ import annotations

import json
import math
import statistics
from functools import lru_cache
from pathlib import Path

from .constants import ALPHA, HEATMAP_HIGH
from .features import SENT_INITIALS, cjk_count
from .preprocess import split_sentences

ARTIFACT_PATH = Path(__file__).with_name("data") / "meta_v1.json"

# 与 scripts/fit_meta.FEATURES 保持一致；工件加载时会校验。
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

STYLE_FEATURES = (
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
)

ENUM_MARKERS = (
    "第一",
    "第二",
    "第三",
    "其一",
    "其二",
    "一是",
    "二是",
    "首先",
    "其次",
    "再者",
    "最后",
    "此外",
    "另外",
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@lru_cache(maxsize=1)
def load_artifact(path: str | Path | None = None) -> dict:
    target = Path(path) if path else ARTIFACT_PATH
    artifact = json.loads(target.read_text(encoding="utf-8"))
    if artifact.get("schema") != "aigc-selfcheck/meta@v1":
        raise ValueError(f"不支持的元分类器工件：{target}")
    if list(artifact.get("features") or []) != FEATURES:
        raise ValueError("元分类器特征顺序与产品代码不一致，请重新导出工件")
    for key in ("imputer_median", "scaler_mean", "scaler_scale", "coef"):
        if len(artifact.get(key) or []) != len(FEATURES):
            raise ValueError(f"元分类器工件字段 {key} 长度不正确")
    return artifact


def _style_score(features: dict[str, float], artifact: dict) -> float:
    cfg = artifact["meta_c"]
    total = float(cfg["style_bias"])
    for name, st in cfg["style_stats"].items():
        value = features.get(name)
        if value is None or value != value:
            continue
        std = float(st["std"]) or 1.0
        total += (
            float(st["weight"])
            * float(st["direction"])
            * (float(value) - float(st["mean"]))
            / std
        )
    return _sigmoid(total)


def _ppl_score(log_ppl: float | None, artifact: dict) -> float:
    if log_ppl is None or log_ppl != log_ppl:
        return 0.5
    cfg = artifact["meta_c"]
    return _sigmoid((float(cfg["mu"]) - float(log_ppl)) / float(cfg["sigma"]))


def c_meta_score(
    features: dict[str, float], log_ppl: float | None, artifact: dict
) -> tuple[float, float, float]:
    """返回 (c_meta, p_style, p_ppl)，常量取自工件而非产品部署常量。"""
    p_style = _style_score(features, artifact)
    p_ppl = _ppl_score(log_ppl, artifact)
    weight = float(artifact["meta_c"]["ppl_weight"])
    return weight * p_ppl + (1.0 - weight) * p_style, p_style, p_ppl


def _sent_len_step(lengths: list[int]) -> float:
    if len(lengths) < 2:
        return 0.0
    mean_len = statistics.fmean(lengths)
    if mean_len <= 0:
        return 0.0
    step = statistics.fmean(
        abs(lengths[i + 1] - lengths[i]) for i in range(len(lengths) - 1)
    )
    return step / mean_len


def _enum_density(text: str) -> float:
    cjk = max(1, cjk_count(text))
    return sum(text.count(word) for word in ENUM_MARKERS) / cjk * 1000


def feature_row(
    text: str,
    scores: dict[str, float],
    c_meta: dict,
    artifact: dict,
) -> dict[str, float]:
    """按训练脚本 scripts/fit_meta._feature_row 的口径构造一行特征。"""
    features = c_meta.get("features") or {}
    sentences = [s.text.strip() for s in split_sentences(text) if s.text.strip()]
    lengths = [len(s) for s in sentences] or [0]
    log_ppl = c_meta.get("log_ppl")
    c_value, p_style, p_ppl = c_meta_score(features, log_ppl, artifact)
    b = float(scores["b"])
    a = float(scores["a"])
    active = ALPHA["b"] + ALPHA["c"]
    fused = b * ALPHA["b"] / active + c_value * ALPHA["c"] / active

    row = {
        "score_a": a,
        "score_b": b,
        "score_c": c_value,
        "p_style": p_style,
        "p_ppl": p_ppl,
        "log_ppl": float("nan") if log_ppl is None else float(log_ppl),
        "fused": fused,
        "cjk": float(cjk_count(text)),
        "n_sents": float(len(sentences)),
        "mean_sent_len": statistics.fmean(lengths),
        "max_sent_len": float(max(lengths)),
        "b_minus_c": b - c_value,
        "c_minus_a": c_value - a,
    }
    for name in STYLE_FEATURES:
        value = features.get(name)
        row[name] = float("nan") if value is None else float(value)
    row["sent_initial_rate"] = (
        sum(1 for s in sentences if s.startswith(SENT_INITIALS)) / len(sentences)
        if sentences
        else 0.0
    )
    row["sent_len_step"] = _sent_len_step(lengths)
    row["enum_density"] = _enum_density(text)
    return row


def probability(row: dict[str, float], artifact: dict) -> float:
    """按工件做缺失值填充 → 标准化 → 逻辑回归。"""
    total = float(artifact["intercept"])
    for index, name in enumerate(FEATURES):
        value = row.get(name, float("nan"))
        if value is None or value != value:
            value = float(artifact["imputer_median"][index])
        scale = float(artifact["scaler_scale"][index]) or 1.0
        z = (float(value) - float(artifact["scaler_mean"][index])) / scale
        total += float(artifact["coef"][index]) * z
    return _sigmoid(total)


def enhanced_flag(fused: float, meta_prob: float, artifact: dict) -> bool:
    return fused > HEATMAP_HIGH or (
        fused > float(artifact["gate"]) and meta_prob > float(artifact["threshold"])
    )


def apply_enhanced(fused_windows: list, artifact: dict | None = None) -> dict:
    """就地写入 `meta_prob` / `enhanced`；返回报告用摘要。

    需要 a,b,c 三路分数齐全；缺路时抛 ValueError，由调用方决定是否关闭。
    """
    artifact = artifact or load_artifact()
    flagged = []
    skipped = 0
    for window in fused_windows:
        missing = {"a", "b", "c"} - set(window.scores)
        if missing:
            raise ValueError(f"增强召回需要 a,b,c 三路分数，窗口 {window.window_key} 缺少 {sorted(missing)}")
        c_meta = window.metas.get("c") or {}
        if not c_meta.get("features"):
            # 旧缓存或异常路径可能没有风格特征；跳过该窗，不用中位数硬凑。
            window.meta_prob = None
            window.enhanced = False
            skipped += 1
            continue
        row = feature_row(window.text, window.scores, c_meta, artifact)
        prob = probability(row, artifact)
        window.meta_prob = prob
        window.enhanced = enhanced_flag(window.p_fuse, prob, artifact)
        if window.enhanced:
            flagged.append(window)
    return {
        "enabled": True,
        "gate": float(artifact["gate"]),
        "threshold": float(artifact["threshold"]),
        "n_flagged": len(flagged),
        "n_extra": sum(1 for w in flagged if w.p_fuse <= HEATMAP_HIGH),
        "n_skipped": skipped,
    }
