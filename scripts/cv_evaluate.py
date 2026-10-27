"""分组交叉验证（P1 协议）：只用 train split，test 不参与任何计算或决策。

背景：单次 80/20 切分在 15~20 个 test 窗口上波动大；分组 CV 给出方差的
诚实估计。分组单位是「会引入相关性的自然簇」——人写按作者、AI 按生成器：
同一作者的多个窗口共享文风，同一生成器的多个窗口共享模型文风，切分时
必须整组进/整组出，否则 AUC 虚高（v2 单 GLM 语料的满分即是教训）。

两个互补分析（均在 train 上）：
  1. LOGO-生成器：留出一个 AI 生成器，看「对未见生成器」的判别力——
     直接量化 DeepSeek/豆包 这类漏检是普遍泛化缺口还是个例。
  2. LOGO-作者：留出一位人写作者，C 路常量在其余作者上重拟合（离线
     重算 p_C），看「对未见作者」的误伤稳定性——校准常量是否过拟合到
     特定作者。

离线重算说明：C 路分数 = 0.6·σ((μ−logPPL)/σ) + 0.4·σ(Σw·z+b)，
缓存 meta 存有原始 log_ppl 与特征值，公式与 stats_lm.py 逐行一致，
任意常量组合可离线精确重算，无需重跑参考模型（v6 起为 Qwen3.5-2B-Base）。

--fit-style：风格路权重/方向重拟合（v3.2 起的常量来源）。人写 train 窗口
拟合 z 的 mean/std，再在（人写 + AI）train 窗口上拟合 L2 逻辑回归
（numpy 手写，确定性），打印可直接粘贴进 constants.py 的 STYLE_STATS 块
与嵌套 LOGO-作者校验。test split 不参与（本脚本只读 train）。

v8 起 C 路缓存键含上下文（窗口起点之前的正文，见 `contexts.py`），校准集的
`context` 字段与产品路径同口径；历史无上下文条目只对首窗（空上下文）可命中。

用法：
    uv run python scripts/cv_evaluate.py
    uv run python scripts/cv_evaluate.py --fit-style
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_selfcheck.cache import ScoreCache  # noqa: E402
from aigc_selfcheck.constants import (  # noqa: E402
    ALPHA,
    BOOST_B,
    BOOST_C,
    BOOST_FLOOR,
    HEATMAP_HIGH,
    PPL_LOG_MEAN,
    PPL_LOG_STD,
    PPL_WEIGHT_IN_C,
    STYLE_BIAS,
    STYLE_FIXED_SCALE,
    STYLE_STATS,
)
from aigc_selfcheck.contexts import window_contexts  # noqa: E402
from aigc_selfcheck.detectors.stats_lm import StatsDetector  # noqa: E402
from aigc_selfcheck.features import window_features  # noqa: E402
from aigc_selfcheck.preprocess import split_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "calibration"
HOT = HEATMAP_HIGH  # 与 evaluate.py / 产品红线同源

# C 路 cache_id 并入校准常量指纹——常量一变，旧缓存条目按设计失效。但离线
# 分析真正需要的只有 meta（log_ppl；它只依赖模型与窗口文本，与常量无关）。
# 这里登记历史轮次的常量指纹用于 meta 回溯查找：当前指纹查不到时依次回退，
# 命中即取其 log_ppl。新增常量轮次后在此追加当时的 (μ, σ, w_in_c, bias, stats)。
_LEGACY_C_CONSTANTS: list[tuple[str, tuple]] = [
    # v7（2026-10-26，无上下文口径；v8 修正上下文前的现行常量）
    ("Qwen/Qwen3.5-2B-Base", (4.67, 0.81, 0.6, -2.045, sorted(STYLE_STATS.items()))),
    # v6（2026-10-26，Qwen3.5-2B-Base 首轮；v7 重导 μ/σ 前的现行常量）
    (
        "Qwen/Qwen3.5-2B-Base",
        (
            4.65, 0.84, 0.6, -2.045,
            sorted({
                "period_comma_ratio": {"mean": 0.75, "std": 1.28, "weight": 1.372, "direction": -1},
                "punct_density": {"mean": 104.46, "std": 33.85, "weight": 1.174, "direction": -1},
                "de_density": {"mean": 43.75, "std": 20.33, "weight": 1.044, "direction": -1},
                "char_mtld": {"mean": 47.91, "std": 25.91, "weight": 0.873, "direction": 1},
                "colloquial_density": {"mean": 2.26, "std": 8.61, "weight": 0.648, "direction": 1},
                "punct_diversity": {"mean": 0.29, "std": 0.57, "weight": 0.604, "direction": 1},
                "connective_density": {"mean": 0.00, "std": 1.00, "weight": 0.530, "direction": 1},
                "sent_initial_rate": {"mean": 0.03, "std": 0.11, "weight": 0.324, "direction": 1},
            }.items()),
        ),
    ),
    # v4/v5（2026-10-25，锁定 v3.2 权重 + v4 归一化；v6 换 2B 前的现行常量）
    (
        "Qwen/Qwen2.5-0.5B",
        (
            4.47, 0.74, 0.6, -2.045,
            sorted({
                "period_comma_ratio": {"mean": 0.75, "std": 1.28, "weight": 1.372, "direction": -1},
                "punct_density": {"mean": 104.46, "std": 33.85, "weight": 1.174, "direction": -1},
                "de_density": {"mean": 43.75, "std": 20.33, "weight": 1.044, "direction": -1},
                "char_mtld": {"mean": 47.91, "std": 25.91, "weight": 0.873, "direction": 1},
                "colloquial_density": {"mean": 2.26, "std": 8.61, "weight": 0.648, "direction": 1},
                "punct_diversity": {"mean": 0.29, "std": 0.57, "weight": 0.604, "direction": 1},
                "connective_density": {"mean": 0.00, "std": 1.00, "weight": 0.530, "direction": 1},
                "sent_initial_rate": {"mean": 0.03, "std": 0.11, "weight": 0.324, "direction": 1},
            }.items()),
        ),
    ),
    # v3.2（2026-10-25，pruned8 LR 拟合轮；v4 归一化刷新前的现行常量，指纹 96f9602ea19d）
    (
        "Qwen/Qwen2.5-0.5B",
        (
            4.47, 0.78, 0.6, -2.045,
            sorted({
                "period_comma_ratio": {"mean": 0.71, "std": 1.20, "weight": 1.372, "direction": -1},
                "punct_density": {"mean": 100.16, "std": 32.85, "weight": 1.174, "direction": -1},
                "de_density": {"mean": 42.15, "std": 22.09, "weight": 1.044, "direction": -1},
                "char_mtld": {"mean": 50.62, "std": 23.19, "weight": 0.873, "direction": 1},
                "colloquial_density": {"mean": 1.33, "std": 4.62, "weight": 0.648, "direction": 1},
                "punct_diversity": {"mean": 0.15, "std": 0.40, "weight": 0.604, "direction": 1},
                "connective_density": {"mean": 0.00, "std": 1.00, "weight": 0.530, "direction": 1},
                "sent_initial_rate": {"mean": 0.03, "std": 0.12, "weight": 0.324, "direction": 1},
            }.items()),
        ),
    ),
    # v3.1（2026-10-26，μ=75分位 + a 路归零 + 阈值重定心轮）
    (
        "Qwen/Qwen2.5-0.5B",
        (
            4.47, 0.78, 0.6, -0.3,
            sorted({
                "connective_density": {"mean": 0.0, "std": 1.0, "weight": 0.9, "direction": +1},
                "de_density": {"mean": 42.2, "std": 22.1, "weight": 0.5, "direction": +1},
                "sentence_length_variance": {"mean": 368.1, "std": 470.9, "weight": 0.7, "direction": -1},
                "punct_diversity": {"mean": 0.2, "std": 0.4, "weight": 0.2, "direction": -1},
                "colloquial_density": {"mean": 1.3, "std": 4.6, "weight": 0.1, "direction": -1},
                "char_mtld": {"mean": 50.6, "std": 23.2, "weight": 0.2, "direction": -1},
            }.items()),
        ),
    ),
]


def _legacy_c_cache_ids() -> list[str]:
    import hashlib

    return [
        f"{model_id}@{hashlib.sha256(repr(c).encode()).hexdigest()[:12]}"
        for model_id, c in _LEGACY_C_CONSTANTS
    ]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def style_p(features: dict, stats: dict, bias: float) -> float:
    """stats_lm.style_score 的离线版（常量可注入）。"""
    total = bias
    for name, st in stats.items():
        if name not in features:
            continue
        std = st["std"] or 1.0
        total += st["weight"] * st["direction"] * (features[name] - st["mean"]) / std
    return sigmoid(total)


def finite_log_ppl(meta: dict) -> float | None:
    """缓存里的 log_ppl，None/NaN（PPL 退化窗口）统一返回 None。

    NaN 会由缓存原样写回（json.dumps 允许 NaN 字面量），只判 None 会让它
    传染 median/pstdev 与 p_ppl；生产 detect_batch 用 NaN 检查退回 0.5。
    """
    value = meta.get("log_ppl")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def lane_c(meta: dict, mu: float, sigma: float, stats: dict, bias: float = STYLE_BIAS) -> float | None:
    log_ppl = finite_log_ppl(meta)
    # PPL 退化窗口：短于 MIN_CJK_FOR_PPL 或预测位为 0 时 log_ppl 为 NaN。
    # 生产 StatsDetector.detect_batch 用 NaN 检查退回 p_ppl=0.5，这里必须同口径。
    if log_ppl is None:
        p_ppl = 0.5
    else:
        p_ppl = sigmoid((mu - log_ppl) / sigma)
    features = meta.get("features") or {}
    if not features:
        return None
    p_style = style_p({k: float(v) for k, v in features.items() if v is not None}, stats, bias)
    return PPL_WEIGHT_IN_C * p_ppl + (1 - PPL_WEIGHT_IN_C) * p_style


def fit_constants(human_units: list[dict]) -> tuple[float, float, dict]:
    """与 evaluate.py --fit 同一套拟合：μ=median(log_ppl)，σ=pstdev，风格 mean/std。"""
    log_ppls = [v for u in human_units if (v := finite_log_ppl(u["meta_c"])) is not None]
    rows = [u["meta_c"].get("features") or {} for u in human_units]
    rows = [{k: float(v) for k, v in r.items() if v is not None and v == v} for r in rows if r]
    mu = statistics.median(log_ppls)
    sigma = statistics.pstdev(log_ppls) or 1.0
    stats = {}
    for name in STYLE_STATS:
        if name in STYLE_FIXED_SCALE:  # 单侧证据：mean=0/std=1 固定，不重估
            stats[name] = dict(STYLE_STATS[name])
            continue
        values = [r[name] for r in rows if name in r]
        if len(values) < 2:
            stats[name] = dict(STYLE_STATS[name])
            continue
        std = statistics.pstdev(values) or 1.0
        stats[name] = {**STYLE_STATS[name], "mean": statistics.fmean(values), "std": std}
    return mu, sigma, stats


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


def generator_of(author: str) -> str:
    """生成器分组：模型已记录（API 样本）按模型，否则按平台（网页样本模型未记录）。

    v4 起聚合端点样本的平台字段都是「阿里云百炼(API)」，按平台分组会把多个
    不同模型并成一个生成器，LOGO-生成器折失去意义——故模型可辨时优先取模型。
    """
    m = re.search(r"模型:\s*([^\|]+)", author)
    if m and "未记录" not in m.group(1):
        return m.group(1).strip()
    m = re.search(r"平台:\s*([^\|]+)", author)
    return m.group(1).strip() if m else "unknown"


def load_train() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (ROOT / "out" / "calibration.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    contexts = window_contexts(rows)
    cache = ScoreCache(path=ROOT / "out" / "score-cache.jsonl")
    sd = StatsDetector.__new__(StatsDetector)  # 只取 cache_id，不加载模型
    cids = {"a": "yuchuantian/AIGC_detector_zhv2", "b": "mokawa3018/cmj-chinese-aigc-text-detector", "c": sd.cache_id}
    c_meta_cids = [cids["c"], *_legacy_c_cache_ids()]
    units = []
    for row_index, r in enumerate(rows):
        if r["split"] != "train":
            continue
        context = contexts[row_index]
        # 缓存 meta 的特征是打分时刻的 features.py 产出，可能滞后于当前代码
        # （如 v3.2 新增 period_comma_ratio / punct_density / sent_initial_rate）——
        # 一律按当前代码从窗口文本重算（纯 CPU 毫秒级）；log_ppl 无法离线重算，沿用缓存。
        sentences = [s.text.strip() for s in split_sentences(r["text"])]
        scores: dict[str, float] = {}
        meta_c: dict = {}
        for lane in ("a", "b", "c"):
            cache_context = context if lane == "c" else ""
            hit = cache.get(lane, cids[lane], r["text"], context=cache_context)
            if not hit or "score" not in hit:
                if lane == "c":
                    # 当前指纹无条目（常量已变更）：meta 回溯历史指纹取 log_ppl，
                    # 分数按当前常量离线重算（与 stats_lm.detect 逐行一致）。
                    for legacy_cid in c_meta_cids[1:]:
                        legacy = cache.get("c", legacy_cid, r["text"], context=context)
                        if legacy and "meta" in legacy:
                            meta_c = dict(legacy["meta"])
                            break
                    if not meta_c:
                        print(f"  [miss] c 路 meta（含历史指纹）缺失：{r['uid']}")
                        return []
                    if finite_log_ppl(meta_c) is None:
                        # 极短窗口（如「简单罗列几项：」）PPL 退化；与当前指纹命中时
                        # 的行为保持一致：lane_c 内部按 p_ppl=0.5 处理，不中断分析。
                        print(f"  [warn] c 路 log_ppl 退化，按 p_ppl=0.5 处理：{r['uid']}")
                    meta_c["features"] = window_features(r["text"], sentences)
                    scores["c"] = lane_c(meta_c, PPL_LOG_MEAN, PPL_LOG_STD, STYLE_STATS)
                    continue
                print(f"  [miss] {lane} 路缓存缺失：{r['uid']}")
                return []
            scores[lane] = float(hit["score"])
            if lane == "c":
                meta_c = dict(hit.get("meta") or {})
        meta_c["features"] = window_features(r["text"], sentences)
        units.append({**r, "scores": scores, "meta_c": meta_c})
    return units


def fused(unit: dict, alpha: dict[str, float], c_override: float | None = None) -> float:
    scores = dict(unit["scores"])
    if c_override is not None:
        scores["c"] = c_override
    active = {k: alpha[k] for k in scores if alpha.get(k)}
    p_fuse = sum(scores[k] * alpha[k] for k in active) / sum(active.values())
    if scores.get("b", 0.0) >= BOOST_B and scores.get("c", 0.0) >= BOOST_C:
        p_fuse = max(p_fuse, BOOST_FLOOR)
    return p_fuse


# ── --fit-style：风格路 LR 重拟合（v3.2 起的常量来源，全部 train-only）──

# v3.2 选型的候选特征集（嵌套分组 CV 比较；见 calibration/README.md v3.2 节）。
# 显式列表、不动态引用 STYLE_STATS——保证跨轮次可复现（当前部署 pruned8 + 单侧证据；
# v10_lean12 是全量重拟合候选，因古典散文探针回归被否决，保留供复现）。
STYLE_FEATURE_VARIANTS: dict[str, list[str]] = {
    "pruned6": [
        "punct_density", "period_comma_ratio", "de_density", "char_mtld",
        "punct_diversity", "colloquial_density",
    ],
    "pruned8": [
        "punct_density", "period_comma_ratio", "de_density", "char_mtld",
        "punct_diversity", "colloquial_density", "connective_density", "sent_initial_rate",
    ],
    # full：所有可从窗口文本计算的风格特征 + 增补候选（含 v3.2 淘汰的句长节奏类）
    "full": [
        "connective_density", "de_density", "colloquial_density", "punct_diversity",
        "punct_density", "period_comma_ratio", "sentence_length_variance", "char_mtld",
        "sent_initial_rate", "sent_len_step", "enum_density",
    ],
    # v10c：base8 + 两个单侧数字证据（部署口径）。证据特征归一化固定 0/1，
    # 不参与人写 mean/std 重估；其余 base8 仍按留出集重估，用于检验部署特征集
    # 本身是否比全量重拟合更稳。
    "v10c_base8_ev": [
        "punct_density", "period_comma_ratio", "de_density", "char_mtld",
        "punct_diversity", "colloquial_density", "connective_density", "sent_initial_rate",
        "num_evidence", "digit_evidence",
    ],
    # v10：在 pruned8 上加入数字/正式文体特征（解决官方文体低 PPL 误伤）。
    # 数字两特征在 features.py 内已截断（60‰ / 50‰），避免表格化公报获得过强证据。
    # 注意：该变体走全量重拟合，会把风格截距从 -2.045 抬高，导致古典散文探针
    # 误伤 0%→40%；v10c 已改用「base8 + 单侧证据」两段式，此处仅保留复现。
    "v10_lean12": [
        "punct_density", "period_comma_ratio", "de_density", "char_mtld",
        "punct_diversity", "colloquial_density", "connective_density", "sent_initial_rate",
        "digit_density", "num_unit_density", "dunhao_density", "formal_density",
    ],
}


def _augment_features(units: list[dict]) -> None:
    """为变体比较补齐 v3.2 未纳入 window_features 的候选特征（就地写回 meta）。"""
    sent_initials = (
        "因此", "然而", "此外", "同时", "另外", "总之", "综上", "值得注意",
        "总体而言", "换句话说", "具体而言", "例如", "这意味着", "更重要的是",
    )
    enum_markers = ("第一", "第二", "第三", "其一", "其二", "一是", "二是",
                    "首先", "其次", "再者", "最后", "此外", "另外")
    for u in units:
        feats = u["meta_c"]["features"]
        sentences = [s.text.strip() for s in split_sentences(u["text"]) if s.text.strip()]
        if sentences:
            hits = sum(1 for s in sentences if s.startswith(sent_initials))
            feats["sent_initial_rate"] = hits / len(sentences)
        else:
            feats["sent_initial_rate"] = 0.0
        lengths = [len(s) for s in sentences]
        if len(lengths) >= 2:
            mean_len = statistics.fmean(lengths)
            step = statistics.fmean([abs(lengths[i + 1] - lengths[i]) for i in range(len(lengths) - 1)])
            feats["sent_len_step"] = step / mean_len if mean_len > 0 else 0.0
        else:
            feats["sent_len_step"] = 0.0
        cjk = max(1, sum(1 for ch in u["text"] if "\u4e00" <= ch <= "\u9fff"))
        feats["enum_density"] = sum(u["text"].count(w) for w in enum_markers) / cjk * 1000


def fit_style_lr(
    train_humans: list[dict], train_ais: list[dict], names: list[str]
) -> tuple[dict[str, float], dict[str, float], list[float], float]:
    """人写拟合 z 的 mean/std →（人写+AI）拟合 L2 逻辑回归。

    返回 (mean, std, weights, bias)。weight/direction 与 LR 系数的关系：
    total = bias + Σ sign(w_i)·|w_i|·(x−mean)/std ≡ intercept + Σ coef_i·z_i。
    """
    import numpy as np

    mean = {}
    std = {}
    for f in names:
        if f in STYLE_FIXED_SCALE:
            mean[f], std[f] = 0.0, 1.0
        else:
            mean[f] = statistics.fmean([u["meta_c"]["features"][f] for u in train_humans])
            std[f] = statistics.pstdev([u["meta_c"]["features"][f] for u in train_humans]) or 1.0
    x = [
        [(u["meta_c"]["features"][f] - mean[f]) / std[f] for f in names]
        for u in train_humans + train_ais
    ]
    y = np.array([0.0] * len(train_humans) + [1.0] * len(train_ais))
    xa = np.array(x)
    n, d = xa.shape
    w = np.zeros(d)
    b = 0.0
    l2, lr_rate, iters = 1.0, 0.1, 2000
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xa @ w + b)))
        w -= lr_rate * (xa.T @ (p - y) / n + l2 * w / n)
        b -= lr_rate * float(np.mean(p - y))
    return mean, std, w.tolist(), float(b)


def _c_with_style(unit: dict, names, mean, std, weights, bias, mu: float, sigma: float) -> float:
    """离线版 p_C：PPL 项用缓存 log_ppl + 注入常量，风格项用注入 LR。"""
    features = unit["meta_c"]["features"]
    total = bias + sum(wi * (features[f] - mean[f]) / std[f] for f, wi in zip(names, weights, strict=True))
    p_style = sigmoid(total)
    log_ppl = finite_log_ppl(unit["meta_c"])
    p_ppl = 0.5 if log_ppl is None else sigmoid((mu - log_ppl) / sigma)
    return PPL_WEIGHT_IN_C * p_ppl + (1 - PPL_WEIGHT_IN_C) * p_style


def _mu_rule(log_ppls: list[float], rule: str) -> float:
    ordered = sorted(log_ppls)
    if rule == "median":
        return statistics.median(ordered)
    return ordered[int(len(ordered) * 0.75)]  # 75 分位（v3.1 采用）


def cmd_fit_style(units: list[dict]) -> int:
    humans = [u for u in units if u["class"] == "human"]
    ais = [u for u in units if u["class"] == "ai"]
    _augment_features(units)
    all_lps = [v for u in humans if (v := finite_log_ppl(u["meta_c"])) is not None]
    sigma_all = statistics.pstdev(all_lps) or 1.0

    print("== 风格路 LR 重拟合（train-only；写回 constants.py 的 STYLE_STATS / STYLE_BIAS）==\n")
    print("选型网格：特征集 × PPL μ 规则（折内按同一规则重导出 μ；先看未见作者误伤，")
    print("在 ≤ v3.1 基线（9%）的组合里取 LOGO-生成器召回最优）\n")
    best: tuple[str, str, float, float, float] | None = None  # (variant, rule, max_fpr, rec_mean, rec_min)
    for vname, names in STYLE_FEATURE_VARIANTS.items():
        for rule in ("median", "75分位"):
            # LOGO-作者：折内重拟合 z+LR+μ/σ，看未见作者误伤稳定性
            fprs = []
            for author in sorted({u["author"] for u in humans}):
                fit_humans = [u for u in humans if u["author"] != author]
                held = [u for u in humans if u["author"] == author]
                fold_lps = [v for u in fit_humans if (v := finite_log_ppl(u["meta_c"])) is not None]
                fold_mu = _mu_rule(fold_lps, rule)
                fold_sigma = statistics.pstdev(fold_lps) or 1.0
                m, s, w, b = fit_style_lr(fit_humans, ais, names)
                fused_scores = [
                    fused(u, ALPHA, _c_with_style(u, names, m, s, w, b, fold_mu, fold_sigma)) for u in held
                ]
                fprs.append(sum(1 for v in fused_scores if v > HOT) / len(fused_scores))
            # LOGO-生成器：折内 LR 不见留出生成器（μ/σ 用全 train 候选规则值）
            mu_all = _mu_rule(all_lps, rule)
            recalls = []
            for gen in sorted({generator_of(u["author"]) for u in ais}):
                fit_ais = [u for u in ais if generator_of(u["author"]) != gen]
                held = [u for u in ais if generator_of(u["author"]) == gen]
                m, s, w, b = fit_style_lr(humans, fit_ais, names)
                pos = [
                    fused(u, ALPHA, _c_with_style(u, names, m, s, w, b, mu_all, sigma_all)) for u in held
                ]
                recalls.append(sum(1 for v in pos if v > HOT) / len(pos))
            max_fpr, rec_mean, rec_min = max(fprs), statistics.fmean(recalls), min(recalls)
            print(
                f"{vname:14s} μ={rule:5s} | LOGO-作者误伤 mean {statistics.fmean(fprs):5.1%} "
                f"max {max_fpr:4.0%} | LOGO-生成器召回 mean {rec_mean:5.1%} min {rec_min:4.0%}"
            )
            if max_fpr <= 0.10 and (best is None or (rec_mean, rec_min) > (best[3], best[4])):
                best = (vname, rule, max_fpr, rec_mean, rec_min)

    if best is None:
        print("\n无组合满足未见作者误伤 ≤10%，维持现常量不动。")
        return 1
    vname, rule, max_fpr, rec_mean, rec_min = best
    print(f"\n选型：{vname} + μ={rule}（误伤 max {max_fpr:.0%}，召回 mean {rec_mean:.1%} min {rec_min:.0%}）")
    names = STYLE_FEATURE_VARIANTS[vname]
    mean, std, weights, bias = fit_style_lr(humans, ais, names)
    print(f"\n== 全 train 拟合（{vname}；黏贴进 constants.py）==")
    print("STYLE_STATS: dict[str, dict[str, float]] = {")
    for f, wi in sorted(zip(names, weights, strict=True), key=lambda t: -abs(t[1])):
        print(
            f'    "{f}": {{"mean": {mean[f]:.2f}, "std": {std[f]:.2f}, '
            f'"weight": {abs(wi):.3f}, "direction": {1 if wi >= 0 else -1}}},'
        )
    print("}")
    print(f"STYLE_BIAS = {bias:.3f}")
    print(f"PPL_LOG_MEAN = {_mu_rule(all_lps, rule):.2f}   # μ 规则：{rule}")
    return 0


def main() -> int:
    units = load_train()
    if not units:
        return 1
    if "--fit-style" in sys.argv:
        return cmd_fit_style(units)
    humans = [u for u in units if u["class"] == "human"]
    ais = [u for u in units if u["class"] == "ai"]
    polished = [u for u in units if u["class"] == "polished"]
    mu_all, sigma_all, _ = fit_constants(humans)
    mu75 = sorted(v for u in humans if (v := finite_log_ppl(u["meta_c"])) is not None)
    mu75 = mu75[int(len(mu75) * 0.75)]  # 75 分位：把「顺」的门槛抬高
    print(f"train 单元：human {len(humans)} / ai {len(ais)} / polished {len(polished)}")
    print(f"全 train 拟合：μ(median)={mu_all:.2f} / μ(75分位)={mu75:.2f} σ={sigma_all:.2f}"
          f"（当前部署常量：μ={PPL_LOG_MEAN} / σ={PPL_LOG_STD}）\n")

    # ── 分析 1：LOGO-生成器（常量不变——拟合只依赖人写）──
    print("== LOGO-生成器：留出一个 AI 生成器（train）==")
    print(f"{'留出':10s} {'n':>3s} {'a路AUC':>7s} {'b路AUC':>7s} {'c路AUC':>7s} {'融合AUC':>7s} {'>HOT':>6s}")
    logo_gen = []
    for gen in sorted({generator_of(u["author"]) for u in ais}):
        held = [u for u in ais if generator_of(u["author"]) == gen]
        # 每路 AUC 必须与同路的负类比较；融合 AUC 用融合负类。此前统一用
        # c 路负类，a/b 列与融合列都是跨路比较，数值不可解释。
        neg = {lane: [u["scores"][lane] for u in humans] for lane in ("a", "b", "c")}
        neg["f"] = [fused(u, ALPHA) for u in humans]
        row = {
            "gen": gen,
            "auc_a": auc([u["scores"]["a"] for u in held], neg["a"]),
            "auc_b": auc([u["scores"]["b"] for u in held], neg["b"]),
            "auc_c": auc([u["scores"]["c"] for u in held], neg["c"]),
            "auc_f": auc([fused(u, ALPHA) for u in held], neg["f"]),
        }
        caught = sum(1 for u in held if fused(u, ALPHA) > HOT)
        logo_gen.append(row)
        print(
            f"{gen:10s} {len(held):3d} {row['auc_a']:7.3f} {row['auc_b']:7.3f} "
            f"{row['auc_c']:7.3f} {row['auc_f']:7.3f} {caught:3d}/{len(held)}"
        )
    for key in ("auc_a", "auc_b", "auc_c", "auc_f"):
        vals = [r[key] for r in logo_gen if r[key] == r[key]]
        print(f"  {key} 跨生成器: mean={statistics.fmean(vals):.3f} min={min(vals):.3f}")

    # ── 分析 2：LOGO-作者（C 路常量在其余作者上重拟合）──
    print("\n== LOGO-作者：留出一位人写作者，C 路常量重拟合（train）==")
    print(f"{'留出':22s} {'n':>3s} {'μ':>5s} {'σ':>5s} {'c路AUC':>7s} {'融合AUC':>7s} {'该作者误伤':>8s}")
    logo_author = []
    for author in sorted({u["author"] for u in humans}):
        fit_set = [u for u in humans if u["author"] != author]
        held = [u for u in humans if u["author"] == author]
        mu, sigma, stats = fit_constants(fit_set)
        pos = [lane_c(u["meta_c"], mu, sigma, stats) for u in ais]
        pos = [p for p in pos if p is not None]
        # 人写侧重算后的 C 分（留出作者 + 其余人写都重算，AUC 负类用全体一致常量）
        c_all_humans = [lane_c(u["meta_c"], mu, sigma, stats) for u in humans]
        neg = [c for c in c_all_humans if c is not None]
        held_fused = []
        for u in held:
            c_new = lane_c(u["meta_c"], mu, sigma, stats)
            held_fused.append(fused(u, ALPHA, c_new))
        ai_fused = []
        for u in ais:
            c_new = lane_c(u["meta_c"], mu, sigma, stats)
            if c_new is not None:
                ai_fused.append(fused(u, ALPHA, c_new))
        row = {
            "author": author,
            "auc_c": auc(pos, neg),
            "auc_f": auc(ai_fused, [f for f in held_fused]),
            "fpr": sum(1 for v in held_fused if v > HOT) / len(held_fused),
        }
        logo_author.append(row)
        print(
            f"{author:22s} {len(held):3d} {mu:5.2f} {sigma:5.2f} {row['auc_c']:7.3f} "
            f"{row['auc_f']:7.3f} {row['fpr']:7.0%}"
        )
    print(
        f"  未见作者误伤率: mean={statistics.fmean(r['fpr'] for r in logo_author):.1%} "
        f"max={max(r['fpr'] for r in logo_author):.1%}"
    )

    # ── 分析 3：train-only 决策变体（α 组合 × PPL 校准点）──
    # 决策表固定使用部署的 STYLE_STATS（v10c：base8 + 单侧证据），不刷新归一化；
    # 否则表内数字与 evaluate.py 实际部署分数不是同一口径。
    # 「部署常量」行必须与 evaluate.py --split=train 完全一致（μ=4.20/σ=0.68）。
    print("\n== 决策变体（train 全量；部署 STYLE_STATS；a 路去留与 μ 选择只看这里）==")
    variants = [
        ("部署常量（v10c）, α=0/0.2/0.8", ALPHA, PPL_LOG_MEAN, PPL_LOG_STD),
        ("α=0.2/0.4/0.4（v2 旧配）, μ=median", {"a": 0.2, "b": 0.4, "c": 0.4}, mu_all, sigma_all),
        ("α=0.2/0.4/0.4（v2 旧配）, μ=75分位", {"a": 0.2, "b": 0.4, "c": 0.4}, mu75, sigma_all),
        ("α=0/0.5/0.5（v7 旧配）, μ=median", {"a": 0.0, "b": 0.5, "c": 0.5}, mu_all, sigma_all),
        ("α=0/0.5/0.5（v7 旧配）, μ=75分位", {"a": 0.0, "b": 0.5, "c": 0.5}, mu75, sigma_all),
        ("α=0/0.2/0.8（v10 候选）, μ=median", ALPHA, mu_all, sigma_all),
        ("α=0/0.2/0.8（v10 候选）, μ=75分位", ALPHA, mu75, sigma_all),
    ]
    print(f"{'变体':38s} {'trainAUC':>8s} {'误伤':>6s} {'召回':>6s}")
    for name, alpha, mu, sigma in variants:
        neg = [fused(u, alpha, lane_c(u["meta_c"], mu, sigma, STYLE_STATS)) for u in humans]
        pos = [fused(u, alpha, lane_c(u["meta_c"], mu, sigma, STYLE_STATS)) for u in ais]
        print(
            f"{name:38s} {auc(pos, neg):8.3f} {sum(1 for v in neg if v > HOT) / len(neg):6.1%} "
            f"{sum(1 for v in pos if v > HOT) / len(pos):6.1%}"
        )

    # ── 分析 4：train ROC 阈值参考（产品 HOT 是否合理）──
    print("\n== train ROC 阈值参考（融合，现状 α/μ）==")
    pairs = sorted(
        [(fused(u, ALPHA), u["class"]) for u in units if u["class"] in ("human", "ai")],
        reverse=True,
    )
    n_pos, n_neg = sum(1 for _, c in pairs if c == "ai"), sum(1 for _, c in pairs if c == "human")
    tp = fp = 0
    best = (0.0, -1.0)  # (threshold, Youden J)
    for i, (score, cls) in enumerate(pairs):
        if cls == "ai":
            tp += 1
        else:
            fp += 1
        j = tp / n_pos - fp / n_neg
        if j > best[1]:
            best = (score, j)
        if i + 1 < len(pairs) and pairs[i + 1][0] < score:  # 阈值候选：分数跳变处
            pass
    print(f"  Youden 最优阈值 ≈ {best[0]:.3f}（J={best[1]:.3f}），产品现用 HOT={HOT}")

    # ── 分析 5：(b,c) 权重网格——train 上「误伤 ≤8% 最大化召回」敏感性扫描 ──
    # 固定部署 μ/σ（v10c 4.20/0.68）；α 的最终选择仍以段落级双优为准，
    # 这里的「最大化召回」只作参考，不能单独作为决策依据。
    print(f"\n== 选型网格（a=0；train 全量；部署 STYLE_STATS/μ/σ；阈值 HOT={HOT}）==")
    print(f"{'b:c':>9s} {'μ':>6s} {'AUC':>6s} {'误伤':>6s} {'召回':>6s}")
    grid = []
    for w_b in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        alpha_try = {"a": 0.0, "b": w_b, "c": round(1 - w_b, 1)}
        for mu_name, mu_try in (("部署", PPL_LOG_MEAN), ("median", mu_all), ("75分位", mu75)):
            neg = [fused(u, alpha_try, lane_c(u["meta_c"], mu_try, PPL_LOG_STD, STYLE_STATS)) for u in humans]
            pos = [fused(u, alpha_try, lane_c(u["meta_c"], mu_try, PPL_LOG_STD, STYLE_STATS)) for u in ais]
            fpr = sum(1 for v in neg if v > HOT) / len(neg)
            rec = sum(1 for v in pos if v > HOT) / len(pos)
            grid.append((fpr, rec, auc(pos, neg), alpha_try, mu_name))
            print(f"{w_b:.1f}:{1-w_b:.1f} {mu_name:>6s} {auc(pos, neg):6.3f} {fpr:6.1%} {rec:6.1%}")
    feasible = [g for g in grid if g[0] <= 0.08]
    if feasible:
        # v10 起以 AUC 为主序（段落级双优另见 analyze 的 α 网格）；召回只作次序。
        fpr, rec, a_score, alpha_best, mu_best = max(feasible, key=lambda g: (g[2], g[1]))
        print(f"\n  参考选型（train-only，误伤≤8% 下最大 AUC）：α={alpha_best} μ={mu_best} → AUC {a_score:.3f} 误伤 {fpr:.1%} 召回 {rec:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
