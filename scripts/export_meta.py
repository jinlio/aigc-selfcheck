"""把 train-only 元分类器固化成产品可加载的 JSON 工件（实验性增强召回）。

默认产品仍是固定融合（α=0/0.4/0.6，HOT=0.75）。本工件只服务 `--enhanced`
可选模式：在固定融合已经越线的窗口之外，再标记「融合分处于灰度区且元分类器
高分」的窗口。它**没有**通过默认模式的门槛（LOGO-生成器 FPR 均值 1.7%、
最差 11.1%，召回均值 85.2%，低于预注册的 90%），因此默认关闭。

工件包含：
  - 24 个特征的缺失值中位数、标准化均值/标准差、LR 系数与截距；
  - 元分类器训练时使用的 c 路常量（μ/σ/风格 mean/std/权重），
    保证产品侧算出的 c 特征与训练时一致；
  - 人写 train 概率的 99.5% 分位阈值（默认增强阈值）与 0.55 灰度门；
  - 训练切分指纹，便于追溯。

用法：
    PYTHONPATH=src D:/anaconda/envs/learning/python.exe scripts/export_meta.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import cv_evaluate as cv  # noqa: E402
import fit_meta as fm  # noqa: E402
import numpy as np  # noqa: E402

from aigc_selfcheck.constants import HEATMAP_HIGH  # noqa: E402

OUT = ROOT / "src" / "aigc_selfcheck" / "data" / "meta_v1.json"
GATE = 0.55


def _fingerprints() -> dict[str, str]:
    cal = ROOT / "calibration" / "out" / "calibration.jsonl"
    split = ROOT / "calibration" / "out" / "splits.md5"
    return {
        "calibration_sha256": hashlib.sha256(cal.read_bytes()).hexdigest(),
        "split_fingerprint": split.read_text(encoding="utf-8").strip(),
    }


def main() -> int:
    units = cv.load_train()
    if not units:
        print("没有 train 单元，先运行 build_calibration.py", file=sys.stderr)
        return 1
    cv._augment_features(units)
    humans = [u for u in units if u["class"] == "human"]
    ais = [u for u in units if u["class"] == "ai"]
    if not humans or not ais:
        print("train 缺少人写或 AI 单元", file=sys.stderr)
        return 1

    mu, sigma, style_stats = cv.fit_constants(humans)
    c_values, p_styles, p_ppls = fm._fold_scores(units, humans)
    x = fm._matrix(units, c_values, p_styles, p_ppls)
    y = np.array([1.0 if u["class"] == "ai" else 0.0 for u in units])

    model = fm._model()
    model.fit(x, y)
    human_p = model.predict_proba(x[y == 0])[:, 1]
    thresholds = {str(q): float(np.quantile(human_p, q)) for q in (0.95, 0.99, 0.995)}

    imputer = model.named_steps["simpleimputer"]
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]

    artifact = {
        "schema": "aigc-selfcheck/meta@v1",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": "experimental enhanced-recall gate; not the default detector",
        "features": fm.FEATURES,
        "imputer_median": [float(v) for v in imputer.statistics_],
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "thresholds": thresholds,
        "threshold": thresholds["0.995"],
        "gate": GATE,
        "hot": HEATMAP_HIGH,
        "meta_c": {
            "mu": float(mu),
            "sigma": float(sigma),
            "style_bias": float(cv.STYLE_BIAS),
            "ppl_weight": float(cv.PPL_WEIGHT_IN_C),
            "style_stats": {
                name: {
                    "mean": float(st["mean"]),
                    "std": float(st["std"]),
                    "weight": float(st["weight"]),
                    "direction": float(st["direction"]),
                }
                for name, st in style_stats.items()
            },
        },
        "train": {
            "n_human": len(humans),
            "n_ai": len(ais),
            "n_authors": len({u["author"] for u in humans}),
            "n_generators": len({cv.generator_of(u["author"]) for u in ais}),
            **_fingerprints(),
        },
        "validation": {
            "note": "LOGO estimates come from scripts/fit_meta.py (fold-refit c constants), "
            "not from this final all-train fit.",
            "logo_generator_recall_mean": 0.852,
            "logo_generator_recall_min": 0.531,
            "logo_author_fpr_mean": 0.017,
            "logo_author_fpr_max": 0.111,
            "academic_external": {
                "human_window_fp": 0.035,
                "human_paragraph_fp": 0.0,
                "ai_window_recall": 0.986,
                "ai_paragraph_recall": 1.0,
            },
        },
        "caveats": [
            "默认关闭：未通过预注册门槛（LOGO 召回均值 ≥90%）。",
            "增强标记会增加人写误报；学术探针上人写窗口误报 3.5%，默认固定融合为 2.6%。",
            "只适用于 a,b,c 三路齐全的窗口；缺少任一路时不要启用。",
            "学术探针不参与训练；该工件只用通用 train 拟合。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"写入 {OUT}")
    print(
        f"train human {len(humans)} / ai {len(ais)}；阈值 q95={thresholds['0.95']:.4f} "
        f"q99={thresholds['0.99']:.4f} q99.5={thresholds['0.995']:.4f}；gate={GATE}"
    )
    print("提醒：这是实验性增强召回工件，默认产品路径不加载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
