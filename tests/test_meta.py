"""元分类器工件与增强召回逻辑的纯函数测试（不加载检测模型）。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT / "src", SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv_evaluate as cv  # noqa: E402
import fit_meta as fm  # noqa: E402

from aigc_selfcheck import meta  # noqa: E402
from aigc_selfcheck.fuse import FusedWindow, fuse_scores  # noqa: E402


def test_artifact_schema_and_threshold() -> None:
    artifact = meta.load_artifact()
    assert artifact["schema"] == "aigc-selfcheck/meta@v1"
    assert artifact["features"] == meta.FEATURES
    for key in ("imputer_median", "scaler_mean", "scaler_scale", "coef"):
        assert len(artifact[key]) == len(meta.FEATURES)
    assert 0.0 < artifact["gate"] < 1.0
    assert 0.0 < artifact["threshold"] < 1.0
    assert artifact["thresholds"]["0.995"] == pytest.approx(artifact["threshold"])
    assert artifact["meta_c"]["mu"] > 0
    assert artifact["meta_c"]["sigma"] > 0


def test_probability_matches_manual_logistic_formula() -> None:
    artifact = meta.load_artifact()
    row = {name: 0.0 for name in meta.FEATURES}
    total = artifact["intercept"]
    for index, name in enumerate(meta.FEATURES):
        z = (row[name] - artifact["scaler_mean"][index]) / artifact["scaler_scale"][index]
        total += artifact["coef"][index] * z
    expected = 1.0 / (1.0 + math.exp(-total))
    assert meta.probability(row, artifact) == pytest.approx(expected, abs=1e-12)


def test_probability_imputes_missing_values() -> None:
    artifact = meta.load_artifact()
    row = {name: float("nan") for name in meta.FEATURES}
    total = artifact["intercept"]
    for index, _name in enumerate(meta.FEATURES):
        value = artifact["imputer_median"][index]
        z = (value - artifact["scaler_mean"][index]) / artifact["scaler_scale"][index]
        total += artifact["coef"][index] * z
    expected = 1.0 / (1.0 + math.exp(-total))
    assert meta.probability(row, artifact) == pytest.approx(expected, abs=1e-12)


def _unit() -> dict:
    text = "本文提出一种新的方法，并在实验中验证了有效性。结果表明，该方法优于基线。"
    return {
        "text": text,
        "scores": {"a": 0.02, "b": 0.61, "c": 0.74},
        "meta_c": {
            "log_ppl": 3.9,
            "features": {
                "connective_density": 2.0,
                "de_density": 40.0,
                "colloquial_density": 0.0,
                "punct_diversity": 1.0,
                "punct_density": 95.0,
                "period_comma_ratio": 0.8,
                "sentence_length_variance": 120.0,
                "char_mtld": 55.0,
                "sent_initial_rate": 0.0,
            },
        },
    }


def test_feature_row_matches_training_script() -> None:
    artifact = meta.load_artifact()
    unit = _unit()
    cv._augment_features([unit])
    c_value, p_style, p_ppl = meta.c_meta_score(
        unit["meta_c"]["features"], unit["meta_c"]["log_ppl"], artifact
    )
    ours = meta.feature_row(unit["text"], unit["scores"], unit["meta_c"], artifact)
    reference = fm._feature_row(unit, c_value, p_style, p_ppl)
    for name in meta.FEATURES:
        left, right = ours[name], reference[name]
        if isinstance(left, float) and math.isnan(left):
            assert math.isnan(right), name
        else:
            assert left == pytest.approx(right), name


def test_feature_row_handles_missing_ppl() -> None:
    artifact = meta.load_artifact()
    unit = _unit()
    unit["meta_c"]["log_ppl"] = None
    c_value, _p_style, p_ppl = meta.c_meta_score(
        unit["meta_c"]["features"], None, artifact
    )
    assert p_ppl == pytest.approx(0.5)
    row = meta.feature_row(unit["text"], unit["scores"], unit["meta_c"], artifact)
    assert math.isnan(row["log_ppl"])
    assert row["score_c"] == pytest.approx(c_value)


def test_enhanced_flag_uses_hot_and_gray_gate() -> None:
    artifact = meta.load_artifact()
    gate = artifact["gate"]
    threshold = artifact["threshold"]
    assert meta.enhanced_flag(0.80, 0.0, artifact) is True
    assert meta.enhanced_flag(gate + 0.01, threshold + 0.01, artifact) is True
    assert meta.enhanced_flag(gate - 0.01, 0.99, artifact) is False
    assert meta.enhanced_flag(gate + 0.01, threshold - 0.01, artifact) is False


def _window(unit: dict, fused: float) -> FusedWindow:
    return FusedWindow(
        window_key="0:0",
        block_index=0,
        sentence_ids=[0],
        text=unit["text"],
        scores=dict(unit["scores"]),
        p_fuse=fused,
        delta=0.1,
        metas={"c": dict(unit["meta_c"])},
    )


def test_apply_enhanced_sets_probability_and_flag() -> None:
    artifact = meta.load_artifact()
    unit = _unit()
    p_fuse, _delta = fuse_scores(unit["scores"])
    window = _window(unit, p_fuse)
    info = meta.apply_enhanced([window], artifact)
    assert info["enabled"] is True
    assert info["n_flagged"] == int(window.enhanced)
    assert window.meta_prob is not None
    assert 0.0 <= window.meta_prob <= 1.0
    expected = p_fuse > artifact["hot"] or (
        p_fuse > artifact["gate"] and window.meta_prob > artifact["threshold"]
    )
    assert window.enhanced is expected


def test_apply_enhanced_requires_all_three_lanes() -> None:
    artifact = meta.load_artifact()
    unit = _unit()
    unit["scores"].pop("a")
    with pytest.raises(ValueError, match="a,b,c"):
        meta.apply_enhanced([_window(unit, 0.8)], artifact)


def test_apply_enhanced_skips_window_without_c_features() -> None:
    artifact = meta.load_artifact()
    unit = _unit()
    unit["meta_c"]["features"] = {}
    window = _window(unit, 0.8)
    info = meta.apply_enhanced([window], artifact)
    assert window.meta_prob is None
    assert window.enhanced is False
    assert info["n_skipped"] == 1
