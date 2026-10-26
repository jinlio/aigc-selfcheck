"""C 路基座切换：独立 PPL 常量与缓存指纹。

默认 2B 的 cache_id 必须保持 v9 指纹不变（历史缓存仍可命中）；
0.5B 回退使用自己的 μ/σ，缓存键必须与 2B 分开。
两枚指纹都并入 STYLE_STATS（风格归一化参与哈希），刷归一化会同时变。
"""

from __future__ import annotations

from aigc_selfcheck.cli import build_parser
from aigc_selfcheck.constants import PPL_CALIBRATION
from aigc_selfcheck.detectors.stats_lm import StatsDetector


def test_default_cache_id_stays_v9() -> None:
    detector = StatsDetector.__new__(StatsDetector)
    assert detector.cache_id == "Qwen/Qwen3.5-2B-Base@f40c417121c1"


def test_fallback_model_uses_own_calibration() -> None:
    detector = StatsDetector.__new__(StatsDetector)
    detector.model_id = "Qwen/Qwen2.5-0.5B"
    detector.ppl_mean, detector.ppl_std = PPL_CALIBRATION[detector.model_id]
    assert (detector.ppl_mean, detector.ppl_std) == (4.13, 0.64)
    assert detector.cache_id == "Qwen/Qwen2.5-0.5B@100f29181213"


def test_cli_accepts_c_model_override() -> None:
    args = build_parser().parse_args(["sample.txt", "--c-model=Qwen/Qwen2.5-0.5B"])
    assert args.c_model == "Qwen/Qwen2.5-0.5B"
