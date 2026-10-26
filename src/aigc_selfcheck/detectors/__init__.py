"""检测器注册表：按路径键构建检测路（PLAN §4.1 可插拔）。

spec 形如 "a,b,c"；接入新检测器时在此登记并跑校准集全量回归
（ALGORITHM §4 的黄金测试纪律）。
"""

from __future__ import annotations

from .base import Detector, DetectorResult


def build_detector(path_key: str, device: str = "auto", c_model: str | None = None) -> Detector:
    import os

    # 测试/CI 出入口：环境变量全局切换 mock，不下载任何模型
    if os.environ.get("AIGC_SELFCHECK_MOCK") == "1":
        from .mock import MockDetector

        return MockDetector(path_key)
    if path_key == "a":
        from .zhv2 import Zhv2Detector

        return Zhv2Detector(device)
    if path_key == "b":
        from .cmj_onnx import CmjOnnxDetector

        return CmjOnnxDetector(device)
    if path_key == "c":
        from .stats_lm import StatsDetector

        return StatsDetector(device, model_id=c_model)
    raise ValueError(f"未知检测路：{path_key}（可用：a=zhv2, b=cmj-onnx, c=stats）")


def build_detectors(spec: str, device: str = "auto", c_model: str | None = None) -> list[Detector]:
    keys = [k.strip().lower() for k in spec.split(",") if k.strip()]
    if not keys:
        raise ValueError("检测路 spec 不能为空，如 --detectors a,b,c")
    return [build_detector(k, device, c_model) for k in keys]


__all__ = ["Detector", "DetectorResult", "build_detector", "build_detectors"]
