"""Mock 检测器：确定性规则打分，供单测/CI 使用（CI 不下载任何模型）。

规则：窗口文本含 AI 高频连接词 → 0.9，否则 0.1。附带极简统计 meta，
让解释层的「触发理由」路径也能被测到。通过环境变量
AIGC_SELFCHECK_MOCK=1 全局启用（注册表内拦截）。
"""

from __future__ import annotations

from .base import Detector, DetectorResult


class MockDetector(Detector):
    def __init__(self, path_key: str, device: str = "auto"):
        super().__init__(device)
        self.path_key = path_key
        self.model_id = f"mock/{path_key}"

    def detect(self, text: str, context: str = "") -> DetectorResult:
        from ..features import AI_CONNECTIVES

        is_ai_style = any(word in text for word in AI_CONNECTIVES)
        score = 0.9 if is_ai_style else 0.1
        meta = {"model": self.model_id, "mock": True}
        if self.path_key == "c":
            meta.update(
                {
                    "p_ppl": 0.9 if is_ai_style else 0.2,
                    "p_style": 0.85 if is_ai_style else 0.15,
                    "ppl": 12.0 if is_ai_style else 55.0,
                    "z_values": {"connective_density": 2.1} if is_ai_style else {},
                    "features": {"connective_density": 12.4 if is_ai_style else 0.0},
                }
            )
        return DetectorResult(score=score, meta=meta)
