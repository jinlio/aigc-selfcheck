"""aigc-selfcheck —— 纯本地的中文文本 AIGC 风险自检台。"""

import os

# HF 端点默认值必须在 transformers/huggingface_hub 首次 import 前生效
# （huggingface_hub 的常量在 import 时求值）。模型文件本地有缓存时只是
# 省去每次运行的连网重试；用户显式设置的 HF_ENDPOINT 优先。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

__version__ = "0.2.0"
