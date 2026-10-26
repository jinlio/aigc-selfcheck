"""首次运行下载三个模型权重（默认走 hf-mirror 镜像，可用 HF_ENDPOINT 覆盖）。

下载量约 5.5GB：zhv2（pytorch/safetensors）、cmj（pytorch + onnx）、
Qwen3.5-2B-Base（v6 起的 c 路参考模型，bf16 约 4.4GB）。
纯 CPU 回退另需 Qwen2.5-0.5B（约 1GB）：加 --cpu-fallback。
注意：必须禁用 Xet 传输（HF_HUB_DISABLE_XET=1）——新版 huggingface_hub 的 Xet
会绕过镜像端点直连 xethub CDN，在被墙网络下超时。
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download  # noqa: E402

REPOS = [
    "yuchuantian/AIGC_detector_zhv2",
    "mokawa3018/cmj-chinese-aigc-text-detector",
    "Qwen/Qwen3.5-2B-Base",
]
CPU_FALLBACK = "Qwen/Qwen2.5-0.5B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="额外下载 Qwen2.5-0.5B（纯 CPU 回退，约 1GB）",
    )
    args = parser.parse_args()
    repos = [*REPOS, CPU_FALLBACK] if args.cpu_fallback else REPOS
    for repo in repos:
        print(f"下载 {repo} …", flush=True)
        snapshot_download(repo)
        print(f"完成 {repo}", flush=True)
    print("全部模型就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
