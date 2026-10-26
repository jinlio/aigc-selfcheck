"""M0 冒烟脚本：三路模型加载 + 同一段 AI 文本打分 + 极性确认 + 延迟基线。

退出标准（计划 M0）：三路各产出分数；A/B 路对已知 AI 文本的 AI 概率 > 0.5
（config id2label + 实测双确认）；记录 MPS/CPU 延迟。

用法：uv run python scripts/smoke.py [--device auto|cpu|mps]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, "src")

from aigc_selfcheck.detectors import build_detectors  # noqa: E402

# 已知 AI 生成文本（用于极性校验，写作风格刻意工整）
AI_SAMPLE = (
    "随着人工智能技术的快速发展，文本生成模型在各类应用场景中展现出显著的能力。"
    "首先，大规模预训练模型使得生成文本的流畅度得到了极大的提升；"
    "其次，指令微调进一步增强了模型对用户意图的理解能力。"
    "与此同时，生成内容的质量评估也成为了一个重要的研究方向。"
    "总而言之，文本生成技术正在深刻地改变内容生产的范式，"
    "并在教育、媒体与办公自动化等领域产生广泛而深远的影响。"
)

# 已知人写风格文本（口语化、句长参差，用于对照）
HUMAN_SAMPLE = (
    "折腾了一晚上环境，总算跑通了。说实话一开始没想到这么麻烦，官网文档写得含糊，"
    "论坛里倒是有个帖子提了一句关键的坑——CUDA 版本要对上。我试了下还真行。"
    "明天把剩下的测试补完吧，太晚了，先睡了。"
)


def smoke_device(device: str) -> int:
    import torch

    resolved = device
    if device == "auto":
        resolved = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"[device] 请求 {device} → 实际 {resolved}（mps 可用: {torch.backends.mps.is_available()}）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--skip-a", action="store_true", help="跳过 A 路（纯 CPU 场景）")
    args = parser.parse_args()

    rc = smoke_device(args.device)
    spec = "b,c" if args.skip_a else "a,b,c"
    print(f"[load] 构建检测路 {spec} …（首次运行会从镜像拉权重，约 2GB）")
    detectors = build_detectors(spec, args.device)

    failures = 0
    for detector in detectors:
        name = f"{detector.path_key.upper()} 路 · {detector.model_id}"
        started = time.time()
        try:
            result_ai = detector.detect(AI_SAMPLE, context="")
            ai_seconds = time.time() - started
            started = time.time()
            result_human = detector.detect(HUMAN_SAMPLE, context="")
            human_seconds = time.time() - started
        except Exception as exc:  # noqa: BLE001 —— 冒烟脚本要求完整报错现场
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(
            f"[ok] {name}\n"
            f"     AI 样本  score={result_ai.score:.3f}  {ai_seconds*1000:.0f}ms\n"
            f"     人写样本 score={result_human.score:.3f}  {human_seconds*1000:.0f}ms"
        )
        if detector.path_key in ("a", "b"):
            if result_ai.score <= 0.5:
                print(
                    f"     [极性警告] AI 样本得分 ≤0.5：AI 类标签可能映射反了"
                    f"（meta={result_ai.meta}），请人工核对 config.id2label。"
                )
                failures += 1
            elif result_human.score >= 0.5:
                print("     [极性警告] 人写样本得分 ≥0.5，与 AI 样本区分不开，需人工检查。")
        if detector.path_key == "c":
            meta = result_ai.meta
            print(
                f"     C 路明细: p_ppl={meta['p_ppl']:.3f} p_style={meta['p_style']:.3f} "
                f"log_ppl={meta['log_ppl']:.2f}"
            )
    for detector in detectors:
        detector.close()
    if failures:
        print(f"\n冒烟未通过：{failures} 项失败")
        return 1
    print("\n冒烟通过：三路均可打分，极性确认无异常。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
