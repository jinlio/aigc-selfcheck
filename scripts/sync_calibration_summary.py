"""从探针产物重建 `src/aigc_selfcheck/data/calibration_summary.json`。

报告头引用的数字必须来自产物，不能手抄——历史上就出现过报告写 v8 的
94.9%/2.6%/11.1%、而产物已经是 93.2%/1.8%/17.8% 的漂移。

用法：
    PYTHONPATH=src python scripts/sync_calibration_summary.py
    PYTHONPATH=src python scripts/sync_calibration_summary.py --check   # CI 用，只比对不写
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_selfcheck import __version__  # noqa: E402
from aigc_selfcheck.constants import CALIBRATION_VERSION  # noqa: E402

SUMMARY_PATH = ROOT / "src" / "aigc_selfcheck" / "data" / "calibration_summary.json"
ACADEMIC_HUMAN = ROOT / "calibration" / "academic" / "report.json"
ACADEMIC_AI = ROOT / "calibration" / "ai_academic_lite" / "report.json"

# 通用块来自 v8 train 全量评估（calibration/README.md「部署常量下的 v8 train
# 全量」一节）。train 内结果，不是独立 test——没有产物文件可读，只能记录来源。
GENERAL = {
    "label": "通用校准集（v8 train）",
    "n_human_windows": 268,
    "n_ai_windows": 297,
    "auc": 0.963,
    "human_fp": 0.0,
    "recall": 0.744,
    "note": "v8 train 内结果，非独立 test；样本量小，不能外推。来源 calibration/README.md。",
}


def build() -> dict:
    for path in (ACADEMIC_HUMAN, ACADEMIC_AI):
        if not path.exists():
            raise SystemExit(f"探针产物缺失，先重建探针：{path}")

    human = json.loads(ACADEMIC_HUMAN.read_text(encoding="utf-8"))
    ai = json.loads(ACADEMIC_AI.read_text(encoding="utf-8"))
    h, a = human["overall"], ai["overall"]

    return {
        "_comment": (
            "本文件由 scripts/sync_calibration_summary.py 从探针产物生成，勿手改。"
            "报告头与 README 引用这里，避免出现「报告写一套、实测另一套」。"
        ),
        "generated_at": date.today().isoformat(),
        "tool_version": __version__,
        "deployed_constants": CALIBRATION_VERSION,
        "sources": {
            "academic_human": "calibration/academic/report.json",
            "academic_ai": "calibration/ai_academic_lite/report.json",
        },
        "general": GENERAL,
        "academic_default": {
            "label": "学术探针 · 默认融合",
            "n_human_windows": h["n_windows"],
            "n_ai_windows": a["n_windows"],
            "ai_window_recall": a["window_recall"],
            "human_window_fp": h["window_positive_rate"],
            "human_doc_high": h["doc_band_high"],
            "ai_doc_high": a["doc_band_high"],
            "ai_window_p50": a["window_p50"],
            "ai_window_p90": a["window_p90"],
            "human_window_p50": h["window_p50"],
            "human_window_p90": h["window_p90"],
            "note": (
                f"人写={human['label']}（{h['n_paragraphs']} 段），"
                f"AI={ai['label']}（{a['n_paragraphs']} 段）。"
                "两组均为 report-only 探针，不进训练。"
            ),
        },
        "notes": [
            "学术探针的 AI 占比不代表真实先验，只用于横向比较召回/误伤。",
            "这些数字不预测知网、朱雀等官方检测结果；官方判定以平台为准。",
            "b 路增强与 mtld 裁决会把部分窗口抬到精确的 0.800 下限；论文报告里被抬过的窗口已标 floor_rule。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="同步校准摘要")
    parser.add_argument("--check", action="store_true", help="只比对，不写入（CI 用）")
    args = parser.parse_args()

    fresh = build()
    rendered = json.dumps(fresh, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        current = SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.exists() else ""
        if current != rendered:
            print("校准摘要与探针产物不一致，运行 scripts/sync_calibration_summary.py 重新生成。")
            return 1
        print("校准摘要与探针产物一致。")
        return 0

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(rendered, encoding="utf-8")
    print(f"已写入 {SUMMARY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
