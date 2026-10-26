#!/usr/bin/env python3
"""按「改哪一段最划算」输出修改计划，并标出不该为了降分而改的段落。"""

from __future__ import annotations

import argparse
import json

from _cli import ensure_local, fail, run, setup_stdio


def _short(text: str, limit: int = 180) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def main(argv: list[str] | None = None) -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="生成修改优先级计划（不自动改写）")
    parser.add_argument("file", help="本地 .docx/.pdf/.txt/.md 文件")
    parser.add_argument("--enhanced", action="store_true", help="实验性增强召回")
    parser.add_argument(
        "--profile",
        choices=["lite", "balanced", "max"],
        default=None,
        help="运行档位：lite=纯 CPU/0.5B，balanced=b+c/2B，max=a+b+c/2B",
    )
    parser.add_argument("--detectors", default=None, help="检测路组合；不填用档位推荐")
    parser.add_argument("--device", default=None, help="auto|cpu|cuda|mps；不填用档位推荐")
    parser.add_argument(
        "--llm-advice",
        dest="llm_advice",
        action="store_true",
        default=None,
        help="用本地基座模型生成改写示例（实验性）",
    )
    parser.add_argument(
        "--no-llm-advice",
        dest="llm_advice",
        action="store_false",
        help="关闭档位默认的模型建议",
    )
    parser.add_argument("--raw", action="store_true", help="输出原始 JSON")
    args = parser.parse_args(argv)

    ensure_local(args.file, "待检测文件")
    cmd = [args.file, "--json"]
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.detectors:
        cmd += ["--detectors", args.detectors]
    if args.device:
        cmd += ["--device", args.device]
    if args.llm_advice is True:
        cmd.append("--llm-advice")
    elif args.llm_advice is False:
        cmd.append("--no-llm-advice")
    if args.enhanced:
        cmd.append("--enhanced")
    result = run(cmd)
    if result.returncode != 0:
        return fail(result)

    payload = json.loads(result.stdout)
    if args.raw:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    doc = payload.get("doc") or {}
    print(f"当前：{doc.get('score', '—')} 分 / {doc.get('band', '—')} 带")
    plan = payload.get("plan") or []
    if not plan:
        print("没有单段主导风险。建议按高风险清单逐句复核，不要为了降分大改全文。")
    else:
        print("\n修改计划（按单段影响排序；不是分数承诺）：")
        for i, item in enumerate(plan, 1):
            impact = item.get("impact", 0)
            impact_text = f"预计影响 -{impact} 分" if impact > 0 else "单窗风险高，分数影响小"
            print(
                f"\n{i}. 块 {item.get('block')} · {item.get('section') or '未识别章节'}"
                f" · {impact_text} · 最高风险 {item.get('max_risk', 0):.2f}"
            )
            print(
                f"   若单独处理这一段：综合分 {item.get('score_without', '—')}"
                f" · 区间带 {item.get('band_without', '—')}"
            )
            note = item.get("review_note")
            if note:
                print(f"   先别改：{note.get('label')} —— {note.get('why')}")
            reasons = item.get("reason_items") or []
            if reasons:
                reason = reasons[0]
                print(f"   主要原因：{reason.get('label', '风格特征')} —— {reason.get('why', '')}")
                if reason.get("suggestion"):
                    print(f"   建议：{reason['suggestion']}")
            advice = item.get("llm_advice") or {}
            if advice.get("text"):
                print(f"   模型改写示例（实验性，需人工确认）：{_short(advice['text'], 200)}")
            print(f"   原文：{_short(item.get('text'))}")

    llm_info = payload.get("llm_advice") or {}
    if llm_info.get("enabled") is False:
        print(f"\n模型建议：已跳过（{llm_info.get('reason', '不可用')}）")

    print("\n提醒：先改有实质问题的段落；不要为了降分堆砌口语词、删实验细节或故意写病句。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
