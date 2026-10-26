#!/usr/bin/env python3
"""把自检 JSON 转成普通用户能读的「为什么 + 怎么改」摘要。"""

from __future__ import annotations

import argparse
import json

from _cli import ensure_local, fail, run, setup_stdio


def _short(text: str, limit: int = 160) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _print_reasons(items: list[dict], indent: str = "   ") -> None:
    for item in items[:3]:
        label = item.get("label") or "风格特征"
        why = item.get("why") or ""
        print(f"{indent}为什么：{label} —— {why}")
        if item.get("suggestion"):
            print(f"{indent}怎么改：{item['suggestion']}")


def main(argv: list[str] | None = None) -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="解释 AIGC 自检结果（不自动改写）")
    parser.add_argument("file", help="本地 .docx/.pdf/.txt/.md 文件")
    parser.add_argument("--top", type=int, default=8, help="展示的高风险句子数，默认 8")
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
    guidance = payload.get("guidance") or {}
    evidence = guidance.get("evidence") or {}
    print(f"综合风险分：{doc.get('score', '—')} / 100（区间带：{doc.get('band', '—')}）")
    print(f"依据：{evidence.get('note', '')}")
    if guidance.get("headline"):
        print(guidance["headline"])
    if payload.get("provisional"):
        print(f"注意：{payload['provisional']}")

    sections = sorted(payload.get("sections") or [], key=lambda s: s.get("score", 0), reverse=True)
    if sections:
        print("\n章节风险（从高到低）：")
        for section in sections[:6]:
            hint = f"；{section['hint']}" if section.get("hint") else ""
            print(
                f"- {section.get('title')}：{section.get('score')} 分 / {section.get('band')} 带"
                f"（{section.get('n_windows', 0)} 窗，median {section.get('median', 0):.2f}，"
                f"P90 {section.get('p90', 0):.2f}）{hint}"
            )

    top = payload.get("top_sentences") or []
    print(f"\n最需要看的 {min(args.top, len(top))} 句：")
    for i, item in enumerate(top[: args.top], 1):
        print(f"\n{i}. [风险 {item.get('risk', 0):.2f}] {_short(item.get('text'))}")
        if item.get("section"):
            print(f"   章节：{item['section']}")
        note = item.get("review_note")
        if note:
            print(f"   先别改：{note.get('label')} —— {note.get('why')}")
        _print_reasons(item.get("reason_items") or [])
        fixes = []
        for reason in item.get("reason_items") or []:
            suggestion = reason.get("suggestion")
            if suggestion and suggestion not in fixes:
                fixes.append(suggestion)
        if fixes:
            print(f"   修改建议：{'；'.join(fixes[:3])}")
        advice = item.get("llm_advice") or {}
        if advice.get("text"):
            print(f"   模型改写示例（实验性，需人工确认）：{_short(advice['text'], 200)}")

    llm_info = payload.get("llm_advice") or {}
    if llm_info.get("enabled") is False:
        print(f"\n模型建议：已跳过（{llm_info.get('reason', '不可用')}）")

    print("\n提醒：这是本地启发式风险估计，不是知网/朱雀等官方检测结果；不要为了降分故意写病句。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
