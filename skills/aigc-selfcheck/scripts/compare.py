#!/usr/bin/env python3
"""改稿前后对比：分数、区间带与章节变化。"""

from __future__ import annotations

import argparse
import json
import sys

from _cli import ensure_local, fail, run, setup_stdio


def main(argv: list[str] | None = None) -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="对比改稿前后的本地 AIGC 风险")
    parser.add_argument("new", help="新版本本地文件")
    parser.add_argument("--old", required=True, help="旧版本本地文件")
    parser.add_argument("--out", default=None, help="对比 HTML 输出路径")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    parser.add_argument("--enhanced", action="store_true", help="实验性增强召回")
    parser.add_argument(
        "--profile",
        choices=["lite", "balanced", "max"],
        default=None,
        help="运行档位：lite=纯 CPU/0.5B，balanced=b+c/2B，max=a+b+c/2B",
    )
    parser.add_argument("--detectors", default=None, help="检测路组合；不填用档位推荐")
    parser.add_argument("--device", default=None, help="auto|cpu|cuda|mps；不填用档位推荐")
    args = parser.parse_args(argv)

    ensure_local(args.new, "新版本")
    ensure_local(args.old, "旧版本")
    cmd = [args.new, "--compare-with", args.old, "--json"]
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.detectors:
        cmd += ["--detectors", args.detectors]
    if args.device:
        cmd += ["--device", args.device]
    if args.out:
        cmd += ["--out", args.out]
    if args.enhanced:
        cmd.append("--enhanced")
    result = run(cmd)
    if result.returncode != 0:
        return fail(result)

    comparison = json.loads(result.stdout)
    if args.json:
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        if args.out:
            print(f"\n对比报告：{args.out}", file=sys.stderr)
        return 0

    old, new = comparison["old"], comparison["new"]
    print(f"版本对比：{comparison['verdict']}")
    print(
        f"综合分 {old['score']} → {new['score']}"
        f"（{comparison['score_delta']:+d}）；区间带 {old['band']} → {new['band']}"
    )
    print(f"改稿前：median {old['median']:.2f} / P90 {old['p90']:.2f} / {old['n_windows']} 窗")
    print(f"改稿后：median {new['median']:.2f} / P90 {new['p90']:.2f} / {new['n_windows']} 窗")

    sections = comparison.get("sections") or []
    if sections:
        print("\n章节变化：")
        for item in sections:
            delta = item.get("delta")
            delta_text = "—" if delta is None else f"{delta:+d}"
            print(
                f"- {item.get('title')}：{item.get('old_score', '—')} → {item.get('new_score', '—')}"
                f"（{delta_text}，{item.get('status')}）"
            )
    print(f"\n{comparison.get('note', '')}")
    if args.out:
        print(f"对比报告：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
