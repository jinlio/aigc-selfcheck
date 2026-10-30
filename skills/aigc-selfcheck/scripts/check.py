#!/usr/bin/env python3
"""运行本地 AIGC 自检，生成单文件 HTML 报告。"""

from __future__ import annotations

import argparse
import sys

from _cli import ensure_local, fail, run, setup_stdio


def main(argv: list[str] | None = None) -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="本地 AIGC 自检（不上传文本）")
    parser.add_argument("file", help="本地 .docx/.pdf/.txt/.md 文件")
    parser.add_argument("--out", default=None, help="HTML 输出路径")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument("--enhanced", action="store_true", help="实验性增强召回（可能增加误报）")
    parser.add_argument(
        "--profile",
        choices=["lite", "balanced", "max"],
        default=None,
        help="运行档位：lite=纯 CPU/0.5B，balanced=b+c/2B，max=a+b+c/2B",
    )
    parser.add_argument("--detectors", default=None, help="检测路组合，如 b,c；不填用档位推荐")
    parser.add_argument("--device", default=None, help="auto|cpu|cuda|mps；不填用档位推荐")
    parser.add_argument("--c-model", default=None, help="C 路基座模型；不填用档位推荐")
    parser.add_argument("--no-cache", action="store_true", help="禁用窗口缓存")
    parser.add_argument(
        "--llm-advice",
        dest="llm_advice",
        action="store_true",
        default=None,
        help="用本地基座模型为最高风险窗口生成改写示例（实验性）",
    )
    parser.add_argument(
        "--no-llm-advice",
        dest="llm_advice",
        action="store_false",
        help="关闭档位默认的大模型具体建议",
    )
    parser.add_argument("--quiet", action="store_true", help="不输出进度信息")
    parser.add_argument("--compare-with", default=None, metavar="OLD", help="与旧版本对比")
    parser.add_argument(
        "--band-set",
        default=None,
        choices=["academic", "general"],
        help="判定口径预设：academic（默认）| general",
    )
    parser.add_argument(
        "--exclude-heading",
        action="append",
        default=None,
        metavar="TEXT",
        help="文档级档位不计入标题含 TEXT 的章节（可重复）",
    )
    args = parser.parse_args(argv)

    ensure_local(args.file, "待检测文件")
    if args.compare_with:
        ensure_local(args.compare_with, "对比文件")

    cmd = [args.file]
    if args.out:
        cmd += ["--out", args.out]
    if args.json:
        cmd.append("--json")
    if args.enhanced:
        cmd.append("--enhanced")
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.detectors:
        cmd += ["--detectors", args.detectors]
    if args.device:
        cmd += ["--device", args.device]
    if args.c_model:
        cmd += ["--c-model", args.c_model]
    if args.no_cache:
        cmd.append("--no-cache")
    if args.llm_advice is True:
        cmd.append("--llm-advice")
    elif args.llm_advice is False:
        cmd.append("--no-llm-advice")
    if args.quiet:
        cmd.append("--quiet")
    if args.compare_with:
        cmd += ["--compare-with", args.compare_with]
    if args.band_set:
        cmd += ["--band-set", args.band_set]
    for heading in args.exclude_heading or []:
        cmd += ["--exclude-heading", heading]

    result = run(cmd)
    if result.returncode != 0:
        return fail(result)
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
