"""命令行入口。

用法：
    aigc-selfcheck 论文.docx                # 生成 selfcheck-report.html
    aigc-selfcheck 论文.docx --json         # 机器可读输出（stdout）
    aigc-selfcheck 论文.docx --detectors b,c --device cpu   # 纯 CPU 形态
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 只读常量，导入很轻；choices 需要在 build_parser 时就拿到预署名。
from .constants import BAND_SETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigc-selfcheck",
        description="纯本地的中文文本 AIGC 风险自检台：逐句热力图 + 多检测器投票 + 可解释特征报告。",
    )
    parser.add_argument("file", help="待检测文档（.docx/.pdf/.txt/.md）")
    parser.add_argument("--json", action="store_true", help="向 stdout 输出机器可读 JSON（可与 --out 并用）")
    parser.add_argument("--out", default=None, help="报告输出路径（默认 selfcheck-report.html；--json 且未指定时不生成）")
    parser.add_argument(
        "--compare-with",
        default=None,
        metavar="OLD",
        help="与旧版本对比：分析新文件并与 OLD 生成版本对比报告（默认 selfcheck-compare.html）",
    )
    parser.add_argument(
        "--profile",
        choices=["lite", "balanced", "max"],
        default=None,
        help="运行档位：lite=纯 CPU/0.5B，balanced=b+c/2B，max=a+b+c/2B；显式参数优先",
    )
    parser.add_argument(
        "--detectors",
        default=None,
        help="检测路组合，如 a,b,c 或 b,c；不填则用档位推荐或旧默认 a,b,c",
    )
    parser.add_argument(
        "--c-model",
        default=None,
        help="C 路基座模型（默认 Qwen3.5-2B-Base；CPU 回退可指定 Qwen2.5-0.5B）",
    )
    parser.add_argument("--device", default=None, help="推理设备 auto|cpu|mps|cuda；不填则用档位推荐（默认 auto）")
    parser.add_argument("--no-cache", action="store_true", help="禁用窗口分数缓存")
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="实验性增强召回：灰度区元分类器额外标记（提高召回，但会增加误报；需 a,b,c 三路）",
    )
    parser.add_argument(
        "--llm-advice",
        dest="llm_advice",
        action="store_true",
        default=None,
        help="用本地 C 路基座模型为最高风险窗口生成贴合原文的改写示例（实验性，需人工确认）",
    )
    parser.add_argument(
        "--no-llm-advice",
        dest="llm_advice",
        action="store_false",
        help="关闭档位默认的大模型具体建议，只用规则建议",
    )
    parser.add_argument("--quiet", action="store_true", help="不输出进度信息")
    parser.add_argument(
        "--band-set",
        default=None,
        metavar="NAME",
        choices=sorted(BAND_SETS),
        help=(
            "判定口径预设：academic（默认，学位论文/期刊正文）| general（博客/公文等混合语域）。"
            "低=median<lo 且 p90<tlo；高=median>=hi 或 p90>=thi；边值见报告首屏。"
        ),
    )
    parser.add_argument(
        "--exclude-heading",
        action="append",
        default=None,
        metavar="TEXT",
        help=(
            "文档级档位不计入标题含 TEXT 的章节及其下属内容（可重复）。"
            "用于排除外文翻译、附录、致谢等非原创正文；热力图与章节地图仍完整显示。"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.file)
    if not source.exists():
        print(f"错误：文件不存在 {source}", file=sys.stderr)
        return 2
    compare_source = Path(args.compare_with) if args.compare_with else None
    if compare_source is not None and not compare_source.exists():
        print(f"错误：对比文件不存在 {compare_source}", file=sys.stderr)
        return 2

    from .profiles import resolve_profile

    profile = resolve_profile(
        args.profile,
        detectors=args.detectors,
        device=args.device,
        c_model=args.c_model,
        llm_advice=args.llm_advice,
    )
    detectors_spec = profile["detectors"]
    device = profile["device"]
    c_model = profile["c_model"]
    llm_advice = bool(profile.get("llm_advice"))

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr)

    # 档位默认开启模型建议，但纯 CPU 上生成 2B 文本很慢：
    # 未显式 --llm-advice 时自动跳过，避免低配机器被默认行为拖住。
    llm_skip_note = ""
    if llm_advice and args.llm_advice is None:
        cpu_only = device == "cpu"
        if device == "auto":
            try:
                import torch

                mps = getattr(torch.backends, "mps", None)
                cpu_only = not (
                    torch.cuda.is_available() or (mps is not None and mps.is_available())
                )
            except Exception:  # noqa: BLE001 —— 设备探测失败时保持默认行为
                cpu_only = False
        if cpu_only:
            llm_advice = False
            llm_skip_note = (
                "大模型具体建议：CPU 上默认跳过（2B 生成较慢）；"
                "需要时显式加 --llm-advice。"
            )
    profile["llm_advice"] = llm_advice

    enhanced = args.enhanced
    if enhanced:
        lanes = {lane.strip() for lane in detectors_spec.split(",") if lane.strip()}
        if not {"a", "b", "c"} <= lanes:
            log("警告：--enhanced 需要 a,b,c 三路检测器，已关闭增强召回。")
            enhanced = False
        else:
            log("增强召回已开启（实验性）：会额外标记灰度区窗口，可能增加误报。")

    # 重型依赖在确认参数后导入
    from .cache import ScoreCache
    from .compare import compare_results, render_compare
    from .constants import resolve_band_set
    from .detectors import build_detectors
    from .pipeline import analyze_document
    from .report import render_report
    from .report.model import build_payload

    try:
        band_set = resolve_band_set(args.band_set)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    try:
        log(
            f"运行档位 {profile['label']}（{profile['name']}）："
            f"检测路 {detectors_spec}，device={device} …"
        )
        log(
            f"档位口径 {band_set.label}：低 median<{band_set.robust_low} 且 "
            f"p90<{band_set.tail_low}；高 median>={band_set.robust_high} 或 "
            f"p90>={band_set.tail_high}；少于 {band_set.min_windows} 窗不判带。"
        )
        if llm_skip_note:
            log(llm_skip_note)
        elif llm_advice:
            log("大模型具体建议：开启（只处理最高风险窗口，结果需人工确认）。")
        detectors = build_detectors(detectors_spec, device, c_model=c_model)
        cache = ScoreCache(enabled=not args.no_cache)
        log("开始分析 …")
        result = analyze_document(
            source,
            detectors,
            cache,
            enhanced=enhanced,
            profile=profile,
            llm_advice=llm_advice,
            band_set=band_set,
            exclude_headings=args.exclude_heading,
        )
        comparison = None
        if compare_source is not None:
            log(f"分析旧版本 {compare_source} …")
            old_result = analyze_document(
                compare_source,
                detectors,
                cache,
                enhanced=enhanced,
                profile=profile,
                llm_advice=llm_advice,
                band_set=band_set,
                exclude_headings=args.exclude_heading,
            )
            comparison = compare_results(old_result, result)
        cache.flush()
        payload = build_payload(result) if comparison is None else None
    except Exception as exc:  # noqa: BLE001 —— CLI 顶层兜底
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if comparison is not None:
        if args.json:
            print(json.dumps(comparison, ensure_ascii=False))
        if args.json and args.out is None:
            out_path = None
        else:
            out_path = Path(args.out) if args.out else Path("selfcheck-compare.html")
        if out_path is not None:
            render_compare(comparison, out_path)
        old_doc, new_doc = comparison["old"], comparison["new"]
        log(
            f"完成：{comparison['verdict']} 综合分 {old_doc['score']} → {new_doc['score']}"
            f"（{comparison['score_delta']:+d}）"
        )
        if out_path is not None:
            if args.json:
                print(str(out_path), file=sys.stderr)
            else:
                print(str(out_path))
        return 0

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))

    out_path = Path(args.out) if args.out else (None if args.json else Path("selfcheck-report.html"))
    if out_path is None:
        return 0

    render_report(payload, out_path)
    doc = payload["doc"]
    log(
        f"完成：区间带[{doc['band']}] median={doc['median']:.2f} p90={doc['p90']:.2f}"
        f" 窗口={doc['n_windows']} 用时 {result.duration_s:.1f}s"
    )
    # --json 时 stdout 已被 JSON 占用，路径走 stderr
    if args.json:
        print(str(out_path), file=sys.stderr)
    else:
        print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
