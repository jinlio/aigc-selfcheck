"""语料探针：当前三路检测器的只读命中率评分。

读取任意含 `texts/*.txt` 的探针目录，按产品同款分句/滑窗/上下文/融合路径打分，
报告窗口命中率、文档「高」带比例、文档 median/P90 超 HOT 比例与逐文体拆分。
`--kind ai` 时命中率即召回；`--kind human` 时即误伤率。

report-only：不进入 calibration.jsonl，不参与切分、常量、阈值或权重调整。

用法：
    PYTHONPATH=src python scripts/probe_ark_lite.py --device=cuda
    PYTHONPATH=src python scripts/probe_ark_lite.py \
        --data-dir calibration/ai_academic_lite --kind ai --label "AI 学术章节"
    PYTHONPATH=src python scripts/probe_ark_lite.py \
        --data-dir calibration/academic --kind human --label "人写学术段落"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_selfcheck.cache import ScoreCache  # noqa: E402
from aigc_selfcheck.constants import HEATMAP_HIGH  # noqa: E402
from aigc_selfcheck.contexts import preceding_context  # noqa: E402
from aigc_selfcheck.detectors import build_detectors  # noqa: E402
from aigc_selfcheck.docio import Block  # noqa: E402
from aigc_selfcheck.fuse import FusedWindow, doc_level, fuse_scores  # noqa: E402
from aigc_selfcheck.preprocess import prepare_block  # noqa: E402

DATA = ROOT / "calibration" / "ai_ark_lite"
HOT = HEATMAP_HIGH


def load_paragraphs(data_dir: Path, texts_subdir: str = "texts") -> list[dict]:
    rows = []
    for path in sorted((data_dir / texts_subdir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        provenance = ""
        body = raw.strip()
        if body.startswith("<!--"):
            end = body.index("-->")
            provenance = body[4:end].strip()
            body = body[end + 3 :].strip()
        style_match = re.search(r"文体:\s*([^|\n]+)", provenance)
        prompt_match = re.search(r"提示词:\s*(\d+)", provenance)
        rows.append(
            {
                "id": path.stem,
                "text": body,
                "style": style_match.group(1).strip() if style_match else "unknown",
                "prompt_idx": int(prompt_match.group(1)) if prompt_match else 0,
            }
        )
    return rows


def windows_for(index: int, text: str) -> list[tuple[int, str, str]]:
    _sentences, windows = prepare_block(Block(kind="paragraph", text=text, index=index))
    return [(index, win.text, preceding_context(text, win.start)) for win in windows]


def score_jobs(jobs: list[tuple[int, str, str]], device: str, cache_path: Path):
    detectors = build_detectors("a,b,c", device=device)
    for detector in detectors:
        print(
            f"  路线 {detector.path_key}: {detector.model_id} @ {getattr(detector, 'device', '?')}",
            flush=True,
        )
    cache = ScoreCache(path=cache_path)
    scores = [{} for _ in jobs]
    metas = [{} for _ in jobs]
    for detector in detectors:
        pending = []
        for i, (_index, text, context) in enumerate(jobs):
            cache_context = context if detector.uses_context else ""
            hit = cache.get(detector.path_key, detector.cache_id, text, context=cache_context)
            if hit and "score" in hit:
                scores[i][detector.path_key] = float(hit["score"])
                metas[i][detector.path_key] = hit.get("meta") or {}
            else:
                pending.append(i)
        if pending:
            results = detector.detect_batch(
                [jobs[i][1] for i in pending], [jobs[i][2] for i in pending]
            )
            for i, result in zip(pending, results, strict=True):
                scores[i][detector.path_key] = result.score
                metas[i][detector.path_key] = result.meta
                cache.put(
                    detector.path_key,
                    detector.cache_id,
                    jobs[i][1],
                    {"score": result.score, "meta": _safe(result.meta)},
                    context=jobs[i][2] if detector.uses_context else "",
                )
            cache.flush()
            print(f"  {detector.path_key} 路新增 {len(pending)} 窗", flush=True)
    for detector in detectors:
        detector.close()
    return scores, metas


def _safe(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def aggregate(rows: list[dict], jobs, scores, metas) -> list[dict]:
    by_para: dict[int, list[FusedWindow]] = defaultdict(list)
    for i, (index, text, _context) in enumerate(jobs):
        p_fuse, delta = fuse_scores(scores[i])
        by_para[index].append(
            FusedWindow(
                window_key=f"{index}:{len(by_para[index])}",
                block_index=index,
                sentence_ids=[],
                text=text,
                scores=scores[i],
                p_fuse=p_fuse,
                delta=delta,
                metas=metas[i],
            )
        )
    docs = []
    for index, row in enumerate(rows):
        windows = by_para[index]
        doc = doc_level(windows)
        lanes = {
            lane: statistics.median([w.scores[lane] for w in windows]) if windows else None
            for lane in ("a", "b", "c")
        }
        docs.append(
            {
                **row,
                "n_windows": doc["n_windows"],
                "window_scores": [w.p_fuse for w in windows],
                "median": doc["median"],
                "p90": doc["p90"],
                "band": doc["band"],
                "lane_medians": lanes,
            }
        )
    return docs


def summarize(rows: list[dict]) -> dict:
    window_scores = [score for row in rows for score in row["window_scores"]]
    n_windows = len(window_scores)
    n = len(rows)
    lane_values = {
        lane: [row["lane_medians"][lane] for row in rows if row["lane_medians"][lane] is not None]
        for lane in ("a", "b", "c")
    }
    positive = (
        sum(score > HOT for score in window_scores) / n_windows if n_windows else 0.0
    )
    return {
        "n_paragraphs": n,
        "n_windows": n_windows,
        "window_positive_rate": positive,
        # 兼容旧报告字段；AI 语料下它等于召回，人写语料下它是误伤率
        "window_recall": positive,
        "window_p50": statistics.median(window_scores) if window_scores else 0.0,
        "window_p90": _percentile(window_scores, 0.9),
        "doc_band_high": sum(row["band"] == "高" for row in rows) / n if n else 0.0,
        "doc_median_hot": sum(row["median"] > HOT for row in rows) / n if n else 0.0,
        "doc_p90_hot": sum(row["p90"] > HOT for row in rows) / n if n else 0.0,
        "lane_median_a": statistics.median(lane_values["a"]) if lane_values["a"] else None,
        "lane_median_b": statistics.median(lane_values["b"]) if lane_values["b"] else None,
        "lane_median_c": statistics.median(lane_values["c"]) if lane_values["c"] else None,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(DATA.relative_to(ROOT)),
        help="含 texts/ 的探针目录（默认通用豆包 Lite 500 段）",
    )
    parser.add_argument("--label", default="豆包 Lite 500 段", help="报告标题")
    parser.add_argument(
        "--texts-subdir",
        default="texts",
        help="样本子目录（默认 texts；人写学术集为 human_texts）",
    )
    parser.add_argument(
        "--kind",
        choices=("ai", "human"),
        default="ai",
        help="ai：窗口命中率=召回；human：窗口命中率=误伤率",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=0,
        help="只跑前 N 段（0=全部）；c 路显存紧时可用于子集冒烟",
    )
    parser.add_argument("--cache", default="", help="score-cache.jsonl 路径")
    parser.add_argument("--report", default="", help="report.json 输出路径")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    cache_path = Path(args.cache) if args.cache else data_dir / "score-cache.jsonl"
    report_path = Path(args.report) if args.report else data_dir / "report.json"
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    rows = load_paragraphs(data_dir, args.texts_subdir)
    if args.max_paragraphs:
        rows = rows[: args.max_paragraphs]
    jobs = [job for index, row in enumerate(rows) for job in windows_for(index, row["text"])]
    print(
        f"探针 {args.label}：{len(rows)} 段 / {len(jobs)} 窗，开始打分 …",
        flush=True,
    )
    scores, metas = score_jobs(jobs, args.device, cache_path)
    docs = aggregate(rows, jobs, scores, metas)

    overall = summarize(docs)
    by_style = {}
    for style in sorted({row["style"] for row in docs}):
        by_style[style] = summarize([row for row in docs if row["style"] == style])

    hit_label = "窗口召回" if args.kind == "ai" else "窗口误伤"
    print(f"\n== {args.label}（report-only，不参与任何决策）==")
    print(
        f"{'范围':16s} {'段':>4s} {'窗':>5s} {hit_label:>8s} {'窗P50':>7s} "
        f"{'窗P90':>7s} {'高带':>6s} {'中位>HOT':>9s} {'P90>HOT':>8s} "
        f"{'a':>6s} {'b':>6s} {'c':>6s}"
    )
    for label, stats in [("全部", overall), *by_style.items()]:
        print(
            f"{label:16s} {stats['n_paragraphs']:4d} {stats['n_windows']:5d} "
            f"{stats['window_positive_rate']:8.1%} {stats['window_p50']:7.3f} "
            f"{stats['window_p90']:7.3f} {stats['doc_band_high']:6.1%} "
            f"{stats['doc_median_hot']:9.1%} {stats['doc_p90_hot']:8.1%} "
            f"{_fmt(stats['lane_median_a'])} {_fmt(stats['lane_median_b'])} "
            f"{_fmt(stats['lane_median_c'])}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "note": "report-only; never used for calibration, splits, thresholds or weights",
                "label": args.label,
                "kind": args.kind,
                "data_dir": str(data_dir),
                "model": "doubao-seed-2-1-lite-260915" if args.kind == "ai" else "human",
                "hot": HOT,
                "overall": overall,
                "by_style": by_style,
                "paragraphs": [
                    {
                        "id": row["id"],
                        "style": row["style"],
                        "n_windows": row["n_windows"],
                        "median": row["median"],
                        "p90": row["p90"],
                        "band": row["band"],
                        "max_window": max(row["window_scores"]) if row["window_scores"] else 0.0,
                        "window_scores": row["window_scores"],
                        "lane_medians": row["lane_medians"],
                        "text": row["text"],
                    }
                    for row in docs
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n报告：{report_path}", flush=True)
    return 0


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
