"""跨语域人写探针：只报告、不调参（report-only）。

读取 `calibration/probe/human.jsonl`（build_probe.py 产出，四个目标语域之外的
人写语域），按产品同款分句/滑窗/上下文/融合路径打分，输出每个语域的文档级
中位分、P90、窗口误伤率与段落误伤率。另给一份**train 内目标语域人写**的
同口径基线（in-sample，仅作对照，不参与任何决策）。

探针不进入 calibration.jsonl、不参与切分/常量/阈值/权重；本脚本也不改任何
常量。AI 侧的 fresh 跨语域样本仍被凭据阻塞（无 OPENAI_API_KEY），因此本表
只回答「异语域人写误伤」这一半问题。

用法：
    PYTHONPATH=src python scripts/probe_registers.py --device=cuda
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_selfcheck.cache import ScoreCache  # noqa: E402
from aigc_selfcheck.constants import HEATMAP_HIGH  # noqa: E402
from aigc_selfcheck.contexts import preceding_context, window_contexts  # noqa: E402
from aigc_selfcheck.detectors import build_detectors  # noqa: E402
from aigc_selfcheck.detectors.stats_lm import StatsDetector  # noqa: E402
from aigc_selfcheck.docio import Block  # noqa: E402
from aigc_selfcheck.fuse import (  # noqa: E402
    FusedWindow,
    apply_mtld_arbitration,
    doc_level,
    fuse_scores,
)
from aigc_selfcheck.preprocess import prepare_block  # noqa: E402

PROBE = ROOT / "calibration" / "probe" / "human.jsonl"
PROBE_CACHE = ROOT / "calibration" / "probe" / "score-cache.jsonl"
CALIBRATION = ROOT / "calibration" / "out" / "calibration.jsonl"
MAIN_CACHE = ROOT / "calibration" / "out" / "score-cache.jsonl"
REPORT = ROOT / "calibration" / "probe" / "report.json"
HOT = HEATMAP_HIGH
A_CID = "yuchuantian/AIGC_detector_zhv2"
B_CID = "mokawa3018/cmj-chinese-aigc-text-detector"


def score_jobs(jobs: list[tuple[int, str, str]], device: str, c_model: str | None):
    detectors = build_detectors("a,b,c", device=device, c_model=c_model)
    for detector in detectors:
        print(
            f"  路线 {detector.path_key}: {detector.model_id} @ {getattr(detector, 'device', '?')}",
            flush=True,
        )
    cache = ScoreCache(path=PROBE_CACHE)
    scores = [{} for _ in jobs]
    metas = [{} for _ in jobs]
    for detector in detectors:
        pending = []
        for i, (_probe, text, context) in enumerate(jobs):
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


def _windows_for(probe_index: int, text: str) -> list[tuple[int, str, str]]:
    _sentences, windows = prepare_block(Block(kind="paragraph", text=text, index=probe_index))
    return [(probe_index, win.text, preceding_context(text, win.start)) for win in windows]


def _doc_rows(probe_rows: list[dict], jobs, scores, metas) -> list[dict]:
    by_probe: dict[int, list[FusedWindow]] = {i: [] for i in range(len(probe_rows))}
    for i, (probe_index, text, _context) in enumerate(jobs):
        p_fuse, delta = fuse_scores(scores[i])
        by_probe[probe_index].append(
            FusedWindow(
                window_key=f"{probe_index}:{len(by_probe[probe_index])}",
                block_index=probe_index,
                sentence_ids=[],
                text=text,
                scores=scores[i],
                p_fuse=p_fuse,
                delta=delta,
                metas=metas[i],
            )
        )
    for windows in by_probe.values():
        apply_mtld_arbitration(windows)

    rows = []
    for index, probe in enumerate(probe_rows):
        windows = by_probe[index]
        doc = doc_level(windows)
        lanes = {
            lane: statistics.median([w.scores[lane] for w in windows]) if windows else None
            for lane in ("a", "b", "c")
        }
        rows.append(
            {
                **{k: probe[k] for k in ("id", "register", "book", "gutenberg_id", "cjk")},
                "n_windows": doc["n_windows"],
                "window_fp": sum(w.p_fuse > HOT for w in windows),
                "median": doc["median"],
                "p90": doc["p90"],
                "band": doc["band"],
                "lane_medians": lanes,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    medians = [r["median"] for r in rows]
    p90s = [r["p90"] for r in rows]
    windows = sum(r["n_windows"] for r in rows)
    return {
        "n_paragraphs": len(rows),
        "n_windows": windows,
        "median_of_medians": statistics.median(medians),
        "mean_of_medians": statistics.fmean(medians),
        "paragraph_fp_median": sum(m > HOT for m in medians) / len(medians),
        "paragraph_fp_p90": sum(p > HOT for p in p90s) / len(p90s),
        "window_fp": sum(r["window_fp"] for r in rows) / windows,
        "lane_median_a": statistics.median(
            r["lane_medians"]["a"] for r in rows if r["lane_medians"]["a"] is not None
        ),
        "lane_median_b": statistics.median(
            r["lane_medians"]["b"] for r in rows if r["lane_medians"]["b"] is not None
        ),
        "lane_median_c": statistics.median(
            r["lane_medians"]["c"] for r in rows if r["lane_medians"]["c"] is not None
        ),
    }


def baseline_train_human() -> dict | None:
    """train 人写段落的同口径文档级基线（in-sample，仅对照）。"""
    rows = [
        json.loads(line)
        for line in CALIBRATION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contexts = window_contexts(rows)
    cache = ScoreCache(path=MAIN_CACHE)
    c_cid = StatsDetector.__new__(StatsDetector).cache_id
    cids = {"a": A_CID, "b": B_CID, "c": c_cid}
    by_para: dict[str, list[FusedWindow]] = {}
    for i, row in enumerate(rows):
        if row["split"] != "train" or row["class"] != "human":
            continue
        scores = {}
        for lane in ("a", "b", "c"):
            hit = cache.get(
                lane, cids[lane], row["text"], context=contexts[i] if lane == "c" else ""
            )
            if not hit or "score" not in hit:
                return None
            scores[lane] = float(hit["score"])
        p_fuse, delta = fuse_scores(scores)
        window = FusedWindow(
            window_key=row["uid"],
            block_index=0,
            sentence_ids=[],
            text=row["text"],
            scores=scores,
            p_fuse=p_fuse,
            delta=delta,
            metas={},
        )
        by_para.setdefault(row["para_sha"], []).append(window)
    docs = [doc_level(windows) for windows in by_para.values()]
    medians = [d["median"] for d in docs]
    p90s = [d["p90"] for d in docs]
    total_windows = sum(d["n_windows"] for d in docs)
    return {
        "n_paragraphs": len(docs),
        "n_windows": total_windows,
        "median_of_medians": statistics.median(medians),
        "mean_of_medians": statistics.fmean(medians),
        "paragraph_fp_median": sum(m > HOT for m in medians) / len(medians),
        "paragraph_fp_p90": sum(p > HOT for p in p90s) / len(p90s),
        "window_fp": sum(
            w.p_fuse > HOT for windows in by_para.values() for w in windows
        )
        / total_windows,
    }


def main() -> int:
    device = "auto"
    c_model = None
    for arg in sys.argv:
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        elif arg.startswith("--c-model="):
            c_model = arg.split("=", 1)[1]

    probes = [
        json.loads(line)
        for line in PROBE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jobs = [job for index, row in enumerate(probes) for job in _windows_for(index, row["text"])]
    print(f"探针 {len(probes)} 段 / {len(jobs)} 窗，开始打分 …", flush=True)
    scores, metas = score_jobs(jobs, device, c_model)
    doc_rows = _doc_rows(probes, jobs, scores, metas)

    print("\n== 跨语域人写探针（report-only，不参与任何决策）==")
    print(
        f"{'语域':22s} {'段':>3s} {'窗':>3s} {'中位分':>7s} {'窗误伤':>6s} {'段误伤':>6s} "
        f"{'P90误伤':>7s} {'a中位':>6s} {'b中位':>6s} {'c中位':>6s}"
    )
    summary = {}
    for register in sorted({r["register"] for r in doc_rows}):
        rows = [r for r in doc_rows if r["register"] == register]
        stats = summarize(rows)
        summary[register] = stats
        print(
            f"{register:22s} {stats['n_paragraphs']:3d} {stats['n_windows']:3d} "
            f"{stats['median_of_medians']:7.3f} {stats['window_fp']:6.0%} "
            f"{stats['paragraph_fp_median']:6.0%} "
            f"{stats['paragraph_fp_p90']:7.0%} {stats['lane_median_a']:6.3f} "
            f"{stats['lane_median_b']:6.3f} {stats['lane_median_c']:6.3f}"
        )
    overall = summarize(doc_rows)
    print(
        f"{'全部':22s} {overall['n_paragraphs']:3d} {overall['n_windows']:3d} "
        f"{overall['median_of_medians']:7.3f} {overall['window_fp']:6.0%} "
        f"{overall['paragraph_fp_median']:6.0%} "
        f"{overall['paragraph_fp_p90']:7.0%} {overall['lane_median_a']:6.3f} "
        f"{overall['lane_median_b']:6.3f} {overall['lane_median_c']:6.3f}"
    )

    baseline = baseline_train_human()
    if baseline:
        print("\n== 对照：train 内目标语域人写（in-sample，仅基线）==")
        print(
            f"{'train-human':22s} {baseline['n_paragraphs']:3d} {baseline['n_windows']:3d} "
            f"{baseline['median_of_medians']:7.3f} {baseline['window_fp']:6.0%} "
            f"{baseline['paragraph_fp_median']:6.0%} "
            f"{baseline['paragraph_fp_p90']:7.0%}"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "note": "report-only; never used for calibration, splits, thresholds or weights",
                "hot": HOT,
                "model": c_model or StatsDetector.model_id,
                "summary": summary,
                "overall": overall,
                "baseline_train_human_in_sample": baseline,
                "paragraphs": doc_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
