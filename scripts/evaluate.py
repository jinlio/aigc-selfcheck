"""校准集评测：路级与融合 AUC、人写误伤率；--fit 拟合临时校准常量。

验收线（ALGORITHM §4，对 test split）：
    融合 AUC ≥ 0.85 且 人写误伤率（p_fuse > HEATMAP_HIGH 的比例）≤ 10% 才发布 v0.1；
    调 α/权重/阈值只允许用 train split（含分组 CV），test 只跑一次。

用法：
    uv run python scripts/evaluate.py                  # 评测（需已 build_calibration）
    uv run python scripts/evaluate.py --split=train    # 只给 train 打分/汇总（v8 常量轮）
    uv run python scripts/evaluate.py --allow-test     # 确认要跑新 test（每版本仅一次）
    uv run python scripts/evaluate.py --fit            # 用 train 人写窗口拟合 PPL/风格常量
    uv run python scripts/evaluate.py --split=train --c-model=Qwen/Qwen2.5-0.5B  # CPU 回退复核

v8 起 c 路按 `contexts.preceding_context` 传入窗口起点之前的正文，缓存键并入
上下文；`--split=train` 只重算 train，避免在已消耗的 test 上产生新分数。
test 打分必须显式加 `--allow-test`，防止误跑已消耗的切分。
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

# 本机网络直连 huggingface.co 被墙；与 download_models.py 同一套镜像默认
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_selfcheck.cache import ScoreCache  # noqa: E402
from aigc_selfcheck.constants import HEATMAP_HIGH  # noqa: E402
from aigc_selfcheck.contexts import window_contexts  # noqa: E402
from aigc_selfcheck.detectors import build_detectors  # noqa: E402
from aigc_selfcheck.fuse import fuse_scores, mtld_arbitration  # noqa: E402

CALIBRATION = Path("calibration/out/calibration.jsonl")
FINGERPRINT = Path("calibration/out/splits.md5")
HOT = HEATMAP_HIGH  # 单一来源：报告红线（constants.HEATMAP_HIGH）即评测红线


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Mann-Whitney 秩和 AUC（并列秩取平均）。"""
    if not scores_pos or not scores_neg:
        return float("nan")
    combined = sorted((v, 1) for v in scores_pos) + sorted((v, 0) for v in scores_neg)
    combined.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    index = 0
    while index < len(combined):
        tied = index + 1
        while tied < len(combined) and combined[tied][0] == combined[index][0]:
            tied += 1
        avg_rank = (index + 1 + tied) / 2
        rank_sum_pos += sum(avg_rank for value, label in combined[index:tied] if label == 1)
        index = tied
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def load_units() -> list[dict]:
    expected = FINGERPRINT.read_text().strip()
    lines = CALIBRATION.read_text(encoding="utf-8").splitlines()
    actual = __import__("hashlib").sha256("".join(line + "\n" for line in lines).encode("utf-8")).hexdigest()
    if actual != expected:
        raise SystemExit(f"冻结指纹不符：{actual} != {expected}——切分被改动，禁止评测")
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    fit = "--fit" in sys.argv
    allow_test = "--allow-test" in sys.argv
    device = "auto"
    split = "all"
    c_model = None
    for arg in sys.argv:
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        elif arg.startswith("--split="):
            split = arg.split("=", 1)[1]
        elif arg.startswith("--c-model="):
            c_model = arg.split("=", 1)[1]
    if split not in {"all", "train", "test"}:
        raise SystemExit("--split 只接受 all / train / test")
    units = load_units()
    train = [u for u in units if u["split"] == "train"]
    test = [u for u in units if u["split"] == "test"]
    if split == "train" or fit:
        test = []
    elif split == "test":
        train = []
    if test and not allow_test:
        raise SystemExit(
            "test split 每个语料版本只评测一次；当前切分的 test 已消耗。"
            "如确认这是新语料/新切分的一次性评测，加 --allow-test。"
        )
    print(f"单元数 train={len(train)} test={len(test)}（--split={split}）")
    if test:
        print("  [warn] test 每个语料版本只评测一次；本次已显式 --allow-test。")

    detectors = build_detectors("a,b,c", device=device, c_model=c_model)
    for detector in detectors:
        # 设备可见性：C 路落在 cpu 会慢一个数量级（曾发生 MPS 会话失效静默回退）
        print(f"  路线 {detector.path_key}: {detector.model_id} @ {getattr(detector, 'device', '?')}", flush=True)
    cache = ScoreCache(path=Path("calibration/out/score-cache.jsonl"))

    def score_all(items: list[dict]) -> list[dict]:
        contexts = window_contexts(items)
        scores_by_unit: list[dict] = [{} for _ in items]
        metas_by_unit: list[dict] = [{} for _ in items]
        # 按路批量打分：缓存命中先分离，未命中的整批送 GPU（顺序无关，结果逐窗一致）
        for detector in detectors:
            pending: list[int] = []
            for i, unit in enumerate(items):
                cache_context = contexts[i] if detector.uses_context else ""
                cached = cache.get(
                    detector.path_key, detector.cache_id, unit["text"], context=cache_context
                )
                if cached and "score" in cached:
                    scores_by_unit[i][detector.path_key] = float(cached["score"])
                    metas_by_unit[i][detector.path_key] = cached.get("meta") or {}
                else:
                    pending.append(i)
            if pending:
                results = detector.detect_batch(
                    [items[i]["text"] for i in pending],
                    [contexts[i] for i in pending],
                )
                for i, result in zip(pending, results, strict=True):
                    scores_by_unit[i][detector.path_key] = result.score
                    metas_by_unit[i][detector.path_key] = result.meta
                    cache.put(
                        detector.path_key,
                        detector.cache_id,
                        items[i]["text"],
                        {"score": result.score, "meta": _json_safe(result.meta)},
                        context=contexts[i] if detector.uses_context else "",
                    )
                cache.flush()  # 增量落盘：长跑中断不丢已算分数
                print(f"  {detector.path_key} 路新增 {len(pending)} 窗（{getattr(detector, 'device', '?')}）", flush=True)

        scored = []
        for i, unit in enumerate(items):
            scores = scores_by_unit[i]
            fused_score, _delta = fuse_scores(scores)
            meta_c = metas_by_unit[i].get("c", {})
            # v12：与产品 pipeline 同规则——b 高 / c 低时用 char_mtld 裁决
            fused_score = mtld_arbitration(fused_score, scores, meta_c.get("features"))
            scored.append({**unit, "scores": scores, "fused": fused_score, "meta_c": meta_c})
        return scored

    train_scored = score_all(train)
    test_scored = score_all(test)

    if fit:
        fit_constants([u for u in train_scored if u["class"] == "human"])
        return 0

    if split == "train":
        print("\n== train split 汇总（只报告，不用于调 test）==")
        for path_key in ("a", "b", "c"):
            pos = [u["scores"][path_key] for u in train_scored if u["class"] == "ai"]
            neg = [u["scores"][path_key] for u in train_scored if u["class"] == "human"]
            print(f"  {path_key} 路 AUC: {auc(pos, neg):.3f}")
        pos = [u["fused"] for u in train_scored if u["class"] == "ai"]
        neg = [u["fused"] for u in train_scored if u["class"] == "human"]
        fp = sum(1 for v in neg if v > HOT) / len(neg) if neg else float("nan")
        recall = sum(1 for v in pos if v > HOT) / len(pos) if pos else float("nan")
        print(f"  融合 AUC: {auc(pos, neg):.3f}")
        print(f"  人写误伤率: {fp:.1%}（{sum(1 for v in neg if v > HOT)}/{len(neg)}，p_fuse > {HOT}）")
        print(f"  窗口召回: {recall:.1%}（{sum(1 for v in pos if v > HOT)}/{len(pos)}）")
        for detector in detectors:
            detector.close()
        return 0

    print("\n== AUC（test split，human=负类，ai=正类）==")
    for path_key in ("a", "b", "c"):
        pos = [u["scores"][path_key] for u in test_scored if u["class"] == "ai"]
        neg = [u["scores"][path_key] for u in test_scored if u["class"] == "human"]
        print(f"  {path_key} 路: {auc(pos, neg):.3f}")
    pos = [u["fused"] for u in test_scored if u["class"] == "ai"]
    neg = [u["fused"] for u in test_scored if u["class"] == "human"]
    fused_auc = auc(pos, neg)
    fp = sum(1 for v in neg if v > HOT) / len(neg) if neg else float("nan")
    print(f"  融合:  {fused_auc:.3f}")
    print("\n== 验收线 ==")
    print(f"  融合 AUC ≥ 0.85:  {'✓' if fused_auc >= 0.85 else '✗'} ({fused_auc:.3f})")
    print(f"  人写误伤率 ≤ 10%: {'✓' if fp <= 0.10 else '✗'} ({fp:.1%}，p_fuse > {HOT})")

    polished = [u["fused"] for u in test_scored if u["class"] == "polished"]
    if polished:
        print("\n== 润色类（最难题，无硬门槛）==")
        print(f"  median={statistics.median(polished):.3f}  >{HOT} 占比 {sum(1 for v in polished if v > HOT) / len(polished):.1%}")
    for detector in detectors:
        detector.close()
    return 0


def _json_safe(value):
    if isinstance(value, float):
        from math import isfinite

        return value if isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def fit_constants(human_train: list[dict]) -> None:
    """从 train 人写窗口拟合 PPL 与风格归一化常量（PROVISIONAL 的替换来源）。"""
    log_ppls: list[float] = []
    feature_rows: list[dict[str, float]] = []
    for unit in human_train:
        meta = unit["meta_c"]
        log_ppl = meta.get("log_ppl")
        if log_ppl is not None:
            log_ppls.append(float(log_ppl))
        features = meta.get("features") or {}
        if features:
            feature_rows.append({k: float(v) for k, v in features.items() if v == v})

    if not log_ppls:
        raise SystemExit("没有可用的 log_ppl 数据（C 路未产出？）")
    print("\n== 拟合结果（写入 constants.py 替换 PROVISIONAL 值）==")
    print(f"PPL_LOG_MEAN = {statistics.median(log_ppls):.2f}   # median：稳健中心")
    sorted_ppls = sorted(log_ppls)
    print(f"PPL_LOG_MEAN(75分位候选) = {sorted_ppls[int(len(sorted_ppls) * 0.75)]:.2f}   # v3.1 起采用：PPL 信号更保守")
    print(f"PPL_LOG_STD = {statistics.pstdev(log_ppls):.2f}")

    names = feature_rows[0].keys() if feature_rows else []
    style_lines = []
    for name in names:
        values = [row[name] for row in feature_rows if name in row]
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        std = std if std > 0 else 1.0
        style_lines.append(f'    "{name}": {{"mean": {mean:.1f}, "std": {std:.1f}}},')
    if style_lines:
        print("STYLE_MEANS/STDS（权重与方向沿用现有表）:")
        for line in style_lines:
            print(line)


if __name__ == "__main__":
    raise SystemExit(main())
