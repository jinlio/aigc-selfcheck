from aigc_selfcheck.fuse import (
    FusedWindow,
    apply_mtld_arbitration,
    composite_score,
    doc_level,
    fuse_scores,
    mtld_arbitration,
)


def test_fuse_scores_boosts_b_high_confidence() -> None:
    # b 路高置信 + c 路中等确定：融合分抬到 floor，避免被 c 拉低。
    assert fuse_scores({"a": 0.2, "b": 0.99, "c": 0.50})[0] == 0.80
    # c 不够确定时不抬分。
    assert fuse_scores({"a": 0.2, "b": 0.99, "c": 0.40})[0] < 0.80
    # b 不够确定时不抬分。
    assert fuse_scores({"a": 0.2, "b": 0.97, "c": 0.55})[0] < 0.80


def test_mtld_arbitration_boosts_b_high_c_low() -> None:
    scores = {"a": 0.0, "b": 0.93, "c": 0.52}
    # 落在 [70, 110] 的现代文本 mtld：抬到 floor。
    assert mtld_arbitration(0.62, scores, {"char_mtld": 85.0}) == 0.80
    # 文言/古白话的 mtld 远超上界：不抬分（这是防探针误伤的关键）。
    assert mtld_arbitration(0.62, scores, {"char_mtld": 161.8}) == 0.62
    # 低于下界（重复度高的文本）：不抬分。
    assert mtld_arbitration(0.62, scores, {"char_mtld": 55.0}) == 0.62
    # c 已够高或 b 不够高：交给 v11 的增强规则，本函数不动。
    assert mtld_arbitration(0.62, {"a": 0.0, "b": 0.93, "c": 0.70}, {"char_mtld": 85.0}) == 0.62
    assert mtld_arbitration(0.62, {"a": 0.0, "b": 0.80, "c": 0.52}, {"char_mtld": 85.0}) == 0.62
    # 只抬不压：已经高于 floor 时保持原值。
    assert mtld_arbitration(0.90, scores, {"char_mtld": 85.0}) == 0.90
    # 缺特征或特征退化：行为与 v11 一致。
    assert mtld_arbitration(0.62, scores, None) == 0.62
    assert mtld_arbitration(0.62, scores, {}) == 0.62


def test_apply_mtld_arbitration_updates_windows_in_place() -> None:
    window = FusedWindow(
        window_key="0:0",
        block_index=0,
        sentence_ids=[0],
        text="x",
        scores={"a": 0.0, "b": 0.93, "c": 0.52},
        p_fuse=0.62,
        delta=0.0,
        metas={"c": {"features": {"char_mtld": 85.0}}},
    )
    untouched = FusedWindow(
        window_key="0:1",
        block_index=0,
        sentence_ids=[1],
        text="y",
        scores={"a": 0.0, "b": 0.93, "c": 0.52},
        p_fuse=0.62,
        delta=0.0,
        metas={"c": {"features": {"char_mtld": 200.0}}},
    )
    assert apply_mtld_arbitration([window, untouched]) == 1
    assert window.p_fuse == 0.80
    assert untouched.p_fuse == 0.62


def _window(p_fuse: float) -> FusedWindow:
    return FusedWindow(
        window_key="0:0",
        block_index=0,
        sentence_ids=[0],
        text="x",
        scores={},
        p_fuse=p_fuse,
        delta=0.0,
    )


def test_composite_score_stays_inside_band_ranges() -> None:
    assert 0 <= composite_score(0.20, 0.30, "低") <= 34
    assert 35 <= composite_score(0.50, 0.70, "中") <= 64
    assert 65 <= composite_score(0.71, 0.82, "高") <= 100


def test_composite_score_is_monotonic_within_high_band() -> None:
    low = composite_score(0.60, 0.60, "高")
    high = composite_score(0.90, 0.90, "高")
    assert low == 65
    assert high > low


def test_doc_level_includes_composite_score() -> None:
    stats = doc_level([_window(v) for v in (0.2, 0.4, 0.6, 0.8, 1.0)])
    assert stats["median"] == 0.6
    assert stats["p90"] == 1.0
    assert stats["band"] == "高"
    assert stats["score"] == 100


def test_doc_level_empty_document_scores_zero() -> None:
    stats = doc_level([])
    assert stats["score"] == 0
    assert stats["band"] == "低"
