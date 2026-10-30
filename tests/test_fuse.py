from aigc_selfcheck.constants import BAND_SETS
from aigc_selfcheck.fuse import (
    FusedWindow,
    apply_mtld_arbitration,
    band_distance,
    composite_score,
    doc_level,
    fuse_scores,
    mtld_arbitration,
)

GENERAL = BAND_SETS["general"]
ACADEMIC = BAND_SETS["academic"]


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
    assert 0 <= composite_score(0.20, 0.30, "低", GENERAL) <= 34
    assert 35 <= composite_score(0.50, 0.70, "中", GENERAL) <= 64
    assert 65 <= composite_score(0.71, 0.82, "高", GENERAL) <= 100


def test_composite_score_is_continuous_across_band_boundaries() -> None:
    """跨带不能跳崖：旧实现在低→中边界上从 34 直接跳到 57。"""
    for band_set in (GENERAL, ACADEMIC):
        just_low = composite_score(
            band_set.robust_low - 0.001, band_set.tail_low - 0.001, "低", band_set
        )
        just_mid = composite_score(
            band_set.robust_low + 0.001, band_set.tail_low + 0.001, "中", band_set
        )
        assert just_low == 34
        assert just_mid <= 36, f"{band_set.name}: 低→中 跳了 {just_mid - just_low} 分"
        just_mid_high = composite_score(
            band_set.robust_high - 0.001, band_set.tail_high - 0.001, "中", band_set
        )
        just_high = composite_score(
            band_set.robust_high + 0.001, band_set.tail_high + 0.001, "高", band_set
        )
        assert just_mid_high == 64
        assert just_high == 65


def test_composite_score_mid_band_spans_most_of_its_range() -> None:
    """中带要能用满 35–64：旧实现实际只能取到 55–64。"""
    mid_scores = set()
    for m in range(451, 600):
        for p in range(601, 849):
            median, p90 = m / 1000, p / 1000
            if GENERAL.band(median, p90) == "中":
                mid_scores.add(composite_score(median, p90, "中", GENERAL))
    assert min(mid_scores) <= 37
    assert max(mid_scores) >= 62


def test_composite_score_is_monotonic_within_high_band() -> None:
    low = composite_score(0.60, 0.60, "高", GENERAL)
    high = composite_score(0.90, 0.90, "高", GENERAL)
    assert low == 65
    assert high > low


def test_doc_level_includes_composite_score() -> None:
    # 用 10 个窗而不是 5 个：不足 min_windows 时会返回「样本不足」，那是另一条测试
    stats = doc_level([_window(v) for v in (0.2, 0.4, 0.6, 0.8, 1.0)] * 2, GENERAL)
    assert stats["median"] == 0.6
    assert stats["p90"] == 1.0
    assert stats["band"] == "高"
    assert stats["score"] == 100


def test_doc_level_empty_document_scores_zero() -> None:
    stats = doc_level([])
    assert stats["score"] == 0
    assert stats["band"] == "低"


def test_band_sets_disagree_on_academic_prose() -> None:
    """同一份人写学术文本：通用口径判中，学术口径判低——这正是反馈的核心。"""
    assert GENERAL.band(0.534, 0.658) == "中"
    assert ACADEMIC.band(0.534, 0.658) == "低"
    # 纯 AI 学术文本在两种口径下都必须是高
    assert GENERAL.band(0.863, 0.951) == "高"
    assert ACADEMIC.band(0.863, 0.951) == "高"


def test_doc_level_reports_insufficient_sample() -> None:
    """1–2 个窗的「P90」就是最大值本身，不能拿它判档。"""
    stats = doc_level([_window(0.9), _window(0.95)], ACADEMIC)
    assert stats["band"] == "样本不足"
    assert stats["n_windows"] == 2


def test_band_distance_counts_windows_needed() -> None:
    # 20 窗里 12 个低于 0.58、18 个低于 0.68：median 够、p90 差 2 个（需 19）
    values = [0.3] * 12 + [0.62] * 6 + [0.9] * 2
    distance = band_distance(values, ACADEMIC, target="低")
    assert distance["median"]["need"] == 11
    assert distance["median"]["have"] == 12
    assert distance["p90"]["need"] == 19
    assert distance["p90"]["have"] == 18
    assert distance["gap"] == 1
    assert distance["reachable"] is False


def test_band_distance_zero_gap_when_already_low() -> None:
    values = [0.2] * 20
    distance = band_distance(values, ACADEMIC, target="低")
    assert distance["gap"] == 0
    assert distance["reachable"] is True
