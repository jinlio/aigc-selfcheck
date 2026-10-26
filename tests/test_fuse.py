from aigc_selfcheck.fuse import FusedWindow, composite_score, doc_level


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
