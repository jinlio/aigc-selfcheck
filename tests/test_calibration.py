"""切分协议的整组留出测试（v7 起）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_calibration.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_calibration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _items(sizes: dict[str, int]) -> list[dict]:
    return [
        {"sha256": f"{group}-{index}", "author": group}
        for group, count in sizes.items()
        for index in range(count)
    ]


class _FixedRng:
    """固定洗牌顺序，保证测试可复现；不引入随机性。"""

    def shuffle(self, seq: list) -> None:
        seq.sort(reverse=True)


class TestPickGroups:
    def test_empty_pool(self):
        module = _load()
        assert module.pick_groups([], 5, lambda p: p["author"], _FixedRng()) == []

    def test_whole_groups_only(self):
        module = _load()
        items = _items({"a": 3, "b": 2, "c": 4, "d": 1})
        picked = module.pick_groups(items, 5, lambda p: p["author"], _FixedRng())
        picked_groups = {p["author"] for p in picked}
        for group in picked_groups:
            assert {p["sha256"] for p in picked if p["author"] == group} == {
                p["sha256"] for p in items if p["author"] == group
            }
        assert len(picked) >= 5
        assert len(picked) <= 7  # 软上限 1.2×5 → 7

    def test_no_group_straddles(self):
        module = _load()
        items = _items({"a": 3, "b": 2, "c": 4, "d": 1})
        picked = module.pick_groups(items, 5, lambda p: p["author"], _FixedRng())
        picked_shas = {p["sha256"] for p in picked}
        picked_groups = {p["author"] for p in picked}
        remaining_groups = {p["author"] for p in items if p["sha256"] not in picked_shas}
        assert not (picked_groups & remaining_groups)
        for group in picked_groups:
            assert all(p["sha256"] in picked_shas for p in items if p["author"] == group)

    def test_huge_group_skipped_when_smaller_fits(self):
        module = _load()
        items = _items({"big": 20, "s1": 2, "s2": 2, "s3": 2})
        picked = module.pick_groups(items, 5, lambda p: p["author"], _FixedRng())
        assert "big" not in {p["author"] for p in picked}
        assert len(picked) >= 5
        assert len(picked) <= 7
