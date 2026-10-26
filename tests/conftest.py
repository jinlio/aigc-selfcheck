"""共享夹具与黄金文本。"""

from __future__ import annotations

from pathlib import Path

import pytest

# 黄金样本：一段 AI 风格长文 + 一段口语化人写风格 + 参考文献结构块
GOLDEN_TEXT = (
    "本研究首先回顾了近年来中文文本生成技术的演进历程，其次分析了主流检测方法的局限性。"
    "与此同时，现有检测器在面对新型生成模型时表现出明显的性能下降，"
    "综上所述，构建一个多路融合的检测框架具有重要的现实意义。"
    "实验部分，我们在三个公开数据集上验证了方法的有效性，"
    "结果表明融合策略显著优于单一检测器，总体而言提升幅度达到八个百分点。\n"
    "\n"
    "折腾了一晚上环境，总算跑通了。说实话一开始没想到这么麻烦，官网文档写得含糊，"
    "论坛里有个帖子提了一句关键的坑——版本要对上。我试了下还真行，第二天就把测试补完了。\n"
    "\n"
    "参考文献\n"
    "[1] 张三. 中文检测方法研究[J]. 计算机学报, 2020.\n"
    "[2] 李四. 文本生成综述[M]. 北京: 某出版社, 2019.\n"
)


@pytest.fixture()
def golden_txt(tmp_path: Path) -> Path:
    path = tmp_path / "golden.txt"
    path.write_text(GOLDEN_TEXT, encoding="utf-8")
    return path


@pytest.fixture()
def mock_result(golden_txt: Path, monkeypatch: pytest.MonkeyPatch):
    from aigc_selfcheck.detectors import build_detectors
    from aigc_selfcheck.pipeline import analyze_document

    # 与 CI 对齐（ci.yml 设 AIGC_SELFCHECK_MOCK=1）：golden 快照必须可复现，
    # 本地有模型也不走真模型——否则快照依赖本机权重，CI 必挂。
    monkeypatch.setenv("AIGC_SELFCHECK_MOCK", "1")
    detectors = build_detectors("a,b,c", device="cpu")
    return analyze_document(golden_txt, detectors, cache=None)
