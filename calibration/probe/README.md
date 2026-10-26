# 跨语域人写探针（report-only）

本目录回答一个校准集覆盖不到的问题：**目标语域之外的人写文本，会不会被误判为 AI？**
探针只报告，不参与任何校准、切分、阈值、权重或常量决策；`human.jsonl` 不写入
`calibration/manifest.jsonl`，也不进入 `build_calibration.py` 的切分。

> **公开仓库收录范围**：只保留本 README 与不含正文的 `report.json`；`human.jsonl`、
> `score-cache.jsonl` 与原始文本不随库分发。

## 语域与来源

校准语料集中在科技/学术相邻语域。探针从 Project Gutenberg 中文公版书取四个
目标语域之外的人写语域，每域固定 25 段（100–400 汉字、至少 3 句，与校准语料
同一筛选口径）：

| 语域 | 书目 | 段数 |
|---|---|---|
| `modern_literary` | 鲁迅《彷徨》《阿Q正传》《野草》 | 25 |
| `modern_essay` | 朱自清《欧游杂记》、鲁迅《朝花夕拾》 | 25 |
| `late_qing_fiction` | 刘鹗《老残游记》、曾朴《孽海花》 | 25 |
| `classical_prose` | 《古文观止》、鲁迅《中国小说史略》 | 25 |

抽样用固定种子 `20261026`，按书内位置均匀铺开，避免只取开篇。正文为公版书，
`human.jsonl` 保留 `gutenberg_id`、源 URL、sha256 与抓取时间。

## 口径

- 每段独立成文档，走产品同款 `prepare_block` → 滑窗 → 上下文 →
  a/b/c 三路 → `fuse_scores`；上下文为窗口起点之前的正文（同 `contexts.py`）。
- 默认 2B 基座与 v8 部署常量；`HOT = 0.75`。
- 结果缓存单独放在 `score-cache.jsonl`，不与校准缓存混用。
- 文档级用 `doc_level` 的 median/P90；「段误伤」= 段中位分 > HOT，
  「P90 误伤」= 段 P90 > HOT，「窗误伤」= 单窗融合分 > HOT。

## v8 结果（2026-10-26，100 段 / 309 窗）

| 语域 | 中位分 | 窗误伤 | 段误伤 | P90 误伤 | a 中位 | b 中位 | c 中位 |
|---|---:|---:|---:|---:|---:|---:|---:|
| classical_prose | 0.423 | 2% | 0% | 4% | 0.010 | 0.129 | 0.570 |
| late_qing_fiction | 0.354 | 0% | 0% | 0% | 0.006 | 0.039 | 0.509 |
| modern_essay | 0.241 | 0% | 0% | 0% | 0.002 | 0.018 | 0.348 |
| modern_literary | 0.248 | 0% | 0% | 0% | 0.002 | 0.018 | 0.366 |
| **全部** | **0.289** | **0%** | **0%** | **1%** | 0.003 | 0.040 | 0.445 |

对照：train 内目标语域人写（98 段 / 268 窗，in-sample）中位分 0.335、窗误伤 0%、段误伤 0%。

- 四个语域的**段落中位分全部低于 HOT**；唯一越过 HOT 的是《古文观止》一段的
  P90（0.764，段中位 0.671，c 路 0.737、b 路 0.572），属于文言文的尾部风险。
- c 路是异语域分数的主要来源（文言 0.570、晚清 0.509），但融合后仍被 b 路拉低；
  a 路在所有语域都接近 0，没有区分度。
- **不要据此调参**：每域 25 段、且只有人写侧；AI 侧的 fresh 跨语域样本仍因
  缺少 `OPENAI_API_KEY` / `OPENAI_API_BASE_URL` 无法生成。探针的用途是界定
  当前版本在目标语域外的风险，不是训练数据。

## 复现

```bash
PYTHONPATH=src python scripts/build_probe.py
PYTHONPATH=src python scripts/probe_registers.py --device=cuda
```

`build_probe.py` 复用 `fetch_human_corpus.safe_fetch`（https + 主机白名单 +
私网/环回 IP 拦截 + 手动逐跳重定向）。Gutenberg 的 `ebooks/*.txt.utf-8`
会 302 到 http，被安全策略拒绝，故直接使用 https 的 cache 文本地址。
