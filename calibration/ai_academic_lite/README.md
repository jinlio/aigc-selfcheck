# AI 学术章节探针（豆包 Lite，2026-10-26）

## 这是什么

产品定位是「论文上传知网等 AIGC 检测前自查」，所以除了人写论文段落，还需要
一组 AI 写的论文章节来测召回。本目录是 **report-only 探针**：不进入
`calibration.jsonl`，不参与切分、μ/σ、α 权重或 HOT 阈值调整；它只回答
「当前固定融合在学术 AI 文本上能抓到多少、哪些章节最容易漏」。

> **公开仓库收录范围**：只保留本 README、`manifest.jsonl`、`run-config.json`
> 与不含正文的汇总 `report.json`；`texts/` 与 `score-cache.jsonl` 不随库分发。

生成器只用了用户指定的豆包 Lite（`doubao-seed-2-1-lite-260915`），没有换
更贵的模型。样本由 `scripts/gen_ai_ark_lite.py --pack academic` 生成，
凭据只从环境变量 `ARK_API_KEY` 读取，源码与输出都不落密钥。

## 数据

- 规模：**140 段**，7 个章节 × 20 个跨学科主题。
- 章节：摘要、引言、相关工作、方法、结果、讨论、结论，各 20 段。
- 主题：计算机、教育、医学、经济、管理、心理、环境、材料、能源、交通等
  20 个跨学科题目轮转；每段提示词要求「只写该章节、100–400 字、不要标题、
  不要列表、不要解释」。
- 生成参数：`temperature=1.0`、`top_p=0.95`、`max_tokens=800`、
  `thinking={"type":"disabled"}`（关闭思考），服务端默认采样；每条样本
  文件头带模型名、提示词编号、主题、日期与 token 用量。
- 清洗：剥离 Markdown 代码围栏、平台自报字数、标题行；汉字数不在
  100–450 的样本自动重试。
- 质量：**140/140 成功，0 失败，0 重复**；汉字数 182–428（中位 307），
  句子数 1–6（中位 4），总 token 42,416，全部 `finish_reason=stop`。

## 复现

```bash
# 只打印计划，不发请求
PYTHONPATH=src D:/anaconda/envs/learning/python.exe scripts/gen_ai_ark_lite.py \
  --pack academic --loops 20 --dry-run

# 真实生成（需要 ARK_API_KEY）
ARK_API_KEY=... PYTHONPATH=src D:/anaconda/envs/learning/python.exe \
  scripts/gen_ai_ark_lite.py --pack academic --loops 20 \
  --out-dir calibration/ai_academic_lite --prefix arkacad
```

生成后跑探针（产品同款分句/滑窗/上下文/融合路径）：

```bash
PYTHONPATH=src D:/anaconda/envs/learning/python.exe scripts/probe_ark_lite.py \
  --data-dir calibration/ai_academic_lite --texts-subdir texts \
  --label "AI 学术（豆包 Lite 7 章节）" --kind ai --device cuda
```

## 结果（v10c 部署常量：α=0/0.2/0.8，μ/σ=4.20/0.68，HOT=0.75）

总体：**窗口召回 88.0%**（309/351），文档「高」带 **100%**（140/140），
段落中位 >HOT **91.4%**（128/140），窗口 P50/P90 0.863/0.951。
三路中位：a 0.000、b 0.995、c 0.829。

| 章节 | 段 | 窗 | 窗口召回 | 段中位 >HOT | 文档「高」 |
|---|---:|---:|---:|---:|---:|
| 摘要 | 20 | 34 | 94.1% | 100.0% | 100% |
| 引言 | 20 | 62 | 95.2% | 100.0% | 100% |
| 相关工作 | 20 | 61 | 95.1% | 100.0% | 100% |
| 讨论 | 20 | 47 | 93.6% | 95.0% | 100% |
| 结论 | 20 | 52 | 90.4% | 100.0% | 100% |
| 方法 | 20 | 47 | 78.7% | 90.0% | 100% |
| 结果 | 20 | 48 | 66.7% | 55.0% | 100% |

**主要短板是「结果」和「方法」**：结果章节的数值/图表描述短、句式碎，c 路
容易被拉低（b 中位仍 0.996）；方法章节的公式/流程描述更接近人写论文的
叙述节奏。文档级仍然 100% 判「高」，因为同一篇的多窗中位/P90 足够高；
但如果用户只贴一段「结果」或「方法」，漏检风险明显高于摘要/引言。
与 v8 相比窗口召回从 94.9% 降到 88.0%，主要因为 α 从 0.4/0.6 移到 0.2/0.8、
c 路权重变大，而 b 路在这些章节更稳；这是 v10c 用 b 路换公文人写安全的
代价，v11 需要用新的学术 AI 样本复测。

## 用途与局限

- 这组数字用于**观察跨章节召回**，不是 test split；不能据此调 α、μ/σ、
  HOT 或风格权重。
- 只有一个生成器（豆包 Lite），不能代表其他模型；提示词固定为 7 条，
  生成文本会带提示词诱导的模板性（例如「本文首先…其次…」），真实 AI 论文
  的分布可能更杂。
- 2026 年的模型输出，不能当作「AI 论文普遍长这样」的断言；它只是一个
  便宜、可复现的难例探针。
- 人写侧对照见 `calibration/academic/README.md`（45 段 HansPub 2020 论文），
  两侧合起来才能看误伤/召回的权衡。

## 文件

- `texts/`：140 个 `.txt`，文件头带生成溯源。
- `manifest.jsonl`：逐条记录路径、模型、提示词编号、主题、token 用量、sha256。
- `run-config.json`：本次生成参数。
- `score-cache.jsonl`：三路窗口分数缓存（report-only）。
- `report.json`：探针汇总与逐段明细。
