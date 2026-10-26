# 跨机续训指南（Windows / CUDA）

本仓库为**公开代码仓库**：只分发代码、AI 自产样本与可审计元数据。人写正文
（`calibration/raw/`）、润色派生样本（`calibration/polished/`）、冻结切分/评分缓存/训练日志
（`calibration/out/`）、探针正文与下载缓存（`calibration/probe/`、`calibration/academic/`）
以及额外 AI 学术语料（`calibration/ai_ark_lite/`、`calibration/ai_academic_lite/texts/`）
均不随库分发。文中这些路径指私有训练归档；公开仓库不能直接复现历史指标，需按本文档
重新抓取/生成。

> 本公开仓库由当前工作树生成**单一根提交**，不继承旧私有仓库的历史；旧私有仓库
> `jinlio/aigc-selfcheck` 仍保持私有，仅用于续训。若以后从旧仓库重新发布，仍需先
> 用 `git filter-repo`/BFG 清洗历史。

## 环境（Windows + CUDA）

```powershell
git clone https://github.com/jinlio/aigc-selfcheck-public.git aigc-selfcheck
cd aigc-selfcheck
uv sync                      # Python 3.12；lockfile 里的 torch 是 CPU 版
```

> **2026-10-26 实测坑（v7）**：`uv run` 会先同步 lockfile。一次 `uv run ruff check`
> 把手工装的 CUDA 轮子换成了 `torch 2.14.1+cpu`，导致 v7 test 首次尝试落到 CPU
> （在写入任何新分数前停止，见「评测纪律状态」）。之后 GPU 运行统一改用全局
> conda 环境 `D:/anaconda/envs/learning/python.exe`（torch 2.6.0+cu124、
> transformers 5.19.0）；项目 `.venv` 只做 ruff/pytest 等离线检查，命令一律加
> `--no-sync`。若要在 `.venv` 内跑 GPU，先装 CUDA 轮子，再用 `uv run --no-sync`，
> 不要直接 `uv run`。

- 本机实测（2026-10-26）：GPU 运行环境为全局 `D:/anaconda/envs/learning/python.exe`
  （torch 2.6.0+cu124，transformers 5.19.0），RTX 4060 Laptop 8GB（driver 610.74），
  `torch.cuda.is_available() == True`，空闲显存约 6.9 GiB；项目 `.venv` 当前是
  `torch 2.14.1+cpu`（被 `uv sync` 同步）。
- 设备解析 `cuda > mps > cpu` 已内建（`resolve_device`），插卡即用；`--device cuda` 可强制。
- 模型下载：直连 huggingface.co 可用则无需配置；被墙时设置
  `$env:HF_ENDPOINT="https://hf-mirror.com"`（代码里已 setdefault 同款镜像）。
- 测试不碰模型：`$env:AIGC_SELFCHECK_MOCK="1"; uv run --no-sync pytest -q`。

### GPU 批量与精度口径（2026-10-26 改造）

- 三路都有 `detect_batch`：a 路按 512-token 分块展平批处理；c 路按 token 预算自适应
  组批（`AIGC_BATCH_TOKEN_BUDGET`，默认 1600），右 padding + attention mask，逐行
  `F.cross_entropy`；`evaluate.py`/`pipeline.py` 均按路整批送 GPU。
- **精度口径**：校准缓存是 checkpoint 原生 **bf16**（旧实现在 CPU 上
  `from_pretrained` 未指定 dtype，Qwen2.5 配置为 bf16），log_ppl 被量化到 bf16 分辨率。
  CUDA 默认精度因此也取 bf16（`_pick_dtype`），与校准/缓存一致；实验模型可用
  `dtype=bf16/fp16/fp32` 显式覆盖。
- 实测速度（v7 train 578 窗）：Qwen3.5-2B-Base 约 19.4s（GPU，c 路补算）；
  v8 带上下文重算 578 窗约 20s 量级。v6 时 0.5B 约 5s、2B 约 26s（483 窗），峰值显存
  1.5 / 4.4 GiB。CPU 回退时 0.5B 约 45 分钟，GPU 是数量级提升；v8 的 0.5B
  独立常量在 4060 上重算 578 窗也在数十秒内（含模型加载），纯 CPU 冒烟测试
  4 窗约 35s（含加载）。
- Qwen3.5-2B-Base 是混合线性注意力架构，未装 `causal_conv1d` / `flash-linear-attention`
  时会走 PyTorch 参考实现（正确但慢）；当前速度已够用，未额外编译。

## 评分缓存（重要）

`calibration/out/score-cache.jsonl` 以 (路, 模型/常量指纹, 窗口文本, 可选上下文) 为键，跨平台可用。
a/b 两路不使用上下文，缓存键与常量无关，一直可复用；C 路指纹含校准常量，且**只在上下文非空时**
把上下文并入哈希——各段首窗（上下文为空）仍命中旧的无上下文条目。
- **v8（2026-10-26）现行指纹为 `Qwen/Qwen3.5-2B-Base@42370d51a253`**（μ=4.24/σ=0.68，
  风格常量沿用 v4）。上下文口径为 `contexts.preceding_context`：取**窗口起点之前**的正文
  尾部 100 字，落盘在 `calibration/out/calibration.jsonl` 的 `context` 字段；产品路径与
  校准路径共用该函数。库内 C 路条目已按 v8 常量 + 上下文全量重算 train 578 窗。
  指纹链：v3.1 `Qwen/Qwen2.5-0.5B@d2ac29e4bc15` → v3.2 `…@96f9602ea19d` →
  v4/v5 `…@50b5f5e7a76d` → v6 `Qwen/Qwen3.5-2B-Base@60884ef1fffe` → v7
  `…@f9b9ce144dc9` → v8 `…@42370d51a253`；CPU 回退
  `Qwen/Qwen2.5-0.5B@73abc54f56f1`（v8 独立 μ/σ 4.13/0.64，见下）；
  `cv_evaluate.py` 的
  `_LEGACY_C_CONSTANTS` 已登记 v7（4.67/0.81）及更早条目，meta 回退可用
  （log_ppl 沿用历史条目，特征按当前代码从文本重算，分数按当前常量离线重算，
  与生产路径逐位一致）。退化窗口（`log_ppl=None`）在历史回退路径按 `p_ppl=0.5`
  继续，不再中断分析。
- **按基座分表的 PPL 常量**：`constants.PPL_CALIBRATION` 以 model_id 为键登记
  (μ, σ)，`StatsDetector` 初始化时取用，`cache_id` 指纹也并入所用 μ/σ；换模型
  不会串用 2B 的常量或缓存。未登记的 model_id 回落到 2B 默认值，因此新增基座
  必须先在表里补条目并跑 `scripts/fit_c_model.py` 重导。
- **v8 修复的旧口径 bug**：此前缓存键不含上下文，而 `pipeline.py` 用「前一个重叠窗口的
  尾部」当上下文。滑窗步长 1 句，前窗尾部往往就是当前窗口自身，模型等于看着答案算 PPL，
  log-PPL 被压低约 3.6 nat（中位）。v8 改为窗口起点之前的正文后，μ/σ 从 4.67/0.81
  重导为 4.24/0.68。v7 的无上下文条目保留供历史复现；v8 已删除 @f9b9ce 下因错误上下文
  产生的孤儿条目。
- **上下文与历史回退的边界**：`_LEGACY_C_CONSTANTS` 的历史条目都是无上下文口径，只对
  空上下文行（首窗）可命中；带上下文行若当前指纹缺失，`cv_evaluate.load_train` 会报
  `[miss]` 并退出，要求先用当前常量重算 train，避免静默混用两种口径。
- 新增常量轮次后，把当时的 (模型, 常量元组) 追加进 `cv_evaluate.py` 的 `_LEGACY_C_CONSTANTS`。

## 评测纪律状态（截至 2026-10-26，v8）

- 当前 `calibration/out/calibration.jsonl` 指纹 `96932408…`：v8 只补写 `context` 字段，
  切分构成仍是 v7（human 98 train / 25 test 段 = 268 / 71 窗，ai 116 / 37 = 297 / 96 窗，
  polished 5 / 1 = 13 / 3 窗）；v7 指纹 `7e43be1b…` 为历史。改语料必须重建切分 = 新版本。
- **v7 test 已于 2026-10-26 一次性消耗**。v8 是分数口径变更（上下文 + α + μ/σ），v7 test
  数字与当前版本不可比；当前切分上禁止再跑 test（`evaluate.py` 默认拒绝，需显式
  `--allow-test` 才会打分）。v8 新 test 需先扩语料并重建切分。
- v8 train-only 决策（全部只读 train）：
  - c 路上下文改为**窗口起点之前的正文**（`contexts.preceding_context`，100 字），
    缓存键并入上下文；train 578 窗在 GPU 用 2B 重算。
  - μ/σ 重导：median 3.89 / μ75 4.24 / σ 0.68；沿用 μ75 规则取 **4.24 / 0.68**。
  - α 由 0/0.5/0.5 改为 **0/0.4/0.6**：train AUC 0.954→0.963、召回 73.1%→74.4%、
    人写误伤 0.7%→0.0%；LOGO-生成器融合 AUC mean 0.960→0.972（min 0.872→0.898）。
  - 风格路 `--fit-style` 重跑仍被否决（六种组合 LOGO-作者 max 18%~27%），
    **权重/截距与 v4 归一化维持不动**。
- 部署常量下的 v8 train 全量：AUC **0.963**、人写误伤 **0.0%（0/268）**、召回
  **74.4%（221/297）**；LOGO-生成器 c 路 mean/min **0.977/0.920**、融合 **0.972/0.898**
  （a 0.432/0.163、b 0.926/0.824）；LOGO-作者 median-refit 误伤 mean/max
  **0.0%/0.0%**，部署常量下逐作者 max 0.0%。
- **跨语域人写探针（v8，report-only）**：Gutenberg 公版书四个目标外语域
  （现代文学、现代散文、晚清小说、文言散文）各 25 段，共 100 段 / 309 窗，
  走产品同款窗口/上下文/融合路径。段中位分全部低于 HOT（段误伤 0%）；文言散文
  有 2% 单窗、1/25 段 P90 > 0.75（《古文观止》一段，段中位 0.671）。探针不进入
  manifest/切分，不参与任何调参；构建与复现见 `calibration/probe/README.md`。
  AI 侧 fresh 跨语域样本仍被凭据阻塞。
- **v7 test 的 3 个人写误伤窗口（只读复核）**：2 段人写（sciencenet-jiaoxue 两个
  重叠窗、sciencenet-liuyanzhu 一个窗）都使用「不仅……而且……」「实际上」等
  议论文连接模板，`connective_density` 落在人写 train 的 100% 分位，b 路同时
  0.73–0.91；属于「学术博客议论」与「AI 议论文模板」的边界，不是数据泄漏。
  详见 `calibration/fp-boundary-review.md`；test 已消耗，未据此调参。
- v7 历史（无上下文 c 路、α=0/0.5/0.5、μ/σ=4.67/0.81）：test 融合 AUC 0.976、
  人写误伤 4.2%、召回 89%（fresh 92% / 回填 82%）、b/c 0.974/0.945、a 0.483。
  只作历史记录，不能与 v8 横比，也不得据此调参。
- 条件救援规则（cmj<0.2 且 Mike0307 校准≥0.8 且 c≥0.6）在 v7 test 上命中
  0/11 个漏检窗，却新增 2 个人写误伤（wald、jiaoxue），**再次否决，不进产品**；
  Mike0307 test AUC 0.944、人写 ≥0.8 占 16.9%。v8 不重启该规则。
- 2026-10-26 第七轮扩容（v7）：人写 +28 段（8 位新作者，CC 2021 存档；剔除
  王宏琳 4 段新闻摘要/翻译 + 2 段引导句）；AI +20 段（百炼 API 4 模型 ×5：
  deepseek-v4-flash-0731、qwen3.6-flash、qwen3.7-plus、qwen3.8-flash）。
- 一切 α/μ/阈值/风格权重决策只允许用 train split 或分组 CV（`scripts/cv_evaluate.py`；
  风格路常量的拟合入口是 `uv run --no-sync python scripts/cv_evaluate.py --fit-style`，
  它只读 train，输出可黏贴进 constants.py 的 STYLE_STATS 块与嵌套 LOGO 校验）。
- **学术探针（2026-10-26，report-only）**：人写 45 段 / 15 篇 HansPub 2020 论文
  （114 窗）+ AI 140 段豆包 Lite 学术章节（351 窗）。默认融合在 AI 学术上窗口召回
  94.9%、文档「高」带 100%、段中位 >HOT 97.9%；人写学术窗口误伤 2.6%、文档「高」带
  11.1%、段中位 >HOT 0%。两组都不进入 `calibration.jsonl`，不参与切分/常量/阈值；
  详见 `calibration/README.md` 的「学术语域探针与增强召回」节。
- **增强召回（`--enhanced`，实验性）**：通用 train 拟合的 24 特征元分类器，
  灰度门 0.55、阈值取人写 train 概率 99.5% 分位（0.8137）。通用 LOGO-生成器
  召回均值 85.2%（默认 75.0%）、LOGO-作者误伤均值 1.7% / 最差 11.1%（默认 0%）；
  学术探针上 AI 窗口召回 98.6%、人写窗口误伤 3.5%（默认 94.9% / 2.6%）。
  未通过「LOGO 召回均值 ≥90%」门槛，**默认关闭**；工件由
  `scripts/export_meta.py` 导出到 `src/aigc_selfcheck/data/meta_v1.json`，
  默认路径不加载，报告只把它标为复核线索。
- **v9 待办**：把学术 hard negatives（5 个「高」带段落 + 公式密集方法段）与
  抽样的 AI 方法/结论段并入 train 之前，必须先准备**全新人写作者**与**全新
  AI 生成器**的 test 组；当前 AI 学术样本只有豆包 Lite，不能当 fresh test。
  旧 test 强制进 train，重导 μ/σ/α，LOGO 复核后新 test 只跑一次；学术探针
  保留为未参与训练的第三组外部复核。
- **待办（v8/v9）**：fresh 生成器样本（当前环境无 `OPENAI_API_KEY` / `OPENAI_API_BASE_URL`，
  `calibration/ai` 153 段已全部入库、无备用样本，AI 侧扩容被凭据阻塞）；把 v7 test
  的 3 个人写误伤窗口作为 train 侧的语域边界案例复核（test 已消耗，不再据此调参）。
  0.5B CPU 回退已完成：独立 μ/σ 4.13/0.64，`--c-model Qwen/Qwen2.5-0.5B` 可切换，
  见 `PPL_CALIBRATION` 与 `scripts/fit_c_model.py`。跨语域人写探针已完成
  （report-only，100 段 / 309 窗，见 `calibration/probe/README.md`）：段中位分
  全部低于 0.75，文言散文有 2% 单窗 / 1 段 P90 尾部风险；AI 侧 fresh 跨语域
  样本仍被凭据阻塞。

### 历史轮次（v6 及以前）

- v6 切分指纹 `f30895f6…`；v6 test 已消耗（融合 AUC 0.982 / 人写误伤 0.0% /
  召回 64%；b/c 0.957/0.960）。v6 常量轮把 c 路参考模型换为 Qwen3.5-2B-Base，
  μ/σ 4.65/0.84；风格路全量重拟合被否决（max 18%~50%），风格常量维持 v4。
- 2026-10-26 第五轮扩容（v5）：人写 +14 段（科学网姬扬/王德华/武夷山，CC 2021 存档，
  已审读）；AI +26 段（用户网页批 50 段去重后——批 1 的 24 段与第四轮逐字节相同
  被排除，实增通义批 1 全部 5 段 + 豆包学术摘要 1 段 + 晚间批 2 全部 20 段）。
- 2026-10-26 GPU 实验轮（P0/P1，全部 train-only，未动常量、未碰 v5 test）：
  - **P0 参考模型**：Qwen3.5-2B-Base 通过判据——v5 train 召回 74%→78%（167→176/227），
    人写误伤 1.4% 不变，LOGO-作者 max 9.1% 不变，c 路 AUC 0.922→0.953，
    融合 AUC 0.943→0.953，峰值显存 4.4 GiB；Qwen2.5-1.5B 仅 +1 点（75%）。
    v6 train 复核（483 窗）同样通过：71%→75%，c AUC 0.915→0.945。**v6 test 已确证，
    自 v6 起生产默认 2B**；0.5B 仅作 CPU 回退（v8 已配套独立 μ/σ 4.13/0.64）。
  - **P1 b 路候选**：三个异源检测器作为 drop-in 替换均不通过——Hello-SimpleAI
    AUC 0.477（对 2026 年文本失效）；qiubin 输出饱和（人/AI 均 100% @0.5，无模型卡）；
    Mike0307 校准版 AUC 0.903、救回 75% b 失明窗，但人写误伤 25.2%、LOGO max 25%。
    混合 b'=(1-w)·cmj+w·Mike 救不回失明窗（α=0.5 下需 b'>0.86），已放弃。
    条件救援规则（cmj<0.2 且 Mike 校准≥0.8 且 c≥0.6 → 标记）在 v6 test 上只新增
    2 个命中（64%→67%）却新增 2 个人写误伤（乔中东），**已否决，不进产品**。
  - 实验脚本 `_p0_lm_upgrade.py` / `_p1_b_candidates.py`（仓库外工作区），
    日志 `calibration/out/p0_lm_upgrade.log`、`p1_b_candidates.log`。
- 2026-10-26 第六轮扩容（v6）：人写 +26 段（8 位新作者，CC 2021 存档；另剔除
  刘进平/孙学军/钟茂初引文/王启云报表共 9 段），AI +20 段（DeepSeek v4.1-flash 网页、
  MiMo v2.6-flash 网页、GPT-6.1-sol 网页、Claude-Sonnet-5.5 网页，各 5 文体）；
  v6 切分已重建并完成一次性 test。

## 公开仓库数据边界（2026-10-26 已执行）

已移出仓库，保存在仓库外私有归档 `_aigc_selfcheck_private/2026-10-26-public-cleanup/`：

- 第三方人写正文、审读记录与下载缓存：`calibration/raw/`、`calibration/academic/{human.jsonl,human_texts,raw,report.json,score-cache.jsonl}`、`calibration/probe/{human.jsonl,score-cache.jsonl}`
- 派生润色样本：`calibration/polished/`
- 冻结切分、评分缓存与训练日志：`calibration/out/`
- 额外 AI 学术语料：`calibration/ai_ark_lite/`、`calibration/ai_academic_lite/{texts,score-cache.jsonl}`
- 内部计划/代理状态：`PLAN.md`、`ROADMAP.md`、`COMPETITIVE-LANDSCAPE.md`、`.zcode/plans/`

保留：代码、测试、文档、`calibration/ai/`（AI 自产样本）、`manifest.jsonl`、`seeds.json`、`AI_PROMPTS.md`、探针 README 与不含正文的汇总。

## 历史隔离

本公开仓库由当前工作树生成**单一根提交**，不继承旧私有仓库的提交历史；旧私有仓库
`jinlio/aigc-selfcheck` 仍保持私有，仅用于续训。

如果以后要从旧私有仓库重新发布，仍需在全新 mirror clone 中清洗历史：

```bash
git clone --mirror https://github.com/jinlio/aigc-selfcheck.git aigc-selfcheck-old.git
cd aigc-selfcheck-old.git
git filter-repo --invert-paths \
  --path calibration/raw \
  --path calibration/out \
  --path calibration/polished \
  --path calibration/probe/human.jsonl \
  --path calibration/probe/score-cache.jsonl \
  --path PLAN.md \
  --path .zcode/plans
```

然后把重写后的仓库推到**新的公开仓库**。不要把旧私有仓库直接改为 public；无论哪种方式，提交前都用 `git ls-files` 再核对一次没有私有正文。
