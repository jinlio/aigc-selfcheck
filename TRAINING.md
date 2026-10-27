# 跨机续训指南（Windows / CUDA）

本仓库为**公开代码仓库**：只分发代码、AI 自产样本与可审计元数据。人写正文
（`calibration/raw/`）、润色派生样本（`calibration/polished/`）、冻结切分/评分缓存/训练日志
（`calibration/out/`）、探针正文与下载缓存（`calibration/probe/`、`calibration/academic/`）
以及额外 AI 学术语料（`calibration/ai_ark_lite/`、`calibration/ai_academic_lite/texts/`）
均不随库分发。文中这些路径指本地数据归档；仅凭本仓库无法直接复现历史指标，需按本文档
重新抓取/生成。

> 本本仓库由当前工作树生成**单一根提交**，不继承旧归档仓库的历史；旧归档仓库
> `jinlio/aigc-selfcheck` 仍保持私有，仅用于续训。若以后从旧仓库重新发布，仍需先
> 用 `git filter-repo`/BFG 清洗历史。

## 环境（Windows + CUDA）

```powershell
git clone https://github.com/jinlio/aigc-selfcheck.git aigc-selfcheck
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
  组批（`AIGC_BATCH_TOKEN_BUDGET`，默认 800），右 padding + attention mask，逐行
  `F.cross_entropy`；`evaluate.py`/`pipeline.py` 均按路整批送 GPU。
- **精度口径**：校准缓存是 checkpoint 原生 **bf16**（旧实现在 CPU 上
  `from_pretrained` 未指定 dtype，Qwen2.5 配置为 bf16），log_ppl 被量化到 bf16 分辨率。
  CUDA 默认精度因此也取 bf16（`_pick_dtype`），与校准/缓存一致；实验模型可用
  `dtype=bf16/fp16/fp32` 显式覆盖。
- **2026-10-27（v11）显存教训**：默认 1600 token / 8 窗会把 8GB RTX 4060 推到
  ~7.9 GiB 并回退共享显存，速度反而慢一个数量级。v11 起默认 token 预算降到 **800**、
  `max_batch` 降到 **4**，峰值显存约 6.0 GiB；仍可用环境变量调大。
- 实测速度（v7 train 578 窗）：Qwen3.5-2B-Base 约 19.4s（GPU，c 路补算）；
  v8 带上下文重算 578 窗约 20s 量级。v6 时 0.5B 约 5s、2B 约 26s（483 窗），峰值显存
  1.5 / 4.4 GiB。CPU 回退时 0.5B 约 45 分钟，GPU 是数量级提升；v8 的 0.5B
  独立常量在 4060 上重算 578 窗也在数十秒内（含模型加载），纯 CPU 冒烟测试
  4 窗约 35s（含加载）。
- Qwen3.5-2B-Base 是混合线性注意力架构，未装 `causal_conv1d` / `flash-linear-attention`
  时会走 PyTorch 参考实现（正确但慢）；当前速度已够用，未额外编译。
- **2026-10-27（v10c）实测**：c 路 GPU（conda `learning`，torch 2.6.0+cu124）预热后
  **24.57 窗/秒**，全量 1547 窗（含 test）**1.0 分钟**；同一批窗口 CPU（项目 `.venv`，
  torch 2.14.1+cpu）**0.065 窗/秒**，约 380 倍。CPU/GPU 同窗对比：Δscore 最大 0.0011、
  Δlog_ppl 最大 0.0088（bf16 口径一致），缓存可混用。预热脚本用 `AIGC_DEVICE=cuda`
  切换 c 路设备（默认 cpu，a/b 始终 cpu）。命令模板见本地数据归档 `V10_RUNBOOK.md`
  「运行环境（GPU）」。
- **教训**：`uv run` 会同步 lockfile 把 CUDA 轮子换回 CPU 版；任何 c 路长跑都必须显式用
  `D:/anaconda/envs/learning/python.exe`，不要用 `uv run`。长跑前先确认
  `torch.cuda.is_available()`，并检查是否残留多个 python 进程并行抢 CPU。

## 评分缓存（重要）

`calibration/out/score-cache.jsonl` 以 (路, 模型/常量指纹, 窗口文本, 可选上下文) 为键，跨平台可用。
a/b 两路不使用上下文，缓存键与常量无关，一直可复用；C 路指纹含校准常量，且**只在上下文非空时**
把上下文并入哈希——各段首窗（上下文为空）仍命中旧的无上下文条目。
- **v8（2026-10-26）指纹为 `Qwen/Qwen3.5-2B-Base@42370d51a253`**（μ=4.24/σ=0.68，
  历史）。上下文口径为 `contexts.preceding_context`：取**窗口起点之前**的正文
  尾部 100 字，落盘在 `calibration/out/calibration.jsonl` 的 `context` 字段；产品路径与
  校准路径共用该函数。库内 C 路条目已按 v8 常量 + 上下文全量重算 train 578 窗。
  指纹链：v3.1 `Qwen/Qwen2.5-0.5B@d2ac29e4bc15` → v3.2 `…@96f9602ea19d` →
  v4/v5 `…@50b5f5e7a76d` → v6 `Qwen/Qwen3.5-2B-Base@60884ef1fffe` → v7
  `…@f9b9ce144dc9` → v8 `…@42370d51a253` → v9 `…@42370d51a253`（μ/σ 未变）→
  v10 lean12 `…@c3edfda8ac9c`（被否决）→ v10c `…@eb1b9d9f0cb6` →
  **v11 现行 `Qwen/Qwen3.5-2B-Base@ec724a59bbf7`**（μ=4.20/σ=0.68，base8 保持
  v9 + 数字/单位/官方专名三个单侧证据特征；b 路增强是融合层规则，不进 c 缓存键）；
  CPU 回退 **`Qwen/Qwen2.5-0.5B@daba0858b64e`**（v8 独立 μ/σ 4.13/0.64，见下）；
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

## 评测纪律状态（截至 2026-10-27，v11）

- 当前 `calibration/out/calibration.jsonl` 指纹
  `df3f35119ee8ccc40ebc39916a61f99fbead1c36dbe4174313e7b78bc5b184ab`（v11 切分）：
  train human 148 段 / 416 窗、ai 515 段 / 1228 窗、polished 5 段 / 13 窗；
  test human 37 段 / 93 窗（15 组）、ai 135 段 / 341 窗（3 个 fresh 生成器）、
  polished 1 段 / 3 窗。v10 test 全部并入 train；v10 及更早指纹为历史。
- **v11 test 已于 2026-10-27 一次性消耗**（日志 `calibration/out/eval-v11-test.log`）：
  融合 AUC **0.895**（a 0.436 / b 0.910 / c 0.798）、人写窗误伤 **1.1%（1/93）**，
  过验收线。同一批缓存分数按部署融合（含 b 路增强）复算：AUC **0.904**、
  AI 窗召回 **68.9%**、人写窗误伤仍 1.1%。**禁止在当前切分上再跑 test，也不得
  据此调参**；v12 需先扩语料并重建切分。
- 评分缓存三路覆盖 v11 train 1657/1657、test 437/437；c 路指纹
  `Qwen/Qwen3.5-2B-Base@ec724a59bbf7`（0.5B 回退 `@daba0858b64e`）。
  v10c 的 `@eb1b9d9f0cb6` 与 lean12 的 `@c3edfda8ac9c` 已弃用，不再命中。
- 以下 v8/v9/v10 记录为历史口径，不得与 v11 横比，也不得据此调参。
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
- **v7 test 的 3 个人写误伤窗口（只读复核）**：2 段人写（同一作者的两个
  重叠窗，另一位作者一窗）都使用「不仅……而且……」「实际上」等
  议论文连接模板，`connective_density` 落在人写 train 的 100% 分位，b 路同时
  0.73–0.91；属于「学术博客议论」与「AI 议论文模板」的边界，不是数据泄漏。
  逐窗口位数值与复核记录保存在本地数据归档；test 已消耗，未据此调参。
- v7 历史（无上下文 c 路、α=0/0.5/0.5、μ/σ=4.67/0.81）：test 融合 AUC 0.976、
  人写误伤 4.2%、召回 89%（fresh 92% / 回填 82%）、b/c 0.974/0.945、a 0.483。
  只作历史记录，不能与 v8 横比，也不得据此调参。
- 条件救援规则（cmj<0.2 且 Mike0307 校准≥0.8 且 c≥0.6）在 v7 test 上命中
  0/11 个漏检窗，却新增 2 个人写误伤，**再次否决，不进产品**；
  Mike0307 test AUC 0.944、人写 ≥0.8 占 16.9%。v8 不重启该规则。
- 2026-10-26 第七轮扩容（v7）：人写 +28 段（8 位新作者，CC 2021 存档；剔除
  4 段新闻摘要/翻译与 2 段不完整引导句）；AI +20 段（百炼 API 4 模型 ×5：
  deepseek-v4-flash-0731、qwen3.6-flash、qwen3.7-plus、qwen3.8-flash）。
- 一切 α/μ/阈值/风格权重决策只允许用 train split 或分组 CV（`scripts/cv_evaluate.py`；
  风格路常量的拟合入口是 `uv run --no-sync python scripts/cv_evaluate.py --fit-style`，
  它只读 train，输出可黏贴进 constants.py 的 STYLE_STATS 块与嵌套 LOGO 校验）。
- **学术探针（2026-10-26，v8 历史口径，report-only）**：人写 45 段 / 15 篇 HansPub 2020 论文
  （114 窗）+ AI 140 段豆包 Lite 学术章节（351 窗）。默认融合在 AI 学术上窗口召回
  94.9%、文档「高」带 100%、段中位 >HOT 97.9%；人写学术窗口误伤 2.6%、文档「高」带
  11.1%、段中位 >HOT 0%。两组都不进入 `calibration.jsonl`，不参与切分/常量/阈值；
  v10c 重跑结果见 v10 节。
- **增强召回（`--enhanced`，实验性，v8 历史口径）**：通用 train 拟合的 24 特征元分类器，
  灰度门 0.55、阈值取人写 train 概率 99.5% 分位（0.8137）。通用 LOGO-生成器
  召回均值 85.2%（默认 75.0%）、LOGO-作者误伤均值 1.7% / 最差 11.1%（默认 0%）；
  学术探针上 AI 窗口召回 98.6%、人写窗口误伤 3.5%（默认 94.9% / 2.6%）。
  v10c 下探针内重拟合的增强数字见 v10 节；产品工件未随 v10c 更新。
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
- 2026-10-26 第五轮扩容（v5）：人写 +14 段（3 位学术语域新作者，CC 2021 存档，
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
    2 个命中（64%→67%）却新增 2 个人写误伤，**已否决，不进产品**。
  - 实验脚本 `_p0_lm_upgrade.py` / `_p1_b_candidates.py`（仓库外工作区），
    日志 `calibration/out/p0_lm_upgrade.log`、`p1_b_candidates.log`。
- 2026-10-26 第六轮扩容（v6）：人写 +26 段（8 位新作者，CC 2021 存档；另剔除
  9 段语域不符材料——翻译摘要/书摘转述/原文引用/数据报表），AI +20 段（DeepSeek v4.1-flash 网页、
  MiMo v2.6-flash 网页、GPT-6.1-sol 网页、Claude-Sonnet-5.5 网页，各 5 文体）；
  v6 切分已重建并完成一次性 test。
- 2026-10-26 第七轮扩容（v9，进行中）：人写 +13 段（gov 语域 12 段：国家 2019/2020、
  北京/上海/武汉 2020 统计公报与国务院 2021 政府工作报告，官方文件直连抓取、
  发布日期 ≤2021；另 1 段公版 classic 层）；AI +527 段暂存（MiniMax-M2.5 API，
  新语域提示词包 news/wiki/gov/literary 各 12 主题 × 2 提示词 × 6 轮，正文存
  本地数据归档 ai_register_mm/），其中每语域 10 段并入主语料。v9 切分（指纹
  27b5b983…）test 为「未见生成器 × 新语域」口径：ai/test 40 段全部为
  MiniMax-M2.5（109 窗），human/test 27 段（12 段全新 gov 作者 + 15 段旧
  train 回填，76 窗）；v8 test 段全部强制并入 train。
- 2026-10-27 v9 常量决策（全部 train-only）：
  - v9 train 评分（旧常量口径）：融合 AUC 0.966、窗口召回 78.1%（307/393）、
    窗口误伤 1.0%（3/292，全部科学网作者，其中 2 窗由 b 路 ≥0.85 驱动）；
    c 路 AUC 0.970 首次高于 b 路 0.923。分生成器段落「高」召回最差：
    豆包 75%、DeepSeek 网页版 80%、DeepSeek 93.3%——共性是 c 高 b 低，
    被 0.4b+0.6c 权重拖到线下（未过线 86 窗中 52 窗在 0.5~0.75 近失区）。
  - --fit 重导：μ75=4.20（旧 4.24，Δ 在拟合精度内）、σ=0.68 不变、
    风格归一化 mean/std 刷到 v9 train；权重/截距继续锁定 v3.2。
  - α 网格扫描（0.50~0.90，train 段落级口径）：α_c=0.80 双优——AI 段高
    88.9%→94.8%、人写段高误伤 7.4%→4.6%（8→5 段，无新增误伤作者，
    全部为已知技术博客难例尾部）；窗口级召回 78.1%→77.9%、误伤 1.0%→0.7%。
    LOGO-作者聚合检查最差作者不翻转。**采纳 α=0/0.2/0.8 + μ4.20 +
    风格归一化刷新**（cache_id 指纹 42370d51a253→f40c417121c1，0.5B
    回退键 73abc54f56f1→100f29181213）；守卫测试与黄金快照同步更新。
  - v9 test（MiniMax-M2.5 × news/wiki/gov/literary 四新语域，109 AI 窗 +
    76 人写窗）以新配置一次性评测（2026-10-27，结果 `calibration/out/eval-v9.log`，
    归档副本在本地数据归档）：
    - 验收线全过：融合 AUC **0.973**（a 0.621 / b 0.962 / c 0.899）、
      人写窗口误伤 **2.6%**（2/76）。AI **段落「高」召回 40/40=100%**
      （news/wiki/gov/literary 四语域各自 10/10，整组未见生成器）；
      窗口召回 70.6%（literary 窗级 43% 最低，段落级聚合后全过）。
      润色 test 1/1 段高（66.7% 窗）。
    - **诚实边界：公文体人写误伤**。test 人写段落「高」6/27=22.2%，
      6 段全部为官方公文（国家 2019/2020 公报、北京/武汉公报、国务院 2021
      报告；median 0.60~0.80，仅 2 窗过窗级线 0.75）；旧 train 回填
      段落（原 train 作者）全部落在低/中带。公文是「人写但高度模板化」
      的固有难点（PPL 低、句式规整），v10 计划：人写侧扩公文子语域
      （预算报告/法规正文/不同年份地区）+ 在 train 上重估段落带阈值；
      产品侧对公文类文档保持「疑似+特征解释」呈现并提示复核。
  - v9 训练侧最终口径（新常量，缓存复算）：train 窗召回 77.9%（306/393）、
    窗误伤 0.7%（2/292）、段落高 AI 94.8%（145/153）/ 人 4.6%（5/108）/ 润 4/5；
    train 分生成器段高最差：豆包 75%、DeepSeek 网页版 80%。
  - 暂存 held-out 探针（report-only）：从 487 段未并入的 MiniMax 暂存语料
    每语域抽 15 段（seed=42，共 60 段/158 窗），产品管线全量打分——
    段落「高」59/60=98.3%（gov 14/15，其余三语域 15/15）、「非低」60/60，
    窗口级 news 89% / wiki 58% / gov 68% / literary 22%（文学散文窗级
    分数低但段落聚合全部判出）。与 test 40/40 相互印证：MiniMax-M2.5
    段落级检测在未见段落上稳定复现。日志归档 probe-mm-staging-v9.log。
  - 跨语域人写探针重跑（新常量）：四个语域外人写语域（古典散文/晚清小说/
    现代随笔/现代文学，100 段 309 窗）窗口/段落/P90 误伤全部 **0%**
    （train 内对照 1%/1%/2%）；与 literary AI 段落 100% 召回形成干净的
    跨语域分离——公版书年代层人写不会被新语域 AI 的召回提升带出误伤。
    日志归档 probe-registers-v9.log。

### v10（2026-10-27）：公文误伤专项 + 数字证据两段式（v10c）

- 2026-10-27 第八轮扩容（v10）：
  - AI +302 段：MiniMax 拟人化提示词包 +142；doubao-seed-2-1-lite-260915 +40；
    doubao-seed-2-1-pro / glm-5.3 / kimi-k3.1-pro 各 +40（96 段生成后等距抽样）。
    弃用重复度高的 mm25 暂存 120 段与同家族 ds4p 40 段。
  - 人写 +28 段：统计公报 2014/2015/2017/2018、国务院 2018/2019 报告、审计署 2017、
    北京 2017/2018/2019 公报、财政部 2019 预算报告、粮食安全白皮书、两高 2019 报告
    （1 段预检否决）。最终 AI 455 段 / 人写 163 段（50 位作者）/ 润色 6 段。
  - v10 切分（指纹 `bace7673…`）：train human 128 段/351 窗、ai 335 段/792 窗、
    polished 5/13；test human 35 段/105 窗（16 组：14 个新公文作者 + 2 位科学网）、
    ai 120 段/299 窗（3 个 fresh 生成器 doubao-lite / doubao-pro / glm-5.3，各 40 段，
    覆盖 gov/literary/news/wiki/学术摘要/技术说明），polished 1/3。v9 test 全部并入 train。
- 根因（train 诊断）：公文 log-PPL 极低，**PPL 路对官方文体失效**
  （公文 p_ppl 中位 0.969 > AI 0.869），风格路正常（公文 p_style 0.119）；
  全局调 PPL 权重 / 阈值 / 双信号门控的 Pareto 前沿无法同时满足高召回低误伤
  （AI 84% 时公文误伤 17%；AI 89% 时公文误伤 50%）。
- **第一次尝试（lean12，已否决）**：base8 + `digit_density` / `num_unit_density` /
  `dunhao_density` / `formal_density` 四特征全 train 重拟合 LR，μ75=4.21、σ=0.88、
  STYLE_BIAS=-0.984。train 上确实好看：融合 AUC 0.9745、窗召回 74.2%、误伤
  0.9%（3/351）、段落带 AI 高 97.0%/人写高 10.2%。但跨语域探针暴露回归：
  **古典散文人写误伤从 0% 跳到 40%**（c 中位 0.570→0.834）。原因是全量重拟合
  把风格截距从 -2.045 抬到 -0.984，且 formal/digit 类特征对没有数字的文言文本
  形成负向惩罚。lean12 因此否决，其 v10 test 结果（AUC 0.826、人写误伤 15.2%）
  也只属于这个被否决的候选，不代表 v10c。
- **v10c 两段式（现行）**：base8 保持 v9 的 mean/std/weight/direction 不变，
  μ/σ 回退 4.20/0.68、STYLE_BIAS 回退 -2.045；只新增两个**单侧数字证据**特征：
  `num_evidence = min(num_unit_density,50)/50`、`digit_evidence = min(digit_density,60)/60`，
  mean=0/std=1 固定、direction=-1、weight 1.5/0.5，只在文本出现数字/单位时**减分**，
  缺席为 0、不加分。两段式 λ 网格（λnum 0–3、λdig 0–1，μ/σ 两组）在 train 与
  四个跨语域探针上联合筛选，选定 1.5/0.5：公文误伤压住，同时古典散文探针
  回到 0%。c 路指纹 2B `@eb1b9d9f0cb6` / 0.5B `@4dbdceea83cd`。
- v10c train 复核（部署常量）：融合 AUC **0.959**、窗口召回 **61.6%（488/792）**、
  误伤 **0.6%（2/351）**；段落带 AI 高 **296/335=88.4%**、人写高 **7/128=5.5%**、
  公文高 **2/12**、润色高 4/5；段落带分布为 AI 高/中/低 296/38/1、人写 7/48/73。
  LOGO-生成器融合
  AUC mean **0.979** / min **0.910**；LOGO-作者误伤 mean **0.6%** / max **20%**
  （国家统计局2020公报，与 v9 持平）。`--fit-style` 守卫（新增 `v10c_base8_ev`
  变体，证据特征固定归一化）：作者误伤 mean 1.2% / max 20%，仍无组合 ≤10%，
  维持现常量。**代价**：相对 lean12，窗口召回从 74.2% 降到 61.6%、段落 AI 高
  从 97.0% 降到 88.4%——这是为了跨语域安全主动接受的保守化。
- 外部探针（v10c 重跑，report-only）：
  - 跨语域人写 100 段/309 窗：四语域窗口误伤、段中位误伤、P90 误伤全部 **0%**；
    文言中位 0.480、晚清 0.439、现代散文 0.305、现代文学 0.311。
  - 学术人写 45 段/114 窗：窗误伤 **1.8%**、段中位 >HOT **0%**、文档「高」带
    **17.8%**；学术 AI（豆包 Lite）140 段/351 窗：窗召回 **88.0%**、文档「高」带
    **100%**、段中位 >HOT **91.4%**。增强模式（元分类器，report-only）q=0.99：
    AI 窗 93.2% / 段 95.7%，人写窗 1.8% / 段中位 0%；q=0.95：AI 窗 96.6% /
    段 97.9%，但人写窗升到 9.6%。增强模式仍默认关闭。
- **v10 test 的归属**：2026-10-27 的一次性 test（日志 `calibration/out/eval-v10.log`）
  是在 lean12 候选上消耗的：融合 AUC **0.826**、人写窗误伤 **15.2%（16/105）**、
  窗召回 65.2%、AI 段高 91.7%，未过验收线（AUC≥0.85、误伤≤10%）。失败集中在
  公文人写（14 个高段全部是最高法/审计署/国务院/财政部/国新办）与 doubao-pro
  文学 AI（文学窗召回 5.9%）。**v10c 是在 test 之后基于探针回归做的修订，没有在
  v10 test 上评测，也不允许回头再用该 test 评测**；v10c 的独立 test 留给 v11
  新切分。因此 v10c 当前只有 train + 外部探针证据，不能宣称通用 95%。
### v11（2026-10-27）：公文专名证据 + b 路增强 + 新整组留出 test

- 第九轮扩容：人写 +22 段（2020–2021 年最高法/最高检/审计署/财政部/国新办与
  江苏/浙江/广东政府工作报告，全部为 2021 年前发布、有连续手写记录的新机构/
  新年份组；人工逐段审读）。AI +195 段：glm-5.3 学术 30 段（train）、
  MiniMax-M2.5 文学 30 段（train）、3 个 fresh test 生成器各 45 段
  （glm-5-3-flash、deepseek-v4-1-flash-economy、doubao-seed-2-0-mini，
  覆盖文学/学术/新闻/公文）。最终人写 185 段/61 位作者（公文 62 段/31 组）、
  AI 650 段/27 个生成器标签、润色 6 段。
- 切分（指纹 `df3f3511…`）：train human 148 段/416 窗、ai 515 段/1228 窗、
  polished 5/13；test human 37 段/93 窗（15 组：22 段新公文 + 15 段旧 train
  回填）、ai 135 段/341 窗（3 个 fresh 组）、polished 1/3。v10 test 全部并入 train。
- **`official_evidence` 单侧特征**：官方专名千字密度（党代会/国家机关/固定政治
  表述；不含领导人姓名），`min(density,30)/30`，mean=0/std=1 固定、direction=-1、
  weight=3.0。λ 网格（0/1.5/3/4.5）在 train + 四个跨语域探针上联合筛选：
  λ=3.0 时 train 人写窗误伤 3.61%→0.48%、最高法2019 报告 6/8→0/8、公文窗
  误伤 0.89%、跨语域探针 0%；AI 窗召回 68.40%→60.75%、AI 公文 84.34%。
  该特征只压人写公文误伤，不提高 AI 召回。
- **b 路高置信增强**：`b≥0.98 且 c≥0.45 → p_fuse=max(p_fuse,0.80)`。网格
  b∈{0.95,0.97,0.98}、c∈{0.35..0.60}、floor∈{0.78,0.80,0.82,0.85}；选
  b=0.98/c=0.45/floor=0.80：相对基线人写窗误伤 0.48%→0.72%（+1 窗）、人写
  段高 11.49% 不变、AI 窗召回 60.75%→68.16%、AI 段高 89.51% 不变、学术 AI
  88.03%→93.16%、跨语域探针 0% 不变。更激进的 b=0.97/c=0.40 把误伤推到
  2.16%、b=0.95 系列 1.4%–2.2%，未采纳。`fuse_scores`、`cv_evaluate.fused`、
  `evaluate.py` 三处同规则。
- **重拟合均被否决**：`--fit-style` 在 v11 上所有特征集/μ 规则的 LOGO-作者
  max 75%–88%（最高法2019 折内 75%），无组合 ≤10%；稳健选型也没有候选满足
  「mean 不升、非离群 max 不升、离群不恶化」。base8 权重、μ/σ 4.20/0.68、
  STYLE_BIAS -2.045 维持 v10c。
- **v11 train（部署常量，含增强）**：融合 AUC **0.940**（a 0.378 / b 0.882 /
  c 0.902）、人写窗误伤 **0.7%（3/416）**、AI 窗召回 **68.2%（837/1228）**；
  AI 段高 **461/515=89.5%**、人写段高 **17/148=11.5%**、公文段高 **12/38=31.6%**、
  润色高 4/5。LOGO-生成器融合 AUC mean **0.965** / min **0.836**
  （doubao-seed-2-1-pro；b 0.909/0.756、c 0.951/0.814）；LOGO-作者误伤
  mean **0.4%** / max **20%**（审计署2017，1/5 窗）。最弱生成器：MiniMax-M2.5
  窗召回 53.4%/段高 84.9%、doubao-seed-2-1-pro 43.9%/75.0%、豆包 46.9%/75.0%、
  kimi-k3.1-pro 59.8%/85.0%。
- **一次性 v11 test（已消耗）**：`evaluate.py --allow-test` 一次性打分 437 窗，
  融合 AUC **0.895**（a 0.436 / b 0.910 / c 0.798）、人写窗误伤 **1.1%（1/93）**，
  过验收线；部署增强复算 AUC **0.904**、AI 窗召回 **68.9%**。AI 段高
  **121/135=89.6%**；分生成器 glm-5-3-flash 91.1%、deepseek-v4-1-flash-economy
  100%、doubao-seed-2-0-mini 77.8%；分语域文学 71.1%、新闻 95.8%、公文 100%、
  学术 100%，文学窗召回仅 23.8%。人写段高 **7/37=18.9%**，全部为公文
  （国新办2020白皮书、审计署2020、广东/浙江2021政府工作报告、财政部2020
  预算执行报告）；窗级仅浙江 1 窗 >0.75。润色 1 段/3 窗，段中位 0.783。
- **结论与 v12 方向**：窗口级验收过线，但段落级公文误伤与文学 AI 漏检仍是
  明确短板，不能宣称「所有领域 95%」。v12 需要新的人写作者/新 AI 生成器，
  旧 test 全部并入 train 后重建切分；优先做文学风格特征（句长分布熵/意象密度/
  重复 n-gram）、公文结构特征（条款编号/年份-机构引用）与段落带阈值重估。

## 数据分发边界（2026-10-26 已执行）

已移出仓库，保存在仓库外本地数据归档 `_aigc_selfcheck_private/2026-10-26-public-cleanup/`：

- 第三方人写正文、审读记录与下载缓存：`calibration/raw/`、`calibration/academic/{human.jsonl,human_texts,raw,report.json,score-cache.jsonl}`、`calibration/probe/{human.jsonl,score-cache.jsonl}`
- 逐段溯源与种子清单（含第三方文章 URL 与作者标识，不公开）：`calibration/manifest.jsonl`、`calibration/seeds.json`
- v7 test 人写误伤逐窗复核记录（含来源标识）：`calibration/fp-boundary-review.md`
- 派生润色样本：`calibration/polished/`
- 冻结切分、评分缓存与训练日志：`calibration/out/`
- 额外 AI 学术语料：`calibration/ai_ark_lite/`、`calibration/ai_academic_lite/{texts,score-cache.jsonl}`
- 内部计划/代理状态：`PLAN.md`、`ROADMAP.md`、`COMPETITIVE-LANDSCAPE.md`、`.zcode/plans/`、`.mimosa/`

保留：代码、测试、文档、`calibration/ai/`（AI 自产样本，含 v9 并入的 40 段 MiniMax 样本）、`AI_PROMPTS.md`、探针 README 与不含正文的汇总。作者身份与逐段来源一律只做聚合表述：本仓库不含作者名单、博主 ID 或第三方文章 URL 清单。`fetch_human_corpus.py` 已移除站点专属发现逻辑与站点白名单（种子自备 + `--allow-host`，内置白名单仅 gutenberg.org 与 web.archive.org）。

## 历史隔离

本本仓库由当前工作树生成**单一根提交**，不继承旧归档仓库的提交历史；旧归档仓库
`jinlio/aigc-selfcheck` 仍保持私有，仅用于续训。

如果以后要从旧归档仓库重新发布，仍需在全新 mirror clone 中清洗历史：

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

然后把重写后的仓库推到**新的本仓库**。不要把旧归档仓库直接改为 public；无论哪种方式，提交前都用 `git ls-files` 再核对一次没有私有正文。
