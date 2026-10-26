"""v0.1 临时校准常量 —— 全部标注 PROVISIONAL。

来源：M4 之前的案头估计，用于让管线先跑起来。校准集建齐后由
scripts/fit_constants.py（基于 build_calibration.py 的冻结切分）重新拟合替换，
替换后按 ALGORITHM.md §4 跑全量回归。数字本身没有承诺意义。

PPL 方向说明：ALGORITHM.md §2.3 的公式写作 σ((log PPL − μ)/σ)，但同节直觉
是「AI 文本过于流畅（PPL 低），比人写更顺分越高」——公式应为
p_ppl = σ((μ − log PPL) / σ)。本实现按直觉方向，公式以这里为准。
"""

# ── 融合权重（α 调整仅依据冻结的 train split + 分组 CV）──
# v3.1 决策（2026-10-26，scripts/cv_evaluate.py，全部 train-only）：
#   a 路在 LOGO-生成器 CV 中同口径 AUC 均值 0.432（v7 修正负类口径后的复算；
#   旧记录 0.159 是把 AI 融合分与 c 路人写负类混比，低估了 a 路），仍弱于
#   b/c；α 网格显示 0.2/0.4/0.4 的 train AUC 0.934 < 0/0.5/0.5 的 0.947，
#   故仍归零。a 路仍在 CLI 中运行并在报告「三路详情」里展示（信息价值保留），
#   只是不再参与融合。b 路同口径 LOGO 均值 0.926 为支柱；c 路 0.958 互补。
#   0/0.5/0.5 + μ=75分位：train AUC 0.872→0.901，同样误伤预算下召回 +10pp。
# v7 决策（2026-10-26，v7 train 268 人写窗 / 297 AI 窗，全部 train-only）：
#   选型网格在 train 上略偏 0.4/0.6（AUC 0.954 vs 0.947，召回 72.4% vs 71.4%），
#   但差距在单折噪声内，且换权重会破坏与 v6 的可比性；α 维持锁定值，
#   留待后续轮次用独立证据检验。b 路在豆包/DeepSeek 上相对弱（LOGO AUC
#   0.824/0.846），c 路是跨生成器主力（LOGO mean 0.958 / min 0.888）。
# v8 决策（2026-10-26，修正 C 路上下文口径后重算，全部 train-only）：
#   上下文改为「窗口起点之前的正文」（旧实现取重叠前窗尾部，把当前窗口
#   自身喂给模型，log-PPL 被系统性压低）。修正后 c 路 LOGO-生成器
#   mean 0.968 / min 0.903（b 路 0.926 / 0.824），α=0/0.4/0.6 在 train 上
#   全面优于 0/0.5/0.5：AUC 0.963 vs 0.954、召回 74.4% vs 73.1%、误伤
#   0.0% vs 0.7%；部署常量下逐作者最大误伤 0.0% vs 18.2%（个别技术博客作者）。
#   采纳 0/0.4/0.6。该决定只用 train，v7 test 已消耗、不再重跑。
# v9 决策（2026-10-27，v9 train 292 人写窗/393 AI 窗——新增 12 位统计公报作者与
#   4 个新语域 AI 组（MiniMax 暂存组整组留出 test，train 未见），全部 train-only）：
#   语料扩容后 c 路 AUC 0.970 已高于 b 路 0.923，b 路成为人写误伤主源（3 窗
#   误伤中 2 窗 b≥0.85）与豆包/DeepSeek 召回拖累（这些窗 c 高 b 低）。
#   α 网格扫描（0.50~0.90 步长 0.05，train 段落级口径）：α_c=0.80 双优——
#   AI 段落「高」召回 88.9%→94.8%，人写段落「高」误伤 7.4%→4.6%（8→5 段，
#   无新增误伤作者，全部为已知技术博客难例）；窗口级召回 78.1%→77.9% 基本持平、
#   窗口误伤 1.0%→0.7%。LOGO-作者聚合检查最差作者不翻转。采纳 0/0.2/0.8；
#   v9 test（MiniMax × 四新语域）以本配置一次性评测。
ALPHA = {"a": 0.0, "b": 0.2, "c": 0.8}
DISAGREE_DELTA = 0.5  # 三路两两最大差 > 0.5 → 「⚠ 三路分歧」，不做硬判定

# ── C 路内部合成（文档未定处，这里定为 PPL 为主、风格为辅）──
PPL_WEIGHT_IN_C = 0.6

# ── PPL 映射：人写窗口的 log-PPL 分布 ──
# v6 起参考模型换为 Qwen3.5-2B-Base（train-only 选型：v6 train 206 人写窗 /
# 264 AI 窗上，同口径召回 71%→75%、误伤 1.0% 持平、LOGO-作者 max 9.1% 过线，
# c 路 AUC 0.915→0.945；见 calibration/README.md v6 节）。
# v7 在扩容后的 train（268 人写窗，266 个有效 log_ppl）上重导：median=4.15、
# μ75=4.674、σ=0.814；四舍五入 4.67/0.81。固定常量对比 v6 的 4.65/0.84，
# train AUC 0.9484→0.9485、召回 71.7%→72.1%、误伤 0.7% 持平，LOGO-作者
# max 仍 9.1%。PPL 项的绝对值随基座模型变化，换模型必须重导这两个常量
# （缓存键含 model_id，旧分不会串用）。样本量仍小，PROVISIONAL。
# v8（2026-10-26）：修正上下文口径（窗口起点之前的正文，不再用重叠前窗
# 尾部）后在 v7 train 上重导：median=3.89、μ75=4.24、σ=0.68。沿用 μ75
# 规则取 4.24/0.68；相对 median 口径在同为 0.0% 误伤下召回 72.4%→74.4%。
# 旧 4.67/0.81 是「无上下文」口径，与新口径不可比；v7 test 保留为历史。
# v9（2026-10-27）：v9 train（292 人写窗，新增 12 位统计公报作者）重导
# median=3.84、μ75=4.20、σ=0.68；Δμ=-0.04 在拟合精度内，沿用 μ75 规则取值。
PPL_LOG_MEAN = 4.20  # μ_calib（75 分位拟合，Qwen3.5-2B-Base，v9 train/带上下文）
PPL_LOG_STD = 0.68  # σ_calib（Qwen3.5-2B-Base，v9 train/带上下文）
PPL_CONTEXT_TOKENS = 128  # 条件上下文长度（带上文算 PPL 更稳）

# 每个参考基座独立的 PPL 校准常量（μ75/σ，均在 v9 train + 带上下文口径下导出）。
# 换模型必须在此登记，否则会误用 2B 的 μ/σ；cache_id 已并入所用常量。
# 0.5B 仅作 CPU 回退：train c AUC 0.930 / 融合 0.944 / 召回 66.3% / 人写误伤 0.0%，
# LOGO-生成器融合 AUC mean 0.957 / min 0.869（豆包），低于 2B 的 0.972 / 0.898。
PPL_CALIBRATION: dict[str, tuple[float, float]] = {
    "Qwen/Qwen3.5-2B-Base": (PPL_LOG_MEAN, PPL_LOG_STD),
    "Qwen/Qwen2.5-0.5B": (4.13, 0.64),
}

# ── 风格特征 z 归一化 + 线性合成（v3.2 定权重，v4 刷归一化）──
# direction: +1 表示取值越大越像 AI；-1 表示越小越像 AI。
# 拟合方式：scripts/cv_evaluate.py --fit-style——人写 train 窗口拟合 z 的 mean/std，
# 再在（人写 + AI）train 窗口上拟合 L2 逻辑回归，coef → weight=|coef| / direction=sign，
# 截距 → STYLE_BIAS；等价于 stats_lm.style_score 的 σ(w·z+b) 形式（可解释线性不变）。
# v3.2 要点（嵌套分组 CV 选型，全部 train；见 calibration/README.md v3.2 节）：
#   1. v1 手设方向在多生成器语料上有三个反了：de_density / char_mtld / punct_diversity
#      （v1 依 GLM 单一生成器直觉设定；LR 按 6 生成器 72 窗口重定向）。
#   2. 新增节奏类特征：period_comma_ratio（句读节奏）、punct_density（标点密度）、
#      sent_initial_rate（句首连接词）。纯句长节奏特征（sentence_length_variance）
#      未过嵌套 CV，退出评分；特征仍计算供报告面板展示。
#   3. 选型网格（特征集 × μ 规则，先卡 LOGO-作者误伤 ≤ v3.1 基线 9%）：
#      pruned6+μ75分位 召回最高（mean 81.6%/min 60%）但未见作者误伤 max 18% 被否决；
#      pruned8+μ75分位 误伤 max 9% 且召回 mean 76.8%/min 47% —— 采纳。
#      μ=median 组合召回普遍差 6~10pp（PPL 信号让位过多），一并否决。
#   4. LOGO-生成器融合 AUC：pruned8 mean 0.957 / min 0.852（v3.1 为 0.750 / 0.438）。
# v4 决策（2026-10-25，v4 train 115 人写窗 / 11 生成器 172 窗，全部 train-only）：
#   --fit-style 全量重拟合（权重+归一化）被预设门槛否决——11 位作者下 LOGO-作者
#   误伤 max 18%（3 位技术博客作者各 1 个说明文式段落越线，段落级方差），
#   而 LOGO-生成器召回与锁权重配置持平（79.7% vs ~80%），无收益只有尾部风险。
#   故采纳：**锁定 v3.2 LR 权重/截距，仅把 mean/std 与 PPL μ/σ 刷新到 v4 train**
#   ——该配置在 LOGO-作者上 mean 0.8% / max 9.1%（最差为一位技术博客作者），通过 10% 门槛。
# v7 决策（2026-10-26）：--fit-style 全量重拟合再次被否决——pruned6/pruned8/full
#   的 LOGO-作者 max 18~22%，无组合满足 ≤10% 门槛；权重/截距继续锁定 v3.2。
#   归一化 mean/std 也未单独刷新：v7 与 v4 的作者集有重叠，刷新收益未过门槛。
# 值均 PROVISIONAL，语料扩容后由 --fit-style 重拟合。
# v9 决策（2026-10-27）：权重/截距继续锁定 v3.2（全量重拟合在 v9 上未重跑
# LOGO-生成器嵌套 CV，无证据支持解锁）；归一化 mean/std 刷到 v9 train
# （292 人写窗，新增 12 位统计公报作者），段高误伤 8→5 段、无新增误伤作者。
STYLE_STATS: dict[str, dict[str, float]] = {
    "period_comma_ratio": {"mean": 0.82, "std": 1.51, "weight": 1.372, "direction": -1},
    "punct_density": {"mean": 101.80, "std": 33.52, "weight": 1.174, "direction": -1},
    "de_density": {"mean": 43.45, "std": 21.46, "weight": 1.044, "direction": -1},
    "char_mtld": {"mean": 42.99, "std": 21.47, "weight": 0.873, "direction": 1},
    "colloquial_density": {"mean": 1.87, "std": 6.64, "weight": 0.648, "direction": 1},
    "punct_diversity": {"mean": 0.46, "std": 0.75, "weight": 0.604, "direction": 1},
    "connective_density": {"mean": 0.17, "std": 1.55, "weight": 0.530, "direction": 1},
    "sent_initial_rate": {"mean": 0.02, "std": 0.10, "weight": 0.324, "direction": 1},
}
STYLE_BIAS = -2.045  # LR 截距（v3.2 train-only 拟合，v4 锁定未重拟合，PROVISIONAL）

# ── 聚合与着色阈值（ALGORITHM §3.2/§3.3/§3.4）──
# v3.1 重定心（+0.10，train ROC 依据）：a 路归零 + μ 上移 75 分位后，融合分
# 分布整体上移（train 人写 median 0.244→0.336，AI median 0.605→0.773），
# 阈值族随分布平移。train ROC：0.75 处误伤 3~4/66（≈5%）、召回 50%，
# 与旧配置在 0.65 的工作点（误伤 3%、召回 40%）相比召回 +10pp。
HEATMAP_LOW = 0.45  # < 0.45 绿
HEATMAP_HIGH = 0.75  # > 0.75 红
BAND_ROBUST_LOW = 0.45  # 区间带：低 if robust<0.45 且 tail<0.6
BAND_TAIL_LOW = 0.6
BAND_ROBUST_HIGH = 0.6  # 高 if robust≥0.6 或 tail≥0.85
BAND_TAIL_HIGH = 0.85
HOT_RUN_THRESHOLD = 0.7  # 疑似段落区间：连续 ≥3 窗 > 0.7
HOT_RUN_LENGTH = 3
EXPLAIN_Z_THRESHOLD = 1.5  # 解释理由：|z| ≥ 1.5 的特征
EXPLAIN_TOP_K = 4  # PPL/文本证据/风格特征合计最多 4 条，避免理由刷屏
