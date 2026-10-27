# aigc-selfcheck

**AI-assisted writing is normal. Make the final draft sound like you before you submit.**

Most papers are no longer written entirely by hand: AI helps with research, outlining and polishing. The real problem is that the submitted draft gets flagged as "AI-generated", and you do not want to lower that flag by wrecking the structure, argument or data.

This tool works locally and does three things:

1. Finds the paragraphs and sentences most likely to be flagged as AI.
2. Explains *why* in plain language, quoting the exact words and phrases in your text.
3. Suggests concrete revisions **without changing structure, facts, data or references**. With enough hardware, a local base model also produces a sentence-level rewrite example tied to the original.

Feed it a paper (`.docx`, text-based `.pdf`, `.txt`, `.md`) and get a single-file HTML report: a 0–100 composite risk score, per-sentence heatmap, section risk map, revision plan, and before/after comparison. Your text never leaves your machine.

```bash
uv sync
uv run aigc-selfcheck paper.docx --profile balanced
# → selfcheck-report.html, open it in any browser
```

The goal is to reduce template-like traces left by AI assistance so the writing sounds more like you. It **does not guarantee** a pass on CNKI, Zhuque or any official detector.

## What makes it different

Most tools give you one percentage. This one is built around the decisions authors actually make before submission:

1. **Why was this flagged?** Not just "too many connectives" — it quotes the actual phrase ("随着……不断发展", "首先、其次、综上所述") and explains why that pattern reads like a template.
2. **Which paragraph should I fix first?** The revision plan ranks paragraphs by how much the document score would move if that paragraph alone were addressed.
3. **Which paragraphs should I not touch?** Formula/symbol-dense text, method/results/conclusion sections, high lane disagreement and very short windows get an explicit "verify manually first" note — do not delete experimental detail just to lower the score.
4. **What exactly should I change?** Rule-based advice covers template openings, empty summaries, parallel structures, repeated sentence starts and connective overuse. With `--llm-advice`, the local Qwen base model rewrites the highest-risk windows; output must share enough Chinese bigrams with the source and introduce no new numbers or effect claims (for example "robustness" or "accuracy" when the source does not mention them), otherwise it is discarded and the rule advice is used.
5. **Did the revision help?** `--compare-with` compares the old and new versions and reports score, band and section-level changes.

It is a **self-check reference**, not an oracle for CNKI, Wanfang, Zhuque or any other detection system. A low score does not guarantee a pass, and a high score does not prove AI authorship.

## What is in the report

- **0–100 composite risk score** (0–34 low / 35–64 medium / 65–100 high), band and evidence sufficiency.
- **Per-sentence heatmap**: click a sentence for plain-language reasons, quoted evidence, revision suggestions, per-lane scores and PPL details.
- **Section map**: median / P90 / composite score aggregated by heading.
- **Revision plan**: ranked by single-paragraph sensitivity, with "do not rewrite yet" notes.
- **Model rewrite example (experimental)**: generated only for the highest-risk few windows, labelled "verify manually", never applied automatically.
- **Calibration card**: v15 both-sides-fresh test and general metrics, sample sizes and caveats (v11–v14 figures are historical).
- **Before/after comparison**: a separate `selfcheck-compare.html`.
- Titles, author metadata, references, formulas, captions, quotes and tables are skipped automatically and shown greyed-out.

## Three profiles: runs on low-end machines, sharper on high-end

| Profile | Lanes | Device | Best for |
|---|---|---|---|
| `lite` | b + c (0.5B) | CPU only | Low-end laptops / tight memory; guaranteed to run, lower accuracy, rule advice only |
| `balanced` | b + c (2B) | auto | Default; good speed/accuracy trade-off, model rewrite examples on by default when an accelerator is available |
| `max` | a + b + c (2B) | auto | Enough hardware; full three-lane setup, can add `--enhanced` |

All profiles share the same fusion, bands and explanation layer. Only the lanes, base model and default advice mode change. A low-end machine still gets a complete report; a high-end machine gets higher recall and more specific advice. On pure CPU, `balanced`/`max` automatically skip the 2B rewrite and keep rule advice; pass `--llm-advice` explicitly to force it.

## How it works

Three detectors score independently, then fuse into a per-window score:

| Lane | Model | Notes |
|---|---|---|
| a | `yuchuantian/AIGC_detector_zhv2` (RoBERTa-wwm-ext classifier) | Sensitive to 2023-era model prose; has a generation gap with newer models |
| b | `mokawa3018/cmj-chinese-aigc-text-detector` (ONNX) | Chinese AIGC specialist, the main lane |
| c | `Qwen/Qwen3.5-2B-Base` PPL + statistical style features | LM perplexity plus explainable features: clause rhythm (period/comma ratio), punctuation density, lexical diversity, "de" density, etc. Three one-sided evidence features (digit/unit density and official-name density) can only lower the score when present, never raise it when absent. Directions and weights come from the train split (v11: base8 from v9, evidence features fixed-normalized, μ/σ 4.20/0.68; the 0.5B CPU fallback uses 4.13/0.64) |

Fusion is a weighted sum renormalized over available lanes; the document score takes the **median (baseline) and P90 (tail)** across windows to form a band. Runs of consecutive high windows are flagged as concentrated hot segments. Each sentence's risk comes from the highest-scoring window covering it; its reasons and suggestions are regenerated from the sentence's own text, so quoted evidence always appears in that sentence. Raw z/PPL details are folded under "technical details" so the default view stays readable.

The explanation layer first uses deterministic rules: it reads PPL and style z-values from lane c, then matches template openings, empty summaries, parallel structures, repeated sentence starts and overlong sentences in the source text, producing reasons with quoted evidence. With `--llm-advice`, the already-loaded Qwen base model rewrites the highest-risk windows under explicit constraints; the output must share at least two Chinese bigrams with the source and must not introduce any number or effect claim absent from the source, otherwise it is dropped and the rule advice is used.

Profiles only change which lanes run, which base model is loaded and whether model advice is generated. They do not change the fusion formula or band thresholds, so low-end and high-end machines are judged by the same standard; only the amount of evidence and the specificity of advice differ.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jinlio/aigc-selfcheck-public.git aigc-selfcheck
cd aigc-selfcheck
uv sync
```

Model weights total about 2 GB; download once before first use (fully offline afterwards):

```bash
uv run python scripts/download_models.py
uv run python scripts/download_models.py --cpu-fallback  # also fetch 0.5B for CPU-only fallback (~1 GB)
```

## Restricted networks

Models are fetched from Hugging Face. If huggingface.co is unreachable on your network:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1   # required: Xet transport bypasses the mirror and times out behind firewalls
```

`scripts/download_models.py` sets both by default. If pypi.org is unreachable during dependency install:

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

```bash
uv run aigc-selfcheck paper.docx --profile balanced   # recommended: b+c, 2B, auto device
uv run aigc-selfcheck paper.docx --profile lite       # low-end: CPU only, 0.5B, rule advice only
uv run aigc-selfcheck paper.docx --profile max        # high-end: a+b+c
uv run aigc-selfcheck paper.pdf                       # text-based PDF (scanned PDFs need OCR first)
uv run aigc-selfcheck article.md --out report.html    # custom output path
uv run aigc-selfcheck paper.docx --json               # machine-readable JSON on stdout (composable with --out)
uv run aigc-selfcheck paper.docx --llm-advice         # force local model rewrite examples
uv run aigc-selfcheck paper.docx --no-llm-advice      # disable the profile's default model advice
uv run aigc-selfcheck paper.docx --enhanced           # experimental enhanced recall (higher recall, more false positives)
uv run aigc-selfcheck new.docx --compare-with old.docx --out compare.html
```

Input formats: `.docx`, text-based `.pdf`, `.txt`, `.md`. Scanned PDFs need OCR first; encrypted PDFs must be unlocked. The `--json` payload uses schema `aigc-selfcheck/report@v1`; comparison output uses `aigc-selfcheck/compare@v1`.

`--enhanced` is an opt-in experimental layer: it adds a gray-zone meta-classifier on top of the fixed fusion. In v10c it raised AI recall on the academic probe (98.6% vs 94.9%) at the cost of human false positives (3.5% vs 2.6% of windows); the artifact has not been re-derived for v11 and remains off by default. Treat it as a review hint, not a verdict. See [calibration/README.md](calibration/README.md).

Device is picked as `cuda → mps → cpu` (Apple Silicon uses MPS); the ONNX lane always runs on CPU. A 10k-character document takes tens of seconds on M-series chips, minutes on pure CPU. Model rewrite processes at most the three highest-risk windows; it is on by default when an accelerator is available, and skipped automatically on pure CPU (about two minutes for three examples in our CPU test). Pass `--llm-advice` explicitly to force it. Generation or validation failures fall back to rule advice without blocking the report.

## Calibration status and limitations

See [calibration/README.md](calibration/README.md). **The public repository distributes code, self-generated AI samples and auditable metadata only**; human-written texts, polished derivatives, score caches, training logs and probe texts are not included, and the metrics below come from a private training archive. These metrics measure **how well this tool separates human from AI text**, not any platform's "AI rate". Different platforms use different models, thresholds and training data; a low local score does not guarantee a low platform score, and whether a revision actually lowers the platform's AI rate must be re-tested there. Key points:

- **Small sample**: the current private calibration set has **625 human paragraphs (227 author/institution groups)** — the v11-era 185 / 61 (tech blogs, academic essays, and 62 paragraphs / 31 groups of 2014–2021 official statistical bulletins, government work reports, audit/budget reports and white papers, each with pre-2021 writing history and verifiable dates; some pre-2021 posts were recovered from Common Crawl's 2021 archives) — plus **v14's 175 peer-reviewed Acta Electronica Sinica paragraphs / 61 first authors** and **v15's 101 more Acta Electronica Sinica paragraphs / 37 authors plus 140 Journal of Software (软件学报) paragraphs / 66 authors, all 2015–2021**; the 147 HansPub OA paragraphs were withdrawn as label noise (originals kept privately as probe material). **853 AI-generated paragraphs (30 generator labels**: GLM, DeepSeek, Kimi, Doubao, Tongyi, ChatGPT, Gemini, Claude, MiMo, GPT-6.1-sol web UIs, eight Bailian API models, four Ark API models, MiniMax persona prompts, three v11 fresh models and three v15 fresh models — kimi-k2.7-code, deepseek-v4-1-flash, doubao-seed-2-1-lite) and 6 AI-polished paragraphs. The public repository keeps only AI-generated samples and aggregated metadata; for compliance reasons, author identities, source sites and per-paragraph provenance are not published, and human, polished and probe texts are not included. The numbers below **do not generalize**.
- **Applicable register**: the calibration corpus is concentrated in **tech/academic-adjacent registers**. In v11 we re-ran the out-of-register human probe (modern literary fiction, modern essay, late-Qing fiction and classical prose; 100 paragraphs / 309 windows, report-only): window, paragraph-median and P90 false-positive rates are all **0%**; classical prose has the highest median (0.480). The sample is small and covers only the human side, so **do not treat results outside the target scenario as settled**; see [calibration/probe/README.md](calibration/probe/README.md).
- **v11 calibration (train-only + one-shot test)**: two changes were selected on train. First, `official_evidence` (density of official names such as Party congresses and state organs, leader names deliberately excluded) is a one-sided feature that lowers the score when present; it cut train human window FPR from 3.61% to 0.48% and fixed the 2019 Supreme Court report (6/8 → 0/8), at the cost of AI window recall 68.40% → 60.75%. Second, a high-confidence lane-b boost (`b≥0.98 and c≥0.45 → p_fuse≥0.80`) restored recall to 68.16% with human window FPR 0.72% and unchanged paragraph-high rates. A full style refit was again rejected by the unseen-author gate; base8 weights, μ/σ 4.20/0.68 and STYLE_BIAS −2.045 stay at v10c.
- **v12 mtld arbitration (train-only, shipped)**: eight candidate routes were tested and rejected (fused comma rule, relaxed lane-b boost, lane-c PPL weight, additive `mtld_evidence`, rhythm evidence into lane c, character n-gram fourth lane, rank-calibrated n-gram blending, paragraph-band threshold grid); see [calibration/README.md](calibration/README.md). The one adopted change repurposes `char_mtld` from an additive feature into a **tie-breaker for lane disagreement**: `b≥0.90 and c<0.60 and 70≤char_mtld≤110 → p_fuse=max(p_fuse,0.80)`. A high lane b says "looks AI", a low lane c says "doesn't look like the training register"; the upper bound 110 excludes classical/vernacular Chinese, whose lexical diversity is naturally higher (the two late-Qing probe windows that false-positived sit at mtld 161.8/262.5, while the highest mtld among AI windows caught by the rule is 108.4). On v11 train: AI window recall **68.2% → 69.5%** (853/1228, +16 windows, 15 of them literary) and AI paragraph-high **89.5% → 90.3%** (465/515, +4 paragraphs, no reverse crossings), while human window FPR 0.7%, human paragraph-high 11.5%, LOGO-author max 20%, all four probes 0% and academic-human 1.8% are **unchanged**. Constants `MTLD_B_HI/MTLD_C_LO/MTLD_LO/MTLD_HI/MTLD_FLOOR`; results are identical for `mtld_hi` 100–120 and `floor` 0.78–0.85, and `c_lo=0.65` raises human FPR to 0.96%, so 0.60 is kept. Because the v11 test was already consumed, this change was accepted on train plus external probes only and was **not** recomputed on test.
- **v14 academic corpus expansion + one-shot test (2026-10-27)**: the v13 to-do "expand academic human-written corpus" is done — scrapling (Chrome TLS impersonation + local real Chrome for the anti-bot challenge) opened two sources: *Acta Electronica Sinica* (电子学报) 2015–2021, direct https, 60 papers → **175 paragraphs kept**; HansPub, JS challenge finally beaten, 60 papers → 147 paragraphs **withdrawn** after train-only review (14 of 19 human false-positive windows came from it; every style-refit combination failed the unseen-author gate — OA templated academic prose is statistically indistinguishable from AI in feature space, i.e. label noise; originals stay in the private archive as a high-risk-style probe). Split rebuilt from the v12 frozen state: the **test side now includes 25 never-seen academic authors** (71 paragraphs / 189 windows, whole-group holdout); AI side had no new material so test groups were backfilled from old train (disclosed honestly as fit-level leakage only). Constants and decision layers unchanged. One-shot test: fused AUC **0.993**, AI window recall **79.8%**, AI paragraph-high **94.7%**, human window FPR **0.5% (1/189)**, academic human paragraph-high **1.4% (24/25 authors completely clean)**; train AUC 0.950, human FPR 0.5% (4/790). The v14 test is now consumed and the v14 split is demoted to history (its AI side was old-train backfill, so the 79.8% recall was systematically optimistic). Layout lessons (two-column line buckets, PUA font codepoints as decimal points/hyphens, half-width sentence periods, `citation_authors` field name) are in calibration/README.md.
- **v15 both-sides corpus expansion + first both-sides-fresh one-shot test (2026-10-27)**: scrapling expanded the human side further — 101 more Acta Electronica Sinica paragraphs / 37 unseen first authors (v14 pipeline, new seed, zero code changes) and a brand-new source, *Journal of Software* (jos.org.cn): 140 paragraphs / 66 unseen first authors, extracted from its **directly reachable fulltext HTML endpoint** (`/html/{year}/{issue}/{id}.htm`), which sidesteps the `_guard` JS challenge sitting on the PDF download path. The AI side, authorized by the user's Ark key, added three never-used generators (kimi-k2-250905 served as kimi-k2.7-code, deepseek-v4-1-flash-261017, doubao-seed-2-1-lite-260915), 45 paragraphs each in the v11 test composition (15 literary, 15 academic, 8 news, 7 official), plus 28 "narrative popular-science" paragraphs feeding two existing train groups (first train-side counter to the lane-b popular-science blind spot). New split (fingerprint `a60d49d8…`): test human side = 120 paragraphs / 53 author groups, **all never seen in training**; test AI side = 135 fresh-generator paragraphs + 41 backfilled. One-shot test: fused AUC **0.975**, human window FPR **0.0% (0/317)**, human paragraph-high **0.0% (0/120, all 53 groups clean)**, AI window recall **66.7% (318/477)** — the honest unseen-generator number — AI paragraph-high **86.9% (153/176)**; per generator: kimi 81% / deepseek-flash 73% / **doubao-lite 45% (literary register only 2% — 1/43 windows)**; polished median 0.783. Train: AUC **0.962**, human FPR **0.4% (5/1332)**, the five false positives scattered across five different sources with zero from Journal of Software — the label-noise gate that withdrew HansPub passes for both new sources. Constants and decision layers unchanged (no fitting operations this round). The v15 test is now consumed.
- **Current train figures (v15 split, v12 deployment constants)**: fused AUC **0.962**, human window FPR **0.4% (5/1332)**, AI window recall **69.1% (1125/1627)**; the v14-split figures (0.950 / 0.5% / 68.3%) and the v11-era figures (0.942 / 0.7% / 69.5%) refer to older splits and stay as history.
- **Known blind spot (500-paragraph Doubao Lite probe, first full run in v12)**: overall window recall is 72.6% and paragraph-high 88.0%, but it splits sharply by prompt type — argumentative 97.0%, method advice 91.2%, academic abstract 78.1%, technical explainer 77.1%, and **popular-science storytelling only 17.5%** (paragraph-high 46%). In that last bucket lane b's median collapses to **0.206** (versus 0.957–0.996 for the other four) because the model writes extended metaphors ("imagine the model as a jigsaw player"), while lane c stays normal at 0.648. The v12 mtld arbitration changes **zero** windows here (lane c is above its 0.60 gate), so this is a lane-b blind spot, not a fusion-layer issue; raising lane-c weight was already rejected in v12 route 3 because it pushed human paragraph-high from 11.5% to 18.2%. The probe is read-only and was not used to select any constant.
- **One-shot v11 test (consumed, never re-run)**: fused AUC **0.895** (a 0.436 / b 0.910 / c 0.798), human window FPR **1.1% (1/93)**, passing the acceptance line (AUC ≥0.85, FPR ≤10%). Recomputing the same cached scores with the deployment boost gives AUC **0.904** and AI window recall **68.9%**. AI paragraph-high is **89.6% (121/135)**; by generator glm-5-3-flash 91.1%, deepseek-v4-1-flash-economy 100%, doubao-seed-2-0-mini 77.8%; by register literary 71.1%, news 95.8%, official documents 100%, academic 100%. Literary window recall is only 23.8%. Human paragraph-high is **18.9% (7/37)**, all official documents; only one human window (Zhejiang) exceeds 0.75. **The window-level gate passes, but paragraph-level official-document false positives and literary AI misses remain clear gaps; the current version cannot claim 95% across all registers.**
- **Historical rounds**: v10 test was consumed on the rejected lean12 candidate (AUC 0.826, human FPR 15.2%) and is not comparable; v10c was never evaluated on it. v8/v9 results and the v8 context fix (μ/σ 4.24/0.68, α 0/0.4/0.6) are historical and cannot be mixed with v11. The P1 "cmj-blind → Mike0307 second opinion" rule hit 0/11 missed windows while adding 2 human false positives on v7 test and remains rejected.
- All calibration constants are provisional, centralized in `constants.py`; the style-weight fitting entry point is `scripts/cv_evaluate.py --fit-style`.

## Privacy and security

- Text never goes online, never uploads, never persists; the only network activity is downloading model weights.
- Model rewrite examples are generated entirely on your machine, with no cloud API; they are suggestions only and never replace the source automatically.
- All model loading uses `trust_remote_code=False` and fetches only from official Hugging Face endpoints.
- The corpus fetch script enforces a domain allowlist and private-address filtering; human-written, polished and probe texts are not committed (copyright).

## Disclaimer

This tool outputs a **heuristic risk estimate**. It is unrelated to CNKI, Wanfang, Zhuque or any official detection system, and does not predict any of their results. A low score does not guarantee passing; a high score does not prove AI ghostwriting. Treat it as a reference mirror, not a judge.

## Development

```bash
uv run pytest      # all tests use mock detectors; CI downloads no models
uv run ruff check src tests scripts
```

Calibration pipeline: `scripts/fetch_human_corpus.py` (human corpus with four fidelity gates) → `scripts/build_calibration.py` (seed=42 frozen split) → `scripts/evaluate.py` (AUC / false-positive rate / constant fitting).

Design documents: [ALGORITHM.md](ALGORITHM.md); training and calibration reproduction: [TRAINING.md](TRAINING.md).

## License

[MIT](LICENSE)
