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
- **Calibration card**: v8 general and academic-probe metrics, sample sizes and caveats.
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
| c | `Qwen/Qwen3.5-2B-Base` PPL + statistical style features | LM perplexity plus explainable features: clause rhythm (period/comma ratio), punctuation density, lexical diversity, "de" density, etc. Directions and weights come from logistic regression fitted on the train split of the calibration set (v3.2; reference model upgraded to 2B in v6, μ/σ re-derived to 4.24/0.68 in v8 after the context fix) |

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

`--enhanced` is an opt-in experimental layer: it adds a gray-zone meta-classifier on top of the fixed fusion. It raises AI recall (98.6% vs 94.9% on the academic probe) but also raises human false positives (3.5% vs 2.6% of windows). It is off by default and must be treated as a review hint, not a verdict. See [calibration/README.md](calibration/README.md).

Device is picked as `cuda → mps → cpu` (Apple Silicon uses MPS); the ONNX lane always runs on CPU. A 10k-character document takes tens of seconds on M-series chips, minutes on pure CPU. Model rewrite processes at most the three highest-risk windows; it is on by default when an accelerator is available, and skipped automatically on pure CPU (about two minutes for three examples in our CPU test). Pass `--llm-advice` explicitly to force it. Generation or validation failures fall back to rule advice without blocking the report.

## Calibration status and limitations

See [calibration/README.md](calibration/README.md). **The public repository distributes code, self-generated AI samples and auditable metadata only**; human-written texts, polished derivatives, score caches, training logs and probe texts are not included, and the metrics below come from a private training archive. These metrics measure **how well this tool separates human from AI text**, not any platform's "AI rate". Different platforms use different models, thresholds and training data; a low local score does not guarantee a low platform score, and whether a revision actually lowers the platform's AI rate must be re-tested there. Key points:

- **Small sample**: the current private calibration set has 123 human paragraphs (30 authors across digest, tech-blog and academic-essay registers, each with pre-2021 writing history and verifiable dates; some authors' pre-2021 posts were recovered from Common Crawl's 2021 archives), 153 AI-generated paragraphs (19 generators: GLM, DeepSeek, Kimi, Doubao, ChatGPT, Tongyi, Gemini, Claude, MiMo and GPT-6.1-sol web UIs plus eight models via the Bailian API) and 6 AI-polished paragraphs. The public repository keeps only the 153 AI-generated samples and aggregated metadata; for compliance reasons, author identities, source sites and per-paragraph provenance are not published, and human, polished and probe texts are not included. The numbers below **do not generalize**.
- **Applicable register**: the calibration corpus is concentrated in **tech/academic-adjacent registers** (technical exposition, academic argumentation, popular science, method advice — the "paper self-check" target scenario). Fiction, finance/legal commentary, news writing and other registers are **not calibrated**. In v8 we ran a one-off out-of-register human probe (modern literary fiction, modern essay, late-Qing fiction and classical prose; 100 paragraphs / 309 windows, report-only): every register's paragraph median stayed below the red line (0% paragraph FPR), while classical prose showed tail risk — 2% of windows and 1/25 paragraph P90 above 0.75. The sample is small and covers only the human side, so **do not treat results outside the target scenario as settled**; see [calibration/probe/README.md](calibration/probe/README.md).
- **Directional validation (historical v3 check; not re-run for v7/v8)**: three representative documents through the full pipeline produced a monotonic median gradient — human 0.238 (low), AI 0.788 (high), AI-polished 0.586 (middle). Both the lane-c context definition and the fusion weights changed in v8, so these figures are not a current-version measurement.
- **Test split (v7 historical result, 170 windows; not applicable to the current code)**: v7 ran a one-shot evaluation under the old **context-free** lane-c definition and α=0/0.5/0.5: fused AUC 0.976 (≥ 0.85 ✓); human false-positive rate 4.2% (3/71 windows); lanes b/c AUC 0.974/0.945; window recall 89% (85/96), split into 92% (58/63) on four fresh generators and 82% (27/33) on two whole old-train backfill groups; lane a AUC 0.483. **v8 fixed the lane-c context definition and re-derived the constants, so these numbers are historical only, not comparable with the current version and not to be used for tuning.** A v8 test requires a new corpus/split and has not been run.
- **v8 context fix and constants round (train-only)**: before v8 the lane-c cache key omitted context, and `pipeline.py` used the tail of the previous **overlapping** window as context — with a one-sentence stride this fed the current window back to the model and systematically depressed log-PPL. v8 now uses the text **before the window start** (`contexts.preceding_context`) and includes a context fingerprint in the cache key; the calibration set stores a `context` field so production and calibration share one definition. On the v7 train split (268 human / 297 AI windows) this re-derived μ75/σ = **4.24 / 0.68** and moved fusion weights from 0/0.5/0.5 to **0/0.4/0.6**: train AUC 0.954→0.963, recall 73.1%→74.4%, human FPR 0.7%→0.0%; LOGO-generator lane-c mean/min 0.977/0.920 and fused 0.972/0.898; LOGO-author median-refit FPR 0.0%. Lane-c cache id is `Qwen/Qwen3.5-2B-Base@42370d51a253`; the v7 4.67/0.81 and `@f9b9ce144dc9` remain as history. A full style-weight refit was still rejected by the unseen-author gate; weights and normalization stay locked.
- **v8 CPU fallback (train-only, report-only)**: the 0.5B model gets its own μ75/σ = **4.13 / 0.64** under the same train/context definition: lane-c AUC 0.930, fused AUC 0.944, recall 66.7%, human FPR 0.0%; LOGO-generator fused AUC mean 0.957 / min 0.869 (Doubao). It still clears the acceptance line but is clearly weaker than 2B (0.963 / 74.4% / min 0.898), so use it only on CPU-only machines where 2B is impractical; cache id `Qwen/Qwen2.5-0.5B@73abc54f56f1`.
- **Current weaknesses**: v8 train has a 0.0% human FPR, but that is a train-internal result and does not replace a test; LOGO-generator still shows Doubao (fused AUC 0.898) and DeepSeek (0.949) as weak spots, and lane b is only 0.824 on Doubao. Recall at 74.4% still has headroom, and the sample is small, so these numbers will move as the corpus grows. The P1 "cmj-blind → Mike0307 second opinion" rule hit 0/11 missed windows while adding 2 human false positives on v7 test and remains rejected; v8 does not revive it. The three v7 test human false positives were reviewed as a genuine "academic-blog argument with template connectives" boundary, not leakage (the per-window review record stays in the private training archive and is not distributed).
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
