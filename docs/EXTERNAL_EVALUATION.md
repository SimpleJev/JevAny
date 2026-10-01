# External decision-suite evaluation

## Public-suite additions — October 1, 2026

### Typed Decisions

We ran the complete `LocalLLaMA/typed-decisions` test split at revision
`d0e2f0c4`: 400 cases, five questions per case and 2,000 scored decisions.
Every locally rerun model completed all 2,000 decisions. Runs used one H200,
batch size 1 and the checkpoint's shipped calibration. Latency is per
five-decision case; published comparator latency uses different hardware and is
not compared.
The soft gold label for each decision is the mean of three samples from a
teacher of roughly 4B-class capability. Accuracy is therefore argmax agreement
with that teacher-derived label, not objective correctness.
Rows follow the fixed common-cohort order used in both panels, rather than
being independently sorted by score.

| Locally rerun model | Source | Accuracy ↑ | KL ↓ | Brier ↓ | ECE ↓ | Median / case ↓ |
|---|---|---:|---:|---:|---:|---:|
| **JevAny-Qwen3.8-27B** | Ours | **72.80%** | 0.293 | 0.131 | 0.053 | 279.8 ms |
| JevAny-Muse-Glimmer-30B | Ours | 69.95% | **0.245** | **0.111** | **0.028** | 265.9 ms |
| JevAny-Qwen3.5-4B-Direct-Token | Ours | 67.20% | 0.435 | 0.179 | 0.082 | 89.4 ms |
| JevAny-Qwen3.5-4B | Ours | 63.50% | 0.479 | 0.218 | 0.115 | 85.5 ms |
| JevAny-Gemma-4B | Ours | 66.25% | 0.272 | 0.128 | 0.036 | 107.1 ms |
| OpenDecider-small | Open external | 66.65% | 0.210 | 0.117 | 0.075 | 136.5 ms |
| Bongard-mini | Open external | 59.65% | 0.256 | 0.132 | 0.073 | 163.9 ms |
| Jeff-Gemma4-E2B | Open external | 57.95% | 0.382 | 0.204 | 0.090 | 53.5 ms |
| Jeff-Qwen3.5-2B | Open external | 55.45% | 0.386 | 0.210 | 0.099 | 42.2 ms |
| Jeff-Qwen3.5-0.8B | Open external | 49.15% | 0.558 | 0.272 | 0.136 | 41.7 ms |

#### Supplemental Typed Decisions rows

Laya is retained as a request-conversion check but is not in the common
dual-panel cohort. The pinned dataset card also contains results that were not
rerun locally; their hardware, runtime and item-level outputs are unavailable
under the local protocol. OpenDecider and Bongard appear below only as
published references; the locally rerun rows above are used in the chart.

| Supplemental row | Accuracy ↑ | KL ↓ | Brier ↓ | ECE ↓ | Status |
|---|---:|---:|---:|---:|---:|
| Laya (`55cf4c4`) | 36.20% | 0.576 | 0.316 | 0.174 | Open; local full-split rerun |
| meraGPT Decider 1 | **76.80%** | **0.096** | **0.052** | 0.180 | Published only; not rerun |
| Liquid AI d1 | 74.20% | 0.475 | 0.155 | 0.124 | Published only; not rerun |
| TypeSafe Jev 1.13.0 | 72.70% | 1.442 | 0.148 | 0.144 | Published only; not rerun |
| Featherless Simple Jev | 71.60% | 0.488 | 0.176 | — | Published only; not rerun |
| prima-ratio + Gemma4 12B | 70.20% | 0.564 | 0.234 | 0.146 | Published open result; not rerun |
| OpenDecider-small | 67.10% | 0.211 | 0.117 | — | Superseded by local rerun above |
| Bongard-mini | 59.40% | 0.256 | 0.132 | 0.067 | Superseded by local rerun above |

The exact Laya rerun reproduces its published 36.2%, validating the request
conversion. Among our releases, 27B leads accuracy, Muse has the best
probability metrics, and direct-token improves over the 4B pointer by 3.7
percentage points. Benchmark-trained specialist checkpoints, including
OpenDecider-nano and the Laya Typed Decisions fine-tune, are excluded from the
zero-shot comparison.

### JevJudge-Public

For a common-input comparison, we selected the same ten current/open models
reported on Typed Decisions and evaluated all 724 records declared
`modality=text` in JevJudge-Public v0.3 at revision `4d576ded`. Every row uses
the same native state and typed question with labels and metadata hidden, and
the fixed order matches the chart. Native inputs are not truncated and share a
65,536-token context ceiling; every model completed 724/724 requests.

| Model | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| **JevAny-Qwen3.8-27B** | **66.44%** | **0.824** | **0.476** | 0.103 |
| JevAny-Muse-Glimmer-30B | 62.57% | 0.873 | 0.503 | 0.099 |
| JevAny-Qwen3.5-4B-Direct-Token | 58.56% | 0.915 | 0.516 | 0.089 |
| JevAny-Qwen3.5-4B | 57.87% | 0.997 | 0.554 | 0.074 |
| JevAny-Gemma-4B | 51.10% | 1.030 | 0.590 | 0.100 |
| OpenDecider-small | 54.97% | 0.971 | 0.547 | **0.033** |
| Bongard-mini | 51.24% | 1.048 | 0.594 | 0.068 |
| Jeff-Gemma4-E2B | 46.69% | 1.210 | 0.681 | 0.181 |
| Jeff-Qwen3.5-2B | 42.82% | 1.214 | 0.682 | 0.191 |
| Jeff-Qwen3.5-0.8B | 43.51% | 1.256 | 0.703 | 0.211 |

Kev lacks matching Typed Decisions results and Laya's runtime silently applies
`max_len=512` truncation, so these runs are supplemental and do not enter the
full-context dual-panel cohort:

| Supplemental model | JevJudge text | Status |
|---|---:|---|
| Kev-4B | 54.28% | Open full-context rerun; no matching Typed result |
| Kev-27B | 38.54% | Open full-context rerun; no matching Typed result |
| Laya | 38.67% | Not full-context comparable; silent `max_len=512` truncation |

This text-only slice represents four roles and has no quality-role records, so
it is a diagnostic rather than JevJudge's official five-role full-multimodal
headline. Earlier partial-context runs and an internal Qwen checkpoint remain
archived in
[`jevjudge-public.json`](../results/external-decision-evals-20261001/jevjudge-public.json)
but are not used for the main comparison.

#### Reproducibility and tracked provenance

- The chart source is
  [`external-zero-shot-v1.json`](../results/external-zero-shot-v1.json), and the
  deterministic renderer is
  [`plot_external_zero_shot.py`](../scripts/plot_external_zero_shot.py).
- The tracked artifact contains the aggregate rows needed to regenerate the
  figure. Raw local run locations and SHA-256 hashes are recorded for audit but
  are not included in the repository. JevJudge scores come from
  `runs/jevjudge-open-baselines-20261001/scores-full-context`; detailed reports
  and predictions are under the sibling `results-full-context` directory.
- Typed Decisions uses dataset revision `d0e2f0c4`, parquet SHA-256
  `4f294f…b647c`; JevJudge uses revision `4d576ded`, test SHA-256
  `5c50f8…e83cb` and suite-manifest SHA-256 `ee52a1…15b68`.
- Typed Decisions sends all five questions for a case in one serial request.
  JevJudge sends one question per request. Neither main protocol samples,
  truncates, or tunes calibration on the evaluation set.

### JevBench v1.5.4

JevBench v1.5.4 has 1,624 questions: 904 open and 720 sealed. Its method defines
the 904 open questions as 534 older questions, 120 new public drafts, and a
250-question public draw from the sealed pool. Of the older questions, 231 are
published and 303 remain unpublished, producing the reported 601 published
open total. However, the official page, API, repository history, releases and
Hugging Face Space expose only the original 231 prompts; no downloadable bundle
for the other 370 was published. The API contains system aggregates and
explicitly omits item-level fields.

The five JevAny releases remain directly comparable on those 231 downloadable
questions in the main [benchmark table](../README.md#evaluation). A complete
1,624-question result requires an
[official evaluation request](https://benchmarkheaven.com/jev-models/request-evaluation),
where the maintainer runs a fixed API or public offline checkpoint and returns
aggregate results. We therefore preserve the v1.5.4 official aggregates
without claiming a local 601- or 1,624-question rerun.

Kev has four open-weight, open-code entries in the official aggregate:

| Open system | Frozen checkpoint | v1.5.4 score A | Rank A | Completed |
|---|---|---:|---:|---:|
| Kev-4B | `jaredpalmer/kev-4b@qwen3` | **38.07** | 37 / 106 | 1,624 / 1,624 |
| Kev-8B | `jaredpalmer/kev-8b` | 34.15 | 38 / 106 | 1,624 / 1,624 |
| Kev-0.6B | `jaredpalmer/kev-0.6b` | 1.26 | 78 / 106 | 1,624 / 1,624 |
| Kev-0.5B | `jaredpalmer/kev-0.5b` | 0.00 | 92 / 106 | 1,624 / 1,624 |

The official evaluator ran these public checkpoints in an evaluator-owned
offline pod. Their weights and [code](https://github.com/jaredpalmer/kev) are
available, but the full item-level outcome cannot be independently reproduced
while the sealed questions remain private. The Kev-4B row is the older Qwen3
research preview, not the current Qwen3.5 checkpoint under the repository's
default revision.

No exact current JevAny release appears in the official aggregate. These Kev
values are official composite scores, not the accuracy metric used by the
downloadable 231-item public evaluation below.

[Pinned official aggregates and reproducibility boundary](../results/external-decision-evals-20261001/jevbench-v1.5.4.json)

### Jev Decision Index

The `multimodalart/jev-decision-index` Space at revision `7cdcea3d` is a static
aggregate registry and methodology page, not an item-level evaluation corpus.
It indexes 120,340 requests across 43 suites and 70 model rows, but does not
publish the request records or per-item predictions required for a new local
run. Its Space metadata also declares no license. It produces no new JevAny
score and is not included in the result showcase.

[Pinned index provenance](../results/external-decision-evals-20261001/jev-decision-index.json)

## Kev and earlier JevAny comparison matrix — September 27, 2026

The frozen recipe evaluates both released JevAny checkpoints and all 14 distinct
Kev checkpoints available through the main repositories and release tags on
September 27, 2026. Identical release aliases share a result. Unreleased
development branches are listed separately in the recipe.

The evaluation includes all public nonempty Kev development and test partitions,
the external SemIf, scienthoon, WANLI, TypeSafe, and ekzhang MMLU-Pro panels,
binding diagnostics, and JevBench's public easy, original, and hard tiers.
Historical versions retain separate reports. Training and calibration partitions
are excluded. Night-2 panels are included as training diagnostics for the later
Kev checkpoints that trained on them.

Some Kev evaluation partitions are deliberately private. Their names, expected
counts, hashes, and mirror revisions remain in `manifest.json` under
`unavailable`; reproducing them requires access to those private partitions.
JevBench results cover its 231 public decisions; its sealed leaderboard composite
is outside this public evaluation.

### Results

The completed public matrix contains 16 checkpoints × 67 panels. Each model
attempted all 22,219 unique requests, representing 56,677 original panel
records. Historical panels share identical requests, which are counted once in
the unique-request total.
Fourteen partitions across eleven private Kev suites remain unavailable.

Download the [full accuracy matrix](../results/decision-evaluation-v1/accuracy.csv),
[metrics and latency table](../results/decision-evaluation-v1/scores.csv),
[detailed JSON](../results/decision-evaluation-v1/comparison.json.gz), and
[provenance and unavailable partitions](../results/decision-evaluation-v1/manifest.json).
Empty accuracy cells on unknowable panels mean confidence-only evaluation.
Published references have empty cells where no matching result is available.

JevBench public accuracy uses all 48 easy, 72 original, and 111 hard items:

| Checkpoint | Easy | Original | Hard |
|---|---:|---:|---:|
| JevAny-27B-SFT | 100.00% | 98.61% | 70.27% |
| JevAny-27B-RLCR | 100.00% | 97.22% | 69.37% |
| Kev-0.5B | 95.83% | 48.61% | 30.63% |
| Kev-0.6B | 100.00% | 72.22% | 36.04% |
| Kev-0.8B | 100.00% | 80.56% | 36.94% |
| Kev-0.8B / night2-du-release | 100.00% | 73.61% | 33.33% |
| Kev-0.8B / v7-base | 100.00% | 72.22% | 32.43% |
| Kev-27B | 100.00% | 100.00% | 72.07% |
| Kev-4B | 100.00% | 93.06% | 54.05% |
| Kev-4B / night2-du-release | 100.00% | 90.28% | 48.65% |
| Kev-4B / qwen3 | 100.00% | 88.89% | 37.84% |
| Kev-4B / r8-documents-release | 100.00% | 93.06% | 45.05% |
| Kev-4B / v7-base | 100.00% | 90.28% | 46.85% |
| Kev-8B | 100.00% | 93.06% | 45.05% |
| Kev-9B | 100.00% | 90.28% | 55.86% |
| Kev-9B / v7-base | 100.00% | 90.28% | 55.86% |
| Jev 1.13.0 / published reference | 100.00% | 98.61% | 72.97% |

The official Jev row is copied from JevBench's published per-item outcomes;
Kev's published Jev reports are retained separately in
[the reference archive](../results/decision-evaluation-v1/official-jev-references.json).

Selected external panels for the current main checkpoints are below.
MMLU-Pro here is the separate 1,000-question ekzhang panel. Transfer-v9
contains its own 200-question MMLU-Pro slice. TypeSafe uses equal-case agreement;
its total-variation distance is included in the full metrics table.
Kev's published Jev MMLU-Pro 1,000 result remains an unpaired reference
because its source hash does not match the public sample.

| Checkpoint | MMLU-Pro 1,000 | WANLI-v2 | SemIf | scienthoon | TypeSafe agreement |
|---|---:|---:|---:|---:|---:|
| JevAny-27B-SFT | 66.80% | 74.25% | 94.44% | 72.85% | 76.79% |
| JevAny-27B-RLCR | 66.30% | 74.15% | 94.44% | 72.85% | 77.63% |
| Kev-0.8B | 23.30% | 60.18% | 72.22% | 53.38% | 54.61% |
| Kev-4B | 52.40% | 69.26% | 89.58% | 72.28% | 71.85% |
| Kev-9B | 50.70% | 73.95% | 90.97% | 75.49% | 72.79% |
| Kev-27B | 63.10% | 74.55% | 97.22% | 79.73% | 79.01% |

All-request scores count context rejections and out-of-memory failures as
wrong. Under this run's math-attention configuration and 80GB GPU limit,
each JevAny checkpoint ran out of memory on seven longstate-v2 records.
Their JevBench, transfer-v9, and MMLU-Pro panels had no out-of-memory failures.
Per-panel rejection counts and error details remain in the reports.
The worker used Transformers 5.17.0, PEFT 0.21.0, and Accelerate 1.15.0;
each run records its Python, PyTorch, checkpoint, precision, and GPU details.

This rerun gives both JevAny checkpoints 82.12% on transfer-v9's 1,046 clean,
knowable questions, compared with the earlier release's 82.41% for SFT and
82.31% for RLCR. Adapter and head hashes match that release, and the old
and current text encoders produced identical encodings on all 1,264 current
transfer records. The original per-item outputs and runtime records were
unavailable for this comparison, so the small aggregate differences have
no established cause. The earlier release measurements remain separate.

The first GPU queue gave ekzhang MMLU-Pro a larger context than the native
Kev protocol. The final results apply 384/1,024/2,048-token limits using
the verified context replay described below. Original GPU predictions
are reused only when the complete native encoding is identical; each
changed request retains its parent key and proof. Timing for reused
predictions is the original GPU timing.

## Reproduce

Use Python 3.12 and the local inference dependencies:

```bash
python -m pip install -e '.[local,multimodal]'
python scripts/build_external_eval.py \
  --sources data/external-sources --download \
  --out data/external-eval --allow-test
```

This reads the exact revisions in
[`recipes/decision-evaluation.json`](../recipes/decision-evaluation.json).
The explicit `--allow-test` enables the fixed test partitions; do not tune or
select checkpoints using these results. No sampling is performed. Original
file hashes and record counts are checked before conversion.

Run one model on a local GPU:

```bash
PYTHONPATH=data/external-sources/kev:$PYTHONPATH \
python -m jevany.external_eval \
  --suite data/external-eval --model jevany-27b-sft \
  --out runs/external-eval/jevany-27b-sft/shard-0
```

Repeat for each model ID in the recipe. Kev uses its pinned upstream
implementation, loaded from `PYTHONPATH`. Each checkpoint retains its shipped
temperature and default evaluation precision: fp32 for the smaller Kev models,
and the trained bf16 backbone for the 27B models. `--dtype` is an explicit
experimental override and is recorded in the run.

For multiple GPUs, give each process its own GPU and output directory. Use
`--num-shards N --shard-index I` to divide a model's queue. Resume with the same
arguments: completed predictions and rejections are retained, and only a torn
final JSONL write is repaired. Changing the model, source, suite, precision,
temperature, or shard assignment requires a new output directory.

Generate reports after all processes finish:

```bash
python scripts/report_external_eval.py \
  --suite data/external-eval --sources data/external-sources \
  --runs runs/external-eval --out runs/external-report
```

The output includes detailed JSON reports, `scores.csv` with one row per
model/panel and explicit metric names, and a wide `accuracy.csv`. Incomplete
panels have empty accuracy cells in both tables. Published official Jev
baselines have separate model IDs and a source column. Local model latency is
reported separately from the predictor's wall time. Local timing columns contain
measurements from local runs.

## What the scores mean

Labels and reference distributions are reserved for scoring. Inference receives
the question inputs. Identical ordered
requests with identical context limits share inference, while each original
panel keeps its own labels, membership, and denominator. Option permutations
remain distinct. Context limits come from each Kev manifest; JevBench uses the
8,192-token state/row and 16,384-token packed context. Inputs are not truncated.
The external ekzhang MMLU-Pro panel has no context manifest and follows
`kev.benchmark --data`: 384 state, 1,024 row, and 2,048 packed tokens.

For existing runs made with larger context limits,
`scripts/replay_external_context.py` can apply narrower limits using the pinned
native text encoder. It reuses an accepted prediction only after comparing the
entire encoding and verifying its length against the recorded GPU input.
New context rejections carry no prediction or GPU latency. Parent run hashes,
encoder provenance, and per-request encoding proofs remain in the corrected
run; the original files are preserved.

`all_requested_accuracy` counts rejected or missing knowable questions as wrong.
`answered_clean` reports accuracy, NLL, Brier, ECE, selective coverage, and ordinal
metrics only where a model returned a valid distribution. Missing predictions
mark a report incomplete. Unknowable questions measure confidence, not accuracy.
Raw, uncalibrated probability metrics are reported separately when logits and
the shipped temperature are available.

JevBench also uses its pinned native scorer, including its probability-sum
tolerance, lexicographic tie rule, ordinal metrics, and family summaries.
TypeSafe reports equal-case modal agreement and total-variation distance to the
reference distribution, with both all-row and answered-row values in `scores.csv`.
Each case has equal weight in these metrics.

Published official Jev results are copied from Kev's committed reports with
their source paths, hashes, API model identity, and measurement dates. They are
marked `published_by_Kev_not_rerun`. Matching normally requires an original
manifest hash; historical aliases require identical partition bytes and context.
Scienthoon's converted rows instead verify every question ID, ordered option
list, and label, with this weaker match recorded explicitly. The API alias may
not expose a provider revision. Unmatched results remain separate references.

JevBench also publishes Jev 1.13.0's per-item public outcomes. Those provide a
separate public-tier accuracy reference, marked `published_by_JevBench_not_rerun`.
These outcomes provide accuracy counts; calibration metrics require probability
distributions, which this source does not include.

Latency measures local model time on the recorded hardware. Cloud price and
the JevBench speed/cost composite are outside this evaluation's scope.
Existing JevAny multimodal and
interactive results remain documented in [EVALUATION.md](EVALUATION.md).
