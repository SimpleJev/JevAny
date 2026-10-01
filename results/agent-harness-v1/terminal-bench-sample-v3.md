# Terminal-Bench 2 Jev harness — sample-v3 snapshot

Predeclared sample: 6 tasks / 12 runs; complete paired rows: 6/6.
Baseline executes the frontier LLM's selected candidate. Optional mode lets Jev choose bounded routine candidates while the frontier LLM retains diagnosis, edits, verification, and completion.
Positive `saved` percentages favor Jev optional mode; negative values are regressions.

## Per-task trade-off

| Level | Task | Status | Success B→J | Frontier calls B→J | Tokens B→J | Cost B→J | Time B→J | Jev replaced decisions |
|---|---|---|---:|---:|---:|---:|---:|---:|
| easy (historical 4/4) | `regex-log` | complete | 0→1 (+1) | 30→18 (+40.0%) | 186,712→196,687 (-5.3%) | $1.536→$1.328 (+13.5%) | 287.0s→178.8s (+37.7%) | 5/10 (50.0%) |
| easy (historical 4/4) | `sqlite-db-truncate` | complete | 1→1 (+0) | 17→15 (+11.8%) | 202,102→222,728 (-10.2%) | $1.271→$1.459 (-14.7%) | 146.2s→163.8s (-12.0%) | 9/13 (69.2%) |
| medium (historical 2/4) | `configure-git-webserver` | complete | 0→0 (+0) | 30→30 (+0.0%) | 267,467→308,860 (-15.5%) | $1.717→$1.942 (-13.1%) | 249.5s→326.0s (-30.7%) | 23/31 (74.2%) |
| medium (historical 2/4) | `llm-inference-batching-scheduler` | complete | 0→0 (+0) | 30→30 (+0.0%) | 572,053→600,218 (-4.9%) | $3.494→$3.655 (-4.6%) | 210.6s→302.6s (-43.6%) | 19/24 (79.2%) |
| hard (historical 0/4) | `query-optimize` | complete | 0→1 (+1) | 30→29 (+3.3%) | 366,138→348,370 (+4.9%) | $2.234→$2.177 (+2.5%) | 831.1s→937.1s (-12.7%) | 27/29 (93.1%) |
| hard (historical 0/4) | `path-tracing` | complete | 0→0 (+0) | 30→30 (+0.0%) | 574,798→361,174 (+37.2%) | $3.488→$2.227 (+36.1%) | 558.0s→339.3s (+39.2%) | 17/26 (65.4%) |

## Level aggregates

| Level | Complete/predeclared | Success B→J | Calls saved | Tokens saved | Cost saved | Time saved | Jev decision share |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy (historical 4/4) | 2/2 | 50%→100% (+50.0%) | +29.8% | -7.9% | +0.7% | +20.9% | 14/23 (60.9%) |
| medium (historical 2/4) | 2/2 | 0%→0% (+0.0%) | +0.0% | -8.3% | -7.4% | -36.6% | 42/55 (76.4%) |
| hard (historical 0/4) | 2/2 | 0%→50% (+50.0%) | +1.7% | +24.6% | +23.0% | +8.1% | 44/55 (80.0%) |
| all completed tiers | 6/6 | 17%→50% (+33.3%) | +9.0% | +6.0% | +6.9% | +1.5% | 100/133 (75.2%) |

## Quality-preserving Pareto frontier

Non-dominated among completed, quality-preserving rows with actual Jev delegation over reward delta plus calls/tokens/cost/time saved: `regex-log`, `query-optimize`.

## Demo candidate

`regex-log` is selected by the predeclared ranking: reward delta first, then frontier-call, cost, and time reduction.

- Outcome: 0→1; frontier calls 30→18; Jev executed 5 of 10 command decisions.
- Candidate surface: 19 candidate slots across 10 command turns; 9 turns exposed multiple options.
- The complete selected-command trace and raw result hashes are retained in the JSON companion.

| Turn | Subgoal | Routine | Options | Controller / selected label |
|---:|---|---|---:|---|
| 1 | Inspect /app directory | True | 2 | frontier: List and preview log |
| 4 | Enumerate typical folders | True | 2 | jev: list common dirs |
| 6 | Inspect logs staging environment | True | 2 | frontier: detailed listing |
| 7 | Look for log files | True | 2 | jev: list log subdirs |
| 9 | Draft and test regex | False | 2 | frontier: test regex |
| 11 | wait | True | 2 | jev: wait |
| 12 | Analyze failures | True | 2 | jev: think |
| 13 | Retest new pattern | False | 2 | frontier: test regex v2 |
| 14 | Save regex to file | False | 1 | frontier: save regex |
| 16 | Read regex from file and verify | True | 2 | jev: verify final |

Note: sample-v3 persisted each option count, the frontier index, and executed choice, but not the text of unselected options. A v4 trace is required for a fully replayable option-menu demo.

## Interpretation and provenance

- This is one exploratory paired attempt per task, not a statistically powered benchmark. Do not infer population-level significance.
- Difficulty is fixed from four retained historical frontier-agent runs (4/4, 2/4, 0/4), not assigned after observing these Jev results.
- `Jev decision share` is delegated commands / (delegated + direct commands), not Jev calls / frontier calls. Failed or low-confidence Jev calls therefore do not inflate replacement.
- Wall time uses Harbor trial start/finish timestamps. Bedrock dollars are the run-recorded estimates; Jev serving cost is not monetized.
- Every source result is preserved. The JSON companion records repository-relative source paths, SHA-256 hashes, exact model IDs, and selected-command traces.

Generated: `2026-10-01T04:21:13.265004+00:00`

Run root: `runs/terminal-bench/sample-v3`

Manifest: `configs/agent_task_ladder.json`
