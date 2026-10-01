# Agent-harness task and delegation ladder

Status: protocol design plus an evidence snapshot from 2026-09-30. A result is called
validated only when the referenced run already exists; proposed levels and pending
runs are not presented as results.

## Two independent axes

Task difficulty and Jev autonomy are different variables. A difficult task can use no
delegation, and a simple task can delegate almost every action. Results should therefore
be indexed by `(task tier, delegation level, frontier model, Jev checkpoint, protocol
version)` instead of placing all experiments on one ambiguous "easy to hard" ladder.

### Task tiers

| Tier | Environment | What it tests | Current examples |
|---|---|---|---|
| T0: controlled mechanism | Small, deterministic state/action space | Selection correctness, regressions, paired-seed sensitivity | FrozenLake, Sokoban |
| T1: bounded language/simulator | Longer language-conditioned plans, but finite affordances and resettable simulator state | Subgoal ordering and state tracking | ALFWorld; other games only as diagnostics |
| T2: structured interactive product | Open-text query plus bounded page controls and a task-specific verifier | Search/refinement under interface state | WebShop |
| T3: dynamic real web | Heterogeneous sites, popups, accounts, navigation, semantic page state | Robust browser interaction | WebArena frozen six-task sample |
| T4: open terminal | Open command space, persistent side effects, diagnosis, editing, and explicit verification | General software/system task completion | Terminal-Bench 2 easy/medium/hard strata |

These labels describe environment openness, not a guaranteed total order for every
individual task. Baseline success and historical pass rate remain the empirical
difficulty measures.

#### Why ALFWorld is not the high-difficulty endpoint

ALFWorld can be difficult because it requires a multi-step plan, object-state tracking,
and recovery from a wrong subgoal. It is nevertheless a bounded text simulator: actions
come from a small templated grammar, observations are compact, resets are deterministic,
and there is no live UI, authentication, package state, arbitrary file editing, or open
shell command space. It belongs at T1 as a useful compositional-planning test, not above
WebArena or Terminal-Bench as evidence of open-world harness capability. Existing
ALFWorld runs are excluded from difficulty claims until a predeclared sample and verifier
policy are recorded.

#### Why WebArena should not be increased now

The WebArena benchmark row is the already predeclared task set
`[264, 278, 300, 158, 4, 763]`. The popup-aware run reuses task 264 and is a mechanism
demo; it is not a seventh benchmark task. Adding tasks after seeing Jev outcomes would
change the estimand and invite selection bias. Any future expansion must be a separately
versioned, predeclared sample and must not overwrite the six-task row. Breadth should now
come from the already declared Terminal-Bench strata, not from silently growing
WebArena.

Terminal-Bench occupies T4 and keeps its internal historical strata:

- easy (4/4 historical pass): `regex-log`, `sqlite-db-truncate`
- medium (2/4): `configure-git-webserver`, `llm-inference-batching-scheduler`
- hard (0/4): `query-optimize`, `path-tracing`

The historical rate is used only to predeclare strata; it is not a claim about results
from the new candidate-selection harness.

### Delegation levels

At every level the frontier LLM owns task interpretation, reasoning, candidate
generation, exception recovery, verification, and the final completion decision. Jev
selects only among bounded options supplied by the LLM; it never invents a shell command
or browser action.

| Level | Jev role | Execution policy | Expected eligible-decision coverage |
|---|---|---|---:|
| D0: direct baseline | None | LLM emits options and its `frontier_choice` executes | 0% |
| D1: shadow judge | Chooses, but choice is logged and not executed | LLM choice executes; agreement and regret are measured | 0% executed |
| D2: selective delegation | Chooses only when the LLM explicitly opts in | One bounded action; frontier fallback on low confidence/error | 1–50% |
| D3: basic-decision default | Chooses every eligible routine/reversible decision | Normally one action; at most two only when the LLM declares the sequence composable | 50–85% |
| D4: bounded subgoal harness | Repeatedly chooses within an LLM-authored option library until a stop/exception condition | Several actions without a fresh frontier call; LLM resumes on novelty, risk, or verification | 85–100% of eligible decisions |

Coverage bands are reporting bins, not success criteria. The configured policy level and
the realized coverage must both be shown. The current Terminal-Bench candidate protocol
is in the D3 family: routine decisions go to Jev, precise non-routine edits can execute
directly, and composable execution is capped at two commands. The existing cross-domain
optional-delegation matrix is D2 because the frontier autonomously gates delegation.
D1 and D4 are proposed ablations and have no result yet.

An **eligible decision** is a low-risk, reversible action for which the LLM produced at
least two locally valid options. Examples include inspection commands, waiting versus
polling, choosing the next page control, or choosing among equivalent verification
steps. Precise file content, irreversible actions, security-sensitive operations,
singleton commands, and final completion are ineligible and remain with the frontier.

In the current implementation, `routine`/non-routine eligibility is declared by the
frontier model. The harness locally enforces the response schema, multi-option rule,
confidence fallback, short sequence cap, and narrow command-policy checks; it does not
semantically prove that arbitrary shell commands are reversible or safe. The
Terminal-Bench container limits task-side effects, but the harness is not a general
shell sandbox.

Report both:

```text
all-command replacement = Jev-executed commands / (Jev-executed + frontier-executed commands)
eligible coverage        = Jev-executed eligible decisions / all executed eligible decisions
```

The first metric is available in the current Terminal-Bench logs. The second requires an
explicit `eligible` field and should be added to future result schemas. Neither metric is
the same as frontier-call savings: D3 still calls the LLM to generate each choice set.
D4 is the level that can directly avoid repeated frontier calls by amortizing one LLM
subgoal/options response across several Jev decisions.

## Required efficiency/performance table

Every row must report quality and resource use together:

| Group | Required fields |
|---|---|
| Identity | task tier, benchmark/task ID, frontier model, Jev checkpoint, protocol version, seed/attempt |
| Quality | verifier reward/success, completion reason, quality delta versus D0 |
| Frontier efficiency | frontier calls, input/output/total tokens, estimated API cost |
| End-to-end efficiency | wall time, Jev calls and latency, separately reported Jev GPU/runtime cost when available |
| Delegation | configured level, all-command replacement, eligible coverage, direct/delegated commands, mean delegated horizon |
| Robustness | protocol repairs, Jev failures, low-confidence fallbacks, invalid actions, paired confidence interval when sample size permits |

All savings use `1 - optional / baseline`; positive means saved. A row must never hide a
negative metric behind an average score. Cold-start exclusions and unpriced cache/GPU
cost must be called out.

## Pareto-frontier and demo selection

Compute frontiers only within the same task, frontier model, checkpoint, and evaluation
protocol. A point is dominated if another point has no worse verifier quality and no
higher frontier calls, tokens, estimated API cost, or end-to-end time, with at least one
strict improvement. Jev runtime cost should be a separate axis until it is monetized.
Delegation coverage is a color/annotation, not an objective to maximize: more delegation
is useful only if quality and efficiency survive.

The publication rule is:

1. Freeze the sample and protocol before inspecting outcomes; retain and display every
   attempted cell, including failures and aborted infrastructure runs.
2. Require optional verifier success and no quality regression for an efficiency demo.
   A quality-recovery demo may instead require `optional > baseline`, but must disclose
   every resource regression.
3. For a validated headline, require paired repetitions and uncertainty intervals. A
   one-task or one-attempt result is explicitly a mechanism/demo, never a benchmark win.
4. Among eligible demos, rank by quality delta, then frontier-call reduction, token
   reduction, API-cost reduction, and wall-time reduction. Break ties with fewer repairs
   and fallbacks.
5. A runnable decision demo additionally requires the complete LLM option set, Jev's
   chosen index/confidence, executed action, observation, and the LLM's final verification.

Recommended showcase panels based on evidence available now:

- **Validated controlled efficiency:** GPT-5.6-sol + 27B Jev on FrozenLake, with success
  100% to 100%, 64.4% calls saved, 63.1% tokens saved, and 37.6% time saved. Label it T0
  mechanism evidence, not general-agent difficulty.
- **Real-web mechanism:** WebArena task 264 popup-aware smoke, where both passed and the
  observed trace reduced Opus calls 8 to 2 and tokens 76,674 to 13,407. Label it a
  single-task smoke; the frozen six-task aggregate is 50% to 50% with worse calls/tokens.
- **Open-terminal candidate trace:** reserve this panel for a completed final-protocol
  paired run with full candidate logging. The v3 `regex-log` run is an exploratory
  quality-recovery candidate (0 to 1 reward and 30 to 18 calls), but tokens rose and v3
  did not retain the complete candidate list required for the final demo.

## Existing evidence snapshot

The validated cross-domain matrix contains 96 paired comparisons (192 agent episodes)
across ten cells. Its
mixed results are important: the GPT/FrozenLake cell is a clear multi-metric win, while
several Sokoban cells get more expensive and the six-task WebArena aggregate has equal
success but no efficiency win. WebShop improves from 50% to 60% in one ten-pair cell,
but its confidence intervals cross zero. See
[`results/agent-harness-v1/formal-matrix.md`](../../results/agent-harness-v1/formal-matrix.md).

Terminal-Bench development history demonstrates why the delegation level must be
controlled rather than maximized:

- v1 `regex-log`: both verifier rewards were 1, but optional delegation increased calls
  20 to 30 and tokens 231,070 to 395,501; this is an over-delegation failure.
- v2 `regex-log`: both rewards were 1, with calls 14 to 12 and tokens 159,509 to 147,212.
- v2 `sqlite-db-truncate`: both rewards were 1, but calls rose 15 to 17 and tokens
  152,107 to 268,907. The same harness can help one task and hurt another.

The v3 sample is complete: 12 task-mode trials across the six predeclared tasks.

| Stratum/task | D0 reward / calls / tokens | D3 reward / calls / tokens | D3 command replacement | Status |
|---|---:|---:|---:|---|
| easy / `regex-log` | 0 / 30 / 186,712 | 1 / 18 / 196,687 | 5 / 10 = 50.0% | exploratory pair complete |
| easy / `sqlite-db-truncate` | 1 / 17 / 202,102 | 1 / 15 / 222,728 | 9 / 13 = 69.2% | exploratory pair complete |
| medium / `configure-git-webserver` | 0 / 30 / 267,467 | 0 / 30 / 308,860 | 23 / 31 = 74.2% | exploratory pair complete |
| medium / `llm-inference-batching-scheduler` | 0 / 30 / 572,053 | 0 / 30 / 600,218 | 19 / 24 = 79.2% | exploratory pair complete |
| hard / `query-optimize` | 0 / 30 / 366,138 | 1 / 29 / 348,370 | 27 / 29 = 93.1% | exploratory pair complete |
| hard / `path-tracing` | 0 / 30 / 574,798 | 0 / 30 / 361,174 | 17 / 26 = 65.4% | exploratory pair complete |

Across all six tasks, success was 1/6 for D0 and 3/6 for D3, with 9.0% frontier-call,
6.0% token, 6.9% estimated Bedrock-cost, and 1.5% wall-time savings. These are single
stochastic attempts and protocol-development evidence, not formal performance estimates.
v3 also spent many frontier calls repairing candidate schemas.
The implementation after v3 accepts valid singleton/non-routine proposals and records
complete candidate sets. At the time of that snapshot, no v4 outcome existed.

### Exploratory v4 delegation-rate sweep

A later six-cell mechanism sweep sampled two easy Terminal-Bench tasks at configured
eligible-decision rates 0%, 50%, and 100%. All six verifier runs passed. The useful
frontier came from `sqlite-db-truncate`:

| Target rate | Actual eligible coverage | Frontier calls | Tokens | Cost | Time |
|---:|---:|---:|---:|---:|---:|
| 0% | 0/2 (0.0%) | 15 | 202,050 | $1.292 | 187.9s |
| 50% | 1/4 (25.0%) | 11 | 161,261 | $1.009 | 155.1s |
| 100% | 2/3 (66.7%) | 8 | 121,293 | $0.764 | 144.7s |

Relative to 0%, the 100% target preserved reward while saving 46.7% of frontier calls,
40.0% of tokens, 40.9% of estimated Bedrock cost, and 23.0% of wall time. Actual
coverage stayed below target because low-confidence Jev returns fell back to the
frontier. This is a single-attempt demonstration, not a confidence-qualified result.

The `regex-log` sweep also passed all three rates, but the model mislabeled all basic
steps as non-routine singletons, so realized Jev coverage was 0% and higher configured
rates merely produced stochastic trajectory regressions. That row is retained as a
negative protocol diagnostic, not presented as a Jev comparison. The post-run parser
now rejects routine singleton menus, and the prompt explicitly requires basic
inspection/navigation/wait/compile/verification steps to expose 2-4 real options.
See [`results/agent-harness-v1/terminal-bench-frontier-v4.md`](../../results/agent-harness-v1/terminal-bench-frontier-v4.md)
for all rows and full candidate traces.

## Minimal next matrix

Do not run every benchmark in full. First run the predeclared six Terminal-Bench tasks at
D0 and D3 with the final protocol, preserving v1-v3. Add D1 shadow cheaply from the same
candidate traces where possible. Run D4 only on one easy and one medium task after D3 is
stable. This yields a useful task-by-delegation grid without expanding WebArena or
mistaking ALFWorld for the hard tier.
