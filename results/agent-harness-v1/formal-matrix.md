# Jev optional-delegation agent benchmark

Validated 96 paired seeds across 10 model/benchmark/checkpoint cells.
The frontier model autonomously chose between direct action and bounded Jev delegation.

| Frontier model | Benchmark | Jev | n | Success base → optional | Calls saved | Tokens saved | Bedrock $ saved | Time saved | Jev calls/episode |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `us.anthropic.claude-opus-4-7` | `frozen_lake` | 4B | 10 | 100% → 90% | +55.9% | +51.4% | +45.6% | -14.0% | 0.70 |
| `us.anthropic.claude-opus-4-7` | `frozen_lake` | 27B | 10 | 100% → 90% | +51.5% | +44.6% | +38.3% | -21.8% | 0.70 |
| `us.anthropic.claude-opus-4-7` | `sokoban` | 4B | 10 | 100% → 100% | -2.1% | -33.2% | -31.7% | -28.7% | 0.70 |
| `us.anthropic.claude-opus-4-7` | `sokoban` | 27B | 10 | 100% → 100% | -29.5% | -84.4% | -76.2% | -51.0% | 0.60 |
| `us.anthropic.claude-opus-4-7` | `webarena` | 27B | 6 | 50% → 50% | -5.6% | -28.2% | -26.8% | +0.4% | 0.17 |
| `us.anthropic.claude-opus-4-7` | `webshop` | 27B | 10 | 50% → 60% | +7.7% | +2.9% | +3.4% | +6.1% | 0.70 |
| `us.anthropic.claude-sonnet-4-6` | `frozen_lake` | 27B | 10 | 100% → 100% | +36.6% | +34.4% | +28.1% | -14.1% | 0.80 |
| `us.anthropic.claude-sonnet-4-6` | `sokoban` | 27B | 10 | 90% → 90% | -10.1% | -14.8% | -14.0% | +3.8% | 0.60 |
| `us.openai.gpt-5.6-sol` | `frozen_lake` | 27B | 10 | 100% → 100% | +64.4% | +63.1% | n/a | +37.6% | 0.70 |
| `us.openai.gpt-5.6-sol` | `sokoban` | 27B | 10 | 100% → 100% | -26.2% | -122.0% | n/a | -7.5% | 0.90 |

## Evidence-backed reading

- Clear multi-metric win: `us.openai.gpt-5.6-sol` on `frozen_lake` with SimpleJev/JevAny-Qwen3.8-27B-LoRA@09c9e9102d5b8cc7d56558d25da1202a761b6c0d: success 100.0% → 100.0%, calls saved 64.4%, tokens saved 63.1%, and time saved 37.6%. Paired bootstrap intervals for calls, tokens, and latency exclude zero.
- Other cells are mixed or negative and remain in the table; they are not presented as wins.
- Dollar figures are estimated Bedrock list-price costs, not Cost Explorer billing. `n/a` means the run used cached tokens whose applicable price could not be verified.
- Jev GPU rental/energy cost is not monetized. One already-allocated H200 served the 27B matrix; cold start was excluded after readiness, while per-request Jev latency remains in wall time.
- FrozenLake and Sokoban are controlled mechanism tests. WebShop and WebArena are external interactive-agent checks; the WebArena row is a predeclared six-task sample, not the full benchmark.

## Provenance

Generated: `2026-10-01T04:21:13.150018+00:00`

Pricing basis: `2026-09-30`

Every row retains the raw JSON path, SHA-256, exact model IDs, seeds, paired deltas, and bootstrap intervals in the machine-readable matrix.
