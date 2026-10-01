# Frontier model + Jev on WebShop

Seed: `3105`

Frontier model: `us.anthropic.claude-opus-4-7` · decision model: `SimpleJev/JevAny-Qwen3.5-4B-LoRA`

Goal: Instruction: Find me men's sleep & lounge with long sleeve, elastic waistband for daily wear with color: multi 9, and size: medium, and price lower than 70.00 dollars

| Mode | Success | Reward | Actions | Bedrock calls | Bedrock tokens | Jev decisions | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frontier only | yes | 1.000 | 9 | 9 | 34,822 | 0 | 15.40s |
| Frontier + optional Jev | yes | 1.000 | 9 | 10 | 47,962 | 0 | 19.69s |

## Delegation decisions

- Turn 3: the LLM proposed a five-option size group and a ten-option color group.
- Both exceeded the enforced 2–4 candidate limit, so the harness rejected the delegation and returned control to the LLM.
- The LLM completed the purchase directly. This is the recorded fallback case, not a Jev efficiency win.

## Optional-Jev action trace

| # | Controller | Action | Confidence | Reward | Terminal |
|---:|---|---|---:|---:|---:|
| 1 | bedrock | `search[men's sleep lounge long sleeve elastic waistband]` | — | 0.000 | no |
| 2 | bedrock | `click[b09nd8p2qr]` | — | 0.000 | no |
| 3 | bedrock | `click[medium]` | — | 0.000 | no |
| 4 | bedrock | `click[multi 9]` | — | 0.000 | no |
| 5 | bedrock | `click[description]` | — | 0.000 | no |
| 6 | bedrock | `click[< prev]` | — | 0.000 | no |
| 7 | bedrock | `click[medium]` | — | 0.000 | no |
| 8 | bedrock | `click[multi 9]` | — | 0.000 | no |
| 9 | bedrock | `click[buy now]` | — | 1.000 | yes |

This case is selected mechanically from a paired benchmark run. The complete machine-readable trajectories and selection rule are retained with the result.
