#!/usr/bin/env python3
"""Paired Bedrock baseline/optional-Jev evaluation on WebShop."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

from jevany.agent_harness import estimated_bedrock_cost
from jevany.harness import HTTPDecisionClient
from jevany.suite import write_json
from jevany.webshop_harness import BedrockJevWebShopAgent


REGION_PREFIXES = ("us.", "eu.", "ap.", "global.")
TOKEN_FIELDS = (
    "inputTokens", "outputTokens", "totalTokens", "cacheReadInputTokens",
    "cacheWriteInputTokens",
)


def runtime_client(profile: str, region: str, model: str, timeout: int = 900):
    import boto3
    from botocore.config import Config

    user_config = Path.home() / ".aws" / "config"
    if user_config.is_file():
        os.environ["AWS_CONFIG_FILE"] = str(user_config)
    session = boto3.Session(profile_name=profile, region_name=region)
    model_id = model
    if model.startswith(REGION_PREFIXES):
        account = session.client("sts", region_name=region).get_caller_identity()["Account"]
        model_id = f"arn:aws:bedrock:{region}:{account}:inference-profile/{model}"
    client = session.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            connect_timeout=10,
            read_timeout=timeout,
            max_pool_connections=4,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )
    return client, model_id


def _revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_webshop_environment(ragen_repo: Path, webshop_repo: Path, *, num_products: int):
    """Load only RAGEN's WebShop adapter, without importing its training stack."""
    package = webshop_repo / "webshop_minimal"
    required = [
        package / "__init__.py",
        package / "data" / "small" / "items_ins_v2_1000.json",
        package / "data" / "small" / "items_shuffle_1000.json",
        package / "search_engine" / "indexes_1k",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WebShop checkout is incomplete; initialize RAGEN's pinned external/webshop-minimal "
            f"submodule. Missing: {missing}"
        )
    sys.path.insert(0, str(webshop_repo))
    for package_name, path in (
        ("ragen", ragen_repo / "ragen"),
        ("ragen.env", ragen_repo / "ragen" / "env"),
    ):
        module = types.ModuleType(package_name)
        module.__path__ = [str(path)]
        sys.modules[package_name] = module
    from ragen.env.webshop.config import WebShopEnvConfig
    from ragen.env.webshop.env import WebShopEnv

    return WebShopEnv(WebShopEnvConfig(
        dataset="small",
        num_products=num_products,
        observation_mode="text",
    ))


def summarize(episodes: list[dict], input_price=None, output_price=None,
              cache_read_price=None, cache_write_price=None) -> dict:
    if not episodes:
        raise ValueError("cannot summarize an empty episode list")
    usage = {key: sum(item["usage"].get(key, 0) for item in episodes) for key in TOKEN_FIELDS}
    for key in ("cacheWriteInputTokens5m", "cacheWriteInputTokens1h", "cacheWriteInputTokens30m"):
        usage[key] = sum(item["usage"].get(key, 0) for item in episodes)
    summary = {
        "episodes": len(episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "mean_reward": statistics.fmean(item["reward"] for item in episodes),
        "mean_actions": statistics.fmean(len(item["actions"]) for item in episodes),
        "mean_bedrock_calls": statistics.fmean(item["bedrock_calls"] for item in episodes),
        "mean_bedrock_failures": statistics.fmean(
            item.get("bedrock_failures", 0) for item in episodes
        ),
        "mean_direct_actions": statistics.fmean(item["direct_actions"] for item in episodes),
        "mean_search_actions": statistics.fmean(item["search_actions"] for item in episodes),
        "mean_delegation_calls": statistics.fmean(item["delegation_calls"] for item in episodes),
        "mean_jev_decisions": statistics.fmean(item["jev_decisions"] for item in episodes),
        "mean_jev_failures": statistics.fmean(item.get("jev_failures", 0) for item in episodes),
        "error_rate": sum(item.get("status") == "error" for item in episodes) / len(episodes),
        "mean_total_latency_ms": statistics.fmean(item["total_latency_ms"] for item in episodes),
        "mean_bedrock_latency_ms": statistics.fmean(item["bedrock_latency_ms"] for item in episodes),
        "mean_jev_latency_ms": statistics.fmean(item["jev_latency_ms"] for item in episodes),
        "usage": usage,
    }
    cost = estimated_bedrock_cost(
        usage,
        input_price,
        output_price,
        cache_read_per_million=cache_read_price,
        cache_write_per_million=cache_write_price,
    )
    if cost is not None:
        summary["estimated_bedrock_cost_usd"] = cost
    return summary


def _reduction(baseline: float, optional: float) -> float | None:
    return None if baseline == 0 else (baseline - optional) / baseline


def bootstrap_interval(values: list[float], *, samples: int = 10_000,
                       seed: int = 20260930) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]]


def compare(summaries: dict, episodes: list[dict]) -> dict:
    baseline = summaries.get("baseline")
    optional = summaries.get("optional")
    if not baseline or not optional:
        return {}
    baseline_tokens = baseline["usage"]["inputTokens"] + baseline["usage"]["outputTokens"]
    optional_tokens = optional["usage"]["inputTokens"] + optional["usage"]["outputTokens"]
    indexed = {}
    for record in episodes:
        key = (record["mode"], record["seed"])
        if key in indexed:
            raise ValueError(f"duplicate paired result: {key[0]}/{key[1]}")
        indexed[key] = record
    baseline_seeds = {seed for mode, seed in indexed if mode == "baseline"}
    optional_seeds = {seed for mode, seed in indexed if mode == "optional"}
    if baseline_seeds != optional_seeds:
        raise ValueError(f"unpaired seed sets: {baseline_seeds} != {optional_seeds}")
    deltas = {key: [] for key in ("success", "reward", "bedrock_calls", "tokens", "latency_ms")}
    for paired_seed in sorted(baseline_seeds):
        left = indexed[("baseline", paired_seed)]
        right = indexed[("optional", paired_seed)]
        deltas["success"].append(float(right["success"]) - float(left["success"]))
        deltas["reward"].append(right["reward"] - left["reward"])
        deltas["bedrock_calls"].append(right["bedrock_calls"] - left["bedrock_calls"])
        deltas["tokens"].append(
            right["usage"].get("totalTokens", 0) - left["usage"].get("totalTokens", 0)
        )
        deltas["latency_ms"].append(right["total_latency_ms"] - left["total_latency_ms"])
    result = {
        "success_rate_delta": optional["success_rate"] - baseline["success_rate"],
        "reward_delta": optional["mean_reward"] - baseline["mean_reward"],
        "bedrock_call_reduction": _reduction(
            baseline["mean_bedrock_calls"], optional["mean_bedrock_calls"],
        ),
        "bedrock_token_reduction": _reduction(baseline_tokens, optional_tokens),
        "wall_time_reduction": _reduction(
            baseline["mean_total_latency_ms"], optional["mean_total_latency_ms"],
        ),
        "paired": {
            key: {
                "mean_optional_minus_baseline": statistics.fmean(values),
                "bootstrap_95_ci": bootstrap_interval(values),
            }
            for key, values in deltas.items()
        },
    }
    if "estimated_bedrock_cost_usd" in baseline:
        result["estimated_bedrock_cost_reduction"] = _reduction(
            baseline["estimated_bedrock_cost_usd"],
            optional["estimated_bedrock_cost_usd"],
        )
    return result


def select_demo(episodes: list[dict], *, seed: int | None = None) -> dict | None:
    paired: dict[int, dict[str, dict]] = defaultdict(dict)
    for item in episodes:
        paired[item["seed"]][item["mode"]] = item
    if seed is None and paired:
        seed = min(paired)
    for episode_seed, pair in sorted(paired.items()):
        if episode_seed != seed:
            continue
        if "baseline" not in pair or "optional" not in pair:
            continue
        return {
            "seed": episode_seed,
            "selection": "predeclared first paired seed",
            "baseline": pair["baseline"],
            "optional": pair["optional"],
        }
    return None


def render_demo(case: dict, model: str, jev_model: str) -> str:
    baseline, optional = case["baseline"], case["optional"]

    def tokens(item):
        return item["usage"].get("inputTokens", 0) + item["usage"].get("outputTokens", 0)

    lines = [
        "# Frontier model + Jev on WebShop",
        "",
        f"Seed: `{case['seed']}`  ",
        f"Frontier model: `{model}` · decision model: `{jev_model}`",
        "",
        f"Goal: {optional['goal']}",
        "",
        "| Mode | Success | Reward | Actions | Bedrock calls | Bedrock tokens | Jev decisions | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("Frontier only", baseline), ("Frontier + optional Jev", optional)):
        lines.append(
            f"| {label} | {'yes' if item['success'] else 'no'} | {item['reward']:.3f} | "
            f"{len(item['actions'])} | {item['bedrock_calls']} | {tokens(item):,} | "
            f"{item['jev_decisions']} | {item['total_latency_ms'] / 1000:.2f}s |"
        )
    lines.extend([
        "",
        "## Delegation decisions",
        "",
    ])
    delegated = False
    for turn in optional["transcript"]:
        if turn.get("tool") == "delegate_clicks":
            delegated = True
            lines.append(f"- Turn {turn['turn'] + 1}: {turn.get('text') or '(no public rationale)'}")
            for index, decision in enumerate(
                (turn.get("tool_input") or {}).get("decisions", []), start=1,
            ):
                lines.append(
                    f"  - Decision {index}: {decision.get('subgoal', '(missing subgoal)')} · "
                    f"candidate keys `{decision.get('candidate_keys', [])}`"
                )
            for step in (turn.get("delegation") or {}).get("steps", []):
                lines.append(
                    f"  - Jev selected `{step.get('selected_action')}` from "
                    f"`{step.get('candidate_actions', [])}` at "
                    f"{float(step.get('confidence', 0)):.3f} confidence"
                )
    if not delegated:
        lines.append("- The frontier model did not delegate in this episode.")
    lines.extend([
        "",
        "## Optional-Jev action trace",
        "",
        "| # | Controller | Action | Confidence | Reward | Terminal |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for action in optional["actions"]:
        confidence = "—" if action["confidence"] is None else f"{action['confidence']:.3f}"
        lines.append(
            f"| {action['index'] + 1} | {action['controller']} | `{action['action']}` | "
            f"{confidence} | {action['reward']:.3f} | {'yes' if action['done'] else 'no'} |"
        )
    lines.extend([
        "",
        "This case is selected mechanically from a paired benchmark run. The complete "
        "machine-readable trajectories and selection rule are retained with the result.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragen-repo", required=True)
    parser.add_argument("--webshop-repo")
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--profile", default="bedrock")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--jev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--jev-model", default="SimpleJev/JevAny-Qwen3.5-4B-LoRA")
    parser.add_argument("--mode", action="append", choices=("baseline", "optional"), dest="modes")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-products", type=int, choices=(100, 1000), default=1000)
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-delegate-steps", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--cache-read-price-per-million", type=float)
    parser.add_argument("--cache-write-price-per-million", type=float)
    parser.add_argument("--mode-order", choices=("fixed", "counterbalanced"),
                        default="counterbalanced")
    parser.add_argument("--demo-seed", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--demo-out")
    args = parser.parse_args()
    if args.episodes < 1 or min(args.max_actions, args.max_turns, args.max_delegate_steps) < 1:
        parser.error("episode, action, turn, and delegation counts must be positive")

    ragen_repo = Path(args.ragen_repo).resolve()
    webshop_repo = Path(args.webshop_repo).resolve() if args.webshop_repo else (
        ragen_repo / "external" / "webshop-minimal"
    )
    if not (ragen_repo / "ragen" / "env" / "webshop" / "env.py").is_file():
        parser.error(f"RAGEN WebShop adapter not found under {ragen_repo}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client, resolved_model = runtime_client(args.profile, args.region, args.bedrock_model)
    agent = BedrockJevWebShopAgent(
        client,
        resolved_model,
        HTTPDecisionClient(args.jev_url),
        jev_model=args.jev_model,
        max_output_tokens=args.max_output_tokens,
        max_turns=args.max_turns,
        max_delegate_steps=args.max_delegate_steps,
        confidence_threshold=args.confidence_threshold,
    )

    modes = args.modes or ["baseline", "optional"]
    episodes = []
    for index in range(args.episodes):
        seed = args.seed + index
        ordered_modes = modes if args.mode_order == "fixed" or index % 2 == 0 else list(reversed(modes))
        for mode in ordered_modes:
            env = None
            try:
                env = load_webshop_environment(
                    ragen_repo, webshop_repo, num_products=args.num_products,
                )
                result = agent.run(
                    env,
                    seed=seed,
                    split=args.split,
                    mode=mode,
                    max_actions=args.max_actions,
                ).as_dict()
                result["status"] = (
                    "error" if result["termination_reason"] == "bedrock_backend_error"
                    else "complete"
                )
            except Exception as error:
                result = {
                    "mode": mode,
                    "split": args.split,
                    "seed": seed,
                    "goal": "",
                    "status": "error",
                    "success": False,
                    "reward": 0.0,
                    "actions": [],
                    "bedrock_calls": 0,
                    "bedrock_failures": 0,
                    "direct_actions": 0,
                    "search_actions": 0,
                    "delegation_calls": 0,
                    "jev_decisions": 0,
                    "jev_failures": 0,
                    "low_confidence_returns": 0,
                    "protocol_repairs": 0,
                    "termination_reason": "backend_error",
                    "error_type": type(error).__name__,
                    "usage": {},
                    "bedrock_latency_ms": 0.0,
                    "jev_latency_ms": 0.0,
                    "total_latency_ms": 0.0,
                    "transcript": [],
                }
            finally:
                if env is not None:
                    env.close()
            episodes.append(result)
            write_json(output_path, {
                "schema_version": 1,
                "status": "running",
                "benchmark": "WebShop paired agent evaluation",
                "bedrock_model": args.bedrock_model,
                "jev_model": args.jev_model,
                "episodes": episodes,
            })
            print(json.dumps({
                "seed": seed,
                "mode": mode,
                "success": result["success"],
                "reward": result["reward"],
                "actions": len(result["actions"]),
                "bedrock_calls": result["bedrock_calls"],
                "jev_decisions": result["jev_decisions"],
                "tokens": result["usage"].get("totalTokens", 0),
            }, sort_keys=True), flush=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in episodes:
        grouped[item["mode"]].append(item)
    summaries = {
        key: summarize(
            value,
            args.input_price_per_million,
            args.output_price_per_million,
            args.cache_read_price_per_million,
            args.cache_write_price_per_million,
        )
        for key, value in sorted(grouped.items())
    }
    demo_seed = args.seed if args.demo_seed is None else args.demo_seed
    if not args.seed <= demo_seed < args.seed + args.episodes:
        parser.error("--demo-seed must be inside the evaluated seed range")
    demo = select_demo(episodes, seed=demo_seed)
    output = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "WebShop paired agent evaluation",
        "ragen_revision": _revision(ragen_repo),
        "webshop_revision": _revision(webshop_repo),
        "bedrock_model": args.bedrock_model,
        "resolved_bedrock_model": resolved_model,
        "jev_model": args.jev_model,
        "config": {
            "split": args.split,
            "modes": modes,
            "episodes": args.episodes,
            "seed_start": args.seed,
            "num_products": args.num_products,
            "max_actions": args.max_actions,
            "max_turns": args.max_turns,
            "max_delegate_steps": args.max_delegate_steps,
            "confidence_threshold": args.confidence_threshold,
            "max_output_tokens": args.max_output_tokens,
            "mode_order": args.mode_order,
            "demo_seed": demo_seed,
            "input_price_per_million": args.input_price_per_million,
            "output_price_per_million": args.output_price_per_million,
            "cache_read_price_per_million": args.cache_read_price_per_million,
            "cache_write_price_per_million": args.cache_write_price_per_million,
        },
        "summaries": summaries,
        "comparison": compare(summaries, episodes),
        "demo_case": demo,
        "episodes": episodes,
    }
    write_json(output_path, output)
    if args.demo_out and demo:
        demo_path = Path(args.demo_out)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_path.write_text(
            render_demo(demo, args.bedrock_model, args.jev_model), encoding="utf-8",
        )
    print(json.dumps({"summaries": summaries, "comparison": output["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
