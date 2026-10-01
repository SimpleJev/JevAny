"""Harbor agent: frontier reasoning plus optional Jev command selection."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from .api import validate_response
from .harness import HTTPDecisionClient
from .terminal_harness import (
    FRONTIER_TOOLS,
    SYSTEM_PROMPT,
    TerminalCandidate,
    TerminalCompletion,
    TerminalProposal,
    delegate_at_rate,
    delegation_limit,
    jev_terminal_request,
    parse_frontier_response,
)


_REGION_PREFIXES = ("us.", "eu.", "ap.", "global.")
_MODEL_PRICES = {
    "us.anthropic.claude-opus-4-7": (5.5, 27.5),
    "us.anthropic.claude-sonnet-4-6": (3.3, 16.5),
    "us.openai.gpt-5.6-sol": (4.4, 22.0),
}


def _as_int(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _as_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _as_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _bedrock_runtime(profile: str, region: str, model: str, timeout: int = 900):
    """Build a public boto3 Bedrock client and resolve cross-region profiles."""

    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile or None, region_name=region)
    model_id = model
    if model.startswith(_REGION_PREFIXES):
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


class AgentHarness(BaseAgent):
    """Generate bounded command options with a frontier model; let Jev choose routine ones."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        *args,
        mode: str = "optional",
        jev_url: str = "http://127.0.0.1:18201",
        jev_model: str = "SimpleJev/JevAny-Qwen3.8-27B-LoRA",
        max_turns: int | str = 30,
        max_output_tokens: int | str = 2048,
        confidence_threshold: float | str = 0.55,
        harness_level: int | str | None = None,
        delegation_rate: float | str = 1.0,
        aws_profile: str = "bedrock",
        aws_region: str = "us-west-2",
        history_limit: int | str = 8,
        input_price_per_million: float | str | None = None,
        output_price_per_million: float | str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mode not in {"baseline", "optional"}:
            raise ValueError("mode must be baseline or optional")
        self.mode = mode
        self.jev_url = jev_url
        self.jev_model = jev_model
        self.max_turns = _as_int(max_turns, 30)
        self.max_output_tokens = _as_int(max_output_tokens, 2048)
        self.confidence_threshold = _as_float(confidence_threshold, 0.55)
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        # Preserve the original two-mode behavior: baseline is level 0 and
        # optional is the prior safe two-command maximum (level 3).
        self.harness_level = (
            0 if mode == "baseline" else 3
        ) if harness_level is None else _as_int(harness_level, 3)
        requested_delegation_rate = _as_float(delegation_rate, 1.0)
        # A baseline is always pure frontier execution even though the default
        # argument remains 1.0 for backward-compatible optional runs.
        self.delegation_rate = 0.0 if mode == "baseline" else requested_delegation_rate
        self.history_limit = _as_int(history_limit, 8)
        self.input_price = _as_optional_float(input_price_per_million)
        self.output_price = _as_optional_float(output_price_per_million)
        if self.max_turns < 1 or self.max_output_tokens < 1 or self.history_limit < 1:
            raise ValueError("turn, token, and history limits must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0 <= self.harness_level <= 3:
            raise ValueError("harness_level must be in [0, 3]")
        if not 0 <= requested_delegation_rate <= 1:
            raise ValueError("delegation_rate must be in [0, 1]")
        if mode == "baseline" and self.harness_level != 0:
            raise ValueError("baseline mode requires harness_level=0")
        self._session: TmuxSession | None = None
        self._marker_seq = 0
        self._pending_marker: str | None = None

    @staticmethod
    @override
    def name() -> str:
        return "jev-terminal-candidates"

    @override
    def version(self) -> str:
        return "1.2.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        env_paths = EnvironmentPaths.for_os(environment.os)
        self._session = TmuxSession(
            session_name=f"jev-terminal-{uuid.uuid4().hex[:8]}",
            environment=environment,
            logging_path=env_paths.agent_dir / "jev-terminal.pane",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=180,
            pane_height=50,
            extra_env=self.extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    @staticmethod
    def _limit(text: str, maximum: int = 12000) -> str:
        if len(text) <= maximum:
            return text
        half = maximum // 2
        return text[:half] + f"\n[... {len(text)-maximum} characters omitted ...]\n" + text[-half:]

    async def _execute(self, candidate: TerminalCandidate) -> tuple[str, bool]:
        if self._session is None:
            raise RuntimeError("terminal session is not initialized")
        command = candidate.command.strip()
        if command == "<WAIT>":
            await asyncio.sleep(candidate.wait_seconds)
            output = await self._session.get_incremental_output()
            screen = await self._session.capture_pane(capture_entire=True)
            completed = self._pending_marker is None or self._pending_marker in screen
            if completed:
                self._pending_marker = None
            return self._limit(output), completed
        if command == "<CTRL_C>":
            await self._session.send_keys(keys=["C-c"], min_timeout_sec=0.2)
            self._pending_marker = None
            return self._limit(await self._session.get_incremental_output()), True

        self._marker_seq += 1
        marker = f"__JEV_CMD_END_{self._marker_seq}__"
        self._pending_marker = marker
        await self._session.send_keys(command.rstrip("\n") + "\n", block=False)
        await self._session.send_keys(f"printf '\\n{marker}\\n'\n", block=False)
        deadline = time.monotonic() + candidate.wait_seconds
        completed = False
        while time.monotonic() < deadline:
            screen = await self._session.capture_pane(capture_entire=True)
            if marker in screen:
                completed = True
                self._pending_marker = None
                break
            await asyncio.sleep(min(0.5, max(0.05, deadline - time.monotonic())))
        output = await self._session.get_incremental_output()
        output = "\n".join(line for line in output.splitlines() if marker not in line)
        return self._limit(output), completed

    @staticmethod
    def _add_usage(total: dict, usage: dict | None) -> None:
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._session is None:
            raise RuntimeError("setup must run before run")
        frontier_model = self.model_name or "us.anthropic.claude-opus-4-7"
        default_prices = _MODEL_PRICES.get(frontier_model)
        input_price = self.input_price if self.input_price is not None else (
            default_prices[0] if default_prices else None
        )
        output_price = self.output_price if self.output_price is not None else (
            default_prices[1] if default_prices else None
        )
        bedrock, resolved_model = _bedrock_runtime(
            self.aws_profile, self.aws_region, frontier_model
        )
        decide = HTTPDecisionClient(self.jev_url)
        usage: dict[str, float] = {}
        metrics: dict[str, Any] = {
            "schema_version": 2,
            "mode": self.mode,
            "harness_level": self.harness_level,
            "delegation_rate": self.delegation_rate,
            "frontier_model": self.model_name,
            "resolved_frontier_model": resolved_model,
            "jev_model": self.jev_model,
            "frontier_price_per_million": {
                "input": input_price,
                "output": output_price,
                "cache": None,
            },
            "frontier_calls": 0,
            "frontier_failures": 0,
            "jev_calls": 0,
            "jev_failures": 0,
            "jev_low_confidence_returns": 0,
            "direct_commands": 0,
            "delegated_commands": 0,
            "eligible_decisions": 0,
            "target_delegated_decisions": 0,
            "actual_delegated_decisions": 0,
            "protocol_repairs": 0,
            "protocol_errors": [],
            "completed_by_frontier": False,
            "termination_reason": "running",
            "turns": [],
        }
        history: list[dict] = []
        initial = self._limit(await self._session.get_incremental_output())
        messages = [{"role": "user", "content": [{"text":
            f"Terminal-Bench task:\n{instruction}\n\nInitial terminal state:\n{initial}"}]}]
        started = time.perf_counter()
        last_observation = initial
        try:
            for turn_index in range(self.max_turns):
                request_messages = (
                    messages if len(messages) <= 1 + 2 * self.history_limit
                    else [messages[0], *messages[-2 * self.history_limit:]]
                )
                request = {
                    "modelId": resolved_model,
                    "system": [{"text": SYSTEM_PROMPT}],
                    "messages": request_messages,
                    "toolConfig": {"tools": FRONTIER_TOOLS, "toolChoice": {"any": {}}},
                    "inferenceConfig": {"maxTokens": self.max_output_tokens},
                }
                call_started = time.perf_counter()
                metrics["frontier_calls"] += 1
                try:
                    response = await asyncio.to_thread(bedrock.converse, **request)
                except Exception:
                    metrics["frontier_failures"] += 1
                    raise
                frontier_latency = (time.perf_counter() - call_started) * 1000
                self._add_usage(usage, response.get("usage"))
                message = response.get("output", {}).get("message", {})
                content = message.get("content", [])
                messages.append({"role": "assistant", "content": content})
                try:
                    tool_id, decision = parse_frontier_response(response)
                except ValueError as error:
                    metrics["protocol_repairs"] += 1
                    metrics["protocol_errors"].append(str(error))
                    raw_calls = [
                        block.get("toolUse") for block in content
                        if isinstance(block, dict) and block.get("toolUse")
                    ]
                    if len(raw_calls) == 1 and raw_calls[0].get("toolUseId"):
                        messages.append({"role": "user", "content": [{"toolResult": {
                            "toolUseId": raw_calls[0]["toolUseId"],
                            "content": [{"json": {"error": str(error)}}],
                            "status": "error",
                        }}]})
                    else:
                        messages.append({"role": "user", "content": [{"text":
                            f"Protocol error: {error}. Call exactly one tool with valid fields."}]})
                    continue
                if isinstance(decision, TerminalCompletion):
                    metrics["completed_by_frontier"] = True
                    metrics["termination_reason"] = "frontier_complete"
                    metrics["completion"] = {
                        "summary": decision.summary,
                        "verification": decision.verification,
                    }
                    metrics["turns"].append({
                        "turn": turn_index, "kind": "complete",
                        "frontier_latency_ms": frontier_latency,
                    })
                    break

                executed = []
                remaining = list(decision.candidates)
                eligible = decision.routine and len(decision.candidates) >= 2
                if eligible:
                    metrics["eligible_decisions"] += 1
                level_limit = delegation_limit(decision, self.harness_level)
                policy_delegates = bool(level_limit and eligible and delegate_at_rate(
                    metrics["eligible_decisions"], self.delegation_rate
                ))
                if policy_delegates:
                    metrics["target_delegated_decisions"] += 1
                    delegated_limit = level_limit
                    prior_observation = last_observation
                    for _ in range(delegated_limit):
                        jev_request = jev_terminal_request(
                            instruction, decision, last_observation, remaining,
                            history[-self.history_limit:], self.jev_model,
                        )
                        jev_started = time.perf_counter()
                        metrics["jev_calls"] += 1
                        try:
                            jev_response = validate_response(jev_request, decide(jev_request))
                        except Exception as error:
                            metrics["jev_failures"] += 1
                            executed.append({"controller": "jev", "error": type(error).__name__})
                            break
                        jev_latency = (time.perf_counter() - jev_started) * 1000
                        answer = jev_response["answers"]["action"]
                        choice = str(answer["choice"])
                        confidence = float(answer["confidence"])
                        selected = next((item for item in remaining if str(item.index) == choice), None)
                        if selected is None:
                            metrics["jev_failures"] += 1
                            break
                        if confidence < self.confidence_threshold:
                            metrics["jev_low_confidence_returns"] += 1
                            break
                        output, completed = await self._execute(selected)
                        metrics["delegated_commands"] += 1
                        executed.append({
                            "controller": "jev", "candidate": selected.index,
                            "label": selected.label, "command": selected.command,
                            "confidence": confidence, "jev_latency_ms": jev_latency,
                            "command_completed": completed, "output": output,
                        })
                        history.append({"controller": "jev", "action": selected.choice_text,
                                        "output": output[-2000:]})
                        last_observation = output
                        remaining.remove(selected)
                        if not completed:
                            break
                        # Stop a multi-command batch when the terminal produced no
                        # substantive new state. The frontier model should reassess.
                        compact = "".join(output.split())
                        prior_compact = "".join(prior_observation.split())
                        if not compact or compact == prior_compact:
                            break
                        prior_observation = output
                actual_delegated = any(
                    item.get("controller") == "jev" and item.get("command")
                    for item in executed
                )
                if actual_delegated:
                    metrics["actual_delegated_decisions"] += 1
                if not executed or not any(item.get("command") for item in executed):
                    selected = decision.candidates[decision.frontier_choice]
                    output, completed = await self._execute(selected)
                    metrics["direct_commands"] += 1
                    executed.append({
                        "controller": "frontier", "candidate": selected.index,
                        "label": selected.label, "command": selected.command,
                        "confidence": None, "command_completed": completed, "output": output,
                    })
                    history.append({"controller": "frontier", "action": selected.choice_text,
                                    "output": output[-2000:]})
                    last_observation = output
                result = {
                    "subgoal": decision.subgoal,
                    "routine": decision.routine,
                    "composable_sequence": decision.composable_sequence,
                    "executed": executed,
                    "current_terminal": last_observation,
                }
                metrics["turns"].append({
                    "turn": turn_index, "kind": "commands",
                    "reasoning_summary": decision.reasoning_summary,
                    "subgoal": decision.subgoal, "routine": decision.routine,
                    "composable_sequence": decision.composable_sequence,
                    "eligible_for_delegation": eligible,
                    "policy_targeted": policy_delegates,
                    "actual_delegated": actual_delegated,
                    "frontier_choice": decision.frontier_choice,
                    "candidate_count": len(decision.candidates),
                    "candidates": [
                        {"index": candidate.index, "label": candidate.label,
                         "command": candidate.command, "expected": candidate.expected}
                        for candidate in decision.candidates
                    ],
                    "frontier_latency_ms": frontier_latency,
                    "executed": executed,
                })
                messages.append({"role": "user", "content": [{"toolResult": {
                    "toolUseId": tool_id,
                    "content": [{"json": result}],
                    "status": "success",
                }}]})
        finally:
            if metrics["termination_reason"] == "running":
                metrics["termination_reason"] = "max_turns"
            metrics["usage"] = usage
            eligible_count = metrics["eligible_decisions"]
            command_count = metrics["delegated_commands"] + metrics["direct_commands"]
            metrics["target_replacement_rate"] = (
                metrics["target_delegated_decisions"] / eligible_count if eligible_count else 0.0
            )
            metrics["actual_replacement_rate"] = (
                metrics["actual_delegated_decisions"] / eligible_count if eligible_count else 0.0
            )
            metrics["actual_command_replacement_rate"] = (
                metrics["delegated_commands"] / command_count if command_count else 0.0
            )
            metrics["total_latency_ms"] = (time.perf_counter() - started) * 1000
            cache_tokens = sum(
                usage.get(key, 0) for key in (
                    "cacheReadInputTokens", "cacheWriteInputTokens",
                    "cacheWriteInputTokens5m", "cacheWriteInputTokens1h",
                )
            )
            estimated_cost = None
            if input_price is not None and output_price is not None and not cache_tokens:
                estimated_cost = (
                    usage.get("inputTokens", 0) * input_price
                    + usage.get("outputTokens", 0) * output_price
                ) / 1_000_000
            metrics["estimated_bedrock_cost_usd"] = estimated_cost
            metrics["cost_estimate_caveat"] = (
                "unavailable: model price unknown or cache pricing not configured"
                if estimated_cost is None else None
            )
            context.n_input_tokens = int(usage.get("inputTokens", 0))
            context.n_cache_tokens = int(usage.get("cacheReadInputTokens", 0))
            context.n_output_tokens = int(usage.get("outputTokens", 0))
            context.cost_usd = estimated_cost
            context.metadata = metrics
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "jev-terminal-metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            await self._session.stop()
