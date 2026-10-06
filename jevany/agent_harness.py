"""Let a frontier Bedrock agent selectively delegate bounded actions to JevAny."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol

from .agent import action_request
from .api import validate_response


class ConverseClient(Protocol):
    def converse(self, **request) -> dict: ...


DecisionFunction = Callable[[dict], dict]


@dataclass(frozen=True)
class HarnessAction:
    index: int
    controller: str
    observation: Any
    action: Any
    action_name: str
    confidence: float | None
    reward: float
    done: bool
    info: dict


@dataclass(frozen=True)
class HarnessEpisode:
    mode: str
    seed: int | None
    success: bool
    reward: float
    actions: tuple[HarnessAction, ...]
    bedrock_calls: int
    bedrock_failures: int
    direct_actions: int
    delegation_calls: int
    jev_decisions: int
    jev_failures: int
    low_confidence_returns: int
    protocol_repairs: int
    termination_reason: str
    usage: dict
    bedrock_latency_ms: float
    jev_latency_ms: float
    total_latency_ms: float
    transcript: tuple[dict, ...]

    def as_dict(self) -> dict:
        value = asdict(self)
        value["actions"] = [asdict(action) for action in self.actions]
        value["transcript"] = list(self.transcript)
        return value


def _observation(value):
    return value[0] if isinstance(value, tuple) else value


def _usage_sum(total: dict, current: Mapping | None) -> None:
    current = current or {}
    for key, value in current.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    detail_tokens = 0
    for detail in current.get("cacheDetails", []) or []:
        if not isinstance(detail, Mapping):
            raise ValueError("Bedrock usage cacheDetails entries must be objects")
        ttl = str(detail.get("ttl", "")).strip().lower()
        ttl = {"pt5m": "5m", "pt1h": "1h", "pt30m": "30m"}.get(ttl, ttl)
        if ttl not in {"5m", "1h", "30m"}:
            raise ValueError(f"unsupported Bedrock cache write TTL: {ttl!r}")
        tokens = detail.get("inputTokens")
        if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or tokens < 0:
            raise ValueError("Bedrock cacheDetails inputTokens must be non-negative")
        total[f"cacheWriteInputTokens{ttl}"] = total.get(f"cacheWriteInputTokens{ttl}", 0) + tokens
        detail_tokens += tokens
    reported = current.get("cacheWriteInputTokens", 0)
    if detail_tokens and reported != detail_tokens:
        raise ValueError(
            f"Bedrock cacheDetails total {detail_tokens} != cacheWriteInputTokens {reported}"
        )


class BedrockJevAgent:
    """Run a Bedrock tool agent with optional multi-action Jev delegation.

    ``baseline`` exposes only one-step direct actions. ``optional`` also exposes
    ``delegate_to_jev`` and leaves every delegation decision to the Bedrock
    model. ``jev_only`` is a no-Bedrock lower-cost control.
    """

    MODES = {"baseline", "optional", "jev_only"}

    def __init__(
        self,
        client: ConverseClient,
        model: str,
        decide: DecisionFunction,
        *,
        jev_model: str = "jevany-latest",
        max_output_tokens: int = 512,
        max_turns: int = 64,
        max_delegate_steps: int = 8,
        confidence_threshold: float = 0.0,
        history_limit: int = 8,
        temperature: float | None = None,
    ):
        if max_output_tokens < 1 or max_turns < 1 or max_delegate_steps < 1:
            raise ValueError("token, turn, and delegation limits must be positive")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if history_limit < 0:
            raise ValueError("history_limit must be nonnegative")
        self.client = client
        self.model = model
        self.decide = decide
        self.jev_model = jev_model
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns
        self.max_delegate_steps = max_delegate_steps
        self.confidence_threshold = confidence_threshold
        self.history_limit = history_limit
        self.temperature = temperature

    @staticmethod
    def _actions(env) -> tuple[list, dict[str, str]]:
        raw = list(env.get_all_actions())
        names = getattr(env, "ACTION_LOOKUP", {})
        return raw, {str(index): str(names.get(action, action)) for index, action in enumerate(raw)}

    def _tools(self, actions: Mapping[str, str], mode: str) -> list[dict]:
        descriptions = ", ".join(f"{key}={value}" for key, value in actions.items())
        tools = [{"toolSpec": {
            "name": "take_action",
            "description": (
                "Execute exactly one environment action chosen by you. "
                f"Current action keys: {descriptions}."
            ),
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"action_key": {"type": "string", "enum": list(actions)}},
                "required": ["action_key"],
                "additionalProperties": False,
            }},
        }}]
        if mode == "optional":
            tools.append({"toolSpec": {
                "name": "delegate_to_jev",
                "description": (
                    "Temporarily hand routine finite-choice control to the cheaper Jev decision model. "
                    "Jev may execute several environment actions and returns control on completion, "
                    "the requested horizon, or low confidence. Use this only when the current local "
                    "decisions do not require your deeper reasoning."
                ),
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "integer", "minimum": 1, "maximum": self.max_delegate_steps,
                            "description": "Maximum consecutive actions Jev may execute.",
                        },
                        "subgoal": {
                            "type": "string", "minLength": 1, "maxLength": 500,
                            "description": (
                                "A concise local objective and constraints for Jev. Preserve your strategic "
                                "plan without copying the full observation."
                            ),
                        },
                    },
                    "required": ["steps", "subgoal"],
                    "additionalProperties": False,
                }},
            }})
        return tools

    @staticmethod
    def _system(mode: str) -> str:
        common = (
            "You are controlling a discrete environment. Maximize task success and avoid invalid or "
            "irreversible actions. Inspect every tool result because it contains the new trusted state. "
            "Before each tool call, give one short public rationale. Call exactly one state-changing tool "
            "per response. Do not claim success unless the environment reports it."
        )
        if mode == "optional":
            return common + (
                " You may either act directly or delegate a bounded run to Jev. Decide autonomously when "
                "delegation is likely to preserve quality while reducing your own turns. Retain control for "
                "novel, ambiguous, or strategic decisions. When delegating, pass a well-scoped local subgoal "
                "and the constraints from your plan that Jev must preserve."
            )
        return common

    def run(self, env, goal: str, *, seed: int | None = None, mode: str = "optional",
            max_actions: int = 64) -> HarnessEpisode:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if max_actions < 1:
            raise ValueError("max_actions must be positive")

        started = time.perf_counter()
        observation = _observation(env.reset(seed=seed))
        trace: list[HarnessAction] = []
        history: list[dict] = []
        transcript: list[dict] = []
        total_reward = 0.0
        success = done = False
        usage: dict = {}
        bedrock_calls = bedrock_failures = direct_actions = delegation_calls = jev_decisions = 0
        jev_failures = 0
        low_confidence_returns = 0
        protocol_repairs = 0
        termination_reason = "running"
        bedrock_latency = jev_latency = 0.0

        def execute(action_key: str, controller: str, confidence: float | None) -> dict:
            nonlocal observation, total_reward, success, done, direct_actions
            raw_actions, actions = self._actions(env)
            if action_key not in actions:
                raise ValueError(f"unknown action key {action_key!r}; expected one of {sorted(actions)}")
            before = observation
            action = raw_actions[int(action_key)]
            # Gym/Gymnasium discrete spaces may emit NumPy scalar actions.
            # Normalize them at the environment boundary so saved trajectories
            # remain strict JSON without taking a NumPy runtime dependency.
            if not isinstance(action, (str, int, float, bool, type(None))) and hasattr(action, "item"):
                action = action.item()
            next_observation, reward, terminal, info = env.step(action)
            observation = _observation(next_observation)
            done = bool(terminal)
            success = success or bool(info.get("success"))
            total_reward += float(reward)
            if controller == "bedrock":
                direct_actions += 1
            trace.append(HarnessAction(
                len(trace), controller, before, action, actions[action_key], confidence,
                float(reward), done, dict(info),
            ))
            history.append({
                "controller": controller, "action": actions[action_key], "reward": float(reward),
                "effective": info.get("action_is_effective"),
            })
            return {
                "action": actions[action_key], "reward": float(reward), "done": done,
                "success": success, "effective": info.get("action_is_effective"),
                "observation": observation,
            }

        def delegate(steps: int, subgoal: str) -> dict:
            nonlocal jev_decisions, jev_failures, jev_latency, low_confidence_returns
            delegated = []
            reason = "horizon"
            for _ in range(min(steps, self.max_delegate_steps, max_actions - len(trace))):
                raw_actions, actions = self._actions(env)
                if not raw_actions or done:
                    reason = "terminal"
                    break
                delegated_goal = f"Global goal: {goal}\nDelegated subgoal: {subgoal}"
                recent = history[-self.history_limit:] if self.history_limit else []
                request = action_request(
                    delegated_goal, observation, actions, recent, self.jev_model,
                )
                before = time.perf_counter()
                jev_decisions += 1
                try:
                    decision = validate_response(request, self.decide(request))
                except Exception as error:
                    jev_failures += 1
                    reason = "backend_error"
                    return {
                        "subgoal": subgoal, "executed": len(delegated), "stop_reason": reason,
                        "error_type": type(error).__name__, "steps": delegated,
                        "current_observation": observation, "success": success, "done": done,
                    }
                finally:
                    jev_latency += (time.perf_counter() - before) * 1000
                answer = decision["answers"]["action"]
                confidence = float(answer["confidence"])
                if confidence < self.confidence_threshold:
                    low_confidence_returns += 1
                    reason = "low_confidence"
                    break
                delegated.append(execute(str(answer["choice"]), "jev", confidence))
                if done:
                    reason = "terminal"
                    break
                if delegated[-1]["reward"] < 0:
                    reason = "negative_reward"
                    break
                if delegated[-1].get("effective") is False:
                    reason = "ineffective_action"
                    break
            return {
                "subgoal": subgoal, "executed": len(delegated), "stop_reason": reason, "steps": delegated,
                "current_observation": observation, "success": success, "done": done,
            }

        if mode == "jev_only":
            while not done and len(trace) < max_actions:
                result = delegate(
                    min(self.max_delegate_steps, max_actions - len(trace)),
                    "Make the safest available progress toward the global goal.",
                )
                if result["executed"] == 0:
                    break
            return HarnessEpisode(
                mode, seed, success, total_reward, tuple(trace), 0, 0, 0, 0, jev_decisions,
                jev_failures, low_confidence_returns, 0,
                "terminal_success" if success else (
                    "terminal_failure" if done else "jev_returned_without_action"
                ), usage, 0.0, jev_latency,
                (time.perf_counter() - started) * 1000, tuple(transcript),
            )

        messages = [{"role": "user", "content": [{"text": (
            f"Goal:\n{goal}\n\nInitial observation:\n{observation}\n\n"
            "Start acting."
        )}]}]
        repair_turns = 0
        while not done and len(trace) < max_actions and bedrock_calls < self.max_turns:
            _, actions = self._actions(env)
            request = {
                "modelId": self.model,
                "system": [{"text": self._system(mode)}],
                "messages": messages,
                "toolConfig": {"tools": self._tools(actions, mode)},
                "inferenceConfig": {"maxTokens": self.max_output_tokens},
            }
            if self.temperature is not None:
                request["inferenceConfig"]["temperature"] = self.temperature
            before = time.perf_counter()
            bedrock_calls += 1
            try:
                response = self.client.converse(**request)
            except Exception as error:
                bedrock_failures += 1
                transcript.append({
                    "turn": bedrock_calls - 1, "error_type": type(error).__name__,
                    "error": str(error),
                })
                termination_reason = "bedrock_backend_error"
                break
            finally:
                bedrock_latency += (time.perf_counter() - before) * 1000
            _usage_sum(usage, response.get("usage"))
            message = response.get("output", {}).get("message", {})
            content = message.get("content", [])
            messages.append({"role": "assistant", "content": content})
            public_text = "\n".join(block["text"] for block in content if "text" in block)
            tool_uses = [block["toolUse"] for block in content if "toolUse" in block]
            transcript.append({
                "turn": bedrock_calls - 1, "text": public_text,
                "tool": tool_uses[0]["name"] if len(tool_uses) == 1 else None,
                "tool_input": tool_uses[0].get("input") if len(tool_uses) == 1 else None,
            })
            if len(tool_uses) != 1:
                repair_turns += 1
                protocol_repairs += 1
                if repair_turns > 2:
                    termination_reason = "model_protocol_error"
                    break
                if tool_uses and all(call.get("toolUseId") for call in tool_uses):
                    messages.append({"role": "user", "content": [{"toolResult": {
                        "toolUseId": call["toolUseId"],
                        "content": [{"json": {"error": "Call exactly one state-changing tool."}}],
                        "status": "error",
                    }} for call in tool_uses]})
                else:
                    messages.append({"role": "user", "content": [{"text": (
                        "The environment is not terminal. Call exactly one available tool."
                    )}]})
                continue
            repair_turns = 0
            call = tool_uses[0]
            tool_input = call.get("input") or {}
            try:
                if call["name"] == "take_action":
                    result = execute(str(tool_input.get("action_key")), "bedrock", None)
                elif call["name"] == "delegate_to_jev" and mode == "optional":
                    steps = tool_input.get("steps")
                    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= self.max_delegate_steps:
                        raise ValueError(f"steps must be an integer in [1, {self.max_delegate_steps}]")
                    subgoal = tool_input.get("subgoal")
                    if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal) > 500:
                        raise ValueError("subgoal must be a non-empty string of at most 500 characters")
                    delegation_calls += 1
                    result = delegate(steps, subgoal.strip())
                else:
                    raise ValueError(f"unknown tool {call.get('name')!r}")
                status = "success"
            except Exception as error:
                result, status = {"error": str(error), "observation": observation}, "error"
                protocol_repairs += 1
            if not call.get("toolUseId"):
                termination_reason = "model_protocol_error"
                break
            messages.append({"role": "user", "content": [{"toolResult": {
                "toolUseId": call["toolUseId"], "content": [{"json": result}], "status": status,
            }}]})

        if termination_reason == "running":
            termination_reason = (
                "terminal_success" if success else "terminal_failure" if done
                else "max_actions" if len(trace) >= max_actions else "max_turns"
            )
        return HarnessEpisode(
            mode, seed, success, total_reward, tuple(trace), bedrock_calls, bedrock_failures,
            direct_actions,
            delegation_calls, jev_decisions, jev_failures, low_confidence_returns,
            protocol_repairs, termination_reason, usage, bedrock_latency,
            jev_latency, (time.perf_counter() - started) * 1000, tuple(transcript),
        )


def estimated_bedrock_cost(usage: Mapping, input_per_million: float | None,
                           output_per_million: float | None, *,
                           cache_read_per_million: float | None = None,
                           cache_write_per_million: float | None = None,
                           cache_write_5m_per_million: float | None = None,
                           cache_write_1h_per_million: float | None = None,
                           cache_write_30m_per_million: float | None = None) -> float | None:
    """Estimate request cost only when explicit current prices are supplied."""
    if input_per_million is None or output_per_million is None:
        return None
    regular_input = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)
    writes = {
        "5m": usage.get("cacheWriteInputTokens5m", 0),
        "1h": usage.get("cacheWriteInputTokens1h", 0),
        "30m": usage.get("cacheWriteInputTokens30m", 0),
    }
    if cache_read and cache_read_per_million is None:
        return None
    detailed_write = sum(writes.values())
    if detailed_write and detailed_write != cache_write:
        raise ValueError("cache write TTL buckets do not equal cacheWriteInputTokens")
    write_prices = {
        "5m": cache_write_5m_per_million,
        "1h": cache_write_1h_per_million,
        "30m": cache_write_30m_per_million,
    }
    if detailed_write:
        if any(tokens and write_prices[ttl] is None for ttl, tokens in writes.items()):
            return None
        write_cost = sum(writes[ttl] * (write_prices[ttl] or 0) for ttl in writes)
    else:
        if cache_write and cache_write_per_million is None:
            return None
        write_cost = cache_write * (cache_write_per_million or 0)
    return (
        regular_input * input_per_million
        + output_tokens * output_per_million
        + cache_read * (cache_read_per_million or 0)
        + write_cost
    ) / 1_000_000
