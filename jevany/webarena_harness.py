"""Bedrock controller with optional bounded Jev delegation for WebArena."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Protocol

from .agent import action_request
from .api import validate_response


class ConverseClient(Protocol):
    def converse(self, **request) -> dict: ...


DecisionFunction = Callable[[dict], dict]

_ACTION = re.compile(
    r"^(?:click \[\d+\]|hover \[\d+\]|type \[\d+\] \[[^\]\n]*\] \[[01]\]|"
    r"press \[[^\]\n]+\]|scroll \[(?:down|up)\]|new_tab|tab_focus \[\d+\]|"
    r"close_tab|goto \[[^\]\n]+\]|go_back|go_forward|stop \[[^\]\n]*\])$"
)
_ELEMENT = re.compile(r"^\s*\[(\d+)\]\s+(.+?)\s*$")
_OBJECTIVE = re.compile(r"OBJECTIVE:\s*(.*?)(?:\nPREVIOUS ACTION:|\Z)", re.DOTALL)


@dataclass(frozen=True)
class WebArenaAction:
    index: int
    controller: str
    action: str
    confidence: float | None
    reward: float
    terminated: bool


@dataclass(frozen=True)
class WebArenaEpisode:
    mode: str
    task_id: int
    goal: str
    success: bool
    reward: float
    actions: tuple[WebArenaAction, ...]
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


def _usage_sum(total: dict, current: Mapping | None) -> None:
    current = current or {}
    for key, value in current.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    detail_tokens = 0
    for detail in current.get("cacheDetails", []) or []:
        ttl = str(detail.get("ttl", "")).strip().lower()
        ttl = {"pt5m": "5m", "pt1h": "1h", "pt30m": "30m"}.get(ttl, ttl)
        if ttl not in {"5m", "1h", "30m"}:
            raise ValueError(f"unsupported Bedrock cache write TTL: {ttl!r}")
        tokens = detail.get("inputTokens")
        if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or tokens < 0:
            raise ValueError("invalid cacheDetails inputTokens")
        total[f"cacheWriteInputTokens{ttl}"] = total.get(
            f"cacheWriteInputTokens{ttl}", 0
        ) + tokens
        detail_tokens += tokens
    if detail_tokens and current.get("cacheWriteInputTokens", 0) != detail_tokens:
        raise ValueError("cacheDetails token total does not match cacheWriteInputTokens")


def extract_goal(observation: str) -> str:
    match = _OBJECTIVE.search(observation or "")
    return match.group(1).strip() if match else ""


def bounded_observation(observation: str, limit: int = 14000) -> str:
    if len(observation) <= limit:
        return observation
    tail = observation.find("\nURL:")
    if 0 <= tail < limit:
        return observation
    suffix = observation[tail:] if tail >= 0 else ""
    return observation[:limit] + "\n... [accessibility tree truncated] ...\n" + suffix


def navigation_choices(observation: str) -> dict[str, str]:
    """Return finite, locally grounded actions that Jev may safely choose."""
    actions: list[str] = []
    for line in (observation or "").splitlines():
        match = _ELEMENT.match(line)
        if not match:
            continue
        element_id, description = match.groups()
        role = description.lstrip().split(" ", 1)[0].lower()
        has_popup_menu = "haspopup: menu" in description.lower()
        if (
            role in {"link", "button", "menuitem", "checkbox", "radio", "option"}
            and not has_popup_menu
        ):
            actions.append(f"click [{element_id}] — {description[:180]}")
        if has_popup_menu:
            actions.append(f"hover [{element_id}] — reveal submenu for {description[:160]}")
    actions.extend(["scroll [down] — reveal content below", "scroll [up] — reveal content above"])
    return {str(index): action for index, action in enumerate(actions)}


def executable_choice(choice: str) -> str:
    return choice.split(" — ", 1)[0]


def validate_browser_action(action: object, observation: str) -> str:
    if not isinstance(action, str):
        raise ValueError("action must be a string")
    action = action.strip()
    if not _ACTION.fullmatch(action):
        raise ValueError("action does not match the WebArena action grammar")
    match = re.match(r"^(?:click|hover|type) \[(\d+)\]", action)
    if match and not re.search(rf"^\s*\[{re.escape(match.group(1))}\]", observation, re.MULTILINE):
        raise ValueError(f"element id {match.group(1)} is absent from the current observation")
    return action


class BedrockJevWebArenaAgent:
    MODES = {"baseline", "optional"}

    def __init__(
        self,
        client: ConverseClient,
        model: str,
        decide: DecisionFunction,
        *,
        jev_model: str = "jevany-latest",
        max_output_tokens: int = 512,
        max_turns: int = 20,
        max_delegate_steps: int = 4,
        confidence_threshold: float = 0.55,
        history_limit: int = 8,
    ):
        if min(max_output_tokens, max_turns, max_delegate_steps, history_limit) < 1:
            raise ValueError("token, turn, delegation, and history limits must be positive")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.client = client
        self.model = model
        self.decide = decide
        self.jev_model = jev_model
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns
        self.max_delegate_steps = max_delegate_steps
        self.confidence_threshold = confidence_threshold
        self.history_limit = history_limit

    def _tools(self, mode: str) -> list[dict]:
        tools = [{"toolSpec": {
            "name": "browser_action",
            "description": (
                "Execute one WebArena action. Valid forms: click [id], hover [id], "
                "type [id] [content] [0|1], press [key], scroll [down|up], new_tab, "
                "tab_focus [index], close_tab, goto [url], go_back, go_forward, stop [answer]."
            ),
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"action": {"type": "string", "minLength": 3, "maxLength": 1000}},
                "required": ["action"],
                "additionalProperties": False,
            }},
        }}]
        if mode == "optional":
            tools.append({"toolSpec": {
                "name": "delegate_navigation",
                "description": (
                    "Delegate a bounded sequence of routine, finite-choice clicks, hovers, or scrolls to Jev. "
                    "Jev sees the fresh accessibility tree after every action. Retain control for typing, "
                    "free-form answers, ambiguous decisions, and irreversible/high-risk actions."
                ),
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "integer", "minimum": 1,
                                  "maximum": self.max_delegate_steps},
                        "subgoal": {"type": "string", "minLength": 1, "maxLength": 500},
                    },
                    "required": ["steps", "subgoal"],
                    "additionalProperties": False,
                }},
            }})
        return tools

    @staticmethod
    def _system(mode: str) -> str:
        common = (
            "You are controlling WebArena through its accessibility tree. Complete the objective and "
            "ground final answers in visible page evidence. Inspect every tool result as the fresh state. "
            "Call exactly one state-changing tool per response and give one short rationale. Never invent "
            "element ids. Use stop [answer] only when the task is complete."
        )
        if mode == "optional":
            common += (
                " You may autonomously delegate routine finite-choice navigation to the cheaper Jev "
                "decision model when that can replace several of your turns without reducing quality. "
                "Prefer delegation for a predictable chain of two or more menu hovers, clicks, or scrolls. "
                "Give it a concrete local subgoal; keep typing, synthesis, and risky choices yourself."
            )
        return common

    def run(self, env, *, task_id: int, mode: str, max_actions: int = 15) -> WebArenaEpisode:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        started = time.perf_counter()
        env.reset(task_id)
        observation = str(env.observe())
        goal = extract_goal(observation)
        if not goal:
            raise ValueError("WebArena reset returned no OBJECTIVE")
        actions: list[WebArenaAction] = []
        history: list[dict] = []
        transcript: list[dict] = []
        usage: dict = {}
        reward = 0.0
        done = False
        bedrock_calls = bedrock_failures = direct_actions = delegation_calls = 0
        jev_decisions = jev_failures = low_confidence_returns = protocol_repairs = 0
        bedrock_latency = jev_latency = 0.0
        termination_reason = "running"

        def execute(action: str, controller: str, confidence: float | None) -> dict:
            nonlocal observation, reward, done, direct_actions
            action = validate_browser_action(action, observation)
            out = env.step(f"In summary, the next action I will perform is ```{action}```")
            done = bool(out.get("terminated"))
            reward = float(out.get("reward") or 0.0) if done else 0.0
            if not done:
                observation = str(out.get("observation") or env.observe())
            if controller == "bedrock":
                direct_actions += 1
            actions.append(WebArenaAction(len(actions), controller, action, confidence, reward, done))
            history.append({"controller": controller, "action": action, "reward": reward})
            return {
                "action": action, "reward": reward, "done": done,
                "observation": "" if done else bounded_observation(observation),
            }

        def delegate(steps: int, subgoal: str) -> dict:
            nonlocal jev_decisions, jev_failures, jev_latency, low_confidence_returns
            delegated = []
            reason = "horizon"
            for _ in range(min(steps, self.max_delegate_steps, max_actions - len(actions))):
                if done:
                    reason = "terminal"
                    break
                choices = navigation_choices(observation)
                if not choices:
                    reason = "no_finite_actions"
                    break
                request = action_request(
                    f"Web task: {goal}\nDelegated navigation subgoal: {subgoal}",
                    bounded_observation(observation), choices, history[-self.history_limit:],
                    self.jev_model,
                )
                before = time.perf_counter()
                jev_decisions += 1
                try:
                    response = validate_response(request, self.decide(request))
                except Exception as error:
                    jev_failures += 1
                    reason = "backend_error"
                    return {"subgoal": subgoal, "executed": len(delegated),
                            "stop_reason": reason, "error_type": type(error).__name__,
                            "steps": delegated, "current_observation": bounded_observation(observation)}
                finally:
                    jev_latency += (time.perf_counter() - before) * 1000
                answer = response["answers"]["action"]
                key = str(answer["choice"])
                confidence = float(answer["confidence"])
                if confidence < self.confidence_threshold:
                    low_confidence_returns += 1
                    reason = "low_confidence"
                    break
                if key not in choices:
                    raise ValueError(f"Jev returned unknown navigation choice {key!r}")
                delegated.append(execute(executable_choice(choices[key]), "jev", confidence))
                if done:
                    reason = "terminal"
                    break
            return {"subgoal": subgoal, "executed": len(delegated), "stop_reason": reason,
                    "steps": delegated, "current_observation": "" if done else bounded_observation(observation)}

        messages = [{"role": "user", "content": [{"text": (
            f"Objective:\n{goal}\n\nCurrent browser state:\n{bounded_observation(observation)}\n\nStart acting."
        )}]}]
        repair_turns = 0
        while not done and len(actions) < max_actions and bedrock_calls < self.max_turns:
            request = {
                "modelId": self.model,
                "system": [{"text": self._system(mode)}],
                "messages": messages,
                "toolConfig": {"tools": self._tools(mode)},
                "inferenceConfig": {"maxTokens": self.max_output_tokens},
            }
            before = time.perf_counter()
            bedrock_calls += 1
            try:
                response = self.client.converse(**request)
            except Exception as error:
                bedrock_failures += 1
                transcript.append({"turn": bedrock_calls - 1, "error_type": type(error).__name__,
                                   "error": str(error)})
                termination_reason = "bedrock_backend_error"
                break
            finally:
                bedrock_latency += (time.perf_counter() - before) * 1000
            _usage_sum(usage, response.get("usage"))
            message = response.get("output", {}).get("message", {})
            content = message.get("content", [])
            messages.append({"role": "assistant", "content": content})
            text = "\n".join(block["text"] for block in content if "text" in block)
            calls = [block["toolUse"] for block in content if "toolUse" in block]
            transcript.append({"turn": bedrock_calls - 1, "text": text,
                               "tool": calls[0]["name"] if len(calls) == 1 else None,
                               "tool_input": calls[0].get("input") if len(calls) == 1 else None})
            if len(calls) != 1:
                repair_turns += 1
                protocol_repairs += 1
                if repair_turns > 2:
                    termination_reason = "model_protocol_error"
                    break
                messages.append({"role": "user", "content": [{"text":
                    "The task is not terminal. Call exactly one available tool."}]})
                continue
            repair_turns = 0
            call = calls[0]
            tool_input = call.get("input") or {}
            try:
                if call["name"] == "browser_action":
                    result = execute(tool_input.get("action"), "bedrock", None)
                elif call["name"] == "delegate_navigation" and mode == "optional":
                    steps = tool_input.get("steps")
                    subgoal = tool_input.get("subgoal")
                    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= self.max_delegate_steps:
                        raise ValueError("invalid delegation horizon")
                    if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal) > 500:
                        raise ValueError("invalid delegation subgoal")
                    delegation_calls += 1
                    result = delegate(steps, subgoal.strip())
                else:
                    raise ValueError(f"unavailable tool {call.get('name')!r}")
                status = "success"
            except (KeyError, TypeError, ValueError) as error:
                result = {"error": str(error), "observation": bounded_observation(observation)}
                status = "error"
                protocol_repairs += 1
            if not call.get("toolUseId"):
                termination_reason = "model_protocol_error"
                break
            messages.append({"role": "user", "content": [{"toolResult": {
                "toolUseId": call["toolUseId"], "content": [{"json": result}], "status": status,
            }}]})

        success = reward >= 1.0
        if termination_reason == "running":
            termination_reason = (
                "terminal_success" if success else "terminal_failure" if done
                else "max_actions" if len(actions) >= max_actions else "max_turns"
            )
        return WebArenaEpisode(
            mode, task_id, goal, success, reward, tuple(actions), bedrock_calls,
            bedrock_failures, direct_actions, delegation_calls, jev_decisions, jev_failures,
            low_confidence_returns, protocol_repairs, termination_reason, usage,
            bedrock_latency, jev_latency, (time.perf_counter() - started) * 1000,
            tuple(transcript),
        )
