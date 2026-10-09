"""Text planning through an OpenAI-compatible Chat Completions endpoint."""
import json
import urllib.request

from .client import _open_json, _validate_endpoint


class ChatCompletionsGenerator:
    """A TextGenerator for non-streaming text responses, with no optional dependencies.

    `base_url` is the API prefix, usually ending in `/v1`. Local servers can
    omit `api_key`; remote servers require HTTPS. HTTP failures preserve the
    server's explanation through DecisionHTTPError, and malformed responses
    raise ValueError.
    """

    def __init__(
        self, model: str, *, base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None, timeout: float = 120,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("planner model must be a non-empty string")
        self.base_url = _validate_endpoint(base_url, timeout)
        self.model, self.api_key, self.timeout = model, api_key, timeout

    def generate(self, prompt: str, params: dict | None = None) -> dict:
        """Generate one text response; parameter support also depends on the served model."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("planner prompt must be a non-empty string")
        names = {
            "max_output_tokens": "max_completion_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "stop_sequences": "stop",
        }
        params = params or {}
        unknown = set(params) - set(names)
        if unknown:
            raise ValueError(f"unsupported Chat Completions generation parameters: {sorted(unknown)}")
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, **{names[key]: value for key, value in params.items()},
        }
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        response = _open_json(urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            method="POST", headers=headers,
        ), self.timeout)
        choices = response.get("choices") if isinstance(response, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("chat completion must contain a non-empty choices list")
        choice = choices[0]
        message = choice.get("message")
        if (not isinstance(message, dict) or not isinstance(message.get("content"), str)
                or not message["content"].strip()):
            raise ValueError(f"chat completion must contain message text; finish_reason={choice.get('finish_reason')!r}")
        return {
            "text": message["content"], "usage": response.get("usage"),
            "stop_reason": choice.get("finish_reason"),
        }
