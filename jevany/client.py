"""Python clients using the same System One request and response as HTTP."""
import io
import json
import math
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .api import JSONContent, Media, Question, SystemOneRequest, validate_response

DETAIL_LIMIT = 400
# A rejected request can be answered with a long page by a gateway in front of the
# model. Only this much of a failed response is copied; successful decision
# responses are read whole.
ERROR_BODY_LIMIT = 64 * 1024


def error_detail(body: bytes, limit: int = DETAIL_LIMIT) -> str:
    """One readable line explaining a failed response, from a JSON error body or any other payload."""
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        for key in ("detail", "message", "error"):
            if key in payload:
                value = payload[key]
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                break
        else:
            text = json.dumps(payload, ensure_ascii=False)
    elif payload is not None:
        text = json.dumps(payload, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


class DecisionHTTPError(urllib.error.HTTPError):
    """An HTTP failure that reports the server's own explanation.

    `str(error)` adds the server's `detail` (or the first line of a non-JSON body)
    to the usual status line, so an unknown model or a disabled media root says so
    instead of only `HTTP Error 422: Unprocessable Entity`. `code`, `status`,
    `msg`, `reason`, `headers` and `read()` behave as urllib's HTTPError, and
    `detail` and `body` expose the parsed explanation and the raw bytes. The body
    holds at most the first ERROR_BODY_LIMIT bytes of the failed response, and
    `read()` returns that copy. Request headers, including any Authorization
    header, are never part of the message.
    """

    def __init__(self, error: urllib.error.HTTPError, body: bytes) -> None:
        super().__init__(error.filename, error.code, error.msg, error.hdrs, io.BytesIO(body))
        self.body = body
        self.detail = error_detail(body)

    def __str__(self) -> str:
        status = f"HTTP Error {self.code}: {self.msg}"
        return f"{status}: {self.detail}" if self.detail else status


def _response_body(error: urllib.error.HTTPError, limit: int = ERROR_BODY_LIMIT) -> bytes:
    """Up to `limit` bytes of a failed response, releasing the original response.

    The status and the explanation at the start of the body are what callers need;
    reading an unbounded error page into memory is not.
    """
    try:
        return error.read(limit)
    except OSError:
        return b""
    finally:
        try:
            error.close()
        except OSError:
            pass


def _validate_endpoint(base_url: str, timeout: float) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("decision endpoint must use HTTP or HTTPS and have a hostname")
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("non-loopback decision endpoints must use HTTPS")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials, a query or a fragment")
    return base_url.rstrip("/")


def _open_json(http_request: urllib.request.Request, timeout: float) -> Any:
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise DecisionHTTPError(error, _response_body(error)) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{http_request.full_url} did not return JSON: {error}") from error


class DecisionClient:
    """Common convenience methods for local and HTTP inference."""

    def system_one(
        self,
        state: JSONContent,
        questions: dict[str, Question | dict],
        *,
        model: str | None = None,
        media: list[Media | dict] | None = None,
    ) -> dict[str, Any]:
        """Evaluate typed questions; return the JSON-compatible response body."""
        request = SystemOneRequest(
            state=state, questions=questions, model=self.model_id if model is None else model,
            media=media or [],
        )
        return self(request)

    def __call__(self, request: SystemOneRequest | dict) -> dict[str, Any]:
        raise NotImplementedError

    def models(self) -> list[dict[str, Any]]:
        """Return the identities, capabilities and limits available through this client."""
        raise NotImplementedError


class JevClient(DecisionClient):
    """Call a JevAny server or a compatible TypeSafe endpoint.

    Invalid requests raise ValueError before sending. Connection errors propagate
    from urllib; HTTP failures raise DecisionHTTPError, which is an HTTPError
    carrying the server's explanation. Malformed responses raise ValueError.
    """

    def __init__(
        self, base_url: str = "http://127.0.0.1:8008",
        api_key: str | None = "local", timeout: float = 120,
        model: str = "jevany-latest",
    ) -> None:
        self.base_url = _validate_endpoint(base_url, timeout)
        self.url = self.base_url + "/v1/systemone"
        self.api_key, self.timeout, self.model_id = api_key, timeout, model

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _open(self, http_request: urllib.request.Request) -> Any:
        return _open_json(http_request, self.timeout)

    def models(self) -> list[dict[str, Any]]:
        """GET /v1/models: what this deployment serves, with its capabilities and limits.

        One real round trip, so it also proves the endpoint is reachable and ready.
        """
        payload = self._open(urllib.request.Request(
            self.base_url + "/v1/models", method="GET", headers=self._headers(),
        ))
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list) or not models or not all(isinstance(item, dict) for item in models):
            raise ValueError("GET /v1/models did not return a models list; is this a System One endpoint?")
        return models

    def __call__(self, request: SystemOneRequest | dict) -> dict[str, Any]:
        validated = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        http_request = urllib.request.Request(
            self.url, data=validated.model_dump_json().encode(), method="POST",
            headers={**self._headers(), "content-type": "application/json"},
        )
        return validate_response(validated, self._open(http_request))
