"""Serve packaged replays and optional CPU environments on the loopback interface."""
import argparse
import base64
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.util
import io
import json
import mimetypes
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
from tempfile import TemporaryDirectory
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import unquote, urlsplit
import webbrowser

from jevany.api import validate_response
from jevany.client import DecisionClient, JevClient
from . import CASES, DemoEnvironment, make_environment
from .context import decision_request, focus_crafter_request

ROOT = Path(__file__).parent
# One editable choice question needs more room than an action click, and nothing
# the page posts is large. Each path keeps its own bound.
BODY_LIMITS = {"/api/start": 4096, "/api/step": 4096, "/api/connect": 4096,
               "/api/images": 4096, "/api/decide": 32768}
MAX_CUSTOM_OPTIONS = 12
DEFAULT_BASE_URL = "http://127.0.0.1:8008"
# GET /v1/models returns metadata, so the connection test holds the run lock for
# much less time than a decision may take.
PROBE_TIMEOUT = 30


def frame_uri(image) -> str:
    """Encode an RGB array as a JPEG data URI."""
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def text_field(value: Any, name: str, limit: int) -> str:
    """A required, stripped, length-bounded string from the browser."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    value = value.strip()
    if len(value) > limit:
        raise ValueError(f"{name} must be at most {limit} characters")
    return value


def identities(entry: dict[str, Any]) -> set[str]:
    """The ids a /v1/models entry answers to, ignoring anything malformed."""
    identity, aliases = entry.get("id"), entry.get("aliases")
    found = {identity} if isinstance(identity, str) and identity.strip() else set()
    if isinstance(aliases, list):
        found |= {alias for alias in aliases if isinstance(alias, str) and alias.strip()}
    return found


def model_descriptor(base_url: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Validate the /v1/models fields the Playground consumes, before adopting an endpoint.

    A compatible endpoint can report anything; a connection is only adopted when the
    fields the page and the media rules read have the shape they are used with.
    """
    identity = entry.get("id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(f"{base_url} did not report a model id in GET /v1/models")
    aliases = entry.get("aliases")
    if aliases is not None and (not isinstance(aliases, list)
                                or not all(isinstance(alias, str) for alias in aliases)):
        raise ValueError(f"{base_url} reported aliases for {identity!r} that are not a list of names")
    sections = {}
    for name in ("capabilities", "limits"):
        value = entry.get(name)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{base_url} reported {name} for {identity!r} that is not an object")
        sections[name] = value or {}
    media_types = sections["capabilities"].get("media_types")
    if media_types is not None and (not isinstance(media_types, list)
                                    or not all(isinstance(item, str) for item in media_types)):
        raise ValueError(f"{base_url} reported capabilities.media_types for {identity!r} "
                         "that is not a list of media types")
    media_enabled = sections["limits"].get("media_enabled", False)
    if type(media_enabled) is not bool:
        raise ValueError(f"{base_url} reported limits.media_enabled for {identity!r} "
                         "that is not a boolean")
    labels = {}
    for name in ("base", "device", "decision_mode", "readout"):
        value = entry.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{base_url} reported {name} for {identity!r} that is not a name")
        labels[name] = value
    return {
        "id": identity.strip(), "aliases": list(aliases or []), **labels,
        "media_types": list(media_types or []),
        "media_enabled": media_enabled,
        "limits": sections["limits"],
    }


class DemoApplication:
    """One local interactive run; concurrent requests cannot change its state."""

    def __init__(self, client: DecisionClient | None = None, *, images: bool = True,
                 factory: Callable[[str, int], DemoEnvironment] = make_environment,
                 media_root: str | Path | None = None, timeout: float = 120):
        self.client, self.images, self.factory = client, images, factory
        self.timeout = timeout
        self.lock = threading.Lock()
        self.env, self.case = None, None
        self.revision = 0
        self.trace = []
        self.snapshot = None
        # The last successful GET /v1/models for the current client; None means the
        # endpoint is configured but has never answered, so its identity and media
        # support are still unknown.
        self.probe: dict[str, Any] | None = None
        self.checked: str | None = None
        self.media_root = Path(media_root).resolve() if media_root is not None else None
        if self.media_root is not None:
            self.media_root.mkdir(parents=True, exist_ok=True)

    def media_state(self) -> dict[str, Any]:
        """Whether an image can actually reach the connected model, and why not.

        A remote endpoint has to prove it: until GET /v1/models has answered, its
        media support is unknown and image input stays off. An in-process model
        (jevany.runtime.JevModel or a caller's own client) takes trusted local
        paths, so it needs no probe.
        """
        remote = isinstance(self.client, JevClient)
        reasons = []
        if self.client is None:
            reasons.append("connect a model first")
        if remote and self.media_root is None:
            reasons.append("start the playground with --media-root inside the server's JEVANY_MEDIA_ROOT")
        if remote and self.probe is None:
            reasons.append("choose Test and connect first; this endpoint's media support is unknown")
        model_types: list[str] | None = None
        server_enabled: bool | None = None
        if self.probe is not None:
            model_types, server_enabled = self.probe["media_types"], self.probe["media_enabled"]
            if "image" not in model_types:
                reasons.append("this checkpoint is text-only")
            elif not server_enabled:
                reasons.append("the server has no JEVANY_MEDIA_ROOT")
        return {
            "transport": "shared-files" if self.media_root is not None else "inline",
            "model_media_types": model_types, "server_media_enabled": server_enabled,
            "verified": self.probe is not None,
            "usable": not reasons, "reason": "; ".join(reasons) or None,
        }

    @property
    def image_requests(self) -> bool:
        """Images are attached only when requested and actually deliverable."""
        return bool(self.images) and self.media_state()["usable"]

    def connection(self) -> dict[str, Any]:
        """Configured endpoint and, separately, what a real round trip confirmed."""
        remote = isinstance(self.client, JevClient)
        served = self.probe
        return {
            "configured": self.client is not None,
            "editable": self.client is None or remote,
            "base_url": self.client.base_url if remote else None,
            "model": self.client.model_id if self.client is not None else None,
            "reachable": self.probe is not None,
            "checked": self.checked,
            "served": {
                "id": served["id"], "aliases": served["aliases"], "base": served["base"],
                "device": served["device"], "decision_mode": served["decision_mode"],
                "readout": served.get("readout"),
                "limits": served["limits"],
            } if served is not None else None,
            "media": self.media_state(),
            "images": bool(self.images), "image_requests": self.image_requests,
            "default_base_url": DEFAULT_BASE_URL,
        }

    def connect(self, base_url: Any, model: Any = None, timeout: Any = None) -> dict[str, Any]:
        """Adopt an endpoint only after GET /v1/models answers; keep the working one otherwise."""
        if self.client is not None and not isinstance(self.client, JevClient):
            raise ValueError("this playground runs an in-process model; its URL cannot be changed")
        base_url = text_field(base_url, "model URL", 2048)
        requested = text_field(model, "model id", 200) if model not in (None, "") else None
        timeout = self.timeout if timeout in (None, "") else timeout
        if type(timeout) not in (int, float) or not 0 < float(timeout) <= 600:
            raise ValueError("timeout must be a number of seconds above 0 and at most 600")
        # JevClient keeps its URL policy: HTTP only on loopback, no credentials in the URL.
        probe_timeout = min(float(timeout), PROBE_TIMEOUT)
        candidate = JevClient(base_url, timeout=probe_timeout, model=requested or "jevany-latest")
        try:
            served = candidate.models()
        except urllib.error.HTTPError as error:
            raise ValueError(f"{base_url} answered but rejected GET /v1/models: {error}") from error
        except TimeoutError as error:
            raise ValueError(f"{base_url} did not answer within {probe_timeout:g} seconds") from error
        except urllib.error.URLError as error:
            raise ValueError(f"cannot reach {base_url}: {error.reason}. Start a server with "
                             "'jevany serve --checkpoint <checkpoint> --port 8008'.") from error
        except ValueError as error:
            raise ValueError(f"{base_url} is not a System One endpoint: {error}") from error
        entry, names = None, set()
        for item in served:
            answers_to = identities(item)
            names |= answers_to
            if entry is None and (requested is None or requested in answers_to):
                entry = item
        if entry is None:
            raise ValueError(f"{base_url} serves {', '.join(sorted(names)) or 'no named model'}; "
                             f"it does not serve {requested!r}")
        # Everything that could reject this endpoint happens before it is adopted, so a
        # malformed descriptor leaves the previous connection, probe and image setting intact.
        descriptor = model_descriptor(base_url, entry)
        adopted = JevClient(base_url, timeout=float(timeout), model=requested or descriptor["id"])
        # A newly connected model starts text-only; images are enabled explicitly
        # once this probe shows the model and the server both accept them.
        self.client, self.probe, self.images = adopted, descriptor, False
        self.checked = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self.connection()

    def set_images(self, enabled: Any) -> dict[str, Any]:
        """Enable image input only when the model, the server and the transport all allow it."""
        if type(enabled) is not bool:
            raise ValueError("send images as true or false")
        if enabled:
            if self.client is None:
                raise ValueError("connect a model before enabling image input")
            media = self.media_state()
            if not media["usable"]:
                raise ValueError(f"image input is unavailable: {media['reason']}")
        self.images = enabled
        return self.connection()

    def decide(self, state: Any, question: Any, options: Any) -> dict[str, Any]:
        """One editable choice question against the connected model; no environment required."""
        if self.client is None:
            raise ValueError("connect a model first: enter its URL in the model connection panel")
        state = text_field(state, "state", 6000)
        question = text_field(question, "question", 2000)
        if not isinstance(options, list) or not 2 <= len(options) <= MAX_CUSTOM_OPTIONS:
            raise ValueError(f"give between 2 and {MAX_CUSTOM_OPTIONS} candidate options")
        criteria: dict[str, Any] = {}
        for item in options:
            item = {"name": item} if isinstance(item, str) else item
            if not isinstance(item, dict):
                raise ValueError("each option needs a name and an optional description")
            name = text_field(item.get("name"), "option name", 120)
            if name in criteria:
                raise ValueError(f"duplicate option {name!r}")
            description = item.get("description")
            criteria[name] = (text_field(description, "option description", 600)
                              if description not in (None, "") else None)
        request = {"model": self.client.model_id, "state": state,
                   "questions": {"decision": {"type": "choice", "instructions": question,
                                              "criteria": criteria}}}
        started = time.monotonic()
        response = validate_response(request, self.client(request))
        answer = response["answers"]["decision"]
        return {
            "model": response.get("model") or self.client.model_id, "options": list(criteria),
            "choice": answer["choice"], "confidence": answer["confidence"],
            "probabilities": answer["probabilities"],
            "seconds": round(time.monotonic() - started, 3),
            "latency_ms": response.get("latency_ms"),
        }

    def config(self) -> dict[str, Any]:
        return {
            "cases": CASES,
            "model": self.client.model_id if self.client else None,
            "images": self.images,
            "media_transport": "shared-files" if self.media_root is not None else "inline",
            "connection": self.connection(),
            "max_custom_options": MAX_CUSTOM_OPTIONS,
            "installed": {key: all(importlib.util.find_spec(name) is not None
                                   for name in (meta["package"], "PIL", "numpy"))
                          for key, meta in CASES.items()},
        }

    def start(self, case: str, seed: int) -> dict[str, Any]:
        if case not in CASES:
            raise ValueError(f"unknown environment {case!r}")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer between 0 and 2147483647")
        try:
            env = self.factory(case, seed)
        except ImportError as error:
            raise ValueError(f"This environment needs the {CASES[case]['extra']} extra. Run: "
                             f"python -m pip install -e '.[{CASES[case]['extra']}]'") from error
        if self.env is not None:
            self.env.close()
        self.env, self.case, self.seed = env, case, seed
        self.trace = []
        self.revision += 1
        self.snapshot = self._state([frame_uri(env.render())])
        return self.snapshot

    def _state(self, frames: list[str]) -> dict[str, Any]:
        return {
            "case": self.case, "revision": self.revision, "seed": self.seed,
            "observation": self.env.observe(), "frames": frames,
            "actions": {key: self.env.ACTION_LOOKUP[key] for key in self.env.get_all_actions()},
            "step": len(self.trace), "done": self.env.done, "success": self.env.success,
            "feedback": self.env.feedback,
            "frame_duration_ms": getattr(self.env, "frame_duration_ms", 65),
            "step_pause_ms": getattr(self.env, "step_pause_ms", 950),
        }

    def step(self, revision: int, *, action: str | None = None,
             model: bool = False) -> dict[str, Any]:
        if self.env is None:
            raise ValueError("start a new run first")
        if type(revision) is not int or revision != self.revision:
            raise ValueError("this run changed in another request; start a new run")
        if self.env.done:
            raise ValueError("this episode has ended; start a new run")
        if type(model) is not bool or (model and action is not None):
            raise ValueError("choose either a manual action or a model decision")
        before = self.env.observe()
        actions = {key: self.env.ACTION_LOOKUP[key] for key in self.env.get_all_actions()}
        probabilities = None
        priority, subgoal = None, None
        started = time.monotonic()
        if model:
            if self.client is None:
                raise ValueError("connect a model first: enter its URL in the model connection panel, "
                                 "or start the playground with --base-url")
            request = decision_request(self.case, self.client.model_id, before, actions, self.trace)
            builder = getattr(self.env, "decision_request", None)
            if builder is not None:
                history = [{
                    "action": entry["action"], "feedback": entry["feedback"],
                    "holding_peg_now": entry["next_observation"]["object_between_both_fingers"],
                } for entry in self.trace[-3:]]
                request = builder(self.client.model_id, history)
            if isinstance(request["state"], dict):
                subgoal = request["state"].get("current_subgoal")
            images = self.image_requests
            storage = (TemporaryDirectory(prefix="jevany-frame-", dir=self.media_root)
                       if images and self.media_root is not None else nullcontext())
            with storage as directory:
                if images:
                    image = self.env.render()
                    if directory is None:
                        uri = frame_uri(image)
                    else:
                        from PIL import Image
                        path = Path(directory) / "observation.png"
                        Image.fromarray(image).save(path)
                        uri = str(path)
                    request["media"] = [{"type": "image", "uri": uri}]
                if self.case == "crafter" and "position_xy" in before:
                    request, priority = focus_crafter_request(self.client, request, before)
                    subgoal = "Objective: " + priority["choice"].replace("_", " ") + "."
                response = validate_response(request, self.client(request))
            answer = response["answers"]["action"]
            action, probabilities = answer["choice"], answer["probabilities"]
        if not isinstance(action, str) or action not in actions:
            raise ValueError(f"unavailable action {action!r}; choose a displayed control")
        after, reward, done, info = self.env.step(action)
        self.revision += 1
        decision = {
            "controller": "model" if model else "manual", "action": action,
            "probabilities": probabilities, "observation": before, "next_observation": after,
            "reward": reward, "done": done, "success": info["success"],
            "feedback": self.env.feedback, "seconds": round(time.monotonic() - started, 3),
            "subgoal": subgoal, "priority": priority,
        }
        self.trace.append(decision)
        frames = [frame_uri(image) for image in (self.env.frames or [self.env.render()])]
        self.snapshot = {**self._state(frames), "decision": decision}
        return self.snapshot

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def make_server(app: DemoApplication, port: int = 8090) -> ThreadingHTTPServer:
    """Bind only to localhost; serve assets, replay data, and one live run."""
    class Handler(BaseHTTPRequestHandler):
        def send(self, value, status=200, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/config":
                return self.send(app.config())
            if path == "/api/trace":
                if not app.lock.acquire(blocking=False):
                    return self.send({"error": "wait for the current action to finish"}, 409)
                try:
                    return self.send({"case": app.case, "seed": getattr(app, "seed", None),
                                      "model": app.client.model_id if app.client else None,
                                      "steps": app.trace})
                finally:
                    app.lock.release()
            path = "/static/index.html" if path == "/" else path
            if not path.startswith(("/static/", "/recordings/")):
                return self.send({"error": "not found"}, 404)
            file = (ROOT / path.lstrip("/")).resolve()
            if not file.is_relative_to(ROOT.resolve()) or not file.is_file():
                return self.send({"error": "not found"}, 404)
            return self.send(file.read_bytes(), content_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream")

        def do_POST(self):
            # JSON plus same-origin checks keep unrelated browser pages from controlling a run.
            origin = self.headers.get("Origin")
            expected_origin = f"http://{self.headers.get('Host')}"
            if origin is not None and origin != expected_origin:
                return self.send({"error": "cross-origin actions are not allowed"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self.send({"error": "send application/json"}, 415)
            limit = BODY_LIMITS.get(self.path)
            if limit is None:
                return self.send({"error": "not found"}, 404)
            if not app.lock.acquire(blocking=False):
                return self.send({"error": "another action is still running"}, 409)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= limit:
                    raise ValueError(f"JSON action body must be between 1 and {limit} bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("send a JSON object")
                if self.path == "/api/start":
                    result = app.start(body.get("case"), body.get("seed", 17))
                elif self.path == "/api/connect":
                    result = app.connect(body.get("base_url"), body.get("model"), body.get("timeout"))
                elif self.path == "/api/images":
                    result = app.set_images(body.get("enabled"))
                elif self.path == "/api/decide":
                    result = app.decide(body.get("state"), body.get("question"), body.get("options"))
                else:
                    result = app.step(body.get("revision"), action=body.get("action"),
                                      model=body.get("model", False))
                self.send(result)
            except (ValueError, TypeError) as error:
                self.send({"error": str(error)}, 400)
            except urllib.error.HTTPError as error:
                # The status line already names the status; show the server's reason.
                self.send({"error": f"the model server rejected the request: {error}"}, 502)
            except Exception as error:
                self.send({"error": f"{type(error).__name__}: {error}"}, 502)
            finally:
                app.lock.release()

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jevany demo", description="Open the local JevAny playground. Replays need no GPU or model.")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--base-url", help="JevAny HTTP server for live model decisions; "
                                           "the page can also connect one without restarting")
    parser.add_argument("--model", default="jevany-latest", help="model identity sent to the server")
    parser.add_argument("--timeout", type=float, default=120, help="model request timeout in seconds")
    parser.add_argument("--text-only", action="store_true", help="send measured state without an image")
    parser.add_argument("--media-root", help="write request images under this shared JEVANY_MEDIA_ROOT directory")
    parser.add_argument("--no-open", action="store_true", help="print the URL without opening a browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    if args.text_only and args.media_root:
        parser.error("--media-root cannot be combined with --text-only")
    client = JevClient(args.base_url, timeout=args.timeout, model=args.model) if args.base_url else None
    app = DemoApplication(client, images=not args.text_only, media_root=args.media_root,
                          timeout=args.timeout)
    try:
        server = make_server(app, args.port)
    except OSError as error:
        parser.exit(2, f"Cannot open demo port {args.port}: {error}. Try --port 8091.\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"JevAny playground: {url}\nReplays are ready. Connect a model in the page, or pass --base-url. "
          "Press Ctrl+C to stop.", flush=True)
    if client is not None:
        print(f"Model endpoint configured: {args.base_url}. Text decisions work now; press Test and connect "
              "in the page to confirm what it serves and to enable image input.", flush=True)
    if not args.no_open and (os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32")):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with app.lock:
            app.close()
