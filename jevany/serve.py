# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""FastAPI server for prefill-only decisions.

Run: uv run --extra serve python -m jevany.serve --run runs/rlcr --port 8008

TypeSafe-compatible: POST /v1/systemone and GET /v1/models (no auth). JEVANY_PREFIX_CACHE /
JEVANY_PREFIX_MIN_TOKENS size the state-prefix cache; JEVANY_DATE_FACTS=1 enables deterministic date preprocessing.
"""
import argparse, hashlib, os, re, subprocess, threading, time
from dataclasses import dataclass, field, replace
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import Checkpoint, LoadOptions, is_hub_id
from .device import default_device, sync

# Interactive limits are larger than the frozen 2048-token benchmark window.
# Training admission uses the tighter constants in jevany.model.
INFER_MAX_STATE, INFER_MAX_BRANCH, INFER_MAX_PACKED = 8192, 8192, 8192
PREFIX_CACHE_SIZE = int(os.environ.get("JEVANY_PREFIX_CACHE", "4"))          # states kept (KV + hidden); 0 disables
PREFIX_MIN_TOKENS = int(os.environ.get("JEVANY_PREFIX_MIN_TOKENS", "384"))   # below this the branch-only pass is not faster on MPS (per-op overhead dominates)
DATE_FACTS = os.environ.get("JEVANY_DATE_FACTS", "0") == "1"
MEDIA_ROOT = os.environ.get("JEVANY_MEDIA_ROOT")
MEDIA_MAX_BYTES = int(os.environ.get("JEVANY_MEDIA_MAX_BYTES", str(50 * 1024 * 1024)))
MEDIA_TOTAL_BYTES = int(os.environ.get("JEVANY_MEDIA_TOTAL_BYTES", str(100 * 1024 * 1024)))
MEDIA_MAX_PIXELS = int(os.environ.get("JEVANY_MEDIA_MAX_PIXELS", str(4096 * 4096)))
MEDIA_MAX_VIDEO_FRAMES = int(os.environ.get("JEVANY_MEDIA_MAX_VIDEO_FRAMES", "3600"))
if min(MEDIA_MAX_BYTES, MEDIA_TOTAL_BYTES, MEDIA_MAX_PIXELS, MEDIA_MAX_VIDEO_FRAMES) <= 0:
    raise ValueError("JEVANY media limits must be positive")
if MEDIA_TOTAL_BYTES < MEDIA_MAX_BYTES:
    raise ValueError("JEVANY_MEDIA_TOTAL_BYTES must be at least JEVANY_MEDIA_MAX_BYTES")

COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _digest(path):
    path = Path(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _source_revision():
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _public_base_name(base):
    value = str(base)
    return Path(value).name if Path(value).is_absolute() else value


@dataclass
class Server:
    """The loaded checkpoint and the state-prefix cache shared by every request (one model, one lock)."""
    checkpoint: Checkpoint
    tok: object
    model: object
    device: str
    model_id: str = "jevany-27b"
    model_aliases: tuple[str, ...] = ("jevany-latest",)
    strict_model_id: bool = False
    provenance: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    prefix_cache: dict = field(default_factory=dict)   # (state token ids, option_isolation) -> prefix, in LRU order
    prefix_hits: int = 0
    prefix_misses: int = 0

    def probs(self, rec):
        """Score one request, using the state-prefix cache only for the default decide/close readout."""
        try:
            enc = self.model.encode(self.tok, rec, max_state=INFER_MAX_STATE,
                                    max_branch=INFER_MAX_BRANCH, strict=True)
            if len(enc["ids"]) > INFER_MAX_PACKED:
                raise ValueError(f"request exceeds {INFER_MAX_PACKED} packed tokens: {len(enc['ids'])}")
        except ValueError as e: raise HTTPException(422, str(e))
        Ls = enc["seg"].count(0); key = (tuple(enc["ids"][:Ls]), bool(enc.get("option_isolation")))
        cache, hit = self.prefix_cache, False
        with self.lock:
            sync(self.device); t = time.time()
            eligible = (PREFIX_CACHE_SIZE and Ls >= PREFIX_MIN_TOKENS
                        and not enc.get("multimodal") and self.model.supports_prefix_cache())
            if eligible and key in cache:
                prefix = cache.pop(key)                            # pop + reinsert = LRU order
                ps = self.model.probs_with_prefix(enc, prefix); cache[key] = prefix
                self.prefix_hits += 1; hit = True
            elif eligible:
                ps, prefix = self.model.probs_and_prefix(enc)      # one pass, and the state prefix is kept for next time
                cache[key] = prefix
                while len(cache) > PREFIX_CACHE_SIZE: cache.pop(next(iter(cache)))
                self.prefix_misses += 1
            else:
                ps = self.model.probs(enc)
            sync(self.device); dt = time.time() - t
        return [p.tolist() for p in ps], {"tokens": len(enc["ids"]), "state_tokens": Ls, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": hit}

    def answer(self, req):
        """The /v1/systemone response body for one request."""
        if self.strict_model_id and req.model not in {self.model_id, *self.model_aliases}:
            raise HTTPException(404, f"unknown model {req.model!r}")
        rec, meta = to_record(prepare(req))
        ps, m = self.probs(rec)
        answers = to_answers(ps, meta)
        return {"model": self.model_id, "answers": answers,
                "inference_temperature": self.model.head.temperature,
                "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(self.tok, answers)},
                "latency_ms": m["latency_ms"]}


def _validate_media_file(item, path):
    try:
        if item.type == "image":
            from PIL import Image
            with Image.open(path) as image:
                if image.width * image.height > MEDIA_MAX_PIXELS:
                    raise HTTPException(422, f"image exceeds {MEDIA_MAX_PIXELS} pixels")
                image.verify()
        else:
            import av
            with av.open(str(path)) as container:
                streams = container.streams.video
                if not streams:
                    raise HTTPException(422, "video file has no video stream")
                stream = streams[0]
                if stream.width * stream.height > MEDIA_MAX_PIXELS:
                    raise HTTPException(422, f"video frame exceeds {MEDIA_MAX_PIXELS} pixels")
                if not stream.frames:
                    raise HTTPException(422, "video frame count is unavailable")
                if stream.frames > MEDIA_MAX_VIDEO_FRAMES:
                    raise HTTPException(422, f"video exceeds {MEDIA_MAX_VIDEO_FRAMES} frames")
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(422, "install the multimodal extra to serve media")
    except Exception:
        raise HTTPException(422, "media file could not be validated")


def prepare(req):
    """Opt-in preprocessing applied to every request before the model sees it."""
    updates = {}
    if DATE_FACTS:
        updates["state"] = with_date_facts(req.state)
    if req.media:
        if not MEDIA_ROOT:
            raise HTTPException(422, "media is disabled; the server operator must set JEVANY_MEDIA_ROOT")
        try:
            root = Path(MEDIA_ROOT).resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(422, "configured media root is unavailable")
        if not root.is_dir():
            raise HTTPException(422, "configured media root is not a directory")
        resolved, total = [], 0
        for item in req.media:
            if "://" in item.uri:
                raise HTTPException(422, "network media URIs are not allowed")
            candidate = Path(item.uri)
            try:
                path = candidate.resolve(strict=True) if candidate.is_absolute() else (root / candidate).resolve(strict=True)
            except (OSError, RuntimeError):
                raise HTTPException(422, "media file does not exist")
            try:
                path.relative_to(root)
            except ValueError:
                raise HTTPException(422, "media path is outside JEVANY_MEDIA_ROOT")
            if not path.is_file():
                raise HTTPException(422, "media path is not a regular file")
            size = path.stat().st_size
            if size > MEDIA_MAX_BYTES:
                raise HTTPException(422, f"media file exceeds {MEDIA_MAX_BYTES} bytes")
            total += size
            if total > MEDIA_TOTAL_BYTES:
                raise HTTPException(422, f"media files exceed {MEDIA_TOTAL_BYTES} bytes in total")
            _validate_media_file(item, path)
            resolved.append(item.model_copy(update={"uri": str(path)}))
        updates["media"] = resolved
    return req.model_copy(update=updates) if updates else req


app = FastAPI(title="JevAny")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def server() -> Server:
    return app.state.server


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    return server().answer(req)


@app.get("/v1/models")
def models():
    s = server()
    model_id = getattr(s, "model_id", "jevany-27b")
    aliases = getattr(s, "model_aliases", ("jevany-latest",))
    run_label = model_id if getattr(s, "strict_model_id", False) else s.checkpoint.requested
    return {"models": [{"id": model_id, "aliases": list(aliases), "run": run_label,
                        "base": _public_base_name(s.checkpoint.meta.base),
                        "lora": s.checkpoint.meta.lora, "device": s.device, "temperature": s.model.head.temperature,
                        "strict_model_id": getattr(s, "strict_model_id", False),
                        "provenance": getattr(s, "provenance", {}),
                        "limits": {"state_tokens": INFER_MAX_STATE, "branch_tokens": INFER_MAX_BRANCH,
                                   "packed_tokens": INFER_MAX_PACKED,
                                   "media_enabled": bool(MEDIA_ROOT),
                                   "media_max_file_bytes": MEDIA_MAX_BYTES,
                                   "media_max_total_bytes": MEDIA_TOTAL_BYTES,
                                   "media_max_pixels": MEDIA_MAX_PIXELS,
                                   "media_max_video_frames": MEDIA_MAX_VIDEO_FRAMES},
                        "prefix_cache": {"size": PREFIX_CACHE_SIZE, "min_state_tokens": PREFIX_MIN_TOKENS, "hits": s.prefix_hits,
                                         "misses": s.prefix_misses, "cached_states": len(s.prefix_cache)}}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rlcr")
    ap.add_argument("--fallback", default="runs/sft")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model-id", default=os.environ.get("JEVANY_MODEL_ID", "jevany-27b"))
    ap.add_argument("--model-alias", action="append", default=None)
    ap.add_argument("--strict-model-id", action="store_true",
                    default=os.environ.get("JEVANY_STRICT_MODEL_ID", "0") == "1")
    ap.add_argument("--code-revision", default=os.environ.get("JEVANY_CODE_REVISION"))
    a = ap.parse_args()
    actual_revision = _source_revision()
    if a.strict_model_id:
        if not a.model_id or not COMMIT_RE.fullmatch(a.code_revision or ""):
            ap.error("strict model identity requires --model-id and a full 40-character --code-revision")
        if actual_revision != a.code_revision:
            ap.error(f"--code-revision {a.code_revision!r} does not match source revision {actual_revision!r}")
    run = a.run if is_hub_id(a.run) or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    dev = a.device or default_device()
    opts = LoadOptions.from_env()
    if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")   # serving default on Apple GPUs (parity measured)
    ck = Checkpoint(run)
    tok, model = ck.load(dev, opts)
    artifact_hashes = {}
    for name in ("head.pt", "adapter_model.safetensors", "adapter_config.json"):
        path = ck.file(name)
        if path.is_file():
            artifact_hashes[name] = _digest(path)
    base_path = Path(opts.base_load_path) if opts.base_load_path else None
    base_artifacts = {}
    if base_path:
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                     "tokenizer_config.json", "processor_config.json"):
            path = base_path / name
            if path.is_file():
                base_artifacts[name] = _digest(path)
    source_root = Path(__file__).resolve().parents[1]
    source_artifacts = {
        str(path.relative_to(source_root)): _digest(path)
        for path in (Path(__file__), source_root / "jevany/api.py",
                     source_root / "jevany/checkpoint.py", source_root / "jevany/model.py")
    }
    provenance = {
        "checkpoint_source": "hub" if is_hub_id(ck.requested) else "local",
        "checkpoint_artifacts_sha256": artifact_hashes,
        "base_artifacts_sha256": base_artifacts,
        "source_artifacts_sha256": source_artifacts,
        "base_revision": ck.meta.base_revision,
        "head_type": ck.meta.head_type,
        "head_dim": ck.meta.head_dim,
        "head_residual_dim": ck.meta.head_residual_dim,
        "query_readout": ck.meta.query_readout,
        "option_readout": ck.meta.option_readout,
        "weights_dtype": ck.meta.weights_dtype,
        "code_revision": a.code_revision or actual_revision,
    }
    aliases = tuple(a.model_alias or ([] if a.strict_model_id else ["jevany-latest"]))
    app.state.server = Server(ck, tok, model, dev, model_id=a.model_id,
                              model_aliases=aliases, strict_model_id=a.strict_model_id,
                              provenance=provenance)
    print(f"serving {ck.requested} ({ck.path}) on {dev} :{a.port}")   # /v1/models reports the run as given, not the resolved cache path
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
