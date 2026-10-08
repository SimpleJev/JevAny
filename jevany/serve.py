# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""FastAPI server for prefill-only decisions.

Run: uv run --extra serve python -m jevany.serve --run runs/rlcr --port 8008
Use ``--readout choice`` for the training-free choice-token deployment path.

TypeSafe-compatible: POST /v1/systemone and GET /v1/models (no auth). JEVANY_PREFIX_CACHE /
JEVANY_PREFIX_MIN_TOKENS size the state-prefix cache; JEVANY_DATE_FACTS=1 enables deterministic date preprocessing.
"""
import argparse, math, os
from contextlib import asynccontextmanager
from dataclasses import fields, replace
from pathlib import Path
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .api import SystemOneRequest, with_date_facts
from .checkpoint import LoadOptions, add_placement_arguments, load_options_from_args
from .inference import InferenceOptions, add_inference_arguments, inference_options_from_args
from .readout import (
    ChoiceReadoutOptions,
    LetterReadoutOptions,
    add_readout_arguments,
    choice_options_from_args,
    normalize_readout,
)
from .runtime import DEFAULT_CHECKPOINT, DecisionRuntime, JevModel
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
            except PermissionError:
                raise HTTPException(422, "server cannot access media file; check shared directory and file permissions")
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


routes = APIRouter()


def server(request: Request) -> DecisionRuntime:
    runtime = getattr(request.app.state, "server", None)
    if runtime is None:
        raise HTTPException(503, "model is not ready")
    return runtime


@routes.post("/v1/systemone")
def systemone(req: SystemOneRequest, request: Request):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    runtime = server(request)
    try:
        return runtime.answer(prepare(req))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@routes.get("/v1/models")
def models(request: Request):
    description = server(request).describe()
    description["limits"].update({
        "media_enabled": bool(MEDIA_ROOT) and bool(description["capabilities"]["media_types"]),
        "media_max_file_bytes": MEDIA_MAX_BYTES,
        "media_max_total_bytes": MEDIA_TOTAL_BYTES,
        "media_max_pixels": MEDIA_MAX_PIXELS,
        "media_max_video_frames": MEDIA_MAX_VIDEO_FRAMES,
    })
    return {"models": [description]}


@routes.get("/health")
def health(request: Request):
    ready = getattr(request.app.state, "server", None) is not None
    return JSONResponse({"status": "ready" if ready else "loading"}, status_code=200 if ready else 503)


def create_app(
    checkpoint: str | Path | None = None, *,
    model: JevModel | None = None, device: str | None = None,
    dtype: str | None = None, model_name: str | None = None,
    options: LoadOptions | None = None, inference_options: InferenceOptions | None = None,
    readout: str = "native", choice_options: ChoiceReadoutOptions | None = None,
    letter_options: LetterReadoutOptions | None = None,
) -> FastAPI:
    """Build an isolated app, loading one checkpoint during ASGI startup.

    Alternatively inject an already loaded JevModel to share its runtime, lock
    and cache with Python callers. Loading options cannot accompany an injected
    model. Each worker loads its own full model; use one worker per device.
    """
    readout = normalize_readout(readout)
    if choice_options is not None and letter_options is not None:
        raise ValueError("give at most one of choice_options and legacy letter_options")
    if letter_options is not None:
        choice_options = letter_options.as_choice()
    if model is not None and (readout != "native" or choice_options is not None or any(
        value is not None for value in (checkpoint, device, dtype, model_name, options, inference_options)
    )):
        raise ValueError("pass either a loaded model or checkpoint loading options")
    if readout == "native" and choice_options is not None:
        raise ValueError("choice_options require readout='choice'")
    if readout == "choice" and inference_options is not None:
        raise ValueError("inference_options apply only to native readout; use choice_options.max_tokens")
    if readout == "choice" and choice_options is None:
        choice_options = ChoiceReadoutOptions()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if model is not None:
            local = model
        else:
            settings = ({
                "readout": "choice",
                "choice_temperature": choice_options.temperature,
                "choice_native_weight": choice_options.native_weight,
                "choice_max_tokens": choice_options.max_tokens,
            } if choice_options is not None else {
                "inference_options": inference_options,
            })
            local = JevModel.from_pretrained(
                DEFAULT_CHECKPOINT if checkpoint is None else checkpoint,
                device=device, dtype=dtype, model_name=model_name, options=options,
                **settings,
            )
        application.state.server = local.runtime
        try:
            yield
        finally:
            application.state.server = None
            if model is None:
                local.clear_cache()

    application = FastAPI(title="JevAny", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError):
        # Invalid input can contain NaN/Infinity, which JSONResponse cannot encode.
        detail = jsonable_encoder(error.errors(), custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
        })
        return JSONResponse(status_code=422, content={"detail": detail})

    application.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    application.include_router(routes)
    return application


app = create_app()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "--run", dest="run", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model-name", help="identity reported in every response; defaults to the checkpoint name")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"])
    add_readout_arguments(ap)
    add_placement_arguments(ap)
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="capture CUDA graphs at startup (row-mode backbones on one GPU); same as JEVANY_CUDA_GRAPHS=1")
    ap.add_argument("--cuda-graph-max-tokens", type=int,
                    help="largest captured row; longer rows run eagerly (default 2048)")
    add_inference_arguments(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args(argv)
    try:
        choice_options = choice_options_from_args(a)
    except ValueError as error:
        ap.error(str(error))
    if choice_options is not None and (
        a.cuda_graphs or a.cuda_graph_max_tokens is not None
        or any(getattr(a, item.name) is not None for item in fields(InferenceOptions))
    ):
        ap.error("CUDA graph and native inference-limit flags cannot be used with --readout choice; "
                 "use --choice-max-tokens")
    options = load_options_from_args(a)
    if a.cuda_graphs or a.cuda_graph_max_tokens is not None:
        options = options or LoadOptions.from_env()
        options = replace(options, cuda_graphs=True,
                          cuda_graph_max_tokens=(a.cuda_graph_max_tokens if a.cuda_graph_max_tokens is not None
                                                 else options.cuda_graph_max_tokens))
    application = create_app(a.run, device=a.device, dtype=a.dtype, model_name=a.model_name,
                             options=options,
                             inference_options=(inference_options_from_args(a)
                                                if choice_options is None else None),
                             readout=normalize_readout(a.readout), choice_options=choice_options)
    import uvicorn
    uvicorn.run(application, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
