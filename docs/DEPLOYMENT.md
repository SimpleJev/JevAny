# Deploy a checkpoint

Install from the checkout using Python 3.12 or newer. The `local` extra loads
weights in Python; `serve` includes the same runtime plus FastAPI and Uvicorn.
Both use the adapter's recorded base model and temperature.
Training and serving resolve the same saved backbone adapter, tokenizer,
decision tokens and branch layout. Changing model families does not change
the request or response schema.

## Python

```bash
python -m pip install -e '.[local,multimodal]'
```

```python
from jevany import Choice, JevModel

model = JevModel.from_pretrained(
    "SimpleJev/JevAny-Qwen3.8-27B-LoRA", device="cuda", dtype="bf16"
)
result = model.system_one(
    state="I was charged twice.",
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payment problems", "shipping": "Delivery problems"},
        )
    },
)
print(result["answers"]["department"])
```

Use `JevModel.from_pretrained("runs/my-jev", model_name="my-jev")` for your
own checkpoint. Loading happens once; reuse the object between requests.
The common runtime serializes model/processor access and reuses eligible
state prefixes. Calling `model(request_dict)` accepts the complete HTTP body.

Configure inference without changing the checkpoint:

```python
from jevany import InferenceOptions, JevModel

model = JevModel.from_pretrained(
    "runs/my-jev",
    device="cuda",
    inference_options=InferenceOptions(
        max_state_tokens=2048,
        max_branch_tokens=4096,
        max_packed_tokens=8192,
        prefix_cache_size=4,
        prefix_min_tokens=384,
    ),
)
print(model.describe())
model.clear_cache()
```

`describe()` reports the resolved adapter, branch layout, context window, media
types, effective token limits and cache statistics. `clear_cache()` releases
prefixes and resets those statistics. Unknown architectures run without prefix
caching unless their adapter declares support; this optimization is optional.

## HTTP

```bash
python -m pip install -e '.[serve,multimodal]'
jevany serve --checkpoint SimpleJev/JevAny-Qwen3.8-27B-LoRA \
  --device cuda --dtype bf16 --port 8008
```

To deploy your training output:

```bash
jevany serve --checkpoint runs/my-jev --model-name my-jev --port 8008
```

In another terminal:

```bash
jevany decide examples/request.json
curl http://127.0.0.1:8008/health
curl http://127.0.0.1:8008/v1/models
```

The model name in responses identifies the configured deployment, even when the
request uses an alias. Omit `model` or send `jevany-latest` to select the loaded
checkpoint; its exact reported ID is also accepted. Other IDs return HTTP 422.
This is a single-model server, and the request cannot load another checkpoint.
Missing checkpoints fail startup.

The server binds to loopback and has no authentication. Use `--host 0.0.0.0`
only inside a deployment with suitable network access controls; terminate
TLS and authentication at your gateway if exposing the API beyond a trusted host.

For an application embedding the server:

```python
from jevany.serve import create_app

app = create_app("runs/my-jev", device="cuda", model_name="my-jev")
```

The ASGI lifespan loads weights once at startup and releases the runtime on
shutdown. Alternatively, `create_app(model=model)` shares an existing `JevModel`,
including its lock and cache, with local callers. Each app has its own configured
runtime. `/health` returns 200 when ready and 503 when no runtime is loaded;
decision and model-discovery requests also return 503 while unavailable.
Use one Uvicorn worker per device: each worker loads a full model.

## A complete local deployment

Every command below is copyable as-is. It downloads a released checkpoint at a
pinned revision, starts one server, checks it, calls it from the command line and
from Python, scores it, and connects the Playground to it. Replace
`SimpleJev/JevAny-Qwen3.5-4B-LoRA@...` with `runs/my-jev` to deploy your own
training output; nothing else changes.

```bash
python -m pip install -e '.[serve]'
```

Download the adapter into the Hugging Face cache, pinned to the revision this
release was published at. No account or token is needed:

```bash
hf download SimpleJev/JevAny-Qwen3.5-4B-LoRA \
  --revision 1c7aa9bab14ac347aeb917c0bcd757838a8a78ce
```

The base model is fetched on first load at the revision the checkpoint records,
so it needs no separate pin. To prefetch it, or to mirror it for an offline host:

```bash
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Start one server. `--device cpu --dtype bf16` works for a functional check;
`--device cuda` is the deployment path:

```bash
jevany serve \
  --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA@1c7aa9bab14ac347aeb917c0bcd757838a8a78ce \
  --model-name jevany-qwen3.5-4b --device cuda --dtype bf16 --host 127.0.0.1 --port 8008
```

Weights load during ASGI startup, so wait for `/health` before sending traffic:

```bash
until curl -sf http://127.0.0.1:8008/health; do sleep 2; done
curl -s http://127.0.0.1:8008/v1/models
```

`/health` answers 200 `{"status":"ready"}` once the runtime is loaded and 503
`{"status":"loading"}` before that. `/v1/models` reports the served id and
aliases, the base model, the device, the readout, the effective token limits and
whether media is enabled.

One request from the command line, then the same request from Python:

```bash
cat > /tmp/request.json <<'JSON'
{"state": "I was charged twice for order 4182.",
 "model": "jevany-qwen3.5-4b",
 "questions": {"department": {"type": "choice",
   "instructions": "Which team should handle this?",
   "criteria": {"billing": "Payment problems", "shipping": "Delivery problems"}}}}
JSON
jevany decide /tmp/request.json --base-url http://127.0.0.1:8008
```

```bash
python - <<'PY'
from jevany import Choice, JevClient

client = JevClient("http://127.0.0.1:8008", model="jevany-qwen3.5-4b")
print(client.system_one(
    "I was charged twice for order 4182.",
    {"department": Choice(instructions="Which team should handle this?",
                          criteria={"billing": "Payment problems", "shipping": "Delivery problems"})},
)["answers"]["department"])
PY
```

Score the running endpoint on labelled data, without loading a second copy of
the weights:

```bash
jevany data init --out data/starter
jevany eval --remote http://127.0.0.1:8008 --remote-model jevany-qwen3.5-4b \
  --data data/starter/development.jsonl --out /tmp/eval-remote
```

Finally, drive it from the browser:

```bash
jevany demo
```

Enter `http://127.0.0.1:8008` in the Playground's model panel and choose
**Test and connect**. See [PLAYGROUND.md](PLAYGROUND.md) for image input and the
custom decision form.

### Prerequisites

The server needs the backbone's weights in memory plus room for activations. Use
two bytes per base parameter for BF16 and four for FP32, from the table in
[Checkpoints and hardware](#checkpoints-and-hardware):

| Checkpoint | BF16 weights | GPU to plan for | CPU/BF16 RAM to plan for |
|---|---:|---|---|
| `JevAny-Qwen3.5-4B-LoRA` (and Direct-Token) | ~8 GB | one 16 GB card | ~12 GB free |
| `JevAny-Gemma-4B-LoRA` | ~16 GB | one 24 GB card | ~20 GB free |
| `JevAny-Qwen3.8-27B-LoRA` (default) | ~54 GB | one 80 GB card, or `--device-map` over several | ~64 GB free |
| `JevAny-Muse-Glimmer-30B-LoRA` | ~59 GB | one 80 GB card, or `--device-map` over several | ~70 GB free |

The RAM column adds headroom for activations, the adapter and the tokenizer to
the weight estimate; measure your own workload before sizing a deployment. A BF16
CPU load of the 4B release above, answering single-question requests, peaked near
8.3 GiB of resident memory. CPU and MPS are supported for backbones that fit and
are useful for functional checks, not for latency: the published latency figures
in this document are GPU measurements and do not transfer to CPU. Python 3.12 or
newer is required. One
Uvicorn worker loads one full model, so size per worker and keep one worker per
device. Downloading 4B needs roughly 8 GB of cache and 27B roughly 54 GB; set
`HF_HOME` to place it on a large enough filesystem.

## Inference settings

Python and HTTP use `InferenceOptions`. Both `jevany serve` and local
`jevany decide --checkpoint` accept the corresponding CLI flags:

| Python field | CLI flag | Environment variable | Default |
|---|---|---|---|
| `max_state_tokens` | `--max-state-tokens` | `JEVANY_MAX_STATE_TOKENS` | 8192 |
| `max_branch_tokens` | `--max-branch-tokens` | `JEVANY_MAX_BRANCH_TOKENS` | 8192 |
| `max_packed_tokens` | `--max-packed-tokens` | `JEVANY_MAX_PACKED_TOKENS` | 8192 |
| `prefix_cache_size` | `--prefix-cache-size` | `JEVANY_PREFIX_CACHE` | 4 |
| `prefix_min_tokens` | `--prefix-min-tokens` | `JEVANY_PREFIX_MIN_TOKENS` | 384 |

A branch includes its shared state. State and branch limits are capped at the
backbone's context window; the packed limit covers the complete request, including
all branches. Oversized inputs are rejected without truncation. Set the cache size
to zero to disable reuse. Native media requests bypass the text prefix cache.
Prefix reuse is limited to validated FP32 backbones. BF16/FP16 models and the
recurrent Qwen3.5/3.8 architecture use complete forward passes. The runtime reports
the effective cache capability; requesting a cache size does not override it.

Explicit CLI flags override environment values. Passing an `InferenceOptions`
object in Python uses that whole object; otherwise settings are read from the
environment when loading. Invalid values fail before weights load. Weight-loading
controls remain in `jevany.checkpoint.LoadOptions`, shared with training and
evaluation.

### Optional CUDA acceleration

The serving controls below are independent. Published benchmark scores use the
exact checkpoint path unless their report says otherwise.

| Control | Values | 4B recommendation | 27B recommendation | What it changes |
|---|---|---|---|---|
| `--dtype` / `JEVANY_DTYPE` | `fp32`, `fp16`, `bf16` | `bf16` | `bf16` | Backbone arithmetic and memory use |
| `JEVANY_MERGE_BF16` | `0`, `1` | `1` for serving | `1` for serving | Permits an approximate BF16 LoRA merge |
| `JEVANY_ATTN` | `sdpa`, `eager` | `sdpa` | `sdpa` | Backend for the full-attention layers |
| `JEVANY_COMPILE` | `0`, `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` | `0` when direct graphs are enabled | `0` | Optional `torch.compile`; it cannot be combined with direct CUDA Graphs |
| `JEVANY_CUDA_GRAPHS` / `--cuda-graphs` | `0`, `1` | `1` for single-question serving | `1` for single-question serving | Replays whole-backbone CUDA graphs on one GPU |
| `JEVANY_CUDA_GRAPH_MAX_TOKENS` / `--cuda-graph-max-tokens` | positive integer | `2048` | `2048` | Longer rows stay on eager inference instead of being padded into a slower graph |

Qwen3.5 and Qwen3.8 mix full-attention layers with Gated DeltaNet layers.
`JEVANY_ATTN` controls only the full-attention layers. Install
`flash-linear-attention` to accelerate the DeltaNet layers. Transformers detects
it automatically and otherwise uses its slower PyTorch implementation.
`causal-conv1d` is also detected automatically, but an H200 prefill test did not
find a latency benefit from it.

```bash
python -m pip install -e '.[local,fast]'   # flash-linear-attention
# causal-conv1d picks a prebuilt wheel for the torch it builds against; disable
# build isolation so that is the installed torch (and set
# CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE on machines without nvcc).
python -m pip install --no-build-isolation causal-conv1d
```

A wheel built for another torch fails to import, and Transformers then falls
back to the PyTorch path with only a log warning. `describe()` and `GET /v1/models`
report whether the two optional packages import under
`acceleration.linear_attention_kernels` for hybrid backbones (`null` otherwise).
Use a profiler to prove which kernels execute; package availability alone is not
that proof. Keep CUDA-only `causal-conv1d` out of CPU serving and test
environments; Transformers may select the installed extension even for CPU
inputs.

On an H200 with a BF16 Qwen3.5-4B backbone, batch size 1 and SDPA, FLA reduced
long-request GPU p90 from 82.81 ms to 47.62 ms. Short-request p90 changed from
51.35 ms to 47.88 ms. Adding `causal-conv1d` to FLA added about 1 ms in that
prefill-only test. SDPA was about 2% faster than eager. These are warmed GPU
forward measurements, not HTTP latency. On an A100-40GB, FLA with causal-conv1d
kept JevAny-Qwen3.5-4B's development-suite accuracy within one question and
reduced median Transfer latency from 105 ms to 94 ms and JevBench p90 from
475 ms to 282 ms.

The exact BF16 path keeps LoRA weights separate. For latency-sensitive serving,
merge them into the BF16 backbone at load time:

```bash
JEVANY_MERGE_BF16=1 jevany serve \
  --checkpoint SimpleJev/JevAny-Qwen3.8-27B-LoRA --device cuda --dtype bf16
```

On one H200, this reduced warmed Qwen3.8-27B forward latency by 31–33% in the
measured short and long requests. The merge changes BF16 rounding: in the full
release checks it changed two predictions on each of Transfer-v9 and public
JevBench. Leave it disabled when reproducing published metrics.

`torch.compile` is also opt-in. Its `reduce-overhead` mode uses CUDA Graphs for
compatible graph segments:

```bash
JEVANY_COMPILE=reduce-overhead jevany serve \
  --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA --device cuda --dtype bf16
```

Compilation requires the Triton version declared by the installed PyTorch
package. JevAny checks that pair before loading model weights. The first request
can spend about a minute compiling, and new shapes may trigger more work. In the
H200 measurements it improved warmed 4B forwards by roughly 15–21%, but did not
materially improve 27B latency. Use it for a persistent 4B service with recurring
shapes, not for short evaluation jobs. Set `JEVANY_COMPILE=default` to compile
without requesting CUDA Graphs, or `JEVANY_COMPILE=0` to disable compilation.

A separate native CUDA Graph microbenchmark measured fixed 96-token and
512-token 4B forwards at 8.91 ms and 16.44 ms, compared with eager execution at
47.03 ms and 46.08 ms. Capture took about 1.1 seconds per shape and retained
about 64 MiB per graph. The inputs, tensor addresses and shapes were fixed, and
the measurement excluded tokenization, transfers and HTTP work.

`--cuda-graphs` (`JEVANY_CUDA_GRAPHS=1`, `LoadOptions(cuda_graphs=True)`) exposes
this direct capture for serving. When the model loads it records the complete
backbone forward once per padded length, using one shared memory pool, and
replays the smallest bucket that fits each single-question request. Capture is
limited to 2,048 tokens by default because padding longer requests erased the
latency gain in the H200 tests:

```bash
jevany serve --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA --device cuda \
  --dtype bf16 --cuda-graphs --cuda-graph-max-tokens 2048
```

It applies to row-mode backbones (Qwen3.5/3.8, Gemma and the other hybrid or
sliding-window families) with the whole model on one CUDA device. Requests are
right-padded; every layer of these backbones is causal, so padding never reaches
a real token. Multi-question requests and rows above the configured limit run
eagerly. Checkpoints with trainable token embeddings are rejected before capture
because their PEFT embedding adapter allocates tensors during each forward.
`describe()` reports the lengths, capture time and how many calls replayed or ran
eagerly under `acceleration.cuda_graphs`. Direct graphs cannot be combined with
`JEVANY_COMPILE`.

Padded shapes select different kernels, so probabilities are close to, not
identical with, the eager path. On an A100-40GB with FLA and causal-conv1d
installed and SDPA, over every request of the development suites, median
forward latency on Transfer fell from 92 ms to 25 ms for JevAny-Qwen3.5-4B
(accuracy 78.78% before, 78.87% after; JevBench 181/231 both), from 95 ms to
26 ms for the Direct-Token model and from 98 ms to 32 ms for JevAny-Gemma-4B.
The same 231-request JevBench development panel was also measured on one H200
with BF16 merged LoRA, SDPA and FLA. With the 2,048-token capture limit, direct
graphs reduced 4B end-to-end median latency from 60.29 ms to 9.70 ms and mean
latency from 60.18 ms to 22.10 ms. Its p95 fell from 75.75 ms to 70.18 ms. For
27B, median latency fell from 113.54 ms to 30.53 ms and mean latency from
138.49 ms to 81.20 ms; p95 was effectively unchanged at 279.17 versus 281.52
ms. Capture took 6.27 seconds for 4B and 9.88 seconds for 27B. On the matched
BF16-merged paths, 4B changed from 187 to 185 correct answers with three argmax
changes, while 27B stayed at 207 with no argmax changes. Maximum probability
changes were 0.0160 and 0.0146 respectively. These serving measurements do not
replace the exact benchmark scores reported by the model cards.

`scripts/benchmark_latency.py` accepts `--cuda-graphs`,
`--cuda-graph-max-tokens`, `--max-packed` and `--serving-kernels`. The last
option keeps the fused SDPA kernels used by `jevany serve` instead of selecting
the math kernel used for fp32-exact evaluation.

For a latency-oriented 27B deployment, keep compilation off and cap direct
graphs at 2,048 tokens:

```bash
JEVANY_MERGE_BF16=1 JEVANY_ATTN=sdpa JEVANY_COMPILE=0 \
  jevany serve --checkpoint SimpleJev/JevAny-Qwen3.8-27B-LoRA \
  --device cuda --dtype bf16 --cuda-graphs --cuda-graph-max-tokens 2048
```

## A lightweight HTTP client

The base package installs the schema, HTTP client and starter-data tools without
PyTorch. A caller connecting to another machine's server needs only:

```bash
python -m pip install -e .
```

```python
from jevany import JevClient, Noul

client = JevClient("http://127.0.0.1:8008")
result = client.system_one("The job has finished.", {"done": Noul(instructions="Is the job complete?")})
print(result["answers"]["done"]["noul"])
```

`JevClient` requires HTTPS for non-loopback endpoints. Pass `api_key` for your
gateway or the official hosted API. Invalid requests and responses raise
`ValueError`; connection failures propagate from `urllib`.
Calls have a configurable 120-second default timeout and no automatic retries.

`client.models()` performs one `GET /v1/models`, so it both reports what the
endpoint serves and proves it is reachable and ready.

The base package also covers the command line against a remote server. Neither
of these needs PyTorch; only `--checkpoint` loads a model in-process:

```bash
jevany decide --help
jevany decide /tmp/request.json --base-url http://127.0.0.1:8008
```

### HTTP failures

An HTTP failure raises `jevany.client.DecisionHTTPError`, a subclass of
`urllib.error.HTTPError` that reports the server's own explanation:

```python
import urllib.error

try:
    client.system_one("state", {"department": Choice(criteria={"billing": None})}, model="wrong")
except urllib.error.HTTPError as error:
    print(error)            # HTTP Error 422: Unprocessable Entity: unknown model 'wrong'; this deployment serves 'sft'
    print(error.code)       # 422
    print(error.detail)     # unknown model 'wrong'; this deployment serves 'sft'
    print(error.read())     # the raw response body
```

`code`, `status`, `msg`, `reason`, `headers` and `read()` behave as they do on
`urllib.error.HTTPError`, so existing `except urllib.error.HTTPError` handlers and
any code already reading the body keep working. `detail` holds the server's
`detail`, `message` or `error` field; a non-JSON body (a gateway's HTML page, for
example) is reported as a single trimmed line instead. Request headers, including
`Authorization`, are never part of the message. `jevany decide` prints the same
text, so a rejected model name, a disabled media root or an oversized request
says which input to change.

## Use the official SDK

The text request and answer envelopes follow the TypeSafe Jev API:

```bash
python -m pip install typesafe-sdk
```

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(
    api_key="local", base_url="http://127.0.0.1:8008", model="jevany-latest"
) as client:
    result = client.system_one(
        "I was charged twice.",
        {"department": Choice(criteria={"billing": "Payments", "shipping": "Delivery"})},
    )
    print(result.choices["department"].choice)
```

The official SDK has typed response objects. JevAny's Python clients return
ordinary dictionaries with the same wire fields. [API.md](API.md) describes
compatibility and the fields specific to JevAny.

## Checkpoints and hardware

| Checkpoint | Base model | Estimated BF16 weight memory |
|---|---|---:|
| `SimpleJev/JevAny-Gemma-4B-LoRA` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | ~16 GB |
| `SimpleJev/JevAny-Qwen3.5-4B-LoRA` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | ~8 GB |
| `SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | ~8 GB |
| `SimpleJev/JevAny-Qwen3.8-27B-LoRA` (default) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | ~54 GB |
| `SimpleJev/JevAny-Muse-Glimmer-30B-LoRA` | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | ~59 GB |

These estimates use two bytes per base parameter. Allow additional memory for
the adapter and inference activations. Gemma E4B has about 8B total parameters
including its embedding tables; E4B refers to its effective parameter count.
For your own training output, use the parameter count of the base you selected.

First loading downloads both the adapter and its separately distributed base,
unless already cached.
Use `owner/repo@revision` to pin an adapter. For offline deployment, prepopulate
the Hugging Face cache and set `HF_HUB_OFFLINE=1`. `JEVANY_BASE_LOAD_PATH` can point
to a local base mirror while retaining the checkpoint's canonical provenance.

By default the runtime loads one full backbone on one device.
CPU/MPS are available for backbones that fit, including smaller models you train.
Quantization is not implemented. BF16-trained checkpoints retain their recorded
loading behavior.

### Several GPUs

When a base does not fit on one card, `device_map` splits its layers over every
visible GPU with Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1 jevany serve --checkpoint SimpleJev/JevAny-Qwen3.8-27B-LoRA \
  --device cuda --device-map auto --max-memory-gib 31
```

Python callers pass `LoadOptions(device_map="auto", max_memory_gib=31)`; the
command-line tools also read `JEVANY_DEVICE_MAP` and `JEVANY_MAX_MEMORY_GIB`.
`device_map` accepts Accelerate's `auto`, `balanced`, `balanced_low_0` and
`sequential` strategies. `max_memory_gib` caps the weights placed on each GPU;
leave headroom for the activations of long requests. JevAny rejects placements
that spill adapter or direct-token readout weights to disk because those weights
cannot be restored safely after Accelerate dispatch. Increase `max_memory_gib`
or expose another GPU instead.

This is pipeline placement, not tensor parallelism: layers still run one after
another, so it adds memory rather than throughput. The readout head stays on
`device` and hidden states return there before it. Sharded models always run
complete forward passes (no prefix cache). `describe()` and `GET /v1/models`
report `device_map` and the `devices` holding parameters.

Splitting changes where tensors live, not the arithmetic: JevAny-Qwen3.5-4B
forced onto two GPUs gave the same per-question probabilities as one GPU on all
231 JevBench questions. On 3×A100-40GB (`--max-memory-gib 22`), the step 22,160
JevAny-Qwen3.8-27B release scored 85.66% on Transfer (published for that step:
85.76%) with a 240 ms median forward pass.

## Native media and limits

For the server, install the `multimodal` extra and set `JEVANY_MEDIA_ROOT` to a
directory of inputs before startup. Paths in requests resolve inside that root;
network URLs, path escapes and oversized files are rejected. A media request
contains one question for the built-in adapters. In-process inference accepts
trusted local paths directly. Accepted media types come from the saved backbone
adapter; setting a media root does not enable vision in a text checkpoint.

```bash
JEVANY_MEDIA_ROOT="$PWD/media" jevany serve \
  --checkpoint SimpleJev/JevAny-Qwen3.8-27B-LoRA --device cuda
```

See [DATA.md](DATA.md#native-media) for the request format and
`GET /v1/models` for the active limits and capabilities. The default packed limit is 8,192 tokens;
evaluation covers the 2,048-token training window.
Confidence thresholds may need recalibration on your domain.

## Extend backbone support

Serving uses the same `backbone_adapter = "my_package.adapters:MyAdapter"` saved
by training. Install that trusted package in the serving environment; no HTTP
handler or client changes are needed. See [TRAINING.md](TRAINING.md#backbone-support)
for model-loading and media hooks.

An adapter's `inference_capabilities(config)` returns
`jevany.backbones.InferenceCapabilities`. It declares the context window,
prefix-cache support, accepted media types, and optional media question limit.
The built-in implementation reads the context window from the decoder config
and media types from the adapter. For example, a custom decoder can disable cache
reuse while retaining the other constraints:

```python
from dataclasses import replace
from jevany.backbones import BackboneAdapter

class MyAdapter(BackboneAdapter):
    def inference_capabilities(self, config):
        return replace(super().inference_capabilities(config), prefix_cache=False)
```

Only opt into caching after validating `new_cache()`, cache copying and batch
reordering, plus cropping for packed branches. Unknown text architectures use
independent rows and uncached inference by default. A model with a different
context layout should report its usable window explicitly.

The offline serving tests cover the seven maintained families, including recurrent
and MoE text models, Gemma 4 Unified and GLM native vision checkpoints, plus a
GPT-2 fixture for the generic adapter contract and a custom
adapter without cache support. They use tiny real architectures and native
processors to check Python/HTTP parity, media, cache behavior, limits and
lifecycle handling.

```bash
python -m pytest tests/test_serving.py -q
```
