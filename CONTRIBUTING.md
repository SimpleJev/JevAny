# Contributing

Install the development dependencies with Python 3.12 or newer:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -m 'not server' -q
```

To reproduce CI on a CPU machine, use Python 3.12 and uv 0.11.28:

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -e '.[dev]'
uv pip check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests -m 'not server' -q -ra --strict-config --strict-markers
.venv/bin/python -m build
```

CI tests both the latest allowed dependencies and the minimum supported
PyTorch, torchvision and Transformers versions. For the latter, add
`-c .github/constraints/minimum.txt` to the install command in a fresh
environment. Both jobs run the full CPU test suite and build the package;
their artifacts include the resolved dependency versions and JUnit test
results. Keep the constraints aligned with the lower bounds in
`pyproject.toml` when changing the supported ML versions.

The integration tests create a tiny local Qwen backbone, run SFT and RLCR
updates, reload the saved adapter, and check Python/HTTP/official-SDK contracts.
The unit tests also use a Qwen2.5 tokenizer, downloaded on first use.
Released-weight tests are optional and require `JEVANY_TEST_CHECKPOINT`.
Video tests decode local clips with PyAV, which is included in the
`multimodal` and `dev` extras, and retain each model processor's frame sampling.

For tests against an existing server:

```bash
JEVANY_BASE_URL=http://127.0.0.1:8008 python -m pytest tests/test_api.py -q
```

Keep JSON request/response compatibility when changing inference. Add labels only
to training records and keep them out of model-facing inputs. Data converters
should record source revisions and preserve evaluation separation.

The code is organized around user entry points:

| Location | Responsibility |
|---|---|
| `jevany/client.py`, `api.py` | Lightweight Python client and wire schema |
| `jevany/runtime.py`, `serve.py` | Shared local inference and HTTP deployment |
| `jevany/training.py`, `train.py` | Recipes, Python training entry point and training loop |
| `jevany/datasets/`, `data.py` | Bundled starter, public-source builders and JSONL validation |
| `recipes/`, `infra/` | Training configurations and scheduler-neutral launcher |
| `examples/` | Small applications using the public interfaces |
| `scripts/`, `results/` | Research/release tools and recorded measurements |
| `docs/` | Guides, evaluation records and showcase assets |

Build a distributable package with `python -m build`. Keep generated weights,
runs, downloaded data and build outputs outside source control.
