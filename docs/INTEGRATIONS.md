# Integrations

## Choose a tool

The [tool-routing example](../examples/tool_routing.py) maps a support request
to one of four tools, using the same Python client for local and HTTP inference.
With a [JevAny server running](DEPLOYMENT.md):

```bash
python -m examples.tool_routing "Where is order A-104?" --min-probability 0.8
```

Use `--checkpoint runs/my-jev --device cpu` to load a local model instead.
The result includes the selected option's probability and the full decision.
It returns a tool name only when that probability reaches the threshold;
otherwise, `"tool": null` and `"deferred": true` let your application ask for
more information or hand the case to another decision process.

The example prints a routing decision. Your application supplies the tool
arguments and executes the tool. Set the threshold on held-out data from your
own task; `0.8` is an example setting. Use the probability distribution for this
threshold, since the API's choice `confidence` adjusts for chance agreement.

## Compile questions and decision trees

Jev-Harness uses an external LLM to compile tasks into typed questions. The planner sees an evidence schema by default. The caller retains control of the state, which passes unchanged to JevAny for the bounded decision.

Planner output must be a JSON object. Markdown fences are accepted when their
body contains only JSON; place explanations outside the fence.

### Local Chat Completions planner

[`ChatCompletionsGenerator`](../jevany/openai_compat.py) connects to a
non-streaming text Chat Completions endpoint. It needs only the base JevAny
installation. With a planner serving `my-planner` at `http://127.0.0.1:8000/v1`
and a JevAny server at `http://127.0.0.1:8008`:

```bash
python -m examples.harness --planner-model my-planner \
  --task "Choose an execution mode and decide whether rollback is required." \
  --evidence '{"environment":"staging","tests":"passed","snapshot":"available"}'
```

Use `--planner-url` to change the planner's API prefix, and `--base-url` or
`--checkpoint` to choose the JevAny backend. The planner must serve the model
name you pass. Local unauthenticated servers need no key; authenticated
planners use the optional `JEVANY_PLANNER_API_KEY` environment variable.
Non-loopback endpoints require HTTPS.

The adapter supports `max_output_tokens`, `temperature`, `top_p`, and
`stop_sequences`; the served model must support the parameters you use.
It maps the token limit to `max_completion_tokens` and stop sequences to `stop`.
Responses return `text`, provider `usage`, and `stop_reason`.
HTTP errors include the server's explanation; requests are not retried
automatically.

Your own planner can implement `generate(prompt, params=None)` with the same
result fields and be passed to `JevHarness`. The
[command-line example](../examples/harness.py) keeps evidence values with
JevAny unless `--planner-sees-evidence` is enabled. This default is not a
prompt-injection security boundary.

### Bedrock planner

With the JevAny server running, install the Bedrock adapter:

```bash
python -m pip install -e '.[bedrock]'
```

Use standard AWS credentials with access to the chosen Bedrock model. This example uses the same planner model as the [recorded harness run](demos/jev-harness.json):

```python
from jevany.bedrock import BedrockGenerator
from jevany.harness import HTTPDecisionClient, JevHarness

planner = BedrockGenerator("us.anthropic.claude-opus-4-7")
harness = JevHarness(planner, HTTPDecisionClient("http://127.0.0.1:8008"))
result = harness.run(
    "Choose an execution mode and decide whether rollback is required.",
    {"environment": "staging", "tests": "passed", "snapshot": "available"},
)
print(result["decision"]["answers"])
```

Jev-Symbolic asks an LLM to write a compact decision tree, validates branch coverage and acyclicity, then sends each internal node to JevAny. Every result records the outcome ID and complete branch trace.

For command-line entry points, see [the harness example](../examples/bedrock_harness.py) and [the symbolic example](../examples/bedrock_symbolic.py). The saved [harness](demos/jev-harness.json) and [symbolic](demos/jev-symbolic.json) outputs include the compiled requests and decisions.
