# Connect a model to the Playground

The Playground (`jevany demo`) serves packaged replays, the optional CPU
environments, and a form for your own decisions. Replays need no model, no GPU
and no account. This page covers the part that needs a model: connecting one,
confirming what it is, and asking it a question.

The [playground guide](../examples/README.md#play-locally-or-connect-your-model)
covers the environments themselves. [DEPLOYMENT.md](DEPLOYMENT.md) covers the
server.

## Start the Playground

```bash
python -m pip install -e .
jevany demo
```

Open `http://127.0.0.1:8090` if the browser does not open by itself. The
Playground binds to loopback only and accepts same-origin requests only.

## Connect a running model

Start a server in another terminal. A small checkpoint of your own is the
cheapest option:

```bash
jevany serve --checkpoint runs/my-jev --model-name my-jev --port 8008
curl http://127.0.0.1:8008/health
curl http://127.0.0.1:8008/v1/models
```

In the Playground, put that address in **YOUR MODEL URL** and choose
**Test and connect**:

```text
http://127.0.0.1:8008
```

Leave **MODEL ID** empty to use whatever the server reports. Fill it in only
when one endpoint serves several identities, or to pin the exact id your
requests should carry. An id the endpoint does not serve is rejected with the
list of ids it does serve, rather than failing later with HTTP 422.

Connecting runs one real `GET /v1/models`. The status line distinguishes three
states:

| Status | Meaning |
|---|---|
| No model configured | Replays and manual play work; **Run model** is unavailable |
| Configured but never tested | A `--base-url` was passed on the command line and has not answered yet. Text decisions work; image input stays off until the endpoint is tested |
| Connected: `<id>` | `GET /v1/models` answered, and the line below reports the served id, aliases, base model, device, readout and accepted media |

A `--base-url` endpoint is usable for text immediately, so the command-line flow
below still works unchanged. Its identity and media support are only known after a
test, so press **Test and connect** before enabling image input. A connection is
adopted only when the endpoint reports the fields the page uses; a descriptor with,
say, a list where `capabilities` belongs is reported and the previous connection
stays in use.

A failed test never replaces a working connection: the previous model stays in
use and the panel says the attempt failed. Common failures and what they mean:

| Message | Fix |
|---|---|
| `cannot reach http://127.0.0.1:8008` | No server on that port; start `jevany serve` |
| `answered but rejected GET /v1/models: HTTP Error 503: ... model is not ready` | The server is still loading weights; retry |
| `it does not serve 'NAME'` | Use one of the listed ids, or leave the field empty |
| `non-loopback decision endpoints must use HTTPS` | Remote endpoints must be `https://`; this is the same rule `JevClient` applies in Python |

The browser sends requests to the local Playground, which forwards them through
`JevClient`. The panel has no API-key field: for a gateway that needs a key, call
it from Python with
`JevClient(base_url, api_key=...)` as in [DEPLOYMENT.md](DEPLOYMENT.md), or put a
loopback endpoint in front of it.

## Text first, images when they work

A newly connected model starts text-only. **Also send the current image** is
enabled only when four things hold:

1. `GET /v1/models` has answered, so the next two are known rather than assumed,
2. the checkpoint accepts images (`capabilities.media_types` contains `image`),
3. the server has `JEVANY_MEDIA_ROOT` set (`limits.media_enabled` is true), and
4. the Playground can write where the server reads.

The checkbox shows what a request would actually carry, so it stays clear while
any of these is missing, and the note under it names the one to fix.

The fourth condition means `--media-root` for an HTTP server, because a server
rejects media that is not a file inside its own media root:

```bash
mkdir -p /tmp/jevany-media
JEVANY_MEDIA_ROOT=/tmp/jevany-media jevany serve \
  --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA --device cuda --dtype bf16 --port 8008
```

```bash
jevany demo --base-url http://127.0.0.1:8008 --media-root /tmp/jevany-media
```

With those two commands, press **Test and connect** once and then tick the
checkbox; the model answers text requests in the meantime. The environment image
stays visible in the browser either way; only the request changes. `--text-only`
starts with image input off; the checkbox can enable it after a successful
connection test.

Both processes need access to the same absolute file paths. If the Playground
runs in Docker and the model server runs on the host, mount the media directory
at the same path and run the container with the host user's UID/GID
(`--user "$(id -u):$(id -g)"`). Each frame lives in a private temporary directory;
a different user cannot read it.

## Try your own decision

**TRY YOUR OWN DECISION** asks the connected model one `choice` question. Edit
the state, the question and the candidate options, then choose **Ask the
model**. The result shows the selected option, its confidence, and the
probability of every candidate.

It needs no game dependencies, no PyTorch in the Playground process and no JSON
editing. It uses the same connected model, the same `POST /v1/systemone` body
and the same answer validation as the environments, so a model that answers here
answers the Python and HTTP examples in [API.md](API.md) too.

Limits: 2 to 12 options with distinct names, 6,000 characters of state and 2,000
characters of question. The server's own token limits still apply; an oversized
request is rejected with the server's explanation rather than truncated.

## Without a browser

The same endpoints are available for scripting against a running Playground:

| Request | Purpose |
|---|---|
| `GET /api/config` | cases, installed extras, and the current connection |
| `POST /api/connect` `{"base_url": ..., "model": ...}` | test and adopt an endpoint |
| `POST /api/images` `{"enabled": true}` | turn image input on or off |
| `POST /api/decide` `{"state": ..., "question": ..., "options": [...]}` | one custom choice question |
| `POST /api/start`, `POST /api/step` | the live environment controls |

They are loopback-only and same-origin, for the local page and local scripts.
For an application, call the model directly with `JevClient` instead.

[Back to the README](../README.md)
