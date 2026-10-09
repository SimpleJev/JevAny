"""Trusted PR review controller. PR contents are data, never executable inputs."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fnmatch
import hashlib
import html
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

from jsonschema import validate

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "policy.json").read_text())
SCHEMA = json.loads((ROOT / "result.schema.json").read_text())
PROMPT = (ROOT / "prompt.md").read_text()
MARKER = "<!-- jevany-pr-review:v1 -->"
BOT = "github-actions[bot]"
LABELS = {"pass": "ai:approved", "human-review": "ai:human-review"}


class ReviewError(Exception):
    """An incomplete review, with a safe, non-secret explanation."""

    def __init__(self, message, http_status=None):
        super().__init__(message)
        self.http_status = http_status


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ReviewError("API redirect refused")


def request_json(url, token, method="GET", body=None, timeout=60):
    request = urllib.request.Request(
        url, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                 "Content-Type": "application/json", "User-Agent": "JevAny-PR-review"},
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
            raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ReviewError("API response exceeds the review size limit")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        # Model API errors can echo a key: keep them generic. GitHub validation
        # details are limited to public API error fields, never request headers.
        detail = ""
        if urllib.parse.urlparse(url).hostname == "api.github.com":
            try:
                data = json.loads(error.read(10000))
                messages = [data.get("message", "")]
                messages += [json.dumps(item) for item in data.get("errors", [])]
                detail = ": " + "; ".join(m for m in messages if m)[:800]
            except (ValueError, AttributeError):
                pass
        raise ReviewError(f"API request failed (HTTP {error.code}){detail}", http_status=error.code) from None
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise ReviewError("API request timed out, failed, or returned invalid JSON") from None


class GitHub:
    def __init__(self, repo, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ReviewError("Invalid repository")
        self.repo = repo
        self.token = token

    def call(self, path, method="GET", body=None):
        if not path.startswith("/") or ".." in path.split("/"):
            raise ReviewError("Invalid API path")
        return request_json("https://api.github.com" + path, self.token, method, body)

    def repo_call(self, path, method="GET", body=None):
        return self.call(f"/repos/{self.repo}{path}", method, body)

    def pages(self, path, key=None, max_pages=30):
        result = []
        separator = "&" if "?" in path else "?"
        for page in range(1, max_pages + 1):
            data = self.repo_call(f"{path}{separator}per_page=100&page={page}")
            items = data[key] if key else data
            result.extend(items)
            if len(items) < 100:
                return result
        raise ReviewError("API pagination limit reached; coverage is incomplete")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def documentation_only(files):
    return bool(files) and all(
        matches(f["filename"], POLICY["documentation"])
        and matches(f.get("previous_filename", f["filename"]), POLICY["documentation"])
        for f in files
    )


def collaborator_status(gh, pr):
    """Check the PR author against this repository, not the triggering actor.

    author_association=CONTRIBUTOR/MEMBER is not proof of current repository
    collaboration. GitHub's collaborator endpoint also covers access via teams.
    """
    login = pr.get("user", {}).get("login")
    status = {"login": login, "verified": False, "collaborator": False}
    if not isinstance(login, str) or not login:
        return status
    try:
        response = gh.repo_call(f"/collaborators/{urllib.parse.quote(login, safe='')}")
    except ReviewError as exc:
        if exc.http_status == 404:
            status["verified"] = True
        return status
    # A successful membership check returns HTTP 204 with no response body.
    if response is None:
        status.update(verified=True, collaborator=True)
    return status


def publication_plan(gh, pr, record, mode):
    """A clean review is distinct from permission to submit GitHub APPROVE."""
    plan = {"event": "COMMENT", "reason": "Human Review: see the findings and review evidence."}
    if record["decision"] == "pending":
        return {"event": None, "reason": "Waiting for review; no approval is submitted."}
    if record["decision"] != "pass":
        return plan
    if mode != "auto-approve":
        plan["reason"] = "Pass (report only): automatic approval is disabled by the workflow mode."
    elif not documentation_only(record["files"]):
        plan["reason"] = "Pass (report only): code, configuration and other non-documentation changes are never auto-approved."
    else:
        access = collaborator_status(gh, pr)
        plan["author_access"] = access
        if not access["verified"]:
            plan["reason"] = "Pass (report only): the PR author's current repository collaborator status could not be verified."
        elif not access["collaborator"]:
            plan["reason"] = "Pass (report only): the PR author is not a current repository collaborator."
        else:
            plan.update(event="APPROVE", reason="Auto-approve: documentation-only Pass by a verified current repository collaborator.")
    return plan


def applicable_workflows(files):
    paths = [p for f in files for p in (f["filename"], f.get("previous_filename", f["filename"]))]
    return ["ci.yml"] + (["pages.yml"] if any(matches(p, POLICY["pages_paths"]) for p in paths) else [])


def snapshot(pr):
    return {"number": pr["number"], "head": pr["head"]["sha"], "base": pr["base"]["sha"],
            "base_ref": pr["base"]["ref"], "head_repo": pr["head"]["repo"]["full_name"]}


def load_pr(gh, number):
    pr = gh.repo_call(f"/pulls/{number}")
    # GitHub can retain an older base.sha on an unchanged PR after main moves.
    # Resolve the actual branch tip instead of treating that cached value as
    # proof that a previously tested merge is still current.
    ref = urllib.parse.quote(pr["base"]["ref"], safe="/")
    tip = gh.repo_call(f"/git/ref/heads/{ref}")["object"]["sha"]
    pr["base"] = dict(pr["base"], sha=tip)
    return pr


def current(pr, expected):
    return pr["state"] == "open" and not pr["draft"] and snapshot(pr) == expected


def select_ci_run(runs, pr):
    """Never use main/push status, another PR's status, or a different base."""
    matches_pr = []
    for run in runs:
        if run["event"] != "pull_request" or run["head_sha"] != pr["head"]["sha"]:
            continue
        # GitHub returns an empty pull_requests array for cross-repository PRs.
        # The CI's run-name captures immutable event head/base identifiers.
        identity = f"PR #{pr['number']} | base {pr['base']['sha']} | head {pr['head']['sha']}"
        if (run.get("display_title") == identity
                and run.get("head_repository", {}).get("full_name") == pr["head"]["repo"]["full_name"]):
            matches_pr.append(run)
            continue
        for linked in run.get("pull_requests", []):
            if (linked["number"] == pr["number"]
                    and linked["head"]["sha"] == pr["head"]["sha"]
                    and linked["base"]["sha"] == pr["base"]["sha"]):
                matches_pr.append(run)
                break
    return max(matches_pr, key=lambda r: (r["run_number"], r.get("run_attempt", 1)), default=None)


def ci_evidence(gh, pr, files):
    checks = []
    for workflow in applicable_workflows(files):
        runs = gh.pages(f"/actions/workflows/{workflow}/runs?event=pull_request&head_sha={pr['head']['sha']}", "workflow_runs")
        run = select_ci_run(runs, pr)
        if run is None:
            candidates = [candidate for candidate in runs if candidate["event"] == "pull_request"
                          and candidate["head_sha"] == pr["head"]["sha"]
                          and candidate.get("head_repository", {}).get("full_name") == pr["head"]["repo"]["full_name"]]
            if candidates:
                latest = max(candidates, key=lambda item: (item["run_number"], item.get("run_attempt", 1)))
                checks.append({"workflow": workflow, "id": latest["id"], "attempt": latest.get("run_attempt", 1),
                               "state": "incomplete" if latest["status"] == "completed" else "pending",
                               "reason": "CI cannot be bound to the current PR/base. Update the branch from main and run CI again."})
            else:
                checks.append({"workflow": workflow, "state": "missing", "reason": "No CI run for this PR head and base"})
            continue
        record = {"workflow": workflow, "id": run["id"], "attempt": run.get("run_attempt", 1),
                  "head": run["head_sha"], "url": run["html_url"]}
        if run["status"] != "completed":
            record.update(state="pending", reason="CI is still running")
        elif run["conclusion"] != "success":
            record.update(state="failed", reason=f"CI concluded {run['conclusion']}")
        else:
            jobs = gh.pages(f"/actions/runs/{run['id']}/jobs?filter=latest", "jobs")
            required = POLICY["ci"][workflow]
            by_name = {j["name"]: j for j in jobs}
            ok = all(name in by_name and by_name[name]["status"] == "completed"
                     and by_name[name]["conclusion"] == "success" for name in required)
            record.update(state="passed" if ok else "incomplete",
                          reason="All expected jobs passed" if ok else "An expected CI job is missing, skipped, or unsuccessful")
            record["jobs"] = [{"name": name, "conclusion": by_name.get(name, {}).get("conclusion")} for name in required]
        checks.append(record)
    return checks


def ci_state(checks):
    states = {c["state"] for c in checks}
    if states & {"failed", "incomplete"}:
        return "human-review"
    if not checks or states & {"pending", "missing"}:
        return "pending"
    return "passed" if states == {"passed"} else "human-review"


def patch_complete(file):
    patch = file.get("patch")
    if patch is None:
        return False
    added = removed = old_remaining = new_remaining = 0
    for line in patch.splitlines():
        if line.startswith("@@"):
            if old_remaining or new_remaining:
                return False
            match = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
            if not match:
                return False
            old_remaining = int(match[1]) if match[1] is not None else 1
            new_remaining = int(match[2]) if match[2] is not None else 1
        elif line.startswith("+"):
            added += 1
            new_remaining -= 1
        elif line.startswith("-"):
            removed += 1
            old_remaining -= 1
        elif line.startswith(" "):
            old_remaining -= 1
            new_remaining -= 1
        elif not line.startswith("\\ No newline"):
            return False
        if min(old_remaining, new_remaining) < 0:
            return False
    return (old_remaining == new_remaining == 0
            and added == file["additions"] and removed == file["deletions"])


def collect_context(gh, pr, files):
    if len(files) != pr["changed_files"] or len(files) > POLICY["max_files"]:
        raise ReviewError("Changed-file coverage is incomplete or exceeds the review limit")
    compare = gh.repo_call(f"/compare/{pr['base']['sha']}...{pr['head']['sha']}")
    merge_base = compare["merge_base_commit"]["sha"]
    trees = {}
    sources = {}
    used = 0

    def read(repo, ref, path):
        nonlocal used
        key = (repo, ref)
        if key not in trees:
            tree = gh.call(f"/repos/{repo}/git/trees/{ref}?recursive=1")
            if tree.get("truncated"):
                raise ReviewError("Repository tree is truncated")
            trees[key] = {entry["path"]: entry for entry in tree["tree"]}
        entry = trees[key].get(path)
        if entry is None:
            return None
        if entry["type"] != "blob" or entry["mode"] not in ("100644", "100755"):
            raise ReviewError("A changed file is a symlink or submodule; manual inspection is required")
        if entry.get("size", POLICY["max_file_bytes"] + 1) > POLICY["max_file_bytes"]:
            raise ReviewError("A file exceeds the full-content review limit")
        data = gh.call(f"/repos/{repo}/git/blobs/{entry['sha']}")
        if data["encoding"] != "base64":
            raise ReviewError("Unsupported blob encoding")
        try:
            raw = base64.b64decode(data["content"])
            text = raw.decode("utf-8")
        except (UnicodeError, ValueError):
            raise ReviewError("A binary file requires manual inspection") from None
        if "\x00" in text:
            raise ReviewError("A binary file requires manual inspection")
        used += len(raw)
        if used > POLICY["max_context_bytes"]:
            raise ReviewError("Full review context exceeds the size limit")
        return text

    changes = []
    for file in files:
        if not patch_complete(file):
            raise ReviewError("A patch is missing, binary, or truncated; full review is unavailable")
        path = file["filename"]
        old_path = file.get("previous_filename", path)
        old = read(gh.repo, merge_base, old_path) if file["status"] != "added" else None
        new = read(pr["head"]["repo"]["full_name"], pr["head"]["sha"], path) if file["status"] != "removed" else None
        if (file["status"] != "added" and old is None) or (file["status"] != "removed" and new is None):
            raise ReviewError("A changed file could not be read at its pinned revision")
        sources[path] = {"old": old or "", "new": new or "", "old_path": old_path}
        changes.append({"file": path, "old_path": old_path, "status": file["status"],
                        "patch": file["patch"], "old": old, "new": new})
    context = {}
    for path in POLICY["context_files"]:
        # Include trusted base evidence, even when a PR edits the same document.
        context[path] = read(gh.repo, pr["base"]["sha"], path)
    return {"title": pr["title"], "description": (pr.get("body") or "")[:8000],
            "changes": changes, "base_context": context, "merge_base": merge_base}, sources


def validate_result(result, sources):
    validate(result, SCHEMA)
    if len(json.dumps(result)) > 60000:
        raise ReviewError("Model output exceeds the publication limit")
    for finding in result["findings"]:
        source = sources.get(finding["file"])
        if source is None or finding["line"] > len(source[finding["side"]].splitlines()):
            raise ReviewError("Model finding refers to an invalid file or line")
        if not all(finding[key].strip() for key in ("title", "reason", "evidence")):
            raise ReviewError("Model finding lacks actionable evidence")
    return result


def model_configuration():
    provider = os.getenv("PR_REVIEW_PROVIDER", "openai")
    if provider not in ("openai", "openrouter"):
        raise ReviewError("PR_REVIEW_PROVIDER must be openai or openrouter")
    secret = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    return provider, os.getenv("PR_REVIEW_MODEL", ""), os.getenv(secret, "")


def review_model(context, sources, key, model, provider="openai"):
    if provider not in ("openai", "openrouter"):
        raise ReviewError("Unsupported AI provider")
    if not key or not model:
        secret = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
        raise ReviewError(f"AI review is not configured: set {secret} and PR_REVIEW_MODEL")
    content = json.dumps(context, ensure_ascii=False)
    schema = {"name": "pr_review", "strict": True, "schema": SCHEMA}
    if provider == "openrouter":
        response = request_json("https://openrouter.ai/api/v1/chat/completions", key, "POST", {
            "model": model, "stream": False, "max_tokens": POLICY["max_output_tokens"],
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "provider": {"require_parameters": True},
        }, timeout=180)
    else:
        response = request_json("https://api.openai.com/v1/responses", key, "POST", {
            "model": model, "store": False, "max_output_tokens": POLICY["max_output_tokens"],
            "instructions": PROMPT, "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", **schema}},
        }, timeout=180)
    try:
        if not isinstance(response, dict) or response.get("error"):
            raise ReviewError("AI provider returned an error or invalid response")
        if provider == "openrouter":
            choices = response.get("choices", [])
            if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                raise ReviewError("AI response was incomplete")
            message = choices[0]["message"]
            if message.get("refusal") or message.get("tool_calls"):
                raise ReviewError("AI review was refused or requested tools")
            output = message["content"]
            if not isinstance(output, str) or not output.strip():
                raise ReviewError("AI response contained no review text")
        else:
            if response.get("status") != "completed":
                raise ReviewError("AI response was incomplete")
            parts = []
            for item in response.get("output", []):
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise ReviewError("AI review was refused")
                    if part.get("type") == "output_text":
                        parts.append(part["text"])
            output = "".join(parts)
        result = validate_result(json.loads(output), sources)
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("AI output failed schema or evidence-location validation") from None
    return result, {"provider": provider, "model": response.get("model", model), "response_id": response.get("id"),
                    "usage": response.get("usage")}


def decide(files, checks, result, error=None):
    state = ci_state(checks)
    if state == "human-review":
        return state, ["Applicable CI failed or cannot be verified for this PR revision; see the CI evidence below."]
    if state == "pending":
        return state, ["Waiting for CI evidence for this exact PR head and base."]
    if error:
        return "human-review", [error]
    if result is None or not result["coverage_complete"]:
        return "human-review", ["AI review coverage is incomplete."]
    reasons = []
    if result["findings"]:
        reasons.append("The review identified substantive findings; see the evidence below.")
    reasons.extend(result["uncertainties"])
    return ("human-review", reasons) if reasons else ("pass", ["Applicable CI passed and the complete review found no substantive issue."])


def own(item):
    return item.get("user", {}).get("login") == BOT and MARKER in (item.get("body") or "")


def fingerprint(expected, files, checks, model, mode, configured, provider="openai", author_access=None):
    return digest({"snapshot": expected, "files": files, "ci": checks, "model": model,
                   "mode": mode, "configured": configured, "provider": provider,
                   "author_access": author_access,
                   "policy": POLICY, "prompt": PROMPT, "schema": SCHEMA,
                   "controller": hashlib.sha256((ROOT / "review.py").read_bytes()).hexdigest()})


def evaluate_one(gh, pr, model, mode, key, provider="openai"):
    files = gh.pages(f"/pulls/{pr['number']}/files")
    checks = ci_evidence(gh, pr, files)
    record = {"snapshot": snapshot(pr), "files": files, "ci": checks, "mode": mode,
              "result": None, "model": None, "sources": {}, "merge_base": None}
    author_access = collaborator_status(gh, pr) if mode == "auto-approve" and documentation_only(files) else None
    record["fingerprint"] = digest([fingerprint(record["snapshot"], files, checks, model, mode, bool(key), provider, author_access), pr["draft"]])
    comments = gh.pages(f"/issues/{pr['number']}/comments")
    key_marker = f"<!-- completed:{record['fingerprint']} -->"
    if any(own(c) and key_marker in c["body"] for c in comments):
        return None
    error = None
    if ci_state(checks) == "passed" and not pr["draft"]:
        try:
            context, sources = collect_context(gh, pr, files)
            record["result"], record["model"] = review_model(context, sources, key, model, provider)
            record["sources"] = {path: {"old_path": value["old_path"]} for path, value in sources.items()}
            record["merge_base"] = context["merge_base"]
        except ReviewError as exc:
            error = str(exc)
    decision, reasons = decide(files, checks, record["result"], error)
    if pr["draft"]:
        decision, reasons = "pending", ["Draft PR: review will resume when marked ready."]
    elif any(c["state"] == "missing" for c in checks) and ci_state(checks) == "pending":
        age = datetime.now(timezone.utc) - datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00"))
        if age.total_seconds() > 1800:
            decision, reasons = "human-review", ["CI evidence is missing or belongs to an older base. Run CI for the current PR revision."]
    record.update(decision=decision, reasons=reasons, retryable=bool(error))
    return record


def safe(text):
    # Keep model text out of HTML, mentions, image links, and Markdown links.
    text = html.escape(str(text)).replace("@", "@\u200b")
    for char in ("\\", "`", "[", "]", "*", "_", "#", "|"):
        text = text.replace(char, "\\" + char)
    return text


def render(gh, record):
    title = {"pass": "Pass", "human-review": "Human Review", "pending": "Waiting for review"}[record["decision"]]
    expected = record["snapshot"]
    lines = [MARKER, f"### {title}", "", *[f"- {safe(reason)}" for reason in record["reasons"]], ""]
    if record.get("publication"):
        lines += [safe(record["publication"]["reason"]), ""]
    elif record["mode"] == "report-only":
        lines += ["Report-only mode: no approval is submitted.", ""]
    result = record.get("result")
    if result:
        lines += [safe(result["summary"]), ""]
        for finding in result["findings"]:
            path = finding["file"]
            ref = expected["head"] if finding["side"] == "new" else record["merge_base"]
            repo = expected["head_repo"] if finding["side"] == "new" else gh.repo
            if finding["side"] == "old":
                path = record["sources"][path]["old_path"]
            url = f"https://github.com/{repo}/blob/{ref}/{urllib.parse.quote(path, safe='/')}#L{finding['line']}"
            lines += [f"- **{safe(finding['severity'])}: {safe(finding['title'])}** ([location]({url}))",
                      f"  {safe(finding['reason'])} Evidence: {safe(finding['evidence'])}"]
    else:
        lines += ["AI semantic review has not completed.", ""]
    for check in record["ci"]:
        run_link = f" ([run](https://github.com/{gh.repo}/actions/runs/{check['id']}))" if "id" in check else ""
        lines.append(f"- CI `{check['workflow']}`: {safe(check['state'])}{run_link} — {safe(check['reason'])}")
    lines += ["", f"Reviewed head `{expected['head']}`; base `{expected['base']}`.",
              "CPU CI does not verify production-model quality, CUDA or multi-GPU performance."]
    if record.get("model"):
        lines.append(f"Model: {safe(record['model']['model'])}; policy: {POLICY['version']}.")
    # Errors remain retryable on the next dispatch/CI event.
    if record["decision"] != "pending" and not record.get("retryable"):
        lines.append(f"<!-- completed:{record['fingerprint']} -->")
    return "\n".join(lines)


def withdraw_approvals(gh, number, keep_head=None):
    for review in gh.pages(f"/pulls/{number}/reviews"):
        if own(review) and review["state"] == "APPROVED" and review.get("commit_id") != keep_head:
            gh.repo_call(f"/pulls/{number}/reviews/{review['id']}/dismissals", "PUT",
                         {"message": "This automated approval is no longer current; see the updated review summary."})


def invalidate(gh, pr):
    number = pr["number"]
    withdraw_approvals(gh, number)
    if any(label["name"] == LABELS["pass"] for label in pr["labels"]):
        gh.repo_call(f"/issues/{number}/labels/{urllib.parse.quote(LABELS['pass'], safe='')}", "DELETE")
    for comment in gh.pages(f"/issues/{number}/comments"):
        if own(comment):
            body = f"{MARKER}\n### Waiting for review\n\nThe PR head or base changed; the previous review is obsolete. A fresh review is required."
            gh.repo_call(f"/issues/comments/{comment['id']}", "PATCH", {"body": body})


def publish_one(gh, record, mode):
    record = dict(record)
    expected = record["snapshot"]
    number = expected["number"]
    pr = load_pr(gh, number)
    if pr["state"] != "open":
        return "closed"
    if snapshot(pr) != expected:
        # Do not publish the old findings onto the new revision.
        invalidate(gh, pr)
        return "stale"
    checks = ci_evidence(gh, pr, record["files"])
    if checks != record["ci"] or pr["draft"]:
        record = dict(record, ci=checks, decision="pending", result=None, retryable=True,
                      reasons=["PR or CI changed during review; waiting for a fresh evaluation."])
    record["mode"] = mode
    if record["decision"] == "pass" and (ci_state(checks) != "passed"
                    or record["result"] is None or not record["result"]["coverage_complete"]
                    or record["result"]["findings"] or record["result"]["uncertainties"]):
        raise ReviewError("Publication invariants failed; approval refused")

    def refresh_plan(fresh):
        record["publication"] = publication_plan(gh, fresh, record, mode)
        access = record["publication"].get("author_access")
        if access is not None and not access["verified"]:
            record["retryable"] = True
        return record["publication"]["event"] == "APPROVE"

    approve = refresh_plan(pr)
    withdraw_approvals(gh, number, keep_head=expected["head"] if approve else None)
    body = render(gh, record)
    if len(body) > 60000:
        raise ReviewError("Rendered review exceeds GitHub's comment limit")
    comments = [c for c in gh.pages(f"/issues/{number}/comments") if own(c)]
    reviews = gh.pages(f"/pulls/{number}/reviews")

    def submit(event):
        review_key = f"<!-- review:{record['fingerprint']}:{event} -->"
        if not any(own(r) and review_key in r["body"] and r["state"] != "DISMISSED" for r in reviews):
            gh.repo_call(f"/pulls/{number}/reviews", "POST", {
                "commit_id": expected["head"], "event": event,
                "body": f"{MARKER}\n{review_key}\n{record['decision'].title()}. See the maintained PR review summary for evidence."
            })

    if record["decision"] != "pending":
        fresh = load_pr(gh, number)
        if not current(fresh, expected):
            invalidate(gh, fresh)
            return "stale"
        # Recheck the author's membership immediately before any approval API
        # request, including when a previous approval is reused.
        if approve:
            approve = refresh_plan(fresh)
            if not approve:
                withdraw_approvals(gh, number)
        try:
            submit("APPROVE" if approve else "COMMENT")
        except ReviewError as exc:
            if not approve:
                raise
            withdraw_approvals(gh, number)
            approve = False
            record = dict(record, decision="human-review", retryable=True,
                          reasons=[f"GitHub did not accept the approval: {exc}"])
            refresh_plan(fresh)
            submit("COMMENT")
    # Detect head/base or membership changes after the approval request, before
    # publishing the success summary and label. GitHub has no atomic ACL/PR CAS.
    if approve:
        fresh = load_pr(gh, number)
        if not current(fresh, expected):
            invalidate(gh, fresh)
            return "stale"
        approve = refresh_plan(fresh)
        if not approve:
            withdraw_approvals(gh, number)
            submit("COMMENT")
    body = render(gh, record)
    if len(body) > 60000:
        raise ReviewError("Rendered review exceeds GitHub's comment limit")
    # Only apply the success label after GitHub actually accepts the review.
    # Report-only Pass is shown in the summary, not as an approval label.
    label = LABELS.get(record["decision"]) if record["decision"] != "pass" or approve else None
    existing = {item["name"] for item in pr["labels"]}
    for managed in LABELS.values():
        if managed in existing and managed != label:
            gh.repo_call(f"/issues/{number}/labels/{urllib.parse.quote(managed, safe='')}", "DELETE")
    if label and label not in existing:
        gh.repo_call(f"/issues/{number}/labels", "POST", {"labels": [label]})
    # Publish the completion marker only after the review API succeeded.
    if comments:
        if comments[-1]["body"] != body:
            gh.repo_call(f"/issues/comments/{comments[-1]['id']}", "PATCH", {"body": body})
    else:
        gh.repo_call(f"/issues/{number}/comments", "POST", {"body": body})
    return record["decision"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate", "publish"])
    parser.add_argument("--output", default="review-result.json")
    parser.add_argument("--input")
    args = parser.parse_args()
    gh = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    mode = os.getenv("PR_REVIEW_MODE", "report-only")
    if mode not in ("report-only", "auto-approve"):
        raise ReviewError("PR_REVIEW_MODE must be report-only or auto-approve")
    if args.command == "evaluate":
        provider, model, key = model_configuration()
        prs = gh.pages("/pulls?state=open")
        if len(prs) > POLICY["max_open_prs"]:
            raise ReviewError("Too many open PRs for one review run")
        records = []
        for item in prs:
            pr = load_pr(gh, item['number'])
            if not pr["head"]["repo"]:
                continue
            record = evaluate_one(gh, pr, model, mode, key, provider)
            if record is not None:
                records.append(record)
        Path(args.output).write_text(json.dumps({"repository": gh.repo, "records": records}, indent=2))
        print(f"Evaluated {len(records)} PRs; completed unchanged reviews were reused.")
    else:
        artifact = json.loads(Path(args.input).read_text())
        if artifact["repository"] != gh.repo:
            raise ReviewError("Artifact repository does not match")
        for record in artifact["records"]:
            status = publish_one(gh, record, mode)
            print(f"PR #{record['snapshot']['number']}: {status}")


if __name__ == "__main__":
    try:
        main()
    except ReviewError as exc:
        print(f"Review incomplete: {exc}")
        raise SystemExit(1)
