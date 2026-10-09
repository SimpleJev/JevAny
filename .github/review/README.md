# Automated PR review

The review runs entirely on GitHub-hosted Ubuntu runners in `SimpleJev/JevAny`.
No local process or GPU runner is required. Forks can run the rule tests; the
regular publication workflow is restricted to the upstream repository.

## Results

- **Pass**: applicable CI jobs passed; every changed file was covered; no
  substantive findings or unresolved uncertainties. Code and configuration can
  Pass as well as documentation. Publication is a separate decision:
  - In `auto-approve` mode, ordinary documentation by a verified current
    repository collaborator receives `APPROVE`.
  - Code/configuration/mixed changes, documentation by a non-collaborator, an
    unverifiable collaborator status, or `report-only` mode receive a Pass
    summary and `COMMENT`, without approval.
- **Human Review**: substantive issues, failed or
  incomplete CI, or an unavailable/incomplete model review. The bot submits
  `COMMENT`, with evidence in one maintained summary.
- Running CI and draft PRs are temporary waiting states, never approvals.

There is no request-changes, close, merge, code-edit, or branch-protection action.
CI failures remain failures. Style preferences and lack of GPU tests do not
escalate an otherwise clean documentation PR. `ai:approved` and `ai:human-review`
are informational labels, not merge requirements.
`ai:approved` is applied only after GitHub accepts an actual approval, never
for Pass with report-only publication.

## GitHub configuration

1. Enable Actions in `SimpleJev/JevAny`. For automatic approval, enable **Settings →
   Actions → General → Allow GitHub Actions to create and approve pull requests**.
   Workflow jobs request their own minimal permissions; global write access is
   unnecessary.
2. Select the provider with the Actions variable `PR_REVIEW_PROVIDER`:
   `openai` (default) or `openrouter`. Add the corresponding Actions secret,
   `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. Never commit the key. An OpenRouter
   key is sufficient when using OpenRouter; an OpenAI key is not also needed.
3. Add an Actions variable `PR_REVIEW_MODEL` with an accessible model ID for that
   provider. There is deliberately no default model in the regular controller.
   OpenAI uses Responses API Structured Outputs with `store: false`. OpenRouter
   uses Chat Completions with strict JSON Schema and `require_parameters: true`
   so routing requires compatible endpoints. Both paths have no shell/tools and
   validate the returned schema and finding locations. Truncation, refusal,
   missing content and API errors require human review.
4. `PR_REVIEW_MODE` defaults to `report-only`. Set the variable to `auto-approve`
   after inspecting the rule tests and review results. Then manually run **PR
   review** on `main` to reevaluate open PRs with the new configuration.

The OpenRouter configuration used in the fork validation was:

| GitHub setting | Value |
| --- | --- |
| Actions repository secret `OPENROUTER_API_KEY` | An OpenRouter key configured on this repository |
| Actions variable `PR_REVIEW_PROVIDER` | `openrouter` |
| Actions variable `PR_REVIEW_MODEL` | `openai/gpt-5-mini` |
| Actions variable `PR_REVIEW_MODE` | `report-only` to observe, or `auto-approve` for eligible documentation |

Fork secrets and Actions approval settings do not configure the upstream
repository. Configure upstream separately. `auto-approve` never merges a PR;
merge remains a maintainer action.

Missing key/model produces Human Review and explicitly states that AI review has
not completed. API usage is billed to the configured provider account; no key is
needed for **Review workflow tests**.

## Evidence and trust boundaries

The controller checks out the exact trusted workflow commit. PR code, dependency
files, prompts, and workflow changes are never executed by a job with the model
key or review write permission. PR content is fetched through the GitHub API as
data. The evaluation job is read-only; a separate job publishes only its same-run
artifact. Neither job restores PR-controlled caches or artifacts from CI.
PR metadata events pass through a no-checkout, no-secret, no-permission signal
workflow. `workflow_run` executes the controller on the default branch. The
controller ignores the signal's contents and fetches GitHub state itself. No
`pull_request_target` exception or event-policy opt-out is needed. Standard
first-time fork contributor approvals for Actions still apply.

CI is collected from `pull_request` runs for the same PR, head and base, with all
expected jobs required to execute successfully. Main's green checks and skipped
jobs are not evidence. Pages is required only for its configured path patterns.
The rule tests detect drift between policy and the current CI/Pages definitions.
CPU CI is not evidence of GPU correctness or production-model performance.
CI and Pages record the PR number, head and base in `run-name`. This preserves
event identity when GitHub omits the PR association array for an external fork.
It changes the displayed run title, not the existing tests or their scope.
Existing external branches using the older workflow must update from `main` to
record this identity. A successful legacy fork run without base evidence is
reported as Human Review, never silently accepted as current CI.

Full old/new changed text and pinned base reference documents are supplied to the
model. Binary, missing, truncated, symlink/submodule, or oversized content routes
to Human Review. Defaults cap one review at 80 changed files, 60 KB per file,
240 KB source context, and 8,000 output tokens. Larger PRs need human review;
there is no silent truncation followed by approval. The reviewer does not
browse external URLs or execute code to substantiate claims.

The publisher rereads PR/CI state, pins each review to `commit_id`, and dismisses
only its own marked approvals when they become invalid. It also checks after
approval for a concurrent head/base change. Auto-approval requires a successful
GitHub collaborator check for the **PR author on the target repository**, not
the person triggering the workflow, a commit author, a historical contributor,
or an organization membership label. The check uses the collaborator API and
includes collaborator access through teams. It is repeated immediately before
and after approval; a detected loss of membership withdraws the bot approval
and retains Pass with report-only publication. An API failure never authorizes
approval. Membership changes invalidate the cached review on the next workflow
run; there is no continuous membership watcher. GitHub's review API cannot
atomically compare-and-swap head, base and collaborator membership, so these
checks minimize races without installing a merge lock.
No new merge restriction is installed. One repository-level concurrency group
serializes runs; each run scans open PRs so coalesced queued events lose no PRs.
Fingerprints deduplicate completed unchanged reviews. API/model failures remain
retryable. Run the workflow manually to retry a configuration or service failure.
The current base is resolved through the Git ref API: a PR's `base.sha` can
remain cached at an older revision after the target branch moves. Controller
code changes also invalidate the completed-review fingerprint.

## Validation

**Review workflow tests** runs policy, CI matching, schema/coverage, stale result,
publication/deduplication, and workflow-boundary tests on GitHub without a model.
These fixtures verify the program's decisions, not AI bug-detection quality.

The implementation was exercised on real PRs in `YuanchenBei/JevAny` before
preparing this upstream version:

- [Publisher and collaborator integration](https://github.com/YuanchenBei/JevAny/actions/runs/37081183546):
  GitHub confirmed the fixture author's collaborator status, accepted APPROVE
  on the tested head, deduplicated repeated publication, and dismissed the test
  approval when the synthetic result changed to Human Review. This test used a
  synthetic model result; it verifies GitHub APIs, not AI judgment.
- [OpenRouter quality test](https://github.com/YuanchenBei/JevAny/actions/runs/37080022948):
  two real documentation PRs were reviewed twice each with `openai/gpt-5-mini`.
  Both clean reviews passed, and both incorrect-count reviews identified the
  3,220 versus 724 error. The final results were published as comments. These
  four calls establish only limited documentation-fixture quality, not general
  code-review accuracy or a live end-to-end AI-to-auto-approval deployment.

The temporary-approval and paid model-fixture workflows stay in the test fork;
they are not installed in upstream. Production approvals are not withdrawn as
test cleanup. They are dismissed only when the controller finds them obsolete
or no longer eligible under the current publication policy.

The rule suite covers these outcomes (several edge cases use simulated API
responses rather than changing real GitHub permissions):

| Sample | Expected result |
| --- | --- |
| Clean docs spelling/translation | Pass; actual approval only in auto-approve mode |
| Documented metric changed to contradict a pinned source | Human Review, with the actual factual issue |
| Intentional legacy `letter` compatibility alias | No invented naming defect |
| Clean code/configuration change with complete review and passing CI | Pass, report only |
| Clean documentation by a current repository collaborator | Pass; eligible for auto-approve |
| Clean documentation by a non-collaborator or unverifiable author | Pass, report only |
| Author loses collaborator access before/during approval | No approval, or dismiss the test approval |
| Failed CI | Human Review; CI stays red |
| CI still running | Wait; no approval |
| New head/base during review | Discard old result; withdraw obsolete bot approval |
| Repeated callbacks/reruns | Update one summary; no repeated identical review |
| Missing key/API failure/invalid output | Human Review; no claim of AI completion |

After installation, run **Review workflow tests** and inspect a review on the
target repository before enabling routine approvals. Model-quality checks must
inspect actual findings, not merely the Human Review label. Passing these
tests cannot guarantee that AI finds every real bug. CPU CI and AI review do
not establish production-model quality, CUDA correctness or GPU performance.

References: [GitHub workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows),
[secure Actions use](https://docs.github.com/en/actions/reference/security/secure-use),
[reviews API](https://docs.github.com/en/rest/pulls/reviews),
[collaborator API](https://docs.github.com/en/rest/collaborators/collaborators#check-if-a-user-is-a-repository-collaborator),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).
