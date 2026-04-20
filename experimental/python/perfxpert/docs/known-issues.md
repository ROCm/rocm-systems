# PerfXpert — Known Issues

Active, living list. Entries that correspond to shipped fixes have been
removed; check `CHANGELOG.md` for history.

## Tool gate

Per-backend status (see `docs/guides/backends.md` for the full matrix):

- **opencode (bundled):** native `{block, retryWith}` gate via
  `.patches/0020-perfxpert-tool-gate.patch` — real pre-tool-call
  rejection, LLM sees a synthetic retry telling it to call
  `perfxpert_intent_classify` first.
- **Claude Code (`perfxpert-code claude`):** native `PreToolUse` hook
  installed into `.claude/settings.json` — mechanical rejection.
- **Gemini CLI (`perfxpert-code gemini`):** `allowedTools` restriction
  + runtime-state file for event-based lift — mechanical rejection.
- **Codex CLI (`perfxpert-code codex`):** **prompt-layer only** — Codex's
  native `PreToolUse` hook fires on Bash tool calls only, not on MCP
  calls, so we cannot reject perfxpert-tool misuse server-side. The
  installed `AGENTS.md` rejection-language stanza is the only guard.
  DECIDED design, see `docs/decisions/2026-04-19-codex-hook-surface.md`.

**Known limitation (Codex only):** an adversarial / smaller model can
still call `bash` / `read` before `perfxpert_intent_classify`. Cycle-5
E2E with Haiku-4-5 in scenario H2 confirmed the prompt-layer gate is
not 100%-effective on smaller models. Mitigations: stronger rejection
language in `AGENTS.md`, MCP tool-name rewrite so Codex sees all
perfxpert tools under the `perfxpert_` prefix for unambiguous prompt
pattern-matching. Disable the prompt gate via
`PERFXPERT_DISABLE_TOOL_GATE=1` if it interferes with a workflow.

## LLM end-to-end smoke test may fail with 429 insufficient_quota

`tests/test_integration/test_llm_end_to_end.py::test_llm_enabled_produces_rec_type`
executes a real OpenAI Agents SDK call. The test skips automatically
when `OPENAI_API_KEY` is unset OR when the provider returns `429`
`insufficient_quota` / `auth_error` / documented transient errors. If
you're hitting a hard fail, ensure your key has quota or set
`OPENAI_API_KEY=""` to force the skip.

Workarounds to confirm wiring on a different account:

- Different key: `OPENAI_API_KEY=sk-... pytest -k test_llm_enabled_produces_rec_type`
- Provider-specific override: `PERFXPERT_AGENTS_MODEL_OPENAI=gpt-3.5-turbo ...`
- Global model override: `PERFXPERT_LLM_MODEL=gpt-4o-mini ...`
- Bump runner turn budget: `PERFXPERT_AGENTS_MAX_TURNS=5` (default 10)

## LLM provider routing (agentic runtime)

As of the 2026-04-20 Docker install-validation, two provider routing
paths are still broken and being tracked separately:

- **`perfxpert analyze --llm anthropic`** — `framework._resolve_model()`
  passes the resolved model name (`claude-sonnet-4-5`) to the
  openai-agents SDK without a LiteLLM prefix. The SDK defaults to the
  OpenAI endpoint and fails with
  `model 'claude-sonnet-4-5' does not exist`. Fix pending: prefix
  claude models with `litellm/anthropic/` (or equivalent SDK-specific
  routing glue) when the selected provider is `anthropic`.

- **`perfxpert analyze --llm claude-code`** — CLI argparse advertises
  `claude-code` as a `--llm` choice but `agents.runtime.build_session`
  never registers it, so any invocation fails with
  `ValueError: unknown provider 'claude-code'`. Fix pending: either
  register a real claude-code provider mapping to the
  `claude-agent-sdk` package, or drop `claude-code` from the argparse
  choices.

Workaround today: use `PERFXPERT_AIRGAP=1` for deterministic analysis
without LLM round-trips; or use the openai provider against a key
with quota.

## Rate-limit escape hatch is opencode-process-scoped

Setting `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` kills client-side
retries in the opencode process, but the provider's own quota
enforcement is external and not affected. Use
`PERFXPERT_LLM_FALLBACK_CHAIN` to cascade across providers when
rate-limited.

## Opencode submodule pristine requirement

The opencode submodule is pinned at `v1.4.11` (MIT) and all
customizations live in `.patches/*.patch`. Do NOT commit mutations
inside the submodule; the submodule's committed state must stay
pristine so `git submodule update` can fetch upstream fixes. The
`setup.py` build hook resets the submodule + re-applies patches on
every rebuild, so any in-tree submodule edits will be clobbered.

## Ship state (2026-04-20)

- Test suite: 1387 passed / 0 failed / 3 skipped (env-gated: OpenAI
  quota + `gemini` CLI missing + `codex` CLI missing). Phase 8 LLM
  provider routing fix added 5 tests (4 in
  `test_agents/test_framework.py` covering LitellmModel wiring for
  anthropic / claude-code / plain-openai / double-prefix guards, plus 1
  in `test_agents/test_runtime.py` for
  `build_session(provider="claude-code")`).
- All three docs scanners green (`scripts/lint.sh`,
  `scripts/link-checker.py`, `scripts/test-samples.py`) in `--strict`
  mode; enforced by `tests/test_docs_tooling/test_ship_readiness.py`.
- End-to-end Docker install validated in
  `rocm/dev-ubuntu-22.04:latest`: single
  `pip install "perfxpert[all] @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=experimental/python/perfxpert"`
  completes in ~3 min, auto-downloads bun into
  `~/.cache/perfxpert/bun/` if not on PATH, compiles the bundled
  opencode binary, yields `perfxpert doctor` → **ALL CLEAN**.

## Docs-audit scanner scope

The ship-readiness scanners skip paths that are legitimate third-party
or historical content:

`scripts/lint.sh`:
- `**/.git/**`, `**/.pytest_cache/**` — git / test runner internals.
- `**/docs/confluence/**` — Confluence-amend audit artifacts that must
  reference the old symbol names to describe what was scrubbed.
- `**/opencode/**`, `**/node_modules/**` — upstream opencode submodule
  (MIT) and its bun-installed JS dep tree. Contains legitimate
  third-party stream-resume API references (the ban applies only to
  the legacy class we deleted, removed in Phase 7.1 — not Node
  stream APIs), translated READMEs, test fixtures with
  deliberately-incomplete internal links.
- `**/perfxpert/ai_analysis/**` — legacy module kept as a stub to avoid
  bisect-churn; never imported at runtime.

Individual hits on a line containing the literal phrase
`"removed in Phase 7.1"` are also ignored so the CHANGELOG / archive
sections can keep searchable records of removed flags + classes
without re-introducing live guidance.

`scripts/link-checker.py`:
- External URLs (`http://` / `https://`) are not validated.
- Anchor fragments (`#section-id`) are stripped before the existence
  check.
- Same path exclusions as the lint scanner.

## Historical: LLM payload field-name mismatch (obsolete — rocm-systems#4979)

**Status: obsolete. No fix required on perfxpert.**

In the pre-refactor codebase, the now-deleted bridge function
`ai_analysis/api.py::_convert_result_to_llm_format()` emitted kernel
dictionaries with the keys `calls` and `percent_of_total`, but the
consumer `ai_analysis/llm_analyzer.py::_sanitize_data()` expected
`dispatch_count` and `pct_total_time`. Memory directions also leaked
as verbose labels (`Host-to-Device`) instead of compact IDs (`h2d`,
`d2h`, `d2d`). The effect was that the LLM received `None` for every
kernel metric — silent data loss, not a crash.

Upstream PR
[rocm-systems#4979](https://github.com/ROCm/rocm-systems/pull/4979)
added `_MEMORY_DIR_MAP` plus field-name renames inside
`ai_analysis/api.py`. It was never merged.

**Why the bug cannot occur in perfxpert:** the agentic refactor deleted
the entire `perfxpert/ai_analysis/` package — both sides of the
mismatched bridge are gone. The current flow is producer-consumer
symmetric by construction: `analysis/core.py::identify_hotspots()`,
`analysis/recommendations.py`, and `agents/framework.py::_sdk_invoke()`
all share the same key vocabulary with no translation layer between
them. Preserved here for institutional memory so future contributors
who find PR #4979 in the commit history understand why it was closed
without being ported.
## Docs-audit baseline

Tracks docs-audit gaps that cannot be mechanically fixed by the
scanners but are known and tolerated for now. Each entry has a
one-line rationale + optional follow-up tracking id.

### Zero-violation baseline

None. All three scanners (`scripts/lint.sh`, `scripts/link-checker.py`,
`scripts/test-samples.py`) report zero violations live, enforced by
`tests/test_docs_tooling/test_ship_readiness.py` — which runs each
scanner in `--strict` mode and asserts `rc == 0`. Green test = zero
violations today; no frozen JSON snapshot is kept in the repo.

### Scanner scope limitations

Documented here so users reading "zero violations" know what is and
isn't covered.

#### `scripts/link-checker.py`
- **External URLs not validated.** Any `http://` or `https://` link is
  skipped (`is_external_url`). Dead external links will not flag.
- **Anchor fragments not validated.** `#section-id` is stripped before
  the file-existence check. A link pointing at a missing anchor inside
  a real file passes.
- **`--strict` is output-format only.** It suppresses the
  human-readable preamble and only emits CSV rows; it does NOT enable
  stricter checks. The set of validated link classes is identical in
  both modes.

Workaround: rely on Markdown preview in your IDE / GitHub for anchor
correctness; external URL health is covered nightly by a separate
link-health workflow (not part of the zero-violation baseline).

#### `scripts/lint.sh` — banned-string scanner
The banned-string scan excludes these paths (`lint.sh:50-54`) so that
historical context or pre-existing test fixtures don't cause false
positives:

- `**/.git/**` — git internals
- `**/.pytest_cache/**` — test runner cache
- `**/perfxpert/ai_analysis/**` — legacy module removed during the
  agentic refactor; banned terms inside historical fixtures are not live.

The scanner also ignores individual hits whose line contains the
historical-anchor phrase "removed in Phase 7.1"; this lets us keep a
searchable record of removed flags, env vars, and classes (e.g. in
`CHANGELOG.md`) without re-introducing live guidance.

Consequence: banned strings inside the excluded paths (or on lines
carrying the historical anchor) will NOT trip the scanner. If you're
adding a new fixture directory that should be ignored for legitimate
reasons, add it here with a one-line comment.

### Out-of-scope follow-ups

- `tests/test_docs_tooling/test_secret_scanner.py` — three tests fail
  locally because they cd into a fresh `tempfile.TemporaryDirectory()`
  without symlinking `tools/_secret_scanner.py` first. Pre-existing
  bug (commit c2c419ff9e).
  - **Reproducer:** `pytest tests/test_docs_tooling/test_secret_scanner.py -q`
  - **Owner / tracking:** queued in a follow-up sweep; no
    issue number assigned yet. Fix is a one-line tmpdir setup (copy
    or symlink `tools/_secret_scanner.py` into the tmpdir before the
    subprocess call).
  - **Workaround for affected devs:** skip the three `test_secret_scanner_*`
    tests with `-k 'not secret_scanner'` while the fix is queued;
    they don't gate any other scanner.
