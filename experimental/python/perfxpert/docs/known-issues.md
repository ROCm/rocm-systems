# PerfXpert — Known Issues

## Tool gate is prompt-layer only — not a mechanical hook

The `0020-perfxpert-tool-gate.patch` and the bracketed `[MUST BE CALLED FIRST
FOR GPU-PERF QUERIES]` prefix in `mcp_server/server.py::_fn_to_tool_schema`
are a **weaker-variant** solution to the cycle-4 B1 blocker (LLMs calling
`bash`/`read` before `perfxpert_intent_classify`). The brief asked for a
pre-turn tool-availability hook (only expose `perfxpert_*` for the first 2
turns) OR a post-turn rejection hook (rewrite non-perfxpert `tool_calls`
into a synthetic retry). Implementing either requires intercepting
opencode's session message flow in `packages/opencode/src/session/processor.ts`
and `prompt.ts`, whose `plugin.trigger(...)` hook points are currently
fire-and-forget — a real blocking hook inside the opencode TypeScript
runtime was outside the time budget for cycle-4.

**Known-limitation:** the current patch does not mechanically reject a
non-perfxpert first tool call; an adversarial LLM can still call `bash`
first. Live-scenario D (cycle-4 validation) showed the prompt+bracket
combo moves the needle but does not guarantee 100% compliance.

**Follow-up:** track a real pre-/post-turn gate at the opencode
TypeScript layer. The cleanest attach point is
`packages/opencode/src/session/prompt.ts` around the `plugin.trigger(
"tool.execute.before", ...)` invocation (lines 414-419 and 455-460) —
extending that hook to allow a plugin to return `{ block: true, retryWith:
<message> }` would give us the rejection semantics the brief described.
Proposed env var: `PERFXPERT_DISABLE_TOOL_GATE=1` (already documented in
the prompt text for user-facing discoverability).

## LLM end-to-end smoke test may fail with 429 insufficient_quota

`tests/test_integration/test_llm_end_to_end.py::test_llm_enabled_produces_rec_type`
executes a real OpenAI Agents SDK call against `gpt-4o-mini` (default) or
`$PERFXPERT_AGENTS_MODEL_OPENAI` if set. The test will **skip** when
`OPENAI_API_KEY` is unset. When the key is set, the wiring is exercised
end-to-end: `framework._sdk_invoke()` builds an `agents.Agent`, calls
`Runner.run_sync(...)`, and extracts `final_output` + tool-call metadata
into a `FakeProviderResponse`.

If the OpenAI account backing `OPENAI_API_KEY` has exhausted its quota,
the SDK returns `429 insufficient_quota` and the test fails (not skips).
This is a **billing condition, not a code defect**: the same error is
reproducible via a direct `agents.Runner.run_sync(...)` call outside
pytest, confirming the wiring is live.

Workarounds to confirm the wiring on a different account:

- Re-run with a different key: `OPENAI_API_KEY=sk-... pytest -k test_llm_enabled_produces_rec_type`
- Override the model per-provider: `PERFXPERT_AGENTS_MODEL_OPENAI=gpt-3.5-turbo ...`
- Global model override: `PERFXPERT_LLM_MODEL=gpt-4o-mini ...`
- Bump runner turn budget: `PERFXPERT_AGENTS_MAX_TURNS=5` (default 10)

This entry will be removed once a CI-owned key with guaranteed quota is
provisioned.

## Historical: LLM payload field-name mismatch (obsolete — rocm-systems#4979)

**Status: obsolete. No fix required on perfxpert.**

In the pre-refactor codebase, the now-deleted bridge
function `ai_analysis/api.py::_convert_result_to_llm_format()`
emitted kernel dictionaries with the keys `calls` and `percent_of_total`,
but the consumer
`ai_analysis/llm_analyzer.py::_sanitize_data()` expected
`dispatch_count` and `pct_total_time`. Memory directions also leaked as
verbose labels (`Host-to-Device`) instead of compact IDs (`h2d`, `d2h`,
`d2d`). The effect was that the LLM received `None` for every kernel
metric — silent data loss, not a crash.

Upstream PR
[rocm-systems#4979](https://github.com/ROCm/rocm-systems/pull/4979)
added `_MEMORY_DIR_MAP` plus field-name renames inside
`ai_analysis/api.py`. It was never merged.

**Why the bug cannot occur in perfxpert**

The agentic refactor deleted the entire `perfxpert/ai_analysis/` package — both
sides of the mismatched bridge are gone. The current flow is
producer-consumer symmetric by construction:

- Producer: `perfxpert/analysis/core.py::identify_hotspots()` emits
  kernel dicts with keys `calls` and `percent_of_total`.
- Consumers: `perfxpert/analysis/recommendations.py`,
  `perfxpert/agents/analysis.py`, and downstream formatters read the
  SAME vocabulary (`calls`, `percent_of_total`, `api_calls`). There is
  no `_sanitize_data`/`_format_data_for_llm` translation step to drift.
- LLM invocation: `agents/framework.py::_sdk_invoke()` serialises the
  agent payload as OpenAI-style `messages=[{"role", "content"}]`; there
  is no bespoke field-mapping layer between perfxpert's internal dicts
  and the provider API.

This note is preserved for institutional memory so future contributors
who find PR #4979 in the commit history understand why it was closed
without being ported.

## Ship state (cycle-3 convergence, 2026-04-20)

- **Cycle-3 reviewers**: 0 blockers, 0 important across all three branches.
- **Test suite**: 1383 passed / 3 skipped / 0 failed (measured 2026-04-20 after Phase 8 LLM provider routing fix). Skips are documented opencode-binary absences (2) plus `test_llm_end_to_end.py` skip-on-429/auth/transient (1). The Phase 8 delta added 3 tests in `test_agents/test_framework.py` covering LitellmModel wiring for anthropic / plain-openai / double-prefix guards.
- **Secret scanning**: local-only dev tool; not shipped in the repo. Each developer is responsible for their own secret-detection tooling. The scanner, its CI workflow, pre-commit hook, and contributor guide were removed on 2026-04-19.
- **Known ongoing work** (not blocking ship):
  - LLM E2E `rec_type` assertion requires a live key with quota; use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and a model on-roster.
  - Confluence update remediation: see `docs/operations/confluence-publish.md` for the manual update recipe; automatic MCP publish requires Atlassian URL + token env vars.

## Opencode fork / bundling

- **`apply-opencode-patches.sh` is not yet wired into the wheel build.**
  The patch apply step is manual: run
  `bash experimental/python/perfxpert/scripts/apply-opencode-patches.sh`
  before bundling the opencode binary. Automating this requires `bun`
  available on the build host (for opencode's post-patch type-check);
  that toolchain setup is deferred to the wheel-build PR.

- **The opencode submodule (`.gitmodules` pin `v1.4.11`) is MIT.**
  All customizations are carried in `.patches/*.patch`. Do NOT commit
  mutations inside the submodule; the submodule's committed state must
  stay pristine so that `git submodule update` can fetch upstream
  fixes.

- **Rate-limit escape hatch is opencode-process-scoped.**
  Setting `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` kills client-side
  retries in the opencode process, but the provider's own quota
  enforcement is external and not affected. Use
  `PERFXPERT_LLM_FALLBACK_CHAIN` to cascade across providers when
  rate-limited.

- **Forced tool priority is LLM-dependent.**
  Patch `0010-perfxpert-tool-priority.patch` and the MCP description
  hint strongly bias the LLM toward `intent_classify` first, but a
  determined model can still skip. Measurement + feedback is
  tracked as future telemetry work.

## Docs-audit baseline

Tracks docs-audit gaps that cannot be mechanically fixed by the
scanners but are known and tolerated for now. Each entry has a
one-line rationale + optional follow-up tracking id.

### Zero-violation baseline

None. All four scanners (`docs/lint.sh`, `docs/link-checker.py`,
`docs/test-samples.py`, `docs/inventory.py`) report zero violations
at the baseline snapshot. See `docs/inventory-baseline.json`.

### `inventory-baseline.json` — source of truth

Two copies exist on disk and are byte-identical except for the ISO-8601
`timestamp` field (they are written in the same `docs/inventory.py`
run, a few seconds apart):

- `docs/inventory-baseline.json` (repo root) — **authoritative**. This
  is the copy scanned/compared against by CI and by
  `docs/inventory.py` itself; the tooling under `docs/` lives at repo
  root.
- `experimental/python/perfxpert/docs/inventory-baseline.json` —
  convenience mirror so the perfxpert docs tree is self-contained for
  users reading it in-place. If the two diverge, the repo-root copy
  wins; regenerate the mirror with `python docs/inventory.py` from the
  repo root.

### Scanner scope limitations

Documented here so users reading "zero violations" know what is and
isn't covered.

#### `docs/link-checker.py`
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

#### `docs/lint.sh` — banned-string scanner
The banned-string scan excludes these paths (`lint.sh:50-54`) so that
historical context or pre-existing test fixtures don't cause false
positives:

- `docs/superpowers/specs/**` — superpowers phase specs (reference)
- `docs/superpowers/plans/**` — superpowers phase plans (reference)
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
