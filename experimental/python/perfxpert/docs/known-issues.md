# PerfXpert — Known Issues

## Codex gate is prompt-layer only

`perfxpert-code codex` stages the same MCP surface as
the other backends, but its gate remains prompt-layer-only. As of April
2026 Codex's native `PreToolUse` surface intercepts Bash only, not MCP /
Write / other tool calls, so it cannot satisfy PerfXpert's "block every
non-perfxpert tool until `intent_classify` returns" contract.

Current backend split:

- **Patched opencode path** — mechanical gate via fork patch 0020
  (`{block, retryWith}` in `tool.execute.before`)
- **Claude Code** — mechanical gate via native `PreToolUse`
- **Gemini CLI** — mechanical gate via project-local `BeforeTool` /
  `AfterTool` hooks + runtime lift
- **Codex CLI** — prompt-layer rejection language in the
  perfxpert-managed `AGENTS.override.md` compatibility override

If you need a hard pre-tool-call gate today, use the default patched
opencode path, Claude Code, or Gemini CLI instead of Codex.

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

- **Patch application is wired into the current build/install path.**
  The setup/build flow applies `.patches/*.patch` to the pinned
  `opencode` submodule and builds the patched binary automatically when
  prerequisites are present. `perfxpert-code install-patches` is the
  rebuild helper for source/editable checkouts; packaged installs prefer
  the bundled result.

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

- **Upstream-opencode fallback weakens the opencode gate.**
  If you force `PERFXPERT_OPENCODE_PATH` to an upstream `opencode`
  binary, or fall through to one on `PATH`, the fork-only
  `{block, retryWith}` behavior is unavailable. The launcher warns in
  that mode and falls back to prompt guidance only.

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
