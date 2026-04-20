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
  DECIDED design — rationale captured in the local Codex hook-surface
  decision record (referenced inline from `docs/guides/backends.md`).

**Known limitation (Codex only):** an adversarial / smaller model can
still call `bash` / `read` before `perfxpert_intent_classify`. Cycle-5
E2E with Haiku-4-5 in scenario H2 confirmed the prompt-layer gate is
not 100%-effective on smaller models. Mitigations: stronger rejection
language in `AGENTS.md`, MCP tool-name rewrite so Codex sees all
perfxpert tools under the `perfxpert_` prefix for unambiguous prompt
pattern-matching. Disable the prompt gate via
`PERFXPERT_DISABLE_TOOL_GATE=1` if it interferes with a workflow.

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

- Test suite: 1385 passed / 0 failed / 3–5 skipped (env-gated: OpenAI
  quota + `gemini` CLI missing + `codex` CLI missing + 2 opencode-patch
  tests skip when the submodule isn't initialized locally). Phase 8 LLM
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
The banned-string scan excludes these paths (see the `find -not -path`
clauses in `scripts/lint.sh`) so that historical context or pre-existing
test fixtures don't cause false positives:

- `**/.git/**` — git internals
- `**/.pytest_cache/**` — test runner cache
- `**/docs/confluence/**` — Confluence-amend audit artifacts.
- `**/opencode/**`, `**/node_modules/**` — upstream submodule + JS deps.

The scanner also ignores individual hits whose line contains the
historical-anchor phrase "removed in Phase 7.1"; this lets us keep a
searchable record of removed flags, env vars, and classes (e.g. in
`CHANGELOG.md`) without re-introducing live guidance.

Consequence: banned strings inside the excluded paths (or on lines
carrying the historical anchor) will NOT trip the scanner. If you're
adding a new fixture directory that should be ignored for legitimate
reasons, add it here with a one-line comment.

### Environmental / informational

- **LLM billing errors surface as one-line messages.** Quota-exhausted
  (429 `insufficient_quota`), authentication (401 / `invalid_api_key`),
  rate-limit (429 `rate_limit_exceeded`), and transient (5xx / timeout)
  failures from any `--llm <provider>` call are classified by
  `perfxpert.providers._exceptions.classify_sdk_error` and rendered as
  a single actionable stderr line (e.g. `⚠ LLM quota exhausted on
  openai …`). The corresponding `test_llm_enabled_produces_rec_type`
  e2e test skips cleanly on each of these classes; it is no longer a
  defect, just a billing / account condition. Set `PERFXPERT_DEBUG=1`
  to see the raw SDK traceback when diagnosing.
