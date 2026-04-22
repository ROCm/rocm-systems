# Known Issues

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

## Ship state (cycle-3 convergence, 2026-04-18)

- **Cycle-3 reviewers**: 0 blockers, 0 important across all three branches.
- **Test suite**: 1036 passed / 3 skipped / 0 failed (measured 2026-04-19 after secret-scanner removal). Skips are documented opencode-binary absences (2) plus `test_llm_end_to_end.py` skip-on-429/auth/transient (1).
- **Secret scanning**: local-only dev tool; not shipped in the repo. Each developer is responsible for their own secret-detection tooling. The scanner, its CI workflow, pre-commit hook, and contributor guide were removed on 2026-04-19.
- **Known ongoing work** (not blocking ship):
  - LLM E2E `rec_type` assertion requires a live key with quota; use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and a model on-roster.
  - Confluence update remediation: see `docs/operations/confluence-publish.md` for the manual update recipe; automatic MCP publish requires Atlassian URL + token env vars.
