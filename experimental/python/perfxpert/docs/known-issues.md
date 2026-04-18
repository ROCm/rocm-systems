# Known Issues

## LLM end-to-end smoke test may fail with 429 insufficient_quota (Phase 7.1)

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
