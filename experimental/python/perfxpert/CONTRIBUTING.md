# Contributing to PerfXpert

Thank you for your interest in improving PerfXpert!

## Extension surfaces

### Want to change how the LLM analyzes data?

Edit per-agent fence slices, **not** a monolithic reference guide:

- `perfxpert/agents/fence/always.md` — shared across every agent (~100 lines)
- `perfxpert/agents/fence/root.md` — Root agent instructions (incl. bottleneck narration)
- `perfxpert/agents/fence/analysis.md` — Analysis agent instructions
- `perfxpert/agents/fence/recommendation.md` — Recommendation agent instructions
- `perfxpert/agents/fence/correctness.md` — Correctness agent (incl. alternative-proposal prose)
- `perfxpert/agents/fence/compute_specialist.md` — Compute-bound technique expert
- `perfxpert/agents/fence/memory_specialist.md` — Memory-bound technique expert
- `perfxpert/agents/fence/latency_specialist.md` — Latency-bound technique expert

Each fence slice MUST stay ≤ 400 lines (enforced by CI — `tests/test_integration/test_narrow_scope.py`).

### Want to add a new bottleneck class?

See the walkthrough in design spec §6.6. Short version: add a row in
`knowledge/bottleneck_types.yaml`, extend `tools/bottleneck.py::classify_from_metrics`,
and register a new specialist agent. Touches ≤ 7 files; no architectural review needed.

### Want to add structured performance data (counters, GPU specs, etc.)?

Edit the appropriate file in `perfxpert/knowledge/*.yaml`. Schemas live in
`perfxpert/knowledge/_schemas/` and CI validates all YAMLs on every PR.

### Want to add a new tool?

Decide its MCP-exposure class:
- **READ_ONLY** → safe for MCP (external clients like Claude Desktop can call it)
- **EXECUTION** → in-process only (modifies filesystem / spawns subprocesses)

Annotate with `@tool_class(...)`. The MCP server only registers READ_ONLY tools;
`tests/test_integration/test_mcp_exposure.py` enforces this.

### Want to add a new LLM provider?

Add a new `perfxpert/providers/<name>_model.py` implementing the Agents-SDK
`Model` protocol. Add provider smoke test to `tests/test_providers/`. Update
`perfxpert/providers/__init__.py` to register in the provider list.

## Tests

```bash
pytest                                    # agentic default
PERFXPERT_LEGACY=1 pytest                 # legacy safety-net matrix (until vX.Y+1)
pytest --cov=perfxpert                    # with coverage
```

CI runs both matrices. Legacy matrix is secondary but still gating.
