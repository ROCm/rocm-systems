# perfxpert Architecture

_Last refreshed: v0.2.0. Source of truth:
`docs/superpowers/specs/2026-04-17-multi-agent-perfxpert-design.md`._

## High-level shape

perfxpert is a **multi-agent system** for GPU performance analysis on AMD
ROCm. At the top there's a batch CLI (`perfxpert analyze`) and an
interactive wrapper (`perfxpert-code`). Both drive the same agent runtime.

## Agents (spec §2)

```
Root (intent router)
 ├─ Analysis  (classifies bottleneck, gathers metrics)
 ├─ Recommendation (hands off to specialists)
 │   ├─ compute_specialist
 │   ├─ memory_specialist
 │   └─ latency_specialist
 └─ Correctness (enforces 5 gates on proposed changes)
```

7 agents total. Each has ≤ 400 lines of fence + ≤ 5 tools + ≤ 10 input / ≤ 5 output fields. Narrow scope is CI-enforced.

## Tools (spec §3 + Appendix A)

~45 deterministic Python functions split into two MCP-exposure classes:

- **READ_ONLY** — safe for MCP (agent lookups, analysis, classification)
- **EXECUTION** — in-process only (profile.run, compile.build, patch.apply)

Every tool is pure (modulo knowledge YAML loads and SQL reads); < 100 ms p99.

## Knowledge (spec §3 Appendix B)

22 YAML files under `perfxpert/knowledge/`, each paired with a JSON schema
in `_schemas/`.
CI validates every YAML against its schema on every PR.

## Providers (spec §3)

5 LLM provider adapters under `perfxpert/providers/`:

- Anthropic (Claude Sonnet family)
- OpenAI (GPT-4 family)
- Ollama (local)
- Private (any OpenAI-compatible endpoint)
- Opencode (bundled subprocess wrapper)

All five verified nightly via `.github/workflows/perfxpert-nightly.yml`.

## Runtime middleware (spec §5)

`perfxpert/runtime/`:

- `gate_cascade.py` — cascaded 5-gate correctness pipeline
- `intent_classifier.py` — deterministic router (air-gap mode)
- `recursion_guard.py` — blocks opencode-in-opencode recursion

## Data flow

### Batch CLI

```
user  →  perfxpert analyze -i trace.db
        → api.analyze_database()
        → agent runtime (Root → Analysis → bottleneck.classify …)
        → Recommendation hands off to specialist
        → Correctness runs 5 gates
        → formatters → text / json / markdown / html
```

### Interactive

```
user  →  perfxpert-code  (bundled opencode)
        → opencode session with AMD-themed config
        → perfxpert MCP server exposes READ_ONLY tools
        → same agent runtime under the hood
```

### Library API

```python
from perfxpert import analyze_database
result = analyze_database("trace.db", provider="anthropic")
```

## Test pyramid (spec §6)

```
Level 5 — Benchmarks (nightly)        TritonBench + KernelBench
Level 4 — End-to-end CLI + SDK (PR)   pytest tests/e2e
Level 3 — Agent integration (PR)      pytest tests/test_integration
Level 2 — Per-agent isolation (PR)    pytest tests/test_agents
Level 1 — Per-tool unit (PR)          pytest tests/test_tools
Level 0 — Knowledge YAML (PR)         pytest tests/test_knowledge
```

## Correctness gates (spec §5)

1. **Claims** — magnitude within proven_optimizations range
2. **Sakana** — hardware-counter sanity
3. **Schema** — output shape valid
4. **Regression** — no hot kernel regressed > 5% (weighted-geomean definition)
5. **Correctness** — semantic preservation (structural)

## What's NOT in this diagram

The following symbols were deleted during the agentic refactor and are
no longer present:

- `interactive.py`, `llm_conversation.py` — bespoke LLM-session state
  machine, superseded by OpenAI Agents SDK sessions.
- `perfxpert/ai_analysis/` module — superseded by `perfxpert/agents/`.
- `PERFXPERT_LEGACY` env var — no longer recognized.
- `PERFXPERT_USE_AGENTS` env var — no longer recognized.

Consult the git history or [CHANGELOG.md](../CHANGELOG.md) for the
old code.

## Pointers

- Full spec: `docs/superpowers/specs/2026-04-17-multi-agent-perfxpert-design.md`
- Phase plans: `docs/superpowers/plans/2026-04-17-perfxpert-phase*`
- Contributor entry: `CONTRIBUTING.md`
- RFCs: `docs/rfcs/`
