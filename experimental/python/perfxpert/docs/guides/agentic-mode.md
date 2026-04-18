# Agentic Mode Guide

PerfXpert's analysis can run in two modes:

- **Air-gap mode** — no LLM call, deterministic only. All analysis
  goes through the same agentic routing + tools, but every agent
  returns its "airgap" path: rule-based classification + knowledge
  lookups, without model inference.
- **LLM-enabled mode** — identical routing, but agents call out to an
  LLM for free-form narrative + rich recommendations.

This guide explains how to select between them, what changes in the
output, and how provider selection works.

Cross-links:
- [Agent hierarchy](../architecture/agent-hierarchy.md) — who calls
  the LLM and who doesn't
- [Gate cascade](../architecture/gate-cascade.md) — correctness is
  identical in both modes
- [MCP server](../integration/mcp-server.md) — always air-gap-safe

## Air-gap mode

Enable via any of:

```bash
# SKIP-SAMPLE — env-var setup
export PERFXPERT_AIRGAP=1
```

```python
# SKIP-SAMPLE — per-session override (verified elsewhere under the harness)
from perfxpert.agents.runtime import build_session
session = build_session(airgap=True)
```

Behavior:

- No outbound network calls from the agent layer. Provider resolution
  is skipped entirely.
- `recommendations[].type` is **not populated** in the `RootOutput` —
  deterministic rules surface a list, but the rich typed metadata
  (technique id, estimated speedup, code-edit hints) is only
  generated when an LLM is in the loop.
- `primary_bottleneck` is still set (classifier runs deterministically
  against the knowledge YAMLs).
- `narrative` is either empty or a terse deterministic summary.
- Gate cascade behavior is **unchanged** — it's middleware; no LLM
  involvement in gates.

Use air-gap mode for:

- Secure / regulated environments where no LLM provider is allowed
- CI pipelines where reproducibility is critical
- Ad-hoc triage when you just want the deterministic hot-kernel +
  bottleneck classification

## LLM-enabled mode

Pick a provider explicitly:

```python
# SKIP-SAMPLE — requires a configured provider; shown as the canonical call
from perfxpert.agents.runtime import build_session
from perfxpert.agents.schemas import RootInput

session = build_session(provider="anthropic")  # or openai / ollama / private / opencode
output = session.run_root(RootInput(
    user_query="Propose the first optimization for my hot kernel.",
    database_path="/tmp/trace.db",
))

for rec in output.recommendations:
    print(rec.type, rec.title)   # type is populated in LLM mode
```

When `airgap=False` (the default), `build_session`:

1. Resolves the provider from the `provider` arg, falling back to
   `DEFAULT_PROVIDER` (currently `"anthropic"`).
2. Validates it against `PROVIDER_REGISTRY`; raises `ValueError` on
   unknown providers.
3. Calls `runtime.recursion_guard.ensure_not_recursive(prov)` — this
   blocks `provider="opencode"` from being chosen from inside an
   already-running opencode session (would recurse forever).
4. Returns an `AnalysisSession` bound to that provider.

Behavior differences vs air-gap:

- `recommendations[].type` is populated with typed metadata (e.g.
  `"async_stream_overlap"`, `"vgpr_reduction"`).
- `narrative` is a full natural-language summary.
- Each agent's fence slice shapes the LLM's voice + constraints —
  never concatenated with the others.
- Gate cascade still runs deterministically on every edit the LLM
  proposes; a rewarded-hack still gets rejected at gate 2 (SOL).

## Provider ladder

`PROVIDER_REGISTRY` (from `perfxpert/providers/__init__.py`) lists the
supported providers in preference order. Current entries:

| Provider | Source | Typical use |
|----------|--------|-------------|
| `anthropic` | Claude API | Production default; requires `ANTHROPIC_API_KEY` |
| `openai` | OpenAI API | Alternative hosted; requires `OPENAI_API_KEY` |
| `ollama` | Local Ollama | Fully local; requires a running `ollama serve` |
| `private` | Any OpenAI-compatible endpoint | Internal deployments; requires `PERFXPERT_LLM_PRIVATE_URL` + `PERFXPERT_LLM_PRIVATE_MODEL` |
| `opencode` | Bundled opencode CLI | Used by `perfxpert-code`; not callable from inside opencode itself (recursion-guarded) |

`perfxpert doctor` reports which providers are reachable from the
current shell. See [contributing/providers.md](../contributing/providers.md)
for how to register a new one.

## CLI entry points

Two CLI surfaces drive the same agent runtime:

- `perfxpert analyze ...` — batch / one-shot. Reads a .db file or a
  source directory, emits a single report (JSON / markdown / webview).
  Use `--llm <provider>` to pick the provider; omit `--llm` to run
  air-gap.
- `perfxpert-code ...` — TUI (bundled AMD-themed opencode). Same
  runtime under the hood; `perfxpert-code run -m <message>` is the
  non-interactive equivalent for scripts.

Roughly:

| Task | Command |
|------|---------|
| Analyze a trace deterministically | `perfxpert analyze -i trace.db` |
| Analyze with Claude | `perfxpert analyze -i trace.db --llm anthropic` |
| Drive an optimization session conversationally | `perfxpert-code` |
| Drive an optimization session non-interactively | `perfxpert-code run -m "optimize hot kernel"` |

Under the hood, both CLIs call `build_session()` and then one of the
tier-0/tier-1 `run_*` methods. The **only** difference is whether the
output is rendered as a report (batch mode) or streamed into a TUI
(opencode mode).

## When to use which

- **CI / regression gates** → `perfxpert analyze` in air-gap mode.
  Reproducible, no network, deterministic output.
- **Interactive optimization** → `perfxpert-code`. Natural-language
  back-and-forth, session persistence, gate cascade active on every
  edit.
- **External MCP clients** (Claude Desktop, Cursor) → always air-gap
  safe because the MCP server only exposes `READ_ONLY` tools; see
  [mcp-server.md](../integration/mcp-server.md).
- **Scripted LLM runs** → `perfxpert-code run -m "..."` or
  `build_session(provider="anthropic")` in Python.
