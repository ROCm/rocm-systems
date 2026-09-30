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

The same agent/session machinery powers `perfxpert analyze`,
`perfxpert.api.*`, the 8 agent MCP tools exposed by `perfxpert-mcp`,
and every `perfxpert-code` backend. The entry surface changes the
shell/UI and config plumbing, not the analysis brain.

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
- `RootOutput.recommendations` is currently empty in the root air-gap
  path. Deterministic mode still classifies the bottleneck and returns
  the terse template narrative, but it does not yet populate structured
  recommendation dicts until an LLM-backed path is used.
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

![progress spinner](assets/gifs/10-progress-spinner.gif)

*Live agent-phase progress UI captured under `PERFXPERT_AIRGAP=1` for
deterministic rendering of the analysis flow.*

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

# recommendations is a List[Dict[str, Any]] — use dict access, not attribute
for rec in output.recommendations:
    print(rec["name"], rec["title"])
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

- `recommendations[]["name"]` carries the technique identifier (e.g.
  `"async_stream_overlap"`, `"vgpr_reduction"`) — same key as
  air-gap mode, but `title` / `description` / `rationale` are
  LLM-rewritten for the specific kernel instead of being templated
  verbatim from the knowledge YAMLs.
- `narrative` is a full natural-language summary.
- Each agent's fence slice shapes the LLM's voice + constraints —
  never concatenated with the others.
- Gate cascade still runs deterministically on every edit the LLM
  proposes; a rewarded-hack still gets rejected at gate 2 (SOL).

## Provider ladder

![all providers](assets/gifs/15-all-providers.gif)

*The five primary providers come directly from the CLI's `--llm`
registry: `anthropic`, `openai`, `ollama`, `private`, `opencode`.
The CLI also accepts `claude-code` as a compatibility alias for the
patched opencode backend used by `perfxpert-code`.*

`PROVIDER_REGISTRY` is defined in `perfxpert/agents/runtime.py` (with a
hard-coded fallback there; `build_session` imports the richer copy
from `perfxpert.providers` when the optional provider package is
available). It lists the supported providers in preference order.
Current entries:

| Provider | Source | Typical use |
|----------|--------|-------------|
| `anthropic` | Claude API | Production default; requires `ANTHROPIC_API_KEY` |
| `openai` | OpenAI API | Alternative hosted; requires `OPENAI_API_KEY` |
| `ollama` | Local Ollama | Fully local; requires a running `ollama serve` |
| `private` | Any OpenAI-compatible endpoint | Internal deployments; requires `PERFXPERT_LLM_PRIVATE_URL` + `PERFXPERT_LLM_PRIVATE_MODEL`; CLI preflight also needs `PERFXPERT_LLM_PRIVATE_API_KEY` or `--llm-api-key` |
| `opencode` | Bundled opencode CLI | Used by `perfxpert-code`; not callable from inside opencode itself (recursion-guarded) |

`perfxpert doctor` reports which providers are reachable from the
current shell. See [contributing/providers.md](../contributing/providers.md)
for how to register a new one.

Private OpenAI-compatible endpoints use JSON for extra headers:

```bash
# SKIP-SAMPLE — requires a reachable private endpoint
export PERFXPERT_LLM_PRIVATE_URL="https://llm-api.iexample.com/OpenAI"
export PERFXPERT_LLM_PRIVATE_MODEL="gpt-5.3-codex"
export PERFXPERT_LLM_PRIVATE_API_KEY="..."
export PERFXPERT_LLM_PRIVATE_HEADERS='{
  "Ocp-Apim-Subscription-Key": ".......",
  "user": ".....",
  "api-version": "preview"
}'
perfxpert analyze -i trace.db --llm private
```

## Fallback chain (`PERFXPERT_LLM_FALLBACK_CHAIN`)

The multi-provider fallback chain lets a rate-limited or unavailable
primary provider hand off to another, so users don't have to rerun.
Set the env var to a comma-separated list of provider names in
preference order:

```bash
# SKIP-SAMPLE — requires a real trace.db and live LLM credentials
# Try Anthropic first; fall back to OpenAI, then a private endpoint.
export PERFXPERT_LLM_FALLBACK_CHAIN="anthropic,openai,private"
perfxpert analyze -i trace.db --llm anthropic
```

When set, `build_session(provider="anthropic", ...)` keeps the
configured provider selection but still surfaces typed provider errors
from `perfxpert/providers/_exceptions.py`. The concrete exception
surface in this tree is:

| Exception | Meaning |
|-----------|---------|
| `RateLimitError` | HTTP 429 / short-term throttle; eligible for fallback cascade |
| `QuotaExceededError` | Hard quota / billing exhaustion; surfaces immediately |
| `TransientError` | Retryable provider/network/API blip; eligible for fallback cascade |
| `FatalError` | Non-retryable provider failure or malformed provider response |
| `TimeoutError` | Per-call timeout budget exceeded |
| `AuthError` | Bad API key / missing credential |
| `ProviderChainExhausted` | Every provider in the configured fallback chain failed |
| `ProviderError` | Generic provider failure when no narrower typed subclass applies |

The `recursion_guard` still applies to each candidate — `opencode`
cannot be chosen from inside an already-running opencode session.

Unset or empty: no fallback. A typed error surfaces to the caller
unchanged.

Related knob:

- `PERFXPERT_DISABLE_RATE_LIMIT_RETRY=1` disables client-side retry
  before falling through; useful when you want the fallback chain to
  take over immediately on 429.

## CLI entry points

Two CLI surfaces drive the same agent runtime:

- `perfxpert analyze ...` — batch / one-shot. Reads a .db file or a
  source directory, emits a single report (JSON / markdown / webview).
  Use `--llm <provider>` to pick the provider; omit `--llm` to run
  air-gap.
- `perfxpert-code ...` — interactive launcher. Plain `perfxpert-code`
  uses the patched opencode path; `perfxpert-code claude|gemini|codex`
  stages the same prompt/MCP surface into the user's native backend.
  `perfxpert-code run -m <message>` is the non-interactive equivalent
  when the patched opencode path is in use.

Roughly:

| Task | Command |
|------|---------|
| Analyze a trace deterministically | `perfxpert analyze -i trace.db` |
| Analyze with Claude | `perfxpert analyze -i trace.db --llm anthropic` |
| Drive an optimization session conversationally | `perfxpert-code` |
| Drive an optimization session non-interactively | `perfxpert-code run -m "optimize hot kernel"` |

Under the hood, the batch CLI, Python API, MCP wrappers, and
`perfxpert-code` backends all funnel into `build_session()` and the
same `run_*` methods. The differences are output rendering and whether
an external TUI talks to those runners through MCP.

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
