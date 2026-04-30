# Migrating to the Agentic Path

> **Historical — the legacy path has been removed.** This document
> is retained for reference only. New users can disregard `PERFXPERT_LEGACY`
> entirely; the agentic runtime is the only path.

---

As of v0.2.0, perfxpert's execution path is the **agentic runtime**
(OpenAI Agents SDK + ~45 deterministic tools + split fence). This is now
the only runtime — the pre-v0.2.0 execution path and its opt-in toggle
were removed during the agentic refactor, so nothing here is optional.

## What changed

| Concern | Pre-v0.2.0 | Agentic (v0.2.0+) |
|---------|------------|--------------------|
| Interactive TUI | Inline conversational mode on `perfxpert analyze` | `perfxpert-code` (bundled AMD-themed opencode) |
| Session resume | Resume-session flag on `perfxpert analyze` | opencode sessions (persistent by default) |
| LLM "fence" | Monolithic reference guide | Per-agent `agents/fence/*.md` + knowledge YAMLs |
| Recommendation engine | Rule-based function in `analyze.py` | Multi-agent (Analysis → Recommendation → Specialists) |
| Correctness checks | Inline in the interactive-session revert helper | 5-gate cascade in `runtime/gate_cascade.py` |
| Code edits | `WorkflowSession._llm_rewrite_file()` | opencode-native edit + MCP verify |
| Library API | Single-shot LLM call | `perfxpert.agents.runtime.build_session()` + `session.run_root(...)` |

## What stays the same

- `perfxpert analyze` CLI command (the conversational flags are removed —
  see below)
- `AnalysisResult` dataclass and its `to_json/markdown/webview` methods
- JSON schema (`analysis-output.schema.json`)
- All `--format`, `--llm`, `--prompt`, `--source-dir`, `-d`, `-o` flags
- ROCm / rocprofv3 integration — we still read `.db` files and produce reports

## Closed pre-agentic PRs not ported

### rocm-systems#4979: LLM payload field-name mismatch

The pre-agentic bridge
`ai_analysis/api.py::_convert_result_to_llm_format()` emitted kernel
dictionaries with `calls` and `percent_of_total`, while
`ai_analysis/llm_analyzer.py::_sanitize_data()` expected
`dispatch_count` and `pct_total_time`. Memory directions also used verbose
labels such as `Host-to-Device` instead of compact IDs such as `h2d`.
The result was silent metric loss in the LLM payload.

[rocm-systems#4979](https://github.com/ROCm/rocm-systems/pull/4979)
patched that deleted bridge. The fix is intentionally not ported:
the agentic refactor removed the entire `perfxpert/ai_analysis/`
package, and the current agent and deterministic payload paths keep
producer and consumer field names aligned around `calls` and
`percent_of_total`.

## If your workflow was...

### Conversational mode on `perfxpert analyze`

→ **Switch to `perfxpert-code`**, which uses the default AMD-themed patched
opencode path for the conversational TUI.
Calls into the same agent runtime as batch mode, just wrapped in a conversational UI.

### Conversational mode with session resume

→ **Use opencode sessions.** `perfxpert-code` persists sessions in `.perfxpert/`
by default; see `perfxpert-code --list-sessions` and `perfxpert-code --resume <id>`.

### Calling the pre-v0.2.0 conversation class

→ **Use the current library API:**

```python
from perfxpert.agents.runtime import build_session
from perfxpert.agents.schemas import RootInput

session = build_session(airgap=True)  # or provider='anthropic'
output = session.run_root(RootInput(
    user_query="Summarize the primary bottleneck.",
    airgap=True,
))
print(output.primary_bottleneck)
```

The new API is typed (Pydantic), streams via the SDK, and supports all
5 providers uniformly.

### Loading the monolithic reference guide

→ **Use `from perfxpert.agents.fence import load_fence_slice`.** Pass the
agent name (`"root"`, `"analysis"`, etc.) to get the per-agent fence slice.
The monolithic guide was removed during the agentic refactor.

### Modifying `llm-reference-guide.md` to change LLM behavior

→ **Edit the relevant `agents/fence/*.md` slice.** Each agent has its own
narrow-scope fence (≤ 400 lines enforced by CI). See CONTRIBUTING.md for
which file controls which behavior.

## I need the old behavior back

There is no runtime toggle. `PERFXPERT_LEGACY` was removed during the
agentic refactor, along with the pre-v0.2.0 code paths it used to gate.
To run the old pipeline you must check out a release tag ≤ v0.2.x (see
`git tag`); the pre-rename reference-guide override env var is no longer
honored either.

If you hit a behavioral difference between the new runtime and what you
expected from the earlier code path, file an issue with:

1. `perfxpert doctor` output
2. A reproducer DB (minimal fixture preferred)
3. Expected vs actual output

We treat regressions versus documented v0.2.x behavior as bugs in the
agentic runtime and aim to close them quickly.

## Questions?

See CONTRIBUTING.md for extension surface docs or open an issue.
