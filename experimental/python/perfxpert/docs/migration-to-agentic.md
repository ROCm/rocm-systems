# Migrating from Legacy perfxpert to the Agentic Path

As of v0.2.0, perfxpert's execution path is the **agentic runtime** (OpenAI
Agents SDK + ~45 deterministic tools + split fence). The Phase-4 opt-in flag
`PERFXPERT_USE_AGENTS=1` has become a no-op (the behavior is now default).
The Phase-2-Phase-5 legacy path is gated behind `PERFXPERT_LEGACY=1` for
one minor release and will be deleted in **vX.Y+1**.

## What changed

| Concern | Legacy (pre-v0.2.0) | Agentic (v0.2.0+) |
|---------|---------------------|--------------------|
| Interactive TUI | `perfxpert analyze --interactive` | `perfxpert-code` (bundled AMD-themed opencode) |
| Session resume | `perfxpert analyze --resume-session file.json` | opencode sessions (persistent by default) |
| LLM "fence" | Monolithic `ai_analysis/share/llm-reference-guide.md` | Per-agent `agents/fence/*.md` + knowledge YAMLs |
| Recommendation engine | Rule-based in `analyze.py::generate_recommendations` | Multi-agent (Analysis → Recommendation → Specialists) |
| Correctness checks | Inline in `interactive.py::_revert_last_edit` | 5-gate cascade in `runtime/gate_cascade.py` |
| Code edits | `WorkflowSession._llm_rewrite_file()` | opencode-native edit + MCP verify |
| Library API (`analyze_database()`) | Direct LLM call via `LLMAnalyzer.analyze_with_llm` | Routes into agent runtime; same return type |

## What stays the same

- `perfxpert analyze` CLI command (minus `--interactive` / `--resume-session`)
- `AnalysisResult` dataclass and its `to_json/markdown/webview` methods
- JSON schema (`analysis-output.schema.json`)
- All `--format`, `--llm`, `--prompt`, `--source-dir`, `-d`, `-o` flags
- ROCm / rocprofv3 integration — we still read `.db` files and produce reports

## If your workflow was...

### `perfxpert analyze --interactive`

→ **Switch to `perfxpert-code`**, which is the AMD-themed bundled opencode TUI.
Calls into the same agent runtime as batch mode, just wrapped in a conversational UI.

### `perfxpert analyze --interactive --resume-session file.json`

→ **Use opencode sessions.** `perfxpert-code` persists sessions in `.perfxpert/`
by default; see `perfxpert-code --list-sessions` and `perfxpert-code --resume <id>`.

### Calling `from perfxpert.ai_analysis import LLMConversation`

→ **Use `from perfxpert.agents.runtime import create_session`.** The new API is
typed (Pydantic), streams via the SDK, and supports all 5 providers uniformly.

### Calling `from perfxpert.ai_analysis import load_reference_guide`

→ **Use `from perfxpert.agents.fence import load_fence_slice`.** Pass the agent
name (`"root"`, `"analysis"`, etc.) to get the per-agent fence slice. The
legacy monolithic guide is deleted in v0.2.0.

### Modifying `llm-reference-guide.md` to change LLM behavior

→ **Edit the relevant `agents/fence/*.md` slice.** Each agent has its own
narrow-scope fence (≤ 400 lines enforced by CI). See CONTRIBUTING.md for
which file controls which behavior.

## Emergency: I need the old behavior back

Set `PERFXPERT_LEGACY=1`. This routes `analyze_database()` through the
deprecated local-only path.

Legacy-mode caveat: LLM enhancement is unavailable there. Use the default
agentic path or `perfxpert-code` if you need LLM-backed explanations.

**This safety net will be removed in vX.Y+1.** Migrate before then.

If you run into a behavioral difference between legacy and agentic that you
believe is a regression, please file an issue with:

1. `perfxpert doctor` output from both modes
2. A reproducer DB (minimal fixture preferred)
3. Expected vs actual output

## Questions?

See CONTRIBUTING.md for extension surface docs or open an issue.
