# PerfXpert — Known Issues and Operational Limits

This file is for active limitations that affect how users or
contributors operate PerfXpert. Environment prerequisites, required
credentials, and local-only test setup bugs do not belong here unless
they affect shipped behavior.

## Codex uses the prompt-layer gate fallback

`perfxpert-code codex` stages the same MCP surface as
the other backends, and its fallback gate is expected to work through
the staged `AGENTS.override.md` prompt. The fallback includes the
Codex-specific discovery exception for deferred MCP tools: before
`mcp__perfxpert__intent_classify` is visible, the prompt allows only
the discovery metadata tools (`tool_search`, `tool_search_tool`) needed
to expose it, and still blocks shell / SSH / build / edit / profiling
fallbacks until the PerfXpert gate returns.

The limitation is narrower: Codex's native `PreToolUse` surface is not
used as the mechanical enforcement layer. As of April 2026 it intercepts
Bash, but not MCP / Write / other tool calls, so the Codex adapter
intentionally records `gate_hook_installed=False` and relies on the
prompt-layer fallback instead.

Current backend split:

- **Patched opencode path** — mechanical gate via fork patch 0020
  (`{block, retryWith}` in `tool.execute.before`)
- **Claude Code** — mechanical gate via native `PreToolUse`
- **Gemini CLI** — mechanical gate via project-local `BeforeTool` /
  `AfterTool` hooks + runtime lift
- **Codex CLI** — working prompt-layer fallback in the
  perfxpert-managed `AGENTS.override.md` compatibility override

If you need enforcement that is independent of model prompt adherence,
use the default patched opencode path, Claude Code, or Gemini CLI.

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
