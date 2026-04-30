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
