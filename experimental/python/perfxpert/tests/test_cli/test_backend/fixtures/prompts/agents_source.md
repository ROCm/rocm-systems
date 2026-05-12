# PerfXpert agent instructions (source)

Always begin a session by calling `perfxpert_intent_classify` so the
tool-priority gate lifts and you can use any tool afterwards.

After classification, you may call `perfxpert_next_step` and
`perfxpert_report` as the workflow requires.

<!--backend:claude-->
Claude-specific note: surface the MCP server as `perfxpert` (renders to
`mcp__perfxpert__<tool>`).
<!--/backend:claude-->

<!--backend:gemini-->
Gemini-specific note: rendered tool names use the `mcp_perfxpert_<tool>` form.
<!--/backend:gemini-->

<!--backend:codex-->
Codex-specific note: tool-name form probed at install time.
<!--/backend:codex-->

End of source file.
