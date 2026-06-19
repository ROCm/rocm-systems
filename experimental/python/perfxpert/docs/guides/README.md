# PerfXpert user guides

End-user-facing how-tos. If you're contributing code instead of running
PerfXpert, jump to [../contributing/README.md](../contributing/README.md).

## Index

| You want to… | Read |
|--------------|------|
| Install + run your first analysis | [getting-started.md](getting-started.md) |
| Run the first-time wizard (`perfxpert init`) | [getting-started.md §3.3](getting-started.md#33-first-run--perfxpert-init) |
| Understand how the agent brain works at runtime | [agentic-mode.md](agentic-mode.md) |
| Drive PerfXpert from a coding backend (opencode / claude / codex / gemini) | [backends.md](backends.md) |
| Embed the agent hierarchy in your own Python code | [python-api.md](python-api.md) |

## Visual demo (15 short GIFs)

Embedded throughout [getting-started.md](getting-started.md). The
rendered GIFs are checked in under [assets/gifs/](assets/gifs/).

![help overview](assets/gifs/02-help.gif)

| # | scenario | gif |
|---|----------|-----|
| 01 | Install (editable + bundled opencode build) | [01-install.gif](assets/gifs/01-install.gif) |
| 02 | `perfxpert --help` + `analyze --help` | [02-help.gif](assets/gifs/02-help.gif) |
| 03 | `perfxpert doctor` | [03-doctor.gif](assets/gifs/03-doctor.gif) |
| 04 | `--format text` | [04-analyze-text.gif](assets/gifs/04-analyze-text.gif) |
| 05 | `--format json` | [05-analyze-json.gif](assets/gifs/05-analyze-json.gif) |
| 06 | `--format markdown` | [06-analyze-markdown.gif](assets/gifs/06-analyze-markdown.gif) |
| 07 | `--format webview` | [07-analyze-webview.gif](assets/gifs/07-analyze-webview.gif) |
| 08 | Tier-0 source-only | [08-tier0-source-only.gif](assets/gifs/08-tier0-source-only.gif) |
| 09 | Tier-0 combined (DB + source) | [09-tier0-combined.gif](assets/gifs/09-tier0-combined.gif) |
| 10 | Live progress spinner | [10-progress-spinner.gif](assets/gifs/10-progress-spinner.gif) |
| 11 | Pre-flight auth error (clean rc=2) | [11-pre-flight-auth-error.gif](assets/gifs/11-pre-flight-auth-error.gif) |
| 12 | Python API (`perfxpert.api`) | [12-python-api.gif](assets/gifs/12-python-api.gif) |
| 13 | MCP server (56 tools) | [13-mcp-server.gif](assets/gifs/13-mcp-server.gif) |
| 14 | `perfxpert-code` interactive | [14-perfxpert-code.gif](assets/gifs/14-perfxpert-code.gif) |
| 15 | All five `--llm` providers | [15-all-providers.gif](assets/gifs/15-all-providers.gif) |

## See also

- [../README.md](../README.md) — top-level docs index
- [../integration/mcp-server.md](../integration/mcp-server.md) — MCP wire surface
- [../architecture/agent-hierarchy.md](../architecture/agent-hierarchy.md) — what each agent does
