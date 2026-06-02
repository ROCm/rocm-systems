# perfxpert-mcp — stdio MCP server

Exposes the read-only subset of perfxpert tools via Model Context Protocol.
Launched by opencode (`perfxpert-code`), Claude Desktop, Cursor, or any
MCP-compatible client.

## Start

```bash
# SKIP-SAMPLE — server is intended to be spawned by an MCP client
perfxpert-mcp
```

stdio transport only. Intended to be spawned by an MCP client, not run by
humans directly.

## Example client config (opencode)

See `perfxpert/_bundled/opencode_config/mcp.json`.

## What's exposed

Every function in `perfxpert.tools.*` decorated with
`@tool_class(ToolClass.READ_ONLY)` is auto-registered. Execution tools
(`patch.apply`, `compile.build`, `profile.run`, `anchors.check`) are
**never** exposed — this is enforced at startup and by CI
(`tests/test_integration/test_mcp_exposure.py`).

## Security posture

See spec §5.8. Short version: the server is read-only by design; any PR
that moves an execution tool into the registered set fails CI and requires
architectural RFC approval.
