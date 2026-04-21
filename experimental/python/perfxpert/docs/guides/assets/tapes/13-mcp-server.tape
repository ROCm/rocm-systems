# perfxpert getting-started demo — scenario 13-mcp-server
# Generated via: vhs 13-mcp-server.tape
# See docs/guides/getting-started.md

Output ../gifs/13-mcp-server.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 500
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "cd /tmp/perfxpert-vhs-demo"
Enter
Sleep 200ms
Type "clear"
Enter
Sleep 200ms
Show

Type "# perfxpert-mcp — stdio JSON-RPC bridge for MCP clients"
Enter
Sleep 500ms
Type "(perfxpert-mcp & MCP_PID=$!; sleep 3; kill $MCP_PID 2>/dev/null; wait 2>/dev/null) 2>&1 | head -5"
Enter
Sleep 5s
Type "echo '# 58 tools = 8 agent-hierarchy + 49 classifier/knowledge + 1 trace_diff.diff_runs'"
Enter
Sleep 2s
