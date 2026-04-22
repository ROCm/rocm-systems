# perfxpert getting-started demo — scenario 13-mcp-server
# Generated via: vhs 13-mcp-server.tape
# See docs/guides/getting-started.md

Output ../gifs/13-mcp-server.gif

Set Shell "./vhs-demo-shell.sh"
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
Type "timeout 3 perfxpert-mcp 2>&1 | grep -m1 'perfxpert-mcp:'"
Enter
Sleep 5s
