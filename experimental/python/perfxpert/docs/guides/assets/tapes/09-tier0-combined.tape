# perfxpert getting-started demo — scenario 09-tier0-combined
# Generated via: vhs 09-tier0-combined.tape
# See docs/guides/getting-started.md

Output ../gifs/09-tier0-combined.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 700
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "cd /tmp/perfxpert-vhs-demo"
Enter
Sleep 200ms
Type "rm -f combined.html"
Enter
Sleep 100ms
Type "clear"
Enter
Sleep 200ms
Show

Type "# Combined: trace.db + source scan → single webview report"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --source-dir /tmp/mock_src --format webview -d . -o combined"
Enter
Sleep 3s
Type "ls -lh combined.html"
Enter
Sleep 1500ms
Type "grep -A1 'Tier-0 Source Scan' combined.html | head -4"
Enter
Sleep 3s
