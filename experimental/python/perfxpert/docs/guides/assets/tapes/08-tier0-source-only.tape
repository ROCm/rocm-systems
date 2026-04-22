# perfxpert getting-started demo — scenario 08-tier0-source-only
# Generated via: vhs 08-tier0-source-only.tape
# See docs/guides/getting-started.md

Output ../gifs/08-tier0-source-only.gif

Set Shell "./vhs-demo-shell.sh"
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
Type "clear"
Enter
Sleep 200ms
Show

Type "# Tier-0: pure source scan, no .db required, no GPU required"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze --source-dir /tmp/mock_src --format text 2>/dev/null | head -40"
Enter
Sleep 6s
