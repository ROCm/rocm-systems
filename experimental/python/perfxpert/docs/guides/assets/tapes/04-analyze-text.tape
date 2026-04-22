# perfxpert getting-started demo — scenario 04-analyze-text
# Generated via: vhs 04-analyze-text.tape
# See docs/guides/getting-started.md

Output ../gifs/04-analyze-text.gif

Set Shell "./vhs-demo-shell.sh"
Set FontSize 12
Set Width 1200
Set Height 800
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

Type "# Airgap deterministic analysis — no network, no LLM, ~1 s"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --format text"
Enter
Sleep 8s
