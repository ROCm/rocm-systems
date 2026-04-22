# perfxpert getting-started demo — scenario 06-analyze-markdown
# Generated via: vhs 06-analyze-markdown.tape
# See docs/guides/getting-started.md

Output ../gifs/06-analyze-markdown.gif

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

Type "# Markdown output — drop into PR descriptions / wikis"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --format markdown 2>/dev/null | head -40"
Enter
Sleep 6s
