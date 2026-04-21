# perfxpert getting-started demo — scenario 07-analyze-webview
# Generated via: vhs 07-analyze-webview.tape
# See docs/guides/getting-started.md

Output ../gifs/07-analyze-webview.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 650
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "cd /tmp/perfxpert-vhs-demo"
Enter
Sleep 200ms
Type "rm -f analysis.html"
Enter
Sleep 100ms
Type "clear"
Enter
Sleep 200ms
Show

Type "# Webview: self-contained HTML — email-ready"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --format webview -d . -o analysis"
Enter
Sleep 3s
Type "ls -lh analysis.html"
Enter
Sleep 1500ms
Type "bash _section_count.sh"
Enter
Sleep 1500ms
Type "echo '# 7 sections: Overview, Summary, Execution, Hotspots, Counters, Recs, Tier-0'"
Enter
Sleep 2s
