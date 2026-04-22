# perfxpert getting-started demo — scenario 07-analyze-webview
# Generated via: vhs 07-analyze-webview.tape
# See docs/guides/getting-started.md
#
# Generates analysis.html and prints the count of `<h2>` section
# headings it contains (Overview, Execution Breakdown, Top Kernel
# Hotspots, Hardware Counters, Optimization Recommendations,
# Summary = 6).

Output ../gifs/07-analyze-webview.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 650
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/.worktrees/perfxpert-phase10-guide-gif-refresh/experimental/python/perfxpert/docs/guides/assets/scripts/start-demo-container.sh"
Enter
Sleep 200ms
Type "docker exec -it perfxpert-guide-vhs bash --noprofile --norc"
Enter
Sleep 500ms
Type "cd /tmp/perfxpert-vhs-demo"
Enter
Sleep 200ms
Type "rm -f analysis.html"
Enter
Sleep 100ms
Type `cat > _section_count.sh <<'EOF'`
Enter
Type "#!/usr/bin/env bash"
Enter
Type `printf 'sections: '; grep -c '<h2>' analysis.html`
Enter
Type `printf 'headings:\n'; grep -oE '<h2>[^<]+</h2>' analysis.html | sed 's|<[^>]*>||g' | sed 's/^/  • /'`
Enter
Type "EOF"
Enter
Type "chmod +x _section_count.sh"
Enter
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
Sleep 4s
