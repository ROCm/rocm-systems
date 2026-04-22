# perfxpert getting-started demo — scenario 05-analyze-json
# Generated via: vhs 05-analyze-json.tape
# See docs/guides/getting-started.md

Output ../gifs/05-analyze-json.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 700
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
Type "clear"
Enter
Sleep 200ms
Show

Type "# JSON output — ready for jq / scripts / dashboards"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --format json 2>/dev/null | jq 'keys'"
Enter
Sleep 4s
Type "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --format json 2>/dev/null | jq '{bottleneck: .primary_bottleneck, hotspot_count: (.hotspots | length)}'"
Enter
Sleep 4s
