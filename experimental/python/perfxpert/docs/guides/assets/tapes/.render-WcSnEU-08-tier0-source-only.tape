# perfxpert getting-started demo — scenario 08-tier0-source-only
# Generated via: vhs 08-tier0-source-only.tape
# See docs/guides/getting-started.md

Output ../gifs/08-tier0-source-only.gif

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

Type "# Tier-0: pure source scan, no .db required, no GPU required"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze --source-dir /tmp/mock_src --format text 2>/dev/null | head -40"
Enter
Sleep 6s
