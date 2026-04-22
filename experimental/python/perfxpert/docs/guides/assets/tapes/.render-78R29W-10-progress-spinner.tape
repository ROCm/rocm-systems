# perfxpert getting-started demo — scenario 10-progress-spinner
# Generated via: vhs 10-progress-spinner.tape
# See docs/guides/getting-started.md
#
# NOTE: this tape runs under PERFXPERT_AIRGAP=1 for determinism.
# Normally `--llm anthropic` talks to Anthropic; the progress
# narrative format is identical, only the phase timings differ.

Output ../gifs/10-progress-spinner.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 600
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

Type "# Progress spinner (LLM mode) — airgap shown for determinism"
Enter
Sleep 500ms
Type "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db --llm anthropic --format text 2>&1 | grep -E '(perfxpert|Primary|Database)' | head -20"
Enter
Sleep 8s
