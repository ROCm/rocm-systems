# Claude Code — project rules (shared guide)

You are working in **rocprofiler-compute** (ROCm / rocm-systems). Follow the **four layers** (same model as all agents — see [`.ai/guide/workflow.md`](../../.ai/guide/workflow.md)):

1. **Context** — Root [`CLAUDE.md`](../../CLAUDE.md): invariants and pointers.
2. **Skills** — Pick **one primary** playbook from [`.ai/skills/`](../../.ai/skills/) using the tree in [`.ai/guide/taxonomy.md`](../../.ai/guide/taxonomy.md). Chain follow-ups only as a small DAG (e.g. fix → test).
3. **Tools** — Respect [`.ai/rules/tools_policy.md`](../../.ai/rules/tools_policy.md) for bash/git/build.
4. **Hooks** — Project [`PreToolUse`](https://code.claude.com/docs/en/hooks) guards may block destructive shell; do not bypass with obfuscated commands. **Also** respect repo pre-commit / `ai_dev_guide.py` expectations.

Hard constraints: read [`.ai/rules/core.md`](../../.ai/rules/core.md), [`.ai/rules/security.md`](../../.ai/rules/security.md), [`.ai/rules/anti_patterns.md`](../../.ai/rules/anti_patterns.md), and [`.ai/rules/profiling_infra.md`](../../.ai/rules/profiling_infra.md) when touching traces, counters, or output.

Standards: [`.ai/standards/python.md`](../../.ai/standards/python.md), [`cpp.md`](../../.ai/standards/cpp.md), [`cmake.md`](../../.ai/standards/cmake.md) as applicable.

Before merge-oriented work, skim [`.ai/review/checklist.md`](../../.ai/review/checklist.md).
