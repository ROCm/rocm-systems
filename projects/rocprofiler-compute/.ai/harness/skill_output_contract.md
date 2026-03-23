# Skill output contract (strict)

Every skill under [`.ai/skills/`](../skills/) ends with `## Output`. That section must be filled so reviewers can verify work **without re-running the agent**.

## Required (all skills)

1. **Summary** — 1–3 sentences: what changed and where (paths / modules).
2. **Patch scope** — Unified diff or list of edited files; note if behavior is intentionally unchanged.
3. **Validation run** — Exact commands (`ruff`, `pytest`, `cmake …`) and **pass/fail** (or N/A with reason).
4. **Risks / follow-ups** — User-visible impact, or “none”.

## Required when relevant

| Situation | Extra |
|-----------|--------|
| Performance / optimization | **# PERF** block: before/after intuition, time complexity, memory, how measured (or N/A). |
| Trace / CSV / schema | Columns, ordering, or units changed? “Yes + tests” or “No”. |
| Build / CMake | Failing target name, fix summary, **rebuild command** that succeeded. |
| Security-sensitive input | Note validation applied ([`.ai/rules/security.md`](../rules/security.md)). |

## Forbidden

- Vague output (“optimized”, “improved”) without commands or scope.
- Claiming tests passed without naming the command or scope.

Agents: keep `## Output` bullets **concrete**; this contract is the bar for “done.”
