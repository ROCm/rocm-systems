# Agent output contract (strict)

Every skill under [`.ai/skills/`](../skills/) ends with `## Output`. Fill it so reviewers can verify work **without re-running the agent**.

## Required (all skills)

1. **Summary** — 1–3 sentences: what changed and where.
2. **Patch scope** — Diff or file list; note if behavior intentionally unchanged.
3. **Validation run** — Exact commands (`ruff`, `pytest`, `cmake …`) and **pass/fail** (or N/A).
4. **Risks / follow-ups** — User-visible impact, or “none”.

## When relevant

| Situation | Extra |
|-----------|--------|
| Performance | **# PERF:** before/after, complexity, memory, how measured. |
| Trace / CSV / schema | Column/order/unit changes? “Yes + tests” or “No”. |
| Build / CMake | Failing target, fix summary, **rebuild command** that passed. |
| Untrusted input | Note validation ([`../rules/security.md`](../rules/security.md)). |

## Forbidden

- Vague claims without commands or scope.
- “Tests passed” without naming command or scope.
