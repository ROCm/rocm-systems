# Skill: Optimize performance

## Goal

Improve time or memory for an **existing** code path without changing **observable** behavior, unless the issue explicitly allows output deltas.

## Steps

1. Identify the hot path (profiler, `pytest` timing, or user-provided trace). Confirm baseline.
2. Prefer algorithmic wins (complexity, fewer passes) over micro-optimizations.
3. In Python: avoid accidental copies over large traces/CSVs; prefer generators or views where patterns already exist in the file.
4. In C++ (`src/lib/`): avoid extra allocations on hot paths; use references; align with existing style.
5. Add or extend **deterministic** tests so behavior stays locked; document any intentional metric formatting change.
6. Run `ruff` on touched Python; build native targets if C++ changed.

## Constraints

- No new dependencies without approval.
- See [`.ai/rules/profiling_infra.md`](../rules/profiling_infra.md) for determinism and test data.
- If behavior or reported numbers can change, get explicit sign-off and update docs/CHANGELOG as needed.

## Output

- Patch, before/after note (expected gain or complexity class), tests run, and any **# PERF** summary: time complexity, memory impact, risk of numerical drift.

**Output format (strict):** [`.ai/standards/agent_output.md`](../standards/agent_output.md).
