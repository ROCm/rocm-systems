# Skill: Add feature

## Goal

Add behavior with minimal surface area.

## Steps

1. Find the existing module (e.g. `src/rocprof_compute_*`, `src/utils/`) that owns the concern.
2. Extend parsers/CLI only via existing patterns (`src/argparser.py` if new flags).
3. Implement the smallest change that satisfies the request.
4. Add or update `tests/`; run `pytest` on relevant files.
5. Run `ruff check` and `ruff format` on edited Python files.

## Constraints

- No new top-level packages or dependencies without approval.
- See `rules/profiling_infra.md` if output or trace semantics change.

## Output

- Patch-sized code change, tests or justification if not applicable, and how you ran ruff/pytest (or CMake for native).

**Output format (strict):** [`.ai/harness/skill_output_contract.md`](../harness/skill_output_contract.md).
