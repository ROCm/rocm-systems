# Skill: Analyze / roofline

## Goal

Change analysis, roofline, or related utilities.

## Steps

1. Touch only `src/rocprof_compute_analyze/`, `src/utils/roofline_calc.py`, or closely related helpers.
2. Preserve deterministic outputs; document any unit or rounding change.
3. Extend `tests/` for CLI paths, parsers, or math edge cases.
4. Run ruff and relevant pytest modules.

## Constraints

- No silent changes to reported metrics or CSV columns without tests.
- Call out performance impact if processing large traces.

## Output

- Code change, tests for output or edge cases, and brief note if metrics/units changed.

**Output format (strict):** [`.ai/harness/skill_output_contract.md`](../harness/skill_output_contract.md).
