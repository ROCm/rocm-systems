# Skill: Update SoC / counters / YAML

## Goal

Change GPU SoC logic or counter definitions safely.

## Steps

1. Inspect `src/rocprof_compute_soc/` (`soc_*.py`, `profile_configs/counter_defs.yaml`).
2. Keep naming consistent across Python and YAML where applicable.
3. Add or adjust tests that parse or select counters for affected SoCs.
4. Run ruff and targeted pytest.

## Constraints

- Do not rename public counter strings without compatibility notes and tests.
- See `rules/profiling_infra.md` for format and determinism.

## Output

- YAML/Python edits, tests covering selection or parsing, and validation commands run.

**Output format (strict):** [`.ai/harness/skill_output_contract.md`](../harness/skill_output_contract.md).
