# Skill: Add experimental CLI

## Goal

Add a gated CLI flag per project CONTRIBUTING.

## Steps

1. Read **Adding Experimental Features** in `CONTRIBUTING.md`.
2. Update `--experimental` help text in `add_general_group()` in `src/argparser.py`.
3. Register the argument with `ExperimentalAction`, correct `base_action`, and `feature_label`.
4. Implement behavior behind the flag; error clearly if used without `--experimental`.
5. Add a test for help visibility and basic flag behavior.

## Constraints

- Match existing `ExperimentalAction` patterns in the same file.
- Do not expose experimental options in default `--help`.

## Output

- Parser changes, implementation behind the flag, tests for help/behavior, and ruff/pytest commands run.
