# Skill: Write test

## Goal

Increase coverage or lock behavior with deterministic tests.

## Steps

1. Place tests under `tests/`; reuse fixtures under `tests/workloads/` when appropriate.
2. Use stable ordering and explicit data; avoid wall-clock assertions.
3. Respect `pytest` markers and `pythonpath` in `pyproject.toml`.
4. Run: `pytest path/to/test_file.py` (or narrower `-k`).

## Constraints

- No production behavior change unless the issue asks for it.
- Keep fixtures small; do not duplicate huge trace files.

## Output

- New or updated tests, fixture notes if any, and `pytest` invocation used.

**Output format (strict):** [`.ai/standards/agent_output.md`](../standards/agent_output.md).
