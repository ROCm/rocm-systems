# Skill: Fix bug

## Goal

Correct behavior without scope creep.

## Steps

1. Reproduce via test or minimal script; add a failing test first when practical.
2. Patch the narrowest code path; avoid “while we’re here” edits.
3. Run targeted `pytest` and ruff on touched files.
4. Note user-visible impact (CLI output, files written, metrics).

## Constraints

- Do not change unrelated formatting or refactor neighboring code.
- If root cause is unclear, document findings instead of guessing.

## Output

- Minimal fix, test proving the bug (before/after when useful), and validation commands run.

**Output format (strict):** [`.ai/standards/agent_output.md`](../standards/agent_output.md).
