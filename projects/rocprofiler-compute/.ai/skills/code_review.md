# Skill: Code review

## Goal

Produce a structured, project-aware review of a diff or PR without duplicating tooling docs.

## Steps

1. Read context: [`.ai/rules/core.md`](../rules/core.md), [`.ai/rules/security.md`](../rules/security.md), and [`.ai/review/checklist.md`](../review/checklist.md).
2. Apply **priority order** below; cite files/lines when possible.
3. Classify each point as **Issue** (must fix), **Suggestion** (should consider), **Question** (clarify), or **Nit** (style/preference).
4. Close with a short **Summary** and whether tests/docs/CHANGELOG look sufficient for the change type.

## Constraints

- **Do not** restate full Ruff rule lists — enforce via `pyproject.toml` and [`.ai/standards/python.md`](../standards/python.md).
- **Do not** duplicate experimental CLI steps — use [CONTRIBUTING.md](../../CONTRIBUTING.md) and [`.ai/rules/profiling_infra.md`](../rules/profiling_infra.md).
- PR process and template: [`.github/pull_request_template.md`](../../.github/pull_request_template.md).

## Output

- **Summary** → **Issues** → **Suggestions** → **Questions** → **Nits** (omit empty sections).
- Reference validation the author should run: `ruff`, `pytest`, CMake as applicable per [CONTRIBUTING.md](../../CONTRIBUTING.md).

**Output format (strict):** [`.ai/standards/agent_output.md`](../standards/agent_output.md).

---

## Review priority order

1. **Correctness** — Logic, edge cases, CLI/parser behavior, counter or metric formulas, CSV/YAML/trace shape consistency.
2. **Security** — [`.ai/rules/security.md`](../rules/security.md): shell/MCP trust, untrusted input, secrets, scope.
3. **Performance** — Hot paths (especially profile/capture): avoid unnecessary imports, copies, or O(n²) over large traces.
4. **Style** — Match `src/`; Ruff + format per `pyproject.toml` (see [`.ai/standards/python.md`](../standards/python.md)).
5. **Architecture / layering** — Respect package boundaries under `src/`; do not add heavy analysis/UI dependencies into low-level profiler paths without lazy loading and maintainer alignment (see [`.ai/rules/core.md`](../rules/core.md)).
6. **Tests & docs** — Mirror `src/` layout under `tests/`; deterministic tests; user-visible changes documented; CHANGELOG when required for release-facing work.

## Domain checks (when relevant)

| Area | Look for |
|------|-----------|
| **Profiler / SoC** | Counter definitions, YAML under `src/rocprof_compute_soc/`, arch-specific configs ([`tools/config_management/README.md`](../../tools/config_management/README.md)). |
| **Analysis** | Metric math, pandas usage, mode-specific paths (CLI / WebUI / DB / TUI). |
| **GPU / arch** | Compatibility across supported GFX targets; config deltas vs template. |
| **Native** | CMake, `src/lib/`, out-of-source build ([`.ai/standards/cmake.md`](../standards/cmake.md)). |
