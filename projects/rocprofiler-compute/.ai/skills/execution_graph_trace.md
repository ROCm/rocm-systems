# Skill: Execution graph & trace semantics

## Goal

Change or extend how **execution-related** data is interpreted, joined, or presented: kernel/marker **traces**, CSV columns, dispatch ordering, or analysis that depends on **graph-like** structure (kernels ↔ markers ↔ dispatches).

## Steps

1. Locate parsers, loaders, or CLI paths that consume **trace or counter CSVs** (e.g. under `tests/workloads/`, analyze modules, `utils` parsers). Do not invent column names—read fixtures and existing code.
2. Preserve **deterministic** ordering for golden tests; document any new sort keys or stable tie-breaks.
3. If touching **SoC/counter** definitions, coordinate with `update_soc_or_counters` patterns and YAML in `rocprof_compute_soc`.
4. Add or update tests that **pin** parsing output or CLI snippets for a small fixture.
5. Run targeted `pytest` and `ruff` on edited Python files.

## Constraints

- Treat trace/CSV content as **untrusted input** when it could come from external issues ([`.ai/rules/security.md`](../rules/security.md)).
- No silent changes to column names, units, or user-visible tables without tests and release notes when required.
- Large new binaries or huge CSVs: avoid unless maintainers agree ([`.ai/rules/profiling_infra.md`](../rules/profiling_infra.md)).

## Output

- Code + tests; explicit note of **schema/column** or **ordering** changes; commands run (`pytest`, `ruff`).

**Output format (strict):** [`.ai/standards/agent_output.md`](../standards/agent_output.md).
