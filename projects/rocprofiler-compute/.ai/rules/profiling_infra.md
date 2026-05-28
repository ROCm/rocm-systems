# Profiling / trace / infra rules

1. **Determinism** — Tests and examples should use fixed seeds, stable sorting,
   and explicit ordering where output is compared.
2. **Fixtures** — Prefer small, focused tests; large trace/CSV data lives under
   `tests/workloads/`. Extend carefully; do not commit huge binaries without
   need.
3. **Formats** — Changes affecting counter names, YAML schemas, CSV columns, or
   parsed trace shapes require tests that lock behavior and note user-visible
   impact.
4. **Performance** — For hot paths, state time/memory impact briefly; avoid
   extra full-buffer copies and accidental O(n²) patterns in Python over large
   traces.
5. **Experimental features** — Follow CONTRIBUTING: `--experimental` gating,
   `ExperimentalAction`, help text updates.

