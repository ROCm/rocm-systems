# GitHub Copilot Instructions — rocprofiler-compute

**Primary reference:** [`AGENTS.md`](../AGENTS.md)

---

## Python Code Style

Follow [`.ai/standards/python-style.md`](../.ai/standards/python-style.md). Key rules:
- Every function does exactly **one** thing — split if "and" appears in its purpose
- Max **2 levels** of nesting preferred; **3 levels** is the hard limit; use guard clauses
- Descriptive names — no abbreviations, no vague names (`data`, `result`, `tmp`, `val`)
- Module order: docstring → imports → constants → public functions → private helpers → classes
- Never mix I/O with computation in the same function

## Linting & Formatting

Config source of truth: [`pyproject.toml`](../pyproject.toml) — do not duplicate here.

For `src/**` code:
- **Type annotations** required on all arguments and return types (ANN rules)
- **F-strings only** — no `.format()` or `%` formatting (UP rules)
- **`pathlib.Path`** for file system operations (PTH rules)
- **88-character** line limit

## Architecture Rules

See [`.ai/standards/architecture.md`](../.ai/standards/architecture.md) for the full package map.

**Do NOT:**
- Import `rocprof_compute_analyze` from within `rocprof_compute_profile`
- Add cross-module state outside of `src/config.py`
- Hardcode GPU arch identifiers — use the arch config files in `src/rocprof_compute_soc/`

**GPU arch support:** `gfx908`, `gfx90a`, `gfx940`, `gfx941`, `gfx942`, `gfx950`
Panel YAML changes must cover all supported archs and pass the pre-commit `hash-check` hook.

## Contributing

See [`.ai/rules/contributing.md`](../.ai/rules/contributing.md) for the full workflow.

- Branch from `develop`; use `topic-<featureName>` naming
- New features: add happy-path + failure-path tests in `tests/`
- All PRs: use [`.github/pull_request_template.md`](pull_request_template.md)
- YAML config changes: run `tools/config_management/master_config_workflow_script.py`

## Code Review

When reviewing code, follow [`.ai/skills/code-review.md`](../.ai/skills/code-review.md):
1. Correctness & GPU-specific bugs first
2. Security
3. Performance (profile mode overhead)
4. Style (Ruff rules)
5. Architecture (module boundaries)
6. Tests & documentation
