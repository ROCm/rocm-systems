# Core rules

1. **Architecture** — Extend existing packages under `src/`. Do not add new top-level layouts, layers, or public APIs without maintainer approval.
2. **Scope** — One purpose per change; prefer small diffs (soft limit ~200 lines). No drive-by refactors or unrelated formatting sweeps.
3. **Dependencies** — Do not add new runtime or dev dependencies unless the user or issue explicitly requests it and maintainers agree.
4. **Verify** — Changes must be verifiable locally: `ruff check` / `ruff format` (see `pyproject.toml`), and `pytest` for Python; CMake build for native changes.
5. **Truth** — Do not invent APIs, file paths, or ROCm/GPU behavior. If unsure, ask or leave a TODO with what to confirm.
6. **Monorepo paths** — This project lives under `projects/rocprofiler-compute` in rocm-systems. Pre-commit hooks may assume the super-repo root; do not “fix” hook paths without understanding CONTRIBUTING setup.
7. **Security** — Follow [`.ai/rules/security.md`](security.md): no blind shell execution, validate external/untrusted input, keep file and MCP scope minimal.
