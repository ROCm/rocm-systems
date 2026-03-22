# Python standards (rocprofiler-compute)

- **Tooling** — Ruff lint + format per `pyproject.toml` (`line-length = 88`, selected rules include E, W, F, I, ANN, UP, PTH). Run `ruff check` and `ruff format` on touched files.
- **Annotations** — `src/**` is subject to ANN/UP/PTH per config; match existing style in the file you edit.
- **Layout** — Application code under `src/` (`rocprof_compute_*`, `utils`, etc.). Tests under `tests/` with `pytest` and existing `pythonpath` in `pyproject.toml`.
- **Design** — Prefer small functions and clear data flow; avoid deep inheritance unless the file already uses that pattern.
- **CLI** — Argparse and experimental flags follow patterns in `src/argparser.py` and CONTRIBUTING.
