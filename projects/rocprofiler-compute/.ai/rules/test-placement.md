# Test Placement Rules — AI-Authoritative Reference

> These rules govern where to place new or relocated test code in `tests/`.

---

## Module-Mirror Convention

Unit tests live in `tests/` and each test file maps to the source module it
exercises.

- Place unit tests for `src/utils/parser.py` in `tests/test_parser.py`, tests
  for `src/utils/tty.py` in `tests/test_tty.py`, and so on.
- When the source path has subdirectories
  (e.g. `src/utils/metrics/metric_evaluator.py`), use the leaf module name
  (`tests/test_metric_utils.py`) or an existing file that already covers that
  module.

## CLI Suites Stay CLI-Only

- `test_analyze_commands.py` is for end-to-end CLI exit-code tests that call
  `binary_handler_analyze_rocprof_compute`. Do not add pure unit tests to it;
  put them in the module-mirror file instead.

## Shared Fixtures

- Fixtures needed by more than one test file belong in `conftest.py`. Test
  files must not import from each other.

## Stubbing Convention

- Use `types.SimpleNamespace` or `argparse.Namespace` for plain attribute bags.
- Use `unittest.mock.Mock` / `MagicMock` only when you need call tracking or
  attribute auto-creation.

## Follow Host-File Conventions

- Match the destination file's existing patterns for class-vs-function grouping,
  marker usage, import style, and helper naming (e.g. `_`-prefixed factory
  functions).
- Read the file before adding to it.
