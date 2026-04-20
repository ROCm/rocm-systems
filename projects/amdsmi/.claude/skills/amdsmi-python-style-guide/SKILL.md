---
name: amdsmi-python-style-guide
description: "ROCm Python style guide based on TheRock. Use when: writing Python code, reviewing Python PRs, checking Python style, creating Python scripts, type hints, error handling patterns, pathlib usage, argparse CLI design."
---

# Python Style Guide (ROCm / TheRock)

Follows PEP 8, enforced by **Ruff** (amd-smi) or **Black** (TheRock) via pre-commit.
Source: [TheRock python_style_guide.md](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/python_style_guide.md)

## Core Principles

- Fail fast and loud — false positives over silent corruption
- When in doubt, raise an exception

---

## Type Hints

- Type hints on **all** function signatures
- Syntax that is compatible with python 3.6.8+
- **Never** use `Any` except in rare generic code
- **Never** use `Optional[T]`, `Union[X, Y]`, `List`, `Dict` — use built-in equivalents
- **Do NOT** use `from __future__ import annotations`

### Extract Complex Types

- Type appears in multiple signatures → `NamedTuple` or `TypeAlias`
- Dict/tuple with 3+ fields → `dataclass` or `NamedTuple`
- If you document what tuple fields mean → use `NamedTuple`

```python
# Good
class KpackInfo(NamedTuple):
    kpack_path: Path
    size: int
    kernel_count: int

# Bad
def get_info() -> tuple[Path, int, int]: ...
```

### Dataclasses Over Tuples

Structured data with multiple fields → `@dataclass` or `NamedTuple`, not raw tuples.
Tuples OK for: simple pairs `(x, y)`, stdlib unpacking, single-use immediate unpacking.

---

## Function Calls & Arguments

- 3+ parameters or ambiguous types → **use keyword arguments**
- Multiple booleans → **always** use keyword arguments

```python
# Good
result = build(amdgpu_family="gfx942", enable_testing=True, use_ccache=False)

# Bad
result = build("gfx942", True, False)
```

---

## Code Organization

- All imports at **top of file** — inline only for documented circular deps (add comment explaining why)
- Scripts must have `if __name__ == "__main__":` guard
- CLI scripts must use `argparse` (not raw `sys.argv`)
- Access `argparse` attrs directly: `args.foo`, not `getattr(args, "foo", default)`
- No duplicate code — extract to shared functions
- No magic numbers or fake estimates
- Methods < 30 lines, classes < 200 lines (ideally)

---

## Error Handling

| Rule | Detail |
|------|--------|
| Fail fast | Never silently continue or produce incomplete results |
| Specific exceptions | `FileNotFoundError`, `subprocess.CalledProcessError` — not bare `except Exception` |
| Preserve chains | Always use `from e` when re-raising |
| Validate results | After critical ops: check file exists, non-empty, expected size |
| No binutils timeouts | Never `timeout=` on `readelf`, `objcopy`, etc. |

```python
# Good — distinguish error conditions
try:
    output = subprocess.check_output([readelf, "-S", str(file_path)])
except subprocess.CalledProcessError as e:
    if e.returncode == 1:
        return False  # legitimate "not found"
    raise RuntimeError(f"readelf failed: {e.output}") from e
except FileNotFoundError as e:
    raise RuntimeError(f"readelf not found: {readelf}") from e

# Bad — swallows everything
try:
    output = subprocess.check_output([readelf, "-S", str(file_path)])
except Exception:
    return False
```

---

## Filesystem & Paths

- **`pathlib.Path` everywhere** — no `os.path`, no string concatenation for paths
- No assumptions about CWD — derive from `Path(__file__).resolve().parent`
- No hard-coded project paths — use env vars, `tempfile`, or relative to `__file__`

```python
# Good
THIS_DIR = Path(__file__).resolve().parent
config = THIS_DIR / "config.json"

# Bad
config = Path("build_tools/config.json")  # assumes CWD
config = Path("/home/user/project/config.json")  # hard-coded
```

---

## Performance

- Compile regexes at **module level**, not inside functions
- Check cheap conditions before expensive ones (magic bytes before subprocess)
- Cache expensive computations when called repeatedly
- Use generators for large datasets

---

## Testing

- Verify fail-fast behavior with `pytest.raises`
- Use real temp files over mocks for filesystem operations
- Mock only external dependencies (network, expensive tools)
- Integration tests should exercise the full path

---

## amd-smi Python Conventions

### Naming
- `py-interface/amdsmi_interface.py` mirrors C API names (`amdsmi_get_*`, `amdsmi_set_*`)
- `amdsmi_cli/` uses `snake_case` methods
- Private functions/methods: `_` prefix

### Project Layout
| Directory | Purpose |
|-----------|---------|
| `py-interface/` | Python bindings — `amdsmi_interface.py`, `amdsmi_wrapper.py`, `amdsmi_exception.py` |
| `amdsmi_cli/` | CLI tool — `amdsmi_commands.py`, `amdsmi_parser.py`, `amdsmi_helpers.py` |

### Critical Python Files (high churn — review carefully)
- `amdsmi_cli/amdsmi_commands.py` — CLI behavior regressions, output format changes
- `py-interface/amdsmi_interface.py` — must stay in sync with C header `amdsmi.h`
- `py-interface/amdsmi_wrapper.py` — generated bindings + library loader (see loader rules below)
- `amdsmi_cli/amdsmi_parser.py` — argument parsing

### Library Loader Rules (`amdsmi_wrapper.py`)
Changes to `_detect_install_context`, `_build_candidate_paths`, or `_load_library` are **❌ BLOCKING** if they break system or pip install context. Verify:
- `Path(__file__).resolve()` used correctly
- pip detection logic intact
- `_libraries['libamd_smi.so']` key preserved
- Sync with `tools/generator.py`

### Python Testing
- Tests must work with **both** system-installed and pip-installed amdsmi
- CLI tests live in `amdsmi_cli/`
- Verify fail-fast behavior with `pytest.raises`
- Use real temp files over mocks for filesystem operations
- Mock only external dependencies (network, expensive tools)
- Integration tests should exercise the full path

---

## Review Checklist

When reviewing Python code, verify:

- [ ] Type hints on all functions — modern syntax, no `Any`
- [ ] No `from __future__ import annotations`
- [ ] Complex types extracted to `NamedTuple`/`dataclass`
- [ ] Specific exception handling — no bare `except Exception`
- [ ] Fail-fast — no silent `continue` on errors
- [ ] `from e` on all re-raises
- [ ] `pathlib.Path` for all filesystem ops
- [ ] No CWD assumptions, no hard-coded paths
- [ ] No magic numbers or fake estimates
- [ ] No duplicate code
- [ ] `argparse` for CLI, `__main__` guard for scripts
- [ ] Keyword args for complex function calls
- [ ] All imports at top (inline only with circular-dep comment)
- [ ] No timeouts on binutils
- [ ] Output validation after critical operations
- [ ] Methods < 30 lines, classes < 200 lines
