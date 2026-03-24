# Ruff and Tooling Rules — AI-Authoritative Reference

> These rules summarize the project's automated linting and formatting requirements.
> Canonical configuration: [`pyproject.toml`](../../pyproject.toml)

---

## Line Length

**Hard limit: 88 characters per line.** This is enforced by ruff (`line-length = 88` in `pyproject.toml`).

- Do not generate lines that exceed 88 characters.
- Break long function arguments, chained calls, or comments across multiple lines.
- Use implicit string concatenation or backslash continuation for long strings.

```python
# Correct — argument list broken across lines
result = some_function(
    argument_one,
    argument_two,
    argument_three,
)

# Flagged by Ruff (E501)
result = some_function(argument_one, argument_two, argument_three, argument_four_long_name)
```

## Enforced Ruff Rules

The following rules are enforced on all files in `src/` (see `pyproject.toml` for full config):

| Rule Set | Code | Requirement |
|----------|------|-------------|
| Line length | E501 | Maximum 88 characters per line |
| Type annotations | ANN | All function arguments and return values must have type hints (except `self`, `cls`, `_` args) |
| Modern syntax | UP | Use f-strings, not `.format()` or `%` |
| Path handling | PTH | Use `pathlib.Path`, not `os.path` |
| Pyflakes | F | No unused imports or variables |
| pycodestyle | E, W | PEP 8 compliance |
| isort | I | Sorted imports |

## Quick Reference

### Type Annotations

```python
# Correct
def process_data(name: str, values: list[float]) -> dict[str, Any]:
    ...

# Flagged by Ruff (ANN)
def process_data(name, values):
    ...
```

### String Formatting

```python
# Correct
message = f"Processing {name}"

# Flagged by Ruff (UP)
message = "Processing {}".format(name)
```

### Path Handling

```python
# Correct
config_path = Path.cwd() / "config" / "settings.yaml"

# Flagged by Ruff (PTH)
config_path = os.path.join(os.getcwd(), "config", "settings.yaml")
```

## Suppressing Rules

Use sparingly and only when genuinely necessary:

- `# fmt: off` / `# fmt: on` — disable/re-enable formatting for a block
- `# fmt: skip` — skip formatting for a single line
- `# noqa: <RULE>` — suppress a specific lint rule for a line

## Running Ruff Locally

```bash
# Check for issues
ruff check .
ruff format --check .

# Auto-fix
ruff check --fix .
ruff format .

# Check specific rule set
ruff check --select ANN .
```
