# YAML Metric Equation Formatting

> These rules describe the canonical format for metric equations in YAML config files.
> Canonical implementation: [`tools/format_yaml.py`](tools/format_yaml.py)

---

## Scope

Equation formatting applies to YAML files matching these path patterns:

- `src/rocprof_compute_soc/analysis_configs/gfx*/*.yaml`
- `src/rocprof_compute_tui/utils/gfx*/*.yaml`

Template files (`*_template.yaml`) and build artifacts are excluded.

Only values under these YAML keys are treated as equations: `value`, `avg`, `min`, `max`, `peak`.

## Formatting Rules

### Operator Spacing

All binary operators (`+`, `-`, `*`, `/`) must have exactly one space on each side.

```yaml
# Correct
value: SUM(x) / SUM(y)

# Incorrect — missing or extra spaces
value: SUM(x)/SUM(y)
value: SUM(x)  /  SUM(y)
```

### Constant Factoring

When an aggregation function (`SUM`, `AVG`, `MIN`, `MAX`) contains a multiplication
by a numeric constant, factor the constant out of the aggregation.

```yaml
# Correct — constant factored out
value: 128 * SUM(TCP_TOTAL_CACHE_ACCESSES_sum)

# Incorrect — constant inside aggregation
value: SUM(TCP_TOTAL_CACHE_ACCESSES_sum * 128)
```

This applies only to multiplication. Addition inside aggregations is left as-is:

```yaml
# Correct — addition stays inside
value: SUM(x + 128)
```

### Minimal Parentheses

Remove parentheses that are unnecessary given operator precedence, but preserve those
required for correct evaluation.

```yaml
# Correct — unnecessary parens removed
value: a + b + c

# Incorrect — redundant grouping
value: (a + b) + c
```

```yaml
# Correct — required parens preserved (subtraction is not associative on the right)
value: a - (b - c)

# Correct — precedence parens preserved
value: (a + b) * c
```

### Unary Negation

Unary negation of a compound expression must keep its parentheses.

```yaml
# Correct
value: -(a + b)

# Incorrect — changes meaning
value: -a + b
```

### Literal Values

These string values are never parsed as equations and must be written exactly as shown:

- `N/A`, `None`, `null`, `true`, `false`
- `Peak (Empirical)` — a column header, not a function call; the space before `(` is required

```yaml
# Correct
peak: Peak (Empirical)

# Incorrect — parsed as function call
peak: Peak(Empirical)
```

## Enforcement

Formatting is enforced in two ways:

1. **Pre-commit hook** — runs `tools/format_yaml.py --fix` on staged YAML files and auto-corrects equations in place. Re-stage the modified files and commit again.
2. **CI test** — `test_analysis_config_yaml_format` validates all production YAML configs in the CTest suite.

## Running Manually

```bash
# Check for issues (shows diffs)
python tools/format_yaml.py --diff src/rocprof_compute_soc/analysis_configs/gfx950/*.yaml

# Auto-fix in place
python tools/format_yaml.py --fix src/rocprof_compute_soc/analysis_configs/gfx950/*.yaml
```

## Fallback Behavior

Equations containing unsupported syntax (unknown characters, unmatched parentheses,
non-standard constructs) are left unchanged — the formatter never corrupts an equation
it cannot fully parse.
