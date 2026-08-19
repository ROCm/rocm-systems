# YAML Metric Equation Guidelines

Metric equations in YAML config files follow a canonical format. The canonical
implementation is [`tools/format_yaml.py`](tools/format_yaml.py).

## Scope
Equation formatting applies to YAML files under:

- `src/rocprof_compute_soc/analysis_configs/gfx*/*.yaml`
- `src/rocprof_compute_tui/utils/gfx*/*.yaml`

Template files (`*_template.yaml`) and build artifacts are excluded. Only values
under these keys are treated as equations: `value`, `avg`, `min`, `max`, `peak`.

## Rules
- **Operator spacing** — all binary operators (`+`, `-`, `*`, `/`) have exactly
  one space on each side: `SUM(x) / SUM(y)`.
- **Constant factoring** — when an aggregation (`SUM`, `AVG`, `MIN`, `MAX`)
  multiplies by a numeric constant, factor it out:
  `128 * SUM(x)`, not `SUM(x * 128)`. Addition inside aggregations is left as-is.
- **Minimal parentheses** — remove parentheses made redundant by operator
  precedence (`a + b + c`, not `(a + b) + c`), but preserve those required for
  correct evaluation (`a - (b - c)`, `(a + b) * c`).
- **Unary negation** — negation of a compound expression keeps its parentheses:
  `-(a + b)`, not `-a + b`.
- **Literal values** — `N/A`, `None`, `null`, `true`, `false`, and
  `Peak (Empirical)` are never parsed as equations and must be written exactly
  as shown (the space before `(` in `Peak (Empirical)` is required).

Equations with unsupported syntax (unknown characters, unmatched parentheses) are
left unchanged; the formatter never corrupts an equation it cannot fully parse.

## Enforcement
The `yaml-format-fix` pre-commit hook runs `tools/format_yaml.py --fix` on staged config files and auto-corrects equations
in place. Re-stage the modified files and commit again.

```bash
# Show proposed changes
python tools/format_yaml.py --diff src/rocprof_compute_soc/analysis_configs/gfx950/*.yaml

# Auto-fix in place
python tools/format_yaml.py --fix src/rocprof_compute_soc/analysis_configs/gfx950/*.yaml
```
