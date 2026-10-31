# Metric library

Each directory here defines one GPU architecture's metrics, once each, keyed by a stable
id such as `compute.salu_util`. The files hold what a metric is (its formula, unit and
peaks), not how a panel shows it. The design is in
[`lld-phase-a-metric-library.md`](../../../docs/design/analysis-config-redesign/lld-phase-a-metric-library.md).

Production does not read these files yet: `analyze` and `profile` still use
[`analysis_configs/`](../analysis_configs/). Until the cutover, a metric change goes into
both trees, and a change to how a panel shows a metric goes into its view in
[`views/`](../views/).

## Layout

```
metric_library/
  gfx908/           a base arch: every file
    compute.yaml    the compute.* metrics
    l2.yaml         the l2.* metrics
    ...
    variables.yaml  the built-in variables, such as GRBM_GUI_ACTIVE_PER_XCD
    sets.yaml       the profiling sets (--set)
  gfx90a/           a child arch: only the files that change
    compute.yaml    inherit: ../gfx908/compute.yaml
```

- **One file per id category**, named after it: `l2.yaml` holds every `l2.*` metric.
  Categories name hardware blocks (`compute`, `l2`, `vl1d`, `fabric`, ...), never a view.
- **Keys are sorted** in every file; `tools/format_yaml.py --fix` sorts them.
- **Plain ASCII** text only.

## A metric

```yaml
compute.salu_util:
  name: SALU Utilization
  formula: 100 * SUM(SQ_ACTIVE_INST_SCA) / SUM($GRBM_GUI_ACTIVE_PER_XCD * $cu_per_gpu)
  unit: Percent
  description: >-
    Indicates what percent of the kernel's duration the SALU was busy executing
    instructions. ...
  peak: 100
```

| Field | Meaning |
|---|---|
| `name` | A name specific to the hardware, in title case, unique on the arch. An id has one name and one unit on every arch. |
| `formula` | The one equation of the metric on this arch. |
| `unit` | One spelling per unit, such as `GB/s` or `Percent`. |
| `description` | What the metric is, not how a panel shows it. May be empty. |
| `peak` | Optional. The theoretical peak. |
| `peak_empirical` | Optional. The roofline benchmark's ceiling, as `$<ceiling>_empirical_peak`. |

## Sets

`sets.yaml` lists each profiling set by metric id. A set's counters must fit in one
profiling pass.

```yaml
compute_thruput_util:
  title: Compute Throughput Utilization
  metrics:
  - compute.salu_util
  - compute.valu_util
```

## Inheritance

A child arch lists only what differs from its parent; the chain is
`METRIC_LIBRARY_PARENT_ARCH` in `src/utils/metric_library.py`. Each child file starts
with `inherit: <path to the parent's file>` and is merged on it:

- a mapping merges key by key, so a changed metric lists only its changed fields;
- a scalar or a list replaces the parent's value;
- `<id>: !remove` deletes a metric, and `<field>: !remove` deletes a field.

A file the child does not have comes from its nearest ancestor.

## Checks

`tools/validate_metric_library.py` runs as a pre-commit hook on any change here or in
`analysis_configs/`. It checks the schema, inheritance, ids (a formula has one id on
every arch), names, units, roofline ceilings and sets, and proves the library equals
`analysis_configs/`: every equation a panel row shows is a library formula, and the
counters, built-in variables and sets match. Its module docstring lists every check.
