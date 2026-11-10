# Metric library

Each directory here defines one GPU architecture's metrics, once each, keyed by a stable
id such as `compute.salu_util`. The files hold what a metric is (its formula, unit and
peaks), not how a panel shows it. `loader.py` loads them.

Production does not read these files yet: `analyze` and `profile` still use
[`analysis_configs/`](../analysis_configs/). Until production reads the library, a metric
change goes into both trees, and a row added, moved or renamed in a panel also goes into
the arch's views in [`views/`](../views/).

## Layout

```
metric_library/
  loader.py         loads the library of an arch (MetricLibrary)
  gfx908/           a base arch: every file
    compute.yaml    the compute.* metrics
    l2.yaml         the l2.* metrics
    ...
    variables.yaml  the built-in variables, such as GRBM_GUI_ACTIVE_PER_XCD
    sets.yaml       the profiling sets (--set)
  gfx90a/           a child arch: only the files that change
    compute.yaml    merged onto gfx908/compute.yaml
```

- **One file per id category**, named after it: `l2.yaml` holds every `l2.*` metric.
  Categories name hardware blocks (`compute`, `l2`, `vl1d`, `fabric`, ...), never a view.
- **Keys are sorted** in every file, apart from a `remove:` list.
- **Plain ASCII** text only.

## Adding a metric

- Check first whether the arch already computes the same quantity, even with a formula
  written another way. If it does, reuse that metric's id.
- A formula has one id on every arch. A later arch reuses the id from the first arch
  with that quantity.

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
| `description` | The text of one panel with the metric, word for word. May be empty. |
| `peak` | Optional. The theoretical peak. |
| `peak_empirical` | Optional. The roofline benchmark's ceiling, as `$<ceiling>_empirical_peak`. |

A description keeps the panel's words, with only these edits:

- plain ASCII (`->` for an arrow, no trademark mark);
- no sentence about how a panel shows the metric ("also presented as a percent of the
  peak");
- no panel 3000 `Equation:` sentence, since `formula` holds the equation;
- no TODO note.

Better wording goes to the metric owners in a separate change.

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

A child arch lists only what differs from its parent. `METRIC_LIBRARY_PARENT_ARCH` in
`loader.py` names the parent of each arch. A new arch adds its entry there and its
directory here; the validator reports one without the other.

A child file merges onto the parent arch's file of the same name:

- a mapping merges key by key, so a changed metric lists only its changed fields;
- a scalar or a list replaces the parent's value, but a mapping, a list and a scalar
  cannot replace one another;
- a top-level `remove:` list drops parent metrics by id:

```yaml
remove:
  - fabric.rd_bw_norm
```

A file missing from the child comes from its nearest ancestor.

## Checks

`tools/validate_metric_library.py` runs as a pre-commit hook on changes here. It checks
the fields, the ids and their files, names, units, sorted keys, that every formula
parses, that a formula has one id on every arch, the roofline ceilings, and that each set
fits in one pass. Its module docstring lists every check. The `check-yaml` hook rejects
duplicate keys.

While production reads `analysis_configs/`,
`tests/unit/rocprof_compute_soc/metric_library/test_metric_library_sync.py` checks that
the library matches `analysis_configs/` and `profile_configs/sets/`: every equation of a
panel row is a library formula, and the units, peaks, counters, built-in variables and
sets match. CI runs it.
