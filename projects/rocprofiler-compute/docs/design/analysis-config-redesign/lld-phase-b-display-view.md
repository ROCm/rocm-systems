# Phase B -- Display / View (Stages 5-8)

Parent: [LLD index](lld-index.md)

Stage dependencies: [Stage dependencies and parallelism](lld-index.md#stage-dependencies-and-parallelism).


## Stage 5 -- Layer 3 schema

Module `src/utils/analysis_views.py` (schema checks, reader and adapter in one module).
A view is one panel of one arch; it holds metric ids and how each row is shown, never a
formula, unit, peak or description.

View fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `view` | str | yes | View title, e.g. `"Compute Units - Compute Pipeline"` |
| `alias` | str | no | The panel's `-b` alias, e.g. `cu_pipe` |
| `tables` | list | yes | The panel's tables, in order |

The panel ID comes from the file name (`1100_compute_units_compute_pipeline.yaml` is
panel 1100). The top stats and system info panels are built-in views defined in
`analysis_views.py`, not files.

Table fields: `title`, `header` (ordered column names), and optionally `render`
(`metric_table` by default, `raw_csv_table`, `pc_sampling_table`), `placeholder_range`,
`source`, `columnwise`, and `metrics` (the rows of a metric table). A table has no ID or
CLI style field: its ID is its position in the panel, and the CLI derives its style from
its header and panel.

Row fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `metric` | str | one of `metric`, `columns` | Metric library id |
| `columns` | dict[str, str] | one of `metric`, `columns` | Column -> metric id, for per-channel rows |
| `label` | str | with `columns` | Display name where it differs from the metric's `name` |
| `round` | int | no | Decimal places the main column rounds to |
| `peak` | `empirical` | no | Show the metric's `peak_empirical` instead of its `peak` |
| `pct_of_peak` | bool | no | `false` to show no percent of peak |
| `values` | dict[str, str] | no | Literal values of typed columns such as `xfer` and `coherency` |

### Validation

Schema checks: the file name, known fields at each level, render types, one of `metric`
and `columns` per row. Every referenced metric id exists in the arch's `MetricLibrary`,
a row asking for the measured peak has one, and a row whose table shows Min, Max or Std
Dev reads no summed collectable without a `<name>_per_dispatch` partner.


## Stage 6 -- Layer 3 reader

`load_views(arch)` reads `views/<arch>/*.yaml` (the arch's metric library arch, e.g.
`gfx115x` for gfx1151) in file name order.
`views_to_panel_configs(views, metrics, collectables)` translates them into the
`OrderedDict[int, dict]` that `build_dfs()` consumes:

- Panel IDs from the file names, table IDs from the table's position in the panel, and a
  row's numeric id from its position in the table.
- Row name: the `label`, else the metric's `name`.
- Main column: the metric's `formula`, wrapped in `ROUND()` when the row asks; `min`,
  `max` and `std dev`: `MIN`/`MAX`/`STD` of the formula per kernel (each `SUM(x)` and
  `AVG(x)` becomes `(x)`, and each summed collectable becomes its
  [per-dispatch partner](../../../src/rocprof_compute_soc/metric_library/README.md#collectables)).
- `unit`, `peak` (or `peak_empirical`) and the panel's descriptions from the metric.


## Stage 7 -- View file authoring

### Directory

```
src/rocprof_compute_soc/views/
  gfx908/  gfx90a/  gfx942/  gfx950/  gfx115x/  gfx1201/  gfx1250/
    0200_system_speed_of_light.yaml
    ...                       # one file per panel file of the arch
```

Each metric library arch has its own directory with one view per panel file.

### Constraint: preserve current panel structure in Stage 7

Per reviewer feedback: the Stage 7 migration preserves the current panel structure,
grouping, and ordering: the same panels, tables, headers and rows in order. Rows that
show no equation are left out. Values, units, peaks and descriptions come from the
metric library, so they can differ from the panel files' own texts.

### View consolidation (post Stage 7, requires customer approval)

After Stage 7 validates the 1:1 migration, the view layer should be tightened. The
current 18 CDNA panels were inherited from the original tool design without formal
requirements -- there is no evidence that customers use or need all of them.

**Principle:** Views that lack customer evidence of usage should be cut. Views that
customers do use should be refined in collaboration with them. The goal is fewer,
more purposeful views -- not a 1:1 carry-forward of legacy structure.

**Process:**

1. After Stage 7 completes, gather customer input on which views they actively use
   and what workflows they support.
2. Propose a reduced view set based on that input.
3. Each view change is a separate PR requiring customer sign-off.

### Example view file

```yaml
# views/gfx942/1100_compute_units_compute_pipeline.yaml
view: Compute Units - Compute Pipeline
alias: cu_pipe
tables:
- title: Compute Speed-of-Light
  header:
    metric: Metric
    value: Avg
    unit: Unit
    peak: Peak
    pct_of_peak: Percent of Peak
  metrics:
  - metric: compute.valu_flops
  - metric: compute.mfma_flops_f8
  - metric: compute.mfma_flops_bf16
- title: Pipeline Statistics
  header:
    metric: Metric
    avg: Avg
    min: Min
    max: Max
    unit: Unit
  metrics:
  - metric: compute.ipc
  - metric: compute.ipc_issued
```

View files contain no formulas, units, peaks, or descriptions -- those live in Layer 2.
The view declares which metrics to show, in what order, and how to render them.

### Migration tooling

The view files are generated once from the panel files by a local script, which links
each row to its metric id with the Phase A row links, keeps a `label` where the panel
name differs from the metric's name, and keeps the display fields above.

### Validation

`tools/validate_views.py` (pre-commit) checks the view files: their names, fields and render
types, that every metric they show is in the arch's metric library, that their Min,
Max and Std Dev columns have the per-dispatch partners they read and their simple-box
rows read no summed collectable, and that they show every metric and every measured peak
of the library.


## Stage 8 -- Cutover

### Modifications

Analyze, its listings, the memory chart, the `-b` panel aliases and profile's counter
selection read the views, with each metric from the metric library:

| Reader | Where | Source |
|---|---|---|
| Analyze panel configs | `generate_configs()`, `rocprof_compute_analyze/analysis_base.py` | `arch_panel_configs()`, `utils/analysis_views.py`: `views_to_panel_configs()` of the arch's views |
| `--list-metrics` | `_build_arch_metric_details()`, `rocprof_compute_base.py` | `arch_panel_configs()` |
| Memory chart | `PanelConfigs.for_arch()`, `memory_chart/loader.py` | `arch_panel_configs()` |
| `-b` panel aliases (analyze, profile, `--list-metrics`, `--list-blocks`) | `get_arch_panel_id_to_alias()`, `utils/analysis_views.py` | The `alias` of each view |
| Profile counters for metric library ids (`-b`, `--set`) | `detect_counters()`, `rocprof_compute_soc/soc_base.py` | `MetricLibrary.get_counters_for_metrics()` |
| Profile counters (no filter, other `-b` ids) and metric-aware counter grouping | `detect_counters()`, `_append_expressions_for_filter_token()` and `_iter_profile_metrics()`, `rocprof_compute_soc/soc_base.py` | `arch_panel_configs()`, through `_profile_panel_configs()` |
| Collectables and built-in variables | `Collectables.of_arch()`, `rocprof_compute_soc/metric_library/loader.py` | `MetricLibrary.collectable_formulas()` |
| Counter grouping inspector, metric health report, membw tree spec check | `tools/` | `arch_panel_configs()` |

A numeric `-b` id selects the row at that position in the views, the id analyze shows.
`--set` selects the metric ids of a library set (`metric_library/{arch}/sets.yaml`).

**No fallback release:** the cutover release makes the new path the only path and removes
the old one: `analysis_configs/` (the panel files, the config templates and
`collectables/`), `load_panel_configs()` and the tooling listed below
([cutover timeline](lld-index.md#cutover-timeline)).

**`generate_configs()` in `analysis_base.py`** -- reads the panel configs from the views
through `arch_panel_configs()` in `analysis_views.py`:

```python
library = MetricLibrary.load(default_metric_library_dir(), arch)
views_to_panel_configs(load_views(arch), library.metrics, library.collectable_formulas())
```


**`detect_counters()` in `soc_base.py`** -- reads the formulas of the selected rows from
the panel configs of the views, so a row's counters are its library metric's.

**`Collectables.of_arch()` in the metric library loader** -- returns
`MetricLibrary.collectable_formulas()`; `analysis_configs/collectables/` is removed.

**`-b` / `--filter-blocks`** -- string metric IDs are the primary format. Numeric
IDs produce a deprecation warning with the equivalent string ID:

```
WARNING: Numeric metric ID '11.2.3' is deprecated.
         Use 'compute.salu_util' instead.
```

`convert_metric_id_to_panel_info()` remains during the deprecation window for
backward compatibility.

### Tooling cleanup

| Tool | What changes |
|---|---|
| `validate_sets_metric_ids.py` | Deleted with the numeric set files; `tools/validate_metric_library.py` checks the string-ID sets. |
| `verify_against_config_template.py`, `parse_config_template.py` | Deleted with the config templates. `tools/validate_views.py` checks the views and `tools/validate_metric_library.py` the library. |
| `metric_description_manager.py` | Reads the views and the library, and writes one description file per arch, `docs/data/metrics/<arch>_metrics.yaml`, with a section per view table named after its title. |
| `hash_manager.py`, `hash_checker.py`, `master_config_workflow_script.py` | Replaced by the metric lock: `check_metric_config.py` (the `metric-config-check` pre-commit hook) runs the checks and regenerates `metric_lock/<arch>.yaml` and the description files. |
| `format_yaml.py` | Checks the metric library's formulas only. |

### Validation

- Full regression: entire test suite (`ctest` + Python tests) with the new path as
  the default. Zero failures.
- Columns: the views reproduce every column of every analysis config table, not only the
  main one, before any panel file is deleted.
- Golden files: for every architecture, analyze text output and `detect_counters()`
  output match the pre-migration capture
  ([Risk mitigation](lld-index.md#risk-mitigation-metric-value-validation)).
- Pre-commit hooks: all pass with the new file structure.
- Documentation pipeline: Sphinx-generated metric reference is identical to
  pre-migration version.


## Set display

Sets are defined in Layer 2 (see [Sets migration](lld-phase-a-metric-library.md#sets-migration)). What Layer 3 decides is how a
set's results are shown:

- **Which rows analyze shows after `profile --set`.** The set's metric ids are saved as
  `filter_blocks`, and analyze filters rows by them as `-b` does with metric ids: each id
  selects every view row that shows it (`metric_placements()`), so `compute.salu_util`
  shows in both panels 2 and 11. A flat set view is HLD OQ2.
- **What `--list-sets` prints.** Each set's option and title, and the name and metric id
  of each of its metrics.
