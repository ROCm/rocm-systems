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
| `legacy_panel_id` | int | yes | Numeric panel ID for transition |
| `alias` | str | no | The panel's `-b` alias, e.g. `cu_pipe` |
| `tables` | list | yes | The panel's tables, in order |

Table fields: `id`, `title`, `header` (ordered column names), and optionally `render`
(`metric_table` by default, `raw_csv_table`, `pc_sampling_table`), `cli_style`
(`simple_box`, `simple_bar`, `simple_multiple_bar`, `mem_chart`, `Roofline`),
`placeholder_range`, `source`, `columnwise`, and `metrics` (the rows of a metric table).

Row fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `metric` | str | one of `metric`, `metrics` | Metric library id |
| `metrics` | dict[str, str] | one of `metric`, `metrics` | Column -> metric id, for per-channel rows |
| `label` | str | with `metrics` | Display name where it differs from the metric's `name` |
| `round` | int | no | Decimal places the main column rounds to |
| `peak` | `empirical` | no | Show the metric's `peak_empirical` instead of its `peak` |
| `pct_of_peak` | bool | no | `false` to show no percent of peak |
| `values` | dict[str, str] | no | Literal values of typed columns such as `xfer` and `coherency` |
| `legacy_id` | str | no | The row's numeric id in the panel file; one row per metric has it |

### Validation

Schema checks: known fields at each level, render types and CLI styles, one of `metric`
and `metrics` per row. Every referenced metric id exists in the arch's `MetricLibrary`,
and a row asking for the measured peak has one.


## Stage 6 -- Layer 3 reader

`load_views(arch)` reads `views/<arch>/*.yaml` (the arch's metric library arch, e.g.
`gfx115x` for gfx1151) in file name order. `views_to_panel_configs(views, metrics)`
translates them into the `OrderedDict[int, dict]` that `build_dfs()` consumes:

- Panel and table IDs from `legacy_panel_id` and each table's `id`; a row's numeric id
  is its position.
- Row name: the `label`, else the metric's `name`.
- Main column: the metric's `formula`, wrapped in `ROUND()` when the row asks; `min`,
  `max` and `std dev`: `MIN`/`MAX`/`STD` of the formula per kernel (each `SUM(x)` and
  `AVG(x)` becomes `(x)`).
- `unit`, `peak` (or `peak_empirical`) and the panel's descriptions from the metric.


## Stage 7 -- View file authoring

### Directory

```
src/rocprof_compute_soc/views/
  gfx908/  gfx90a/  gfx942/  gfx950/  gfx115x/  gfx1250/
    0000_top_stats.yaml
    0100_system_info.yaml
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
legacy_panel_id: 1100
alias: cu_pipe
tables:
- id: 1101
  title: Compute Speed-of-Light
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
- id: 1102
  title: Pipeline Statistics
  header:
    metric: Metric
    avg: Avg
    min: Min
    max: Max
    unit: Unit
  metrics:
  - metric: compute.ipc
  - metric: compute.ipc_issued
    legacy_id: 11.2.1
```

View files contain no formulas, units, peaks, or descriptions -- those live in Layer 2.
The view declares which metrics to show, in what order, and how to render them.

### Migration tooling

The view files are generated once from the panel files by a local script, which links
each row to its metric id with the Phase A row links, keeps a `label` where the panel
name differs from the metric's name, and keeps the display fields above.

### Validation

`tools/validate_views.py` (pre-commit) checks the view files: their fields, render types
and CLI styles, and that every metric they show is in the arch's metric library.
`test_analysis_views_sync.py`, next to the metric library sync test, checks every arch's
views against the panel files until they are removed: the same panels, tables, headers
and table settings in order, the same rows in order apart from rows that show no
equation, the `-b` aliases of the config templates, and one `legacy_id` per metric,
its row's numeric id in the panel file. It skips the panels the metric library sync test skips (the roofline panel of
an arch without a roofline benchmark, gfx908).


## Stage 8 -- Cutover

### Modifications

Every production reader of the old analysis config tree moves to the new path:

| Reader | Where | New source |
|---|---|---|
| Analyze panel configs | `generate_configs()`, `rocprof_compute_analyze/analysis_base.py` | `views_to_panel_configs()` |
| `--list-metrics` | `_build_arch_metric_details()`, `rocprof_compute_base.py` | `views_to_panel_configs()` |
| Memory chart | `PanelConfigs.for_arch()`, `memory_chart/loader.py` | `views_to_panel_configs()` |
| Profile counters (no filter, `-b`, `--set`, `--roof-only`) | `detect_counters()` and `_append_analysis_yaml_for_filter_token()`, `rocprof_compute_soc/soc_base.py` | `MetricLibrary.get_counters_for_metrics()` |
| Metric-aware counter grouping | `_iter_arch_analysis_yaml_metrics()` and `profiling_counter_grouping_policy.yaml`, `soc_base.py` | Not designed yet: needs metric ids instead of positional ids |
| `-b` panel aliases | `get_arch_panel_id_to_alias()`, `utils/utils_common.py` (reads `*_config_template.yaml`) | Not designed yet: views need to carry the alias |

`load_panel_configs()` in `utils/utils_common.py` then has no production caller. The
unused copy in `utils/file_io.py` is deleted.

**No fallback release:** the cutover release makes the new path the only path and removes
the old one: the old panel config files, `load_panel_configs()`, the YAML counter scan in
`detect_counters()` and the hardcoded built-in variables, along with the tooling listed
below ([cutover timeline](lld-index.md#cutover-timeline)).

**`generate_configs()` in `analysis_base.py`** -- the new path replaces the
`load_panel_configs()` call:

```python
library = MetricLibrary.load(METRIC_LIBRARY_DIR, arch)
views = load_views(view_dirs, arch, library)
ac.panel_configs = views_to_panel_configs(views, library)
```


**`detect_counters()` in `soc_base.py`** -- counter detection goes through
`MetricLibrary.get_counters_for_metrics()`; the YAML-text scan is removed.

**`get_build_in_vars()` in `utils_counter_defs.py`** -- loads from
`MetricLibrary.variables`; the hardcoded dicts are removed.

**`-b` / `--filter-blocks`** -- string metric IDs are the primary format. Numeric
IDs produce a deprecation warning with the equivalent string ID:

```
WARNING: Numeric metric ID '11.2.3' is deprecated.
         Use 'compute.salu_util' instead.
```

`convert_metric_id_to_panel_info()` remains during the deprecation window for
backward compatibility.

**`legacy_panel_id`** -- removed from all Layer 3 view files.

### Tooling cleanup

| Tool | What changes |
|---|---|
| `validate_sets_metric_ids.py` | Delete it with the numeric set files; `tools/validate_metric_library.py` checks the string-ID sets. |
| `verify_against_config_template.py` | Remove Panel Config template validation. Only Layer 2 schema validation. Remove `gfx9_config_template.yaml` and `gfx11_config_template.yaml`. |
| `metric_description_manager.py` | Remove old panel config reading path. |
| `hash_manager.py` | Remove old per-arch directory entries from hash database. Only base arch dirs and `views/`. |
| `format_yaml.py` | Remove Panel Config format handling if no longer needed. |

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

- **Which panels analyze shows after `profile --set`.** Today the set's numeric IDs are
  saved as `filter_blocks` and select panels. A string ID such as `compute.salu_util`
  appears in several panels (e.g. 2 and 11), so the views must decide. This is HLD OQ2.
- **What `--list-sets` prints.** Today it prints numeric IDs.

At the cutover, `--set` switches to the Layer 2 sets, with these display rules in place.
