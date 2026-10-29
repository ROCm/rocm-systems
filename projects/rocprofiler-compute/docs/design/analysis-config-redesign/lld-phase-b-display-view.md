# Phase B -- Display / View (Stages 5-8)

Parent: [LLD index](lld-index.md)

Stage dependencies: [Stage dependencies and parallelism](lld-index.md#stage-dependencies-and-parallelism).


## Stage 5 -- Layer 3 schema

New module `src/utils/layer3_schema.py`.

`ViewDefinition` fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `view` | str | yes | View title, e.g. `"Compute Units - Compute Pipeline"` |
| `render` | str | yes | Render type (see mapping below) |
| `archs` | list[str] | no | View-level arch filter -- entire view skipped for unlisted archs |
| `tables` | list[ViewTableGroup] | yes | Sub-table groupings within the view |
| `legacy_panel_id` | int | no | Numeric panel ID for transition |

`ViewTableGroup` fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `title` | str | yes | Table title, e.g. `"Compute Speed-of-Light"` |
| `header` | dict[str, str] | yes | Column name mapping |
| `cli_style` | str | no | Render hint: `simple_bar`, `mem_chart`, `simple_box` |
| `metrics` | list[ViewMetricRef] | yes | Ordered list of metric references |

`ViewMetricRef` fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Layer 2 metric id |
| `archs` | list[str] | no | Metric-level arch filter |
| `label` | str | no | Display name override for this view |
| `unit` | str | no | Display unit for this view, e.g. `Bytes/s` for a metric defined in `GB/s` |
| `description_note` | str | no | Sentence appended to the metric's description in this view, saying how the view shows it, e.g. "This is also presented as a percent of the peak theoretical bandwidth." |

**Render type mapping to current table types:**

| `render` value | Current equivalent |
|---|---|
| `metric_table` | standard metric table |
| `raw_csv_table` | raw CSV pass-through |
| `pc_sampling_table` | PC sampling data |
| `simple_box` | box-plot style (current `cli_style: simple_box`) |

### Validation

Schema validation function checks: all referenced metric ids exist in a provided
`MetricLibrary`, arch names are known architectures, render type is from the allowed
set.

### Tests

Valid view passes validation; missing metric id fails; invalid arch name fails;
label override preserved in output.


## Stage 6 -- Layer 3 reader

New module `src/utils/layer3_reader.py`.

`load_views(view_dirs: list[Path], arch: str, library: MetricLibrary) -> list[ViewDefinition]`:

1. Discover `*.yaml` files in each directory in `view_dirs`.
2. Load each via `yaml.safe_load` (no `inherit` needed for view files -- they are
   flat by design).
3. Validate against `layer3_schema.py`.
4. Filter: skip views where `archs` is set and the current arch is not in the list.
   Within each view, skip metric refs where `archs` is set and the arch is not listed.
5. Resolve each metric `id` against `library.metrics` -- error if not found.
6. For each `ViewMetricRef`: if `label` is present, use it as display name; otherwise
   use `name` from the Layer 2 `MetricDefinition`.
7. For each `ViewMetricRef` with a `unit` different from the metric's Layer 2 `unit`,
   convert the value in code with one table of known unit conversions (e.g. `GB/s` ->
   `Bytes/s`, factor 1e9). An unknown pair is a validation error. The metric definition
   itself stays uniform; only the view's display changes.

Display adjustments such as unit scaling, rounding and aggregation wrappers belong in code,
not in static YAML. The metric library keeps only each row's main column
([Phase A](lld-phase-a-metric-library.md#layer-2-metric-schema)). A view's `header`
declares every column of the table (`min`, `max`, `std dev`, ...), and the code generates
each column's wrapped form from a per-kernel expression, as the `simple_box` tables
already do for `MIN/Q1/MEDIAN/Q3/MAX`.

`views_to_panel_configs(views: list[ViewDefinition], library: MetricLibrary) -> OrderedDict[int, dict]`:

Translates view objects into the same `OrderedDict[int, dict]` consumed by
`build_dfs()`. It is the translation from the new format to the existing pipeline; Phase
A ships no adapter (see [Stage 3](lld-phase-a-metric-library.md#stage-3----equivalence-with-the-analysis-configs)).

- Panel IDs: from `legacy_panel_id` during transition, or from view ordering when
  legacy IDs are removed.
- Table IDs: synthesized from panel ID + table sequence number.
- Header dicts: generated from the view's `header` field.
- Metric OrderedDicts: built from `ViewMetricRef` ids, resolving formulas and units
  from `MetricLibrary`.

### Tests

- Load test view files, resolve against test `MetricLibrary`, verify resulting
  `panel_configs` structure matches expected shape.
- Arch filtering: views and metric refs correctly included/excluded per arch.
- Label override: display name comes from view label when present, metric name
  otherwise.


## Stage 7 -- View file authoring

### New directory

```
analysis_configs/
  views/
    0000_top_stats.yaml
    0200_system_speed_of_light.yaml
    0400_roofline.yaml
    0600_command_processor.yaml
    0800_shader_processor_input.yaml
    1000_ta_td.yaml
    1200_local_data_share_lds.yaml
    1400_l1_address_processing.yaml
    cdna/
      0500_command_processor_cpc_cpf.yaml
      0700_wavefront.yaml
      1100_compute_units_compute_pipeline.yaml
      1500_l1_data_cache.yaml
      1600_l1_cache.yaml
      1700_l2_cache.yaml
      1800_l2_cache_per_channel.yaml
      1900_l2_fabric_interface.yaml
      2100_fabric_stall.yaml
      2200_fabric_stall_2.yaml
    rdna/
      0500_command_processor_cpc.yaml
      0700_workgroup_processor.yaml
      0900_l0_cache.yaml
      1100_compute_units.yaml
      1300_l2_cache.yaml
      1500_l1_data_cache.yaml
    gfx950/
      3000_mem_bw.yaml
```

The `views/` top level contains views shared across all architectures. `views/cdna/`
and `views/rdna/` contain family-specific views. `views/gfx950/` contains arch-specific
views (e.g., memory bandwidth analysis, present only on gfx950).

### Constraint: preserve current panel structure in Stage 7

Per reviewer feedback: the Stage 7 migration must preserve the current panel structure,
grouping, and ordering exactly. No user-facing changes to views during this
refactoring. This establishes a verified 1:1 mapping between old and new config
formats before any consolidation begins.

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
# views/cdna/1100_compute_units_compute_pipeline.yaml
view: "Compute Units - Compute Pipeline"
legacy_panel_id: 1100
tables:
  - title: "Compute Speed-of-Light"
    header:
      metric: Metric
      value: Avg
      unit: Unit
      peak: Peak
      pct_of_peak: Percent of Peak
    metrics:
      - id: compute.valu_flops
      - id: compute.mfma_flops_f8
        archs: [gfx940, gfx941, gfx942, gfx950]
      - id: compute.mfma_flops_bf16
      - id: compute.mfma_flops_f16
      - id: compute.mfma_flops_f32
      - id: compute.mfma_flops_f64
      - id: compute.mfma_iops_int8
  - title: "Pipeline Statistics"
    header:
      metric: Metric
      avg: Avg
      min: Min
      max: Max
      unit: Unit
    metrics:
      - id: compute.ipc
      - id: compute.ipc_issued
      - id: compute.salu_util
      - id: compute.valu_util
```

View files contain no formulas, units, peaks, or descriptions -- those live in Layer 2.
The view declares which metrics to show, in what order, and how to render them.

### Migration tooling

One-time script `tools/generate_views_from_panels.py`:

1. Read each current panel config YAML.
2. Extract grouping and display information (table titles, headers, metric ordering,
   `cli_style`).
3. Generate the corresponding view file, replacing metric definitions with `id`
   references (using the mapping established in Stage 4).
4. Preserve `label` overrides where the current display name differs from the
   canonical `name` -- the 234 cases of intentional name variation identified in the
   HLD's static analysis.
5. Preserve, as `description_note`, each sentence a panel adds to a metric's description
   about how the panel shows it (the library descriptions leave these out; Phase A
   lists them in the PR #13522 description review).

### Validation

- Golden-file test: load views -> produce `panel_configs` via `views_to_panel_configs()`
  -> compare to old `panel_configs` for each architecture.
- All metrics referenced by views exist in `MetricLibrary`.
- Label overrides are correct (checked against the 234 known cases).


## Stage 8 -- Cutover

### Modifications

Every production reader of the old analysis config tree moves to the new path:

| Reader | Where | New source |
|---|---|---|
| Analyze panel configs | `generate_configs()`, `rocprof_compute_analyze/analysis_base.py` | `views_to_panel_configs()` |
| `--list-metrics`, `--list-blocks` | `_build_arch_metric_list()`, `rocprof_compute_base.py` | `views_to_panel_configs()` |
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
library = load_metric_library(METRIC_LIBRARY_DIR, arch)
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
