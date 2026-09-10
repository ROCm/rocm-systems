# Phase 2 -- Display / View (Stages 5-8)

Parent: [LLD index](lld-index.md)


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
| `comparable` | bool | no | Whether table is comparable across runs |
| `metrics` | list[ViewMetricRef] | yes | Ordered list of metric references |

`ViewMetricRef` fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Layer 2 metric id |
| `archs` | list[str] | no | Metric-level arch filter |
| `label` | str | no | Display name override for this view |

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

### Dependencies

None. Can be developed in parallel with Phase 1 stages.


## Stage 6 -- Layer 3 reader

New module `src/utils/layer3_reader.py`.

`load_views(view_dirs: list[Path], arch: str, library: MetricLibrary) -> list[ViewDefinition]`:

1. Discover `*.yaml` files in each directory in `view_dirs`.
2. Load each via `yaml.safe_load` (no `!inherit` needed for view files -- they are
   flat by design).
3. Validate against `layer3_schema.py`.
4. Filter: skip views where `archs` is set and the current arch is not in the list.
   Within each view, skip metric refs where `archs` is set and the arch is not listed.
5. Resolve each metric `id` against `library.metrics` -- error if not found.
6. For each `ViewMetricRef`: if `label` is present, use it as display name; otherwise
   use `name` from the Layer 2 `MetricDefinition`.

`views_to_panel_configs(views: list[ViewDefinition], library: MetricLibrary) -> OrderedDict[int, dict]`:

Translates view objects into the same `OrderedDict[int, dict]` consumed by
`build_dfs()`. This replaces Stage 3's `layer2_to_panel_configs()` as the permanent
translation from the new format to the existing pipeline.

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

### Dependencies

Stages 2 and 5.


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
   `cli_style`, `comparable`).
3. Generate the corresponding view file, replacing metric definitions with `id`
   references (using the mapping established in Stage 4).
4. Preserve `label` overrides where the current display name differs from the
   canonical `name` -- the 234 cases of intentional name variation identified in the
   HLD's static analysis.

### Validation

- Golden-file test: load views -> produce `panel_configs` via `views_to_panel_configs()`
  -> compare to old `panel_configs` for each architecture.
- All metrics referenced by views exist in `MetricLibrary`.
- Label overrides are correct (checked against the 234 known cases).

### Dependencies

Stage 4 (metrics must be migrated before views can reference them) and Stage 5
(view schema).


## Stage 8 -- Adapter retirement

### Modifications

**`generate_configs()` in `analysis_base.py`** -- remove the feature flag. The
new path is the only path:

```python
library = load_metric_library(Path(config_dir), arch)
views = load_views(view_dirs, arch, library)
ac.panel_configs = views_to_panel_configs(views, library)
```

The call to `load_panel_configs()` for analysis configs is removed. (TUI mode's
config files under `rocprof_compute_tui/utils/` may still use `load_panel_configs()`
until separately migrated.)

**`detect_counters()` in `soc_base.py`** -- remove the old YAML-text-scanning path.
Counter detection goes through `MetricLibrary.get_counters_for_metrics()`
unconditionally.

**`get_build_in_vars()` in `utils_counter_defs.py`** -- remove the hardcoded
dicts. The function loads from `MetricLibrary.variables` unconditionally.

**`-b` / `--filter-blocks`** -- string metric IDs are the primary format. Numeric
IDs produce a deprecation warning with the equivalent string ID:

```
WARNING: Numeric metric ID '11.2.3' is deprecated.
         Use 'compute.salu_util' instead.
```

`convert_metric_id_to_panel_info()` remains during the deprecation window for
backward compatibility.

**`legacy_id`** -- removed from all Layer 2 metric definitions.

**`legacy_panel_id`** -- removed from all Layer 3 view files.

**Old panel config YAML files** -- removed from per-arch directories. The base arch
and `views/` directories are the canonical source.

**`compat_adapter.py`** -- removed.

### Tooling cleanup

| Tool | What changes |
|---|---|
| `validate_sets_metric_ids.py` | Remove numeric ID validation path. Only string IDs. |
| `verify_against_config_template.py` | Remove Panel Config template validation. Only Layer 2 schema validation. Remove `gfx9_config_template.yaml` and `gfx11_config_template.yaml`. |
| `metric_description_manager.py` | Remove old panel config reading path. |
| `hash_manager.py` | Remove old per-arch directory entries from hash database. Only base arch dirs and `views/`. |
| `format_yaml.py` | Remove Panel Config format handling if no longer needed. |

### Validation

- Full regression: entire test suite (`ctest` + Python tests) with the new path as
  the only path. Zero failures.
- Output equivalence: for every architecture, compare analyze text output to a
  golden file captured before migration.
- Counter extraction: for every architecture, compare `detect_counters()` output to
  golden file.
- Pre-commit hooks: all pass with the new file structure.
- Documentation pipeline: Sphinx-generated metric reference is identical to
  pre-migration version.

### Dependencies

Stages 3, 6, 7.
