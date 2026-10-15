# Phase 1 -- Metric Library (Stages 1-4)

Parent: [LLD index](lld-index.md)


## Stage 1 -- Layer 2 schema + inheritance loader

> Layer 1.5 collectables moved out of Phase A into a separate epic. The collectables
> section below is kept as input to that epic.

### Layer 2 metric schema

New developer tool `tools/validate_metric_library.py`, following the existing
`tools/validate_*.py` checks: it loads each arch's metric library through the
inheritance loader, collects every schema error, and exits 1 if there are any. It runs
as a pre-commit hook on `metric_library/` files and from
`master_config_workflow_script.py --validate-only`. It lands with the first metric
library files (Stage 4), not with the loader. As with today's panel files, the runtime
does not re-validate the files it loads.

`MetricDefinition` fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Stable hierarchical string, e.g. `mem.l2_hit_rate` |
| `name` | str | yes | Canonical display name, e.g. `"L2 Cache Hit Rate"` |
| `formula` | str | yes | Base expression without aggregation wrappers; never empty, in the canonical form of `tools/format_yaml.py`. A metric an arch cannot compute does not exist on that arch; a later arch adds it. |
| `unit` | str | yes | One canonical spelling per unit (e.g. `"GB/s"`, `"GFLOP/s"`; aliases such as `Gbps` are rejected); a display scale such as Bytes/s is a view concern (D8 Kind 2); may be empty |
| `description` | str | yes | Human-readable metric description; may be empty |
| `peak` | str (formula) | no | Performance ceiling expression. Required for speed-of-light metrics. |
| `coll_level` | str | no | Accumulation level, e.g. `SQ_LEVEL_WAVES` |
| `avg_mode` | str | no | Phase 2, with runtime wrapper generation; not in Phase A. Default `weighted` (`SUM(X)/SUM(Y)`). Set to `simple` for metrics that use `AVG()` instead. |

**Aggregation is not a metric field.** The metric defines a single base formula
without aggregation wrappers (`SUM`, `AVG`, `MIN`, `MAX`). Aggregation is a display
concern owned by Layer 3 -- the view's `header` dict declares which columns to show
(e.g., `avg`, `min`, `max`, or `value`), and the code auto-generates the wrapped
formulas at evaluation time.

This eliminates the current duplication where ~98% of metrics repeat the same base
formula three times with different wrappers. The `simple_box` tables already prove
this pattern works -- they auto-generate `MIN/Q1/MEDIAN/Q3/MAX` from a single `expr`.

Percent-of-peak is not a schema field. In the target state it is computed automatically
when a metric has a `peak` formula and `unit: Percent`. During the transition, the
analysis configs keep today's explicit `pct_of_peak` per table row, so output does not
change.

A metric's numeric positional ID (`legacy_id`) is not a metric field either. The same
metric appears in several tables, at a different position in each, so the positional IDs
are positions in today's analysis configs (see
[Transition mechanism](lld-index.md#transition-mechanism-legacy_id)). The metric library
does not say where a metric is shown; in Phase A the analysis configs stay the display
source, and the Phase 2 views place metrics by id.

A metric concept has one `id` on every architecture. Each family base (gfx908 for CDNA,
gfx115x for RDNA, and gfx1250) defines the id with that family's formula, as the HLD's
end-to-end example does for `mem.l2_hit_rate`. Within one architecture, every appearance
of an id uses the same formula.

A file maps each metric `id` to its fields, so the `id` is the key rather than a field:

```yaml
compute.salu_util:
  name: "SALU Utilization"
  formula: "100 * SQ_ACTIVE_INST_SCA / ($GRBM_GUI_ACTIVE_PER_XCD * $cu_per_gpu)"
  unit: Percent
  description: >-
    The percentage of time the SALU was busy executing instructions.

compute.valu_flops:
  name: "VALU FLOPs"
  formula: "64 * (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16 + ...) / (End_Timestamp - Start_Timestamp)"
  unit: GFLOPs
  peak: "$max_sclk * $cu_per_gpu * 64 * 2 / 1000"
  description: >-
    The total number of vector ALU floating-point operations per second.
```

**Typed row fields** of fabric stall and per-channel tables (`xfer`, `coherency`,
`expr`, `type`, `transaction`, `read req`, `ea read stall - hbm`, ...) are display
columns, so they stay as display data in the analysis config rows, not on
`MetricDefinition`. There is no separate `FabricStallMetric` schema.

`VariableDefinition` fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | str | yes | The key: variable name without `$`, e.g. `GRBM_GUI_ACTIVE_PER_XCD` |
| `formula` | str | yes | Expression, e.g. `"(GRBM_GUI_ACTIVE / $num_xcd)"` |

Evaluation order stays in code: `calc_builtin_vars()` evaluates the `PER_XCD` variables
first.

```yaml
# gfx908/variables.yaml
GRBM_GUI_ACTIVE_PER_XCD:
  formula: "(GRBM_GUI_ACTIVE / $num_xcd)"
GRBM_COUNT_PER_XCD:
  formula: "(GRBM_COUNT / $num_xcd)"
GRBM_SPI_BUSY_PER_XCD:
  formula: "(GRBM_SPI_BUSY / $num_xcd)"
numActiveCUs:
  formula: >-
    TO_INT(MIN(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) /
    SUM($GRBM_GUI_ACTIVE_PER_XCD), 0) / $max_waves_per_cu * 8 +
    MIN(MOD(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) /
    SUM($GRBM_GUI_ACTIVE_PER_XCD), 0),
    $max_waves_per_cu), 8), $cu_per_gpu))
kernelBusyCycles:
  formula: "ROUND(AVG((((End_Timestamp - Start_Timestamp) / 1000) * $max_sclk)), 0)"
hbmBandwidth:
  formula: "($max_mclk / 1000 * 32 * $num_memory_channels)"
```

### Layer 1.5 -- Collectables

A collectable is a formula fragment whose counters fit in a single hardware pass. It
sits between raw PMC counters (Layer 1) and full metric formulas (Layer 2) in the
layer hierarchy.

#### The hardware constraint

Each IP block has a per-pass counter limit enforced by the hardware. On gfx942:

| IP Block | Max counters per pass |
|---|---|
| SQ | 8 (includes SQC and SP via BLOCK_REMAP) |
| TCC | 4 |
| TCP | 4 |
| TA | 2 |
| TD | 2 |
| GRBM | 2 |

Metric formulas routinely exceed these limits:

- **VALU FLOPs** (single metric): references 12 SQ counters (`SQ_INSTS_VALU_ADD_F16`,
  `SQ_INSTS_VALU_MUL_F16`, ..., `SQ_INSTS_VALU_FMA_F64`). SQ limit is 8 -- this one
  metric must span at least 2 hardware passes.
- **System Speed-of-Light panel**: ~39 SQ/SQC counters total -> 5+ SQ-limited passes.
- **L2 Cache panel**: 32 TCC counters / 4 per pass = 8+ TCC-limited passes.
- **vL1D Cache panel**: 34 TCP counters / 4 per pass = 9+ TCP-limited passes.

The current system handles this by binning counters across passes via
`perfmon_coalesce()`, then evaluating the full formula against merged counter data
from all passes. This produces incorrect results for ratio metrics.

#### Why cross-pass evaluation is mathematically wrong

**Concrete example: L2 Cache Hit Rate on gfx942**

Formula: `100 * TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`

The invariant: in any single kernel execution, `TCC_HIT + TCC_MISS = total requests`,
so `0% <= HitRate <= 100%`.

Suppose TCC_HIT_sum and TCC_MISS_sum land in different passes (TCC limit = 4 counters,
and other TCC counters consumed the remaining slots). Consider a GEMM kernel touching
a large matrix:

| | TCC_HIT | TCC_MISS | True hit rate |
|---|---|---|---|
| Pass 1 (cache cold at start) | 500,000 | 300,000 | 62.5% |
| Pass 2 (cache warm from pass 1) | 750,000 | 50,000 | 93.75% |

The cross-pass formula uses TCC_HIT from pass 1 and TCC_MISS from pass 2:

```
HitRate_cross = 100 * 500,000 / (500,000 + 50,000) = 90.9%
```

This result (90.9%) is neither the pass-1 truth (62.5%) nor the pass-2 truth (93.75%).
It describes no actual kernel execution.

**Formal statement:** Let H(i) and M(i) be hit and miss counts in pass i. The true
hit rate for pass i is R(i) = H(i) / (H(i) + M(i)). Cross-pass collection computes:

```
R_cross = H(1) / (H(1) + M(2))
```

R_cross = R(1) only when M(2) = M(1), i.e., when the workload is perfectly
deterministic across passes. For any workload with state-dependent behavior (cache
warming, OS scheduling jitter, memory allocation differences), M(2) != M(1) and
the error is:

```
|R_cross - R(1)| = |H(1) * (M(1) - M(2))| / ((H(1) + M(1)) * (H(1) + M(2)))
```

**Pathological case:** For the TCP hit rate formula `100 * (REQ - MISS) / REQ`: if
REQ is measured in pass 1 (low-traffic phase, REQ = 1000) and MISS in pass 2
(high-traffic phase, MISS = 2000), the result is `100 * (1000 - 2000) / 1000 = -100%`
-- a physically impossible value.

#### Existing partial mechanisms

| Mechanism | What it does | Why it is insufficient |
|---|---|---|
| `same_bucket_priority_metric_ids` | Greedy bin-packing that tries to keep priority metrics' counters in one pass | Falls through silently when counters don't fit; only 4 metrics configured (gfx115x only) |
| `_metric_aware_coalesce_pass` | Best-effort: sorts metrics by counter count, tries to pack each metric's counters into one bucket | No guarantee; logs a debug message and defers to first-fit when it fails |
| `coll_level` | Marks accumulator counters that need dedicated pass files | Only for ACCUM counters, not for general ratio metrics |
| `NOISE_CLAMP` | Display-time clamp that suppresses results with high noise indicators | Mitigation, not prevention -- the wrong value is still computed |

#### What a collectable formalizes

A collectable declares a set of counters that form an **atomic collection unit** --
they must be collected in the same hardware pass to produce a mathematically correct
intermediate result.

```yaml
# gfx908/collectables.yaml
collectables:
  l2_hit_miss:
    id: collect.l2_hit_miss
    formula: "TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)"
    # counters: [TCC_HIT_sum, TCC_MISS_sum]  -- inferred from formula
```

A Layer 2 metric can reference a collectable by id:

```yaml
L2 Cache Hit Rate:
  id: mem.l2_hit_rate
  name: "L2 Cache Hit Rate"
  formula: "100 * $collect.l2_hit_miss"
  unit: Percent
  peak: 100
  description: >-
    The ratio of cache line requests that hit in the last-level on-chip cache.
```

#### Scope in this LLD

This stage defines the collectable **schema and config structure** so that
collectables have a well-defined home in the layer architecture. Layer 2 metrics can
reference collectables by id in their formulas. During counter extraction,
`get_counters_for_metrics()` resolves collectables transitively to their underlying
Layer 1 counters.

The evaluation pipeline (`eval_metric()`, `build_eval_string()`, `MetricEvaluator`)
is not changed in this LLD. Currently, collectables expand inline like built-in
variables -- they are a formula organization mechanism. Per-pass intermediate
evaluation (collecting a collectable's counters together and evaluating its formula
within that pass) is a future enhancement requiring a separate design.

The bin-packing pipeline (`perfmon_coalesce()`) is not changed. Collectable-aware
counter grouping is a natural follow-up once per-pass evaluation is implemented.

`CollectableDefinition` fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Stable string, e.g. `collect.l2_hit_miss` |
| `formula` | str | yes | References Layer 1 counters only |
| `counters` | list[str] | no | Explicit counter list for pass-limit validation (inferred from formula if absent) |

The `MetricLibrary` (Stage 2) loads collectables from `collectables.yaml` alongside
metrics and variables. `get_counters_for_metrics()` resolves collectable references
in the same way it resolves built-in variables -- transitively extracting the
underlying Layer 1 counters from the collectable's formula.

**Target state for SDK integration:** The HLD notes that collectables should
ultimately be integrated into `sdk_config.yaml` as part of the SDK counter
definitions. This requires collaboration with the SDK team and is not part of this
LLD. Until then, collectables are defined separately by rocprof-compute in the
base arch YAML files.

### Inheritance loader

New module `src/utils/yaml_inherit_loader.py`. A metric library file names its parent
with a top-level `inherit: <path>` key. (A `!inherit <path>` tag on the first line, as
first proposed, is not valid YAML: a document cannot hold a tagged scalar followed by a
mapping.)

GitLab CI's `extends` keyword uses the same approach -- deep merge across files,
multi-level inheritance, later values override earlier ones.
See [GitLab YAML optimization docs](https://docs.gitlab.com/ci/yaml/yaml_optimization/).
Native YAML anchors (`&`/`*`) and merge keys (`<<`) are insufficient -- they only
perform shallow merge, do not work across files, and the merge key is semi-deprecated
in the YAML spec.

**Implementation:** The merge logic is inline, with no new dependency. It must work in
profile mode, which prohibits pip dependencies (`ProfileModeImportGuard` enforces
stdlib-only imports), so yaml is imported from the vendored copy
(`from vendored import yaml`).

`InheritLoader` subclasses `yaml.SafeLoader`. It registers one custom tag, `!remove`,
which marks a key for deletion during the merge, and it rejects a key that appears twice
in one mapping (plain YAML would silently keep the last one, hiding a duplicate metric
id). The repo's `check-yaml`
pre-commit hook needs `--unsafe` for the metric library files because of this tag.

To load a file:

1. Resolve the `inherit` path relative to the current file's directory.
2. Load the parent file recursively (supporting chains:
   `gfx950 -> gfx942 -> gfx940 -> gfx90a -> gfx908`).
3. Merge the current file's content on top of the parent, applying `!remove` as it
   goes. Removing a key that the parent does not have is an error.

**Merge operations:**

| Operation | Input | Behavior |
|---|---|---|
| Modify | Scalar in override | Override replaces base value. |
| Add | New key in override | Appended to the merged map. Supports adding new metrics, tables, or fields that the parent arch does not have. |
| Remove | `key: !remove` | The key is deleted from the merged result. Used when a child arch drops a metric (keyed by its id) or a field of one. |
| Deep merge | Map + map | Recursive deep merge. Override keys win; base-only keys preserved, in file order. A metric in a child file lists only the fields that change. |
| Replace | List in override | A list replaces the base list as a whole. |

Example of all three operations:

```yaml
# gfx90a/sq.yaml
inherit: ../gfx908/sq.yaml

# Modify: change the formula for an existing metric
compute.valu_flops:
  formula: "64 * (SQ_INSTS_VALU_ADD_F16 + ...) / (End_Timestamp - Start_Timestamp)"

# Add: new metric not present in gfx908
compute.mfma_flops_bf16:
  name: "MFMA FLOPs (BF16)"
  formula: "512 * SQ_INSTS_VALU_MFMA_MOPS_BF16 / (End_Timestamp - Start_Timestamp)"
  unit: GFLOPs
  description: >-
    BF16 matrix fused multiply-add operations per second.

# Remove: metric that existed in gfx908 but is not valid for gfx90a
compute.some_deprecated_metric: !remove

# Remove a field: drop the parent's peak
compute.salu_util:
  peak: !remove
```

**Cycle detection:** Maintain a set of resolved absolute paths during the recursive
chain. If a path appears twice, raise `InheritanceCycleError` with the full chain.

**No runtime switch in Phase A:** Production code does not switch between the old and
new paths during Phase A -- no environment variable or hidden option. The Layer 2 path
is exercised by tests only (Stage 3 validation) until the cutover.

### Tests

- Schema validation (`tools/validate_metric_library.py`): valid files pass; missing
  required fields, unknown fields, bad ids, duplicate ids and a child file that does not
  inherit from its parent's file are reported.
- Inheritance loader: single-level inheritance, multi-level chain (4+ deep, matching
  the real `gfx908 -> gfx90a -> gfx940 -> gfx942` chain), cycle detection error.
- Merge operations: key modification (override), key addition (new metric in child),
  removal of a metric or a field (`!remove` deletes from merged result), deep-merge of
  maps, lists replaced, and errors for removing a missing key, for a duplicate key and
  for `!remove` with a value.
- Round-trip: a minimal synthetic YAML family (base + chain of overrides with
  additions, modifications, and removals) loads correctly and produces the expected
  merged structure.
- Golden comparison infrastructure: the Stage 3 equivalence test is a new testing
  pattern for the project (a snapshot test whose reference side is today's tree, read
  in the same run).

### Dependencies

None. This is the foundation stage.


## Stage 2 -- Layer 2 parser

### MetricLibrary

New module `src/utils/metric_library.py`.

`load_metric_library(metric_dir: Path, arch: str) -> MetricLibrary`:

1. Resolve the architecture's base arch and inheritance chain:
   ```python
   METRIC_LIBRARY_PARENT_ARCH = {
       "gfx908": None,         # CDNA base -- no parent
       "gfx90a": "gfx908",
       "gfx940": "gfx90a",     # each arch inherits from its immediate predecessor
       "gfx941": "gfx940",
       "gfx942": "gfx940",
       "gfx950": "gfx942",
       "gfx115x": None,        # RDNA base -- no parent
       "gfx1250": None,        # own base -- shares no file with gfx115x
   }
   ```
   gfx1150-gfx1153 map to the `gfx115x` directory through `canonical_config_arch()`.
2. Load YAML files from `{metric_dir}/{arch}/`. Each file either stands alone (base
   arch) or has an `inherit` key pointing to the parent arch's file.
3. The `InheritLoader` (Stage 1) resolves the `inherit` chain recursively and
   deep-merges overrides on top of the base.
4. Do not re-validate: `tools/validate_metric_library.py` checks the files at commit
   time.
5. Collect all metrics into `MetricLibrary.metrics`, keyed by string `id`. Raise on
   duplicate ids. In Phase A the entries are the loaded mappings, as
   `load_panel_configs()` returns today; a typed model can come with the cutover.
6. Load `variables.yaml` (inherited like the metric files) into
   `MetricLibrary.variables`, in file order.

7. Load `sets.yaml` from the arch directory (inherited along `METRIC_LIBRARY_PARENT_ARCH` like the
   metric files) into `MetricLibrary.sets: dict[str, SetDefinition]`. Sets only reference
   metric ids; see [Sets migration](#sets-migration).

Collectables (Layer 1.5, separate epic) are not loaded in Phase A.

`MetricLibrary` class:

```python
class MetricLibrary:
    metrics: dict[str, dict]        # metric id -> its fields
    variables: dict[str, dict]      # variable name -> {formula}, in file order
    sets: dict[str, SetDefinition]  # Stage 4, sets migration

    def get_counters_for_metrics(
        self, metric_ids: list[str], gpu_series: str
    ) -> set[str]: ...
```

`get_counters_for_metrics()` extracts HW counters from the requested metrics:

1. **Pre-expand collectable references** (once the collectables epic lands): Replace `$collect.xxx` in formulas with the
   collectable's own formula before passing to the regex extractor. This is necessary
   because `VARIABLE_RE` in `utils_counter_defs.py` stops at the dot -- `$collect.l2_hit_miss`
   would be parsed as variable `collect` + stray text `.l2_hit_miss`.
2. **Concatenate formula strings** for the requested metrics (including expanded
   collectables).
3. **Include `coll_level` and `peak` values** in the text passed to the extractor. The current
   `detect_counters()` scans raw YAML text which incidentally picks up
   `coll_level: SQ_LEVEL_WAVES` as a counter name. The MetricLibrary approach must
   explicitly add `coll_level` values to maintain equivalence. The raw-text scan also
   picks up counter names mentioned only in YAML comments or descriptions (on gfx115x
   and gfx1250); the formula-based extraction does not, and the equivalence test lists
   those names as known differences.
4. **Delegate to `extract_counters_and_variables()`** in `utils_counter_defs.py` for
   regex-based HW counter extraction and transitive built-in variable resolution.

### Tests

- Integration test: load a synthetic family chain (`gfx908/` -> `gfx940/` ->
  `gfx942/`) from test fixtures. Verify: correct number of metrics, inheritance
  overrides applied, variable definitions present, duplicate id raises error.
- Counter extraction equivalence: `get_counters_for_metrics()` returns the same
  counter set as `extract_counters_and_variables()` when given the same formula text.
- String metric IDs follow the hierarchical convention
  (`{category}.{subcategory}_{name}`).

### Dependencies

Stage 1.


## Stage 3 -- Equivalence with the analysis configs

Phase A ships no compatibility adapter. The metric library says what each metric is, not
where it is shown: the analysis configs stay the display source and the production
loader, and nothing links a panel row to a metric id until the Phase 2 views do. The
adapter to the existing pipeline comes with the views, as `views_to_panel_configs()`
([Phase 2](lld-phase2-display-view.md)). A Phase A adapter would need a full copy of the
panel layout with each row linked to its metric, kept in sync with the analysis configs
until Stage 8 deletes both; nothing in production would read it.

### Integration point

In Phase A nothing in production changes. `generate_configs()` in `analysis_base.py`
keeps calling `load_panel_configs()`, `detect_counters()` in `soc_base.py` keeps scanning
the panel YAMLs, and `get_build_in_vars()` keeps its hardcoded dicts. Production switches
once, at Stage 8, through the Layer 3 views
([Stage 8](lld-phase2-display-view.md#stage-8----adapter-retirement)).

The Layer 2 files live in a sibling tree, `src/rocprof_compute_soc/metric_library/`
(`default_metric_library_dir()`), not in `analysis_configs/{arch}/`. The old loader
requires a `Panel Config` key in every `*.yaml` of an arch directory, so mixing the two
formats there would break it. A user-supplied `--config-dir` keeps using the old loader.

At the cutover, counters come from the metrics that the selected views place
(`get_counters_for_metrics()`). Phase A checks only the no-filter counter set, because
`-b` tokens and sets select rows, and rows are not linked to metric ids yet.

### Validation

`tests/integration/test_metric_library_equivalence.py`, for every arch that has a metric
library tree, compares sets with today's analysis configs:

- **Formulas:** every equation a row shows (its `avg`, `value` or `expr` form, or each
  column of a multi-column row, without display-only `ROUND()`) is a library formula,
  except rows listed as display variants by positional id and label (another unit, or
  the same equation written differently); every library formula is shown by some row.
- **Counters:** the counters the whole library selects equal the counters
  `detect_counters()` selects with no filter (with `--membw-analysis` where the arch has
  panel 3000).
- **Built-in variables:** the library variables equal `get_build_in_vars()`.

The reference side is today's tree, read through the production loaders in the same run,
so there are no snapshot files. The test runs in ctest and as a pre-commit hook on the
analysis configs, the metric library files and the modules they use; the schema itself is
checked by `tools/validate_metric_library.py`.

### Dependencies

Stage 2.


## Stage 4 -- Metric migration

The largest stage. Parallelizable by hardware concept (compute, memory, system) and by
architecture. The deliverable is the actual Layer 2 YAML files and the migration of
built-in variables and sets.

### New directory structure

```
metric_library/          # sibling of analysis_configs/ (see Stage 3)
  gfx908/                # CDNA base (first arch) -- complete standalone definitions
    variables.yaml
    sets.yaml            # Stage 4, sets migration
    cp.yaml              # CPC/CPF
    spi.yaml
    sq.yaml              # compute.* (instructions, execution) and wave.* (wave state)
    dispatch.yaml        # launch parameters and kernel time from the dispatch record
    grbm.yaml
    lds.yaml
    sqc_icache.yaml      # L1I
    sqc_dcache.yaml      # scalar L1D
    ta_td.yaml
    tcp.yaml             # vector L1D
    tcc.yaml             # L2, L2-Fabric, per channel
    derived.yaml         # results that combine blocks (arithmetic intensity)
  gfx90a/                # inherits gfx908, overrides only what differs
  gfx940/                # inherits gfx90a
  gfx941/                # inherits gfx940
  gfx942/                # inherits gfx940
  gfx950/                # inherits gfx942
  gfx115x/               # RDNA base -- complete standalone definitions
  gfx1250/               # own base -- complete standalone definitions
```

Files are grouped by hardware block, as the HLD's Layer 2 asks ("one file per hardware
concept"), not by today's panels: a summary panel such as Speed-of-Light has no hardware
meaning, and where a metric is shown is a Layer 3 concern. A metric lives in the file of
its id category (table below). The RDNA and gfx1250 bases add their own block files.

No abstract `_base/` directories. The first architecture in each family (gfx908 for
CDNA, gfx115x for RDNA, and gfx1250, which shares no file with gfx115x) serves as the
base -- its files are complete, standalone
definitions. Each subsequent arch inherits from its immediate predecessor in the
hardware lineage, using additions, modifications, and removals (`!remove`) to
express only what changed. This chain-based pattern applies to future families
(gfx12xx).

### Migration order

1. **gfx908** -- the CDNA base. Convert its current panel config files to Layer 2
   format with string IDs, single formulas, and descriptions. This is the largest
   single-arch conversion but produces the foundation for the inheritance chain.
2. **gfx90a** -- inherits gfx908. Override files for metrics added or modified in MI200.
3. **gfx940** -- inherits gfx90a. MI300 generation -- adds MFMA FLOPs variants,
   modifies formulas for new counter names.
4. **gfx941** -- inherits gfx940. Differs in 2 files (one unit override and three
   roofline formulas). The smallest override, ideal for validating the inheritance
   mechanism.
5. **gfx942, gfx950** -- inherits gfx940/gfx942 respectively.
6. **gfx115x** -- RDNA base. Separate standalone definitions, no inheritance from CDNA.
7. **gfx1250** -- own base. Separate standalone definitions.

Each step is validated: run the Stage 3 equivalence test for the migrated architecture
before proceeding to the next.

### Metric ID assignment

Convention: `{category}.{subcategory}_{name}`, all lowercase, underscores for
word separation.

Categories name hardware (or the data source), never a view: `system`, `memchart`,
`roofline` and `wavefront` (today's panels) are not categories, and the validator rejects
them. A metric's category follows the block of the counters it uses, not the panel that
shows it: a metric shown only in a summary panel takes the category of the block it
measures, and SQ instruction counters (`SQ_*INST*`) are `compute`.

**Identity across archs (row continuity).** A child arch keeps its parent's id for every
metric it still shows; a changed equation is an override of that id, never a new id. Two
placements of one id that disagree within an arch are a Kind 3 case for review, and two
ids that come to show the same equation are reported, not merged.

| Category | Hardware | File | Example IDs |
|---|---|---|---|
| `cp` | CPC/CPF | `cp.yaml` | `cp.cpf_util`, `cp.cpc_stall_rate` |
| `spi` | SPI | `spi.yaml` | `spi.workgroups_per_cu` |
| `compute` | SQ instruction issue and execution (`SQ_*INST*`, execution-unit busy, CU busy) | `sq.yaml` | `compute.valu_util`, `compute.mfma_util`, `compute.cu_util`, `compute.inst_mix_valu` |
| `wave` | SQ wave state: occupancy, wave cycles and their wait/active breakdown, save/restore | `sq.yaml` | `wave.occupancy`, `wave.cycles`, `wave.issue_wait_cycles` |
| `dispatch` | The dispatch record (launch parameters, resource allocation, kernel time); no counters | `dispatch.yaml` | `dispatch.grid_size`, `dispatch.vgprs`, `dispatch.kernel_time` |
| `grbm` | GRBM | `grbm.yaml` | `grbm.gui_active_cycles` |
| `lds` | LDS | `lds.yaml` | `lds.util`, `lds.theoretical_bw` |
| `mem.l1i` | SQC instruction cache | `sqc_icache.yaml` | `mem.l1i_hit_rate_incl_dup`, `mem.l1i_fetch_lat` |
| `mem.sl1d` | SQC scalar data cache | `sqc_dcache.yaml` | `mem.sl1d_bw`, `mem.sl1d_l2_rd_bw` |
| `ta` | TA (address processing) | `ta_td.yaml` | `ta.global_insts`, `ta.busy` |
| `td` | TD (data return) | `ta_td.yaml` | `td.busy`, `td.spi_stall` |
| `mem.l1d` | TCP (with TA inputs, e.g. coalescing: AMD defines the vL1D as TA + TC + TD) | `tcp.yaml` | `mem.l1d_cache_bw`, `mem.l1d_l2_rd_bw` |
| `mem.l2`, `mem.l2_chan` | TCC, per channel | `tcc.yaml` | `mem.l2_hit_rate`, `mem.l2_chan_hit_rate_per_chan` |
| `mem.fabric`, `mem.hbm` | TCC-EA (Infinity Fabric, HBM) | `tcc.yaml` | `mem.fabric_rd_lat`, `mem.hbm_bw` |
| `derived` | Results that combine blocks | `derived.yaml` | `derived.ai_hbm`, `derived.ai_l2` |

The numeric positional IDs (e.g. `"11.2.2"`) are row positions in `analysis_configs`;
the metric library does not store them.

### OQ3 resolution -- metrics with both description and formula drift

The HLD analysis counted 39 such metrics by grouping on metric name alone. Grouping by
(table title, metric name), so that generic names such as `Utilization` or `Req` do not
merge unrelated metrics, gives 17 groups at `f8fac5575b`. Every one of them involves
gfx115x or gfx1250.

**Resolution:** a metric concept has one `id` on every architecture, and each family base
defines it with that family's formula (see Stage 1). The comparison criteria below are
still useful for review: where a family's formula measures something noticeably
different (for example, gfx115x `VALU FLOPs` counts lane-ops), the id stays shared and
the difference is tracked as a formula issue, not as a separate id.

**Within one architecture**, every appearance of an id uses the same formula. Where
today's YAMLs differ between tables:

- **A different quantity under the same name** (e.g. `Wavefront Occupancy` as a count in
  table 201 and as a percent in 301) gets separate ids.
- **The same quantity at a different unit scale** (e.g. `L2-Fabric Read BW` in GB/s vs
  Bytes/s) keeps one id with one uniform definition (formula and unit). A different
  display unit is a view concern: the scaled form stays in the analysis configs until
  Phase 2, where a view may request its own display unit (see `ViewMetricRef.unit`).
- **The same quantity with a different formula** (e.g. the gfx1250 cache hit rates,
  HITS/REQ vs HITS/(HITS+MISSES)) is unified to one formula in both the old and the new
  files. Each case is approved individually and listed in the CHANGELOG, because it
  changes displayed values.

### OQ4 resolution -- same description under different names

The HLD analysis counted 43 groups. Recounted at `f8fac5575b` with the grouping above, 30
descriptions are shared by more than one name.

**Resolution criteria:**

1. Select the **most specific, unambiguous name** as the canonical `name` in Layer 2.
   Prefer names that self-document the metric's scope without being excessively verbose.
2. All other names become view-level `label` overrides in Layer 3. Until then they stay
   the row names in the analysis configs.

| Description group | Current names | Canonical `name` |
|---|---|---|
| L2 cache hit ratio | `Cache Hit`, `Hit Rate`, `L2 Cache Hit Rate` | `L2 Cache Hit Rate` |
| Read latency | `L2-Fabric Read Latency`, `Read Latency` | `L2-Fabric Read Latency` |
| INT8 MFMA ops | `MFMA IOPs (INT8)`, `MFMA IOPs (Int8)` | `MFMA IOPs (INT8)` |

Two groups share a description only because of documentation errors: `L1I Hit Rate` and
`L1I BW` have swapped descriptions on gfx908-gfx942, and `Dependency Wait Cycles` reuses
the `Wave Cycles` description. These keep separate ids.

### Sets migration

A set is a Layer 2 concept: its membership is constrained by hardware (the counters of
its metrics must fit in a single pass), which is checked against the metric formulas.
How a set's results are displayed is a Layer 3 concern (see
[Set display](lld-phase2-display-view.md#set-display)).

Sets move from `profile_configs/sets/{arch}_sets.yaml` to
`metric_library/{arch}/sets.yaml` and switch from numeric positional IDs to string metric
IDs:

```yaml
# Before
sets:
- title: Compute Throughput Utilization
  set_option: compute_thruput_util
  description: Placeholder
  metric:
  - 11.2.2: SALU Utilization
  - 11.2.3: VALU Utilization

# After
sets:
- title: Compute Throughput Utilization
  set_option: compute_thruput_util
  description: Placeholder
  metric:
  - compute.salu_util
  - compute.valu_util
```

- **References only.** A set entry references a metric id and carries no metric fields,
  so a metric is still defined exactly once per architecture. Every referenced id must
  exist on that architecture.
- **Inheritance.** `sets.yaml` follows the same `inherit` chain as the metric files; for
  example, gfx941 and gfx942 use gfx940's sets unchanged.
- **Single-pass check.** Validation uses `get_counters_for_metrics()` to confirm that each
  set's counters fit in one pass.
- **Transition.** Production `--set` and `--list-sets` keep reading
  `profile_configs/sets/` until the cutover. The equivalence tests prove that each
  string-ID set yields the same counters as the numeric set it replaces.

### Built-in variable migration

The built-in variables move into each base arch's `variables.yaml`. In Phase A,
`get_build_in_vars()` in `utils_counter_defs.py` keeps its hardcoded dicts, and the
Stage 3 equivalence test asserts that the library variables equal `get_build_in_vars()`
for every arch.
At the cutover, `get_build_in_vars()` loads from `MetricLibrary.variables` and the
hardcoded dicts are removed.

The evaluation pipeline (`calc_builtin_vars()` in `evaluation_pipeline.py`) is
unchanged -- it receives the same `dict[str, str]` regardless of source.

### Tooling updates

| Tool | What changes |
|---|---|
| `validate_sets_metric_ids.py` | Validate `metric_library/{arch}/sets.yaml`: references resolve through `MetricLibrary`, and each set fits one pass. The numeric files keep their positional check until the cutover. |
| `format_yaml.py` | Also checks `metric_library/gfx*` files (its pre-commit hook covers them): the `formula` and `peak` keys, with the `!remove` tag accepted. Core equation formatting logic (factoring constants out of aggregation) is unchanged. Done in Stage 1 (task 03). |
| `hash_manager.py` | Include base arch directories in hash database. Path discovery expands to cover `gfx908/*.yaml` and `gfx115x/*.yaml` as base arch files. |
| `verify_against_config_template.py` | No change: `tools/validate_metric_library.py` checks the metric library files, and this script keeps checking the old panel files until the cutover. |
| `metric_description_manager.py` | No change in Phase A: the old panel YAMLs stay the documentation source until the cutover (Stage 8), when it reads descriptions from Layer 2. Output format for docs YAMLs is unchanged. |

### Migration tooling

A one-time local script converts each arch. It is not committed; each PR commits its
output unedited and says how the files were generated.

1. Read each current panel config YAML for a given architecture.
2. For each metric: generate the Layer 2 format with string `id`, canonical `name`,
   formula, `unit` and `description`.
3. Report every metric whose formula differs between tables of the same architecture,
   for the review described under OQ3.
4. Generate the family base files and arch override files.
5. Keep its row -> metric id decisions locally, keyed by positional id, for the sets
   migration and for generating the Phase 2 views.

The generated files are checked by the Stage 3 equivalence test and by
`tools/validate_metric_library.py`; there is no round trip through an adapter.

### Validation

- Per-arch equivalence test (Stage 3): formulas, no-filter counters and built-in
  variables match today's analysis configs.
- Schema check: `tools/validate_metric_library.py` passes.

### Dependencies

Stages 1, 2. Stage 3 needed for validation (equivalence test).
