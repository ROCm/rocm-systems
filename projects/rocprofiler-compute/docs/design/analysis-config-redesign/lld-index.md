# Implement three-layer analysis config separation

## Motivation

rocprofiler-compute's analysis pipeline consumes a single monolithic `Panel Config`
YAML format. `load_panel_configs()` returns `OrderedDict[int, dict]` to `build_dfs()`,
which builds DataFrame templates with formula strings, and `eval_metric()` evaluates
them against profiled counter data. This format interleaves metric specification, display
configuration, documentation, and grouping -- every change to any concern touches files
that carry all four.

The parent HLD ([`hld-analysis-config-redesign.md`](hld-analysis-config-redesign.md))
defines a three-layer architecture to separate them. This LLD specifies how each layer
connects to the existing pipeline, how the Phase B compatibility adapter translates
between old and new formats, and what each self-contained PR delivers.

**Scope:** Layers 2 and 3 -- metric library, display/view, and the compatibility adapter.
Layer 1.5 (collectables) is designed in the
[single-pass metrics plan](../single-pass-and-multi-pass-metrics/hld-plan.md#terms); the
metric library holds each arch's collectables in `collectables.yaml`.
Layer 1 (`sdk_config.yaml`) is out of scope. The metric formula evaluation engine (`eval_metric`, `build_eval_string`, `MetricEvaluator`) is unchanged --
only where formulas are defined and how they reach the engine changes.

Built-in variables (`$GRBM_GUI_ACTIVE_PER_XCD`, `$numActiveCUs`, ...) are collectables
in the metric library
([Built-in variable migration](lld-phase-a-metric-library.md#built-in-variable-migration)).

**Phases.** The design docs use these two names:

- **Phase A** is Stages 1-4: the metric library, its equivalence validator, sets and
  collectables. Production does not change.
  Detail: [Phase A -- Metric Library](lld-phase-a-metric-library.md).
- **Phase B** is Stages 5-8: the views, `views_to_panel_configs()`, the cutover, and
  string ids in `-b`.
  Detail: [Phase B -- Display / View](lld-phase-b-display-view.md), which also plans
  the view consolidation after Stage 7.


## Pipeline before the cutover

### Analyze mode

```mermaid
flowchart LR
    YAML["Panel Config YAML<br/>(analysis_configs/{arch}/*.yaml)"]
    LPC["load_panel_configs()<br/>(utils_common.py)"]
    BD["build_dfs()<br/>(parser.py)"]
    AC["ArchConfig<br/>(panel_configs + DataFrames)"]
    EM["eval_metric()<br/>(evaluation_pipeline.py)"]
    SA["show_all()<br/>(tty.py)"]

    YAML --> LPC
    LPC -->|"OrderedDict[int, dict]"| BD
    BD --> AC
    AC --> EM
    EM --> SA
```

`load_panel_configs()` globs `*.yaml` from the arch directory, loads each via
`yaml.safe_load`, and produces an `OrderedDict` keyed by panel ID. `build_dfs()`
iterates the panel configs, builds a DataFrame per `metric_table` with formula strings
as cell values, and populates `ArchConfig.dfs`, `dfs_type`, and `dfs_expressions`.
`eval_metric()` evaluates each formula cell against the raw PMC
DataFrame and variable dicts. `show_all()` renders the evaluated DataFrames by panel
order.

### Profile mode

```mermaid
flowchart LR
    YAML["Panel Config YAML"]
    DC["detect_counters()<br/>(soc_base.py)"]
    ECV["extract_counters_and_variables()<br/>(utils_counter_defs.py)"]
    PC["perfmon_coalesce()<br/>(soc_base.py)"]
    PMC["pmc_perf_*.yaml<br/>(one per hardware pass)"]

    YAML -->|metric expressions| DC
    DC -->|expressions| ECV
    ECV -->|counter set| PC
    PC --> PMC
```

`detect_counters()` takes the metric expressions of the panel YAMLs (all of them, or the
sections `-b` / `--set` select), runs `extract_counters_and_variables()` to extract the
HW counter names, and passes the set to `perfmon_coalesce()`, which bins counters into
per-pass files respecting IP-block limits. Comments, descriptions and labels add no
counters.


## Pipeline after the cutover

### Analyze mode

```mermaid
flowchart LR
    L2["Layer 2 YAML<br/>(base arch + {arch}/ overrides)"]
    L3["Layer 3 YAML<br/>(views/)"]
    LML["MetricLibrary.load()<br/>(metric_library/loader.py)"]
    LV["load_views()<br/>(layer3_reader.py)"]
    V2P["views_to_panel_configs()"]
    BD["build_dfs()<br/>(parser.py, unchanged)"]
    AC["ArchConfig"]
    EM["eval_metric()<br/>(unchanged)"]
    SA["show_all()<br/>(unchanged)"]

    L2 --> LML
    LML -->|MetricLibrary| LV
    L3 --> LV
    LV -->|"list[ViewDefinition]"| V2P
    V2P -->|"OrderedDict[int, dict]"| BD
    BD --> AC
    AC --> EM
    EM --> SA
```

Everything below `OrderedDict[int, dict]` is unchanged. `build_dfs()`, `eval_metric()`,
and `show_all()` require zero modifications during the entire migration. The interface
contract is the `OrderedDict[int, dict]` shape that `build_dfs()` expects.

### Profile mode

```mermaid
flowchart LR
    L2["Layer 2 YAML<br/>(base arch + {arch}/ overrides)"]
    LML["MetricLibrary.load()<br/>(metric_library/loader.py)"]
    GCM["get_counters_for_metrics()"]
    PC["perfmon_coalesce()<br/>(soc_base.py, unchanged)"]
    PMC["pmc_perf_*.yaml<br/>(one per hardware pass)"]

    L2 --> LML
    LML -->|MetricLibrary| GCM
    GCM -->|counter set| PC
    PC --> PMC
```

`detect_counters()` calls `MetricLibrary.get_counters_for_metrics()` instead of reading
the panel YAMLs' expressions. The counter set reaches the same `perfmon_coalesce()`
path.


## Transition mechanism: positional ids

The current pipeline -- `build_dfs()`, the `-b` filter, profiling sets -- is built
around numeric positional IDs (e.g., `11.2.3` = panel 1100, table 1102, metric index
3). The new design replaces these with stable string IDs (`compute.salu_util`). During
the transition, both systems must coexist.

The views store no numeric IDs. Each one is computed at run time from the view's
position: the panel ID from the view file name (`1100_compute_units_compute_pipeline.yaml`
is panel 1100), a table's ID from its position in the view (1102 is the second table),
and a row's ID from its position in the table (`11.2.2` is the third row). The
built-in views (top stats, system info) have fixed panel IDs.

`views_to_panel_configs()` uses these IDs to build the numeric `OrderedDict[int, dict]`
that `build_dfs()` expects, and analyze `-b`, `profile -b`, `--list-metrics`, the
database `metric_id` and the memory chart layouts all read the same run-time positions,
so profile and analyze agree by construction. A numeric ID typed in `-b` means the ID
analyze shows, and prints a deprecation warning naming the string ID.

A numeric ID changes when a view moves its row; string IDs do not. Numeric IDs are
removed in a later release, so no positions lock keeps them stable.


## Stage dependencies and parallelism

```mermaid
flowchart LR
    subgraph PA["Phase A -- Metric Library"]
        S1["Stage 1<br/>Schema + loader"]
        S2["Stage 2<br/>Parser"]
        S3["Stage 3<br/>Equivalence"]
        S4["Stage 4<br/>Metric migration"]
    end
    subgraph PB["Phase B -- Display / View"]
        S5["Stage 5<br/>View schema"]
        S6["Stage 6<br/>View reader"]
        S7["Stage 7<br/>View authoring"]
        S8["Stage 8<br/>Cutover"]
    end

    S1 --> S2
    S2 --> S3
    S2 --> S4
    S3 -.->|validates| S4
    S2 --> S6
    S5 --> S6
    S5 --> S7
    S4 --> S7
    S3 --> S8
    S6 --> S8
    S7 --> S8
```

A solid arrow is a build dependency; Stage 4 uses Stage 3 only to check its files.
Stages with no path between them can run in parallel, for example Stages 1 and 5.


## Risk mitigation: metric value validation

Metric values are a core product guarantee. Even unrelated refactoring can cause
breakages through typos or missed propagation.

### Validation strategy

1. **Mega-kernel metric-value check:** validates all metric values for a given
   architecture, not just the 2 workloads `test_metric_validation.py` tests
   (`memcopy`, `hbm_bandwidth`). It runs by hand on real hardware with exclusive GPU
   access; it is not a CI test and is not committed.

2. **Per-stage validation gate:** Before merging each implementation stage, run the
   mega-kernel validation test on at least one architecture (preferably gfx942 as
   the most widely deployed). Output must match expected values within tolerance.

3. **Golden-file comparison:** Capture the full `analyze` text output for all
   architectures before the migration begins. After each stage, compare the new
   output to the golden file. Any difference must be explained and approved.


## Cutover timeline

```mermaid
flowchart LR
    T1["Phase A and Stages 5-7<br/>Production reads the panel YAMLs<br/>The validator keeps the library equal to them<br/>-b: numeric ids only"]
    T2["Stage 8: cutover release<br/>Views and library are the only path<br/>Old panel files, old tooling and<br/>all equivalence checks removed<br/>-b: string ids, numeric ids warn"]
    T4["Later, outside this LLD<br/>-b: numeric ids rejected"]
    T1 --> T2 --> T4
```

Until the Phase B views, numeric ids are positions in the analysis configs, as today;
string ids need the row links that the views add. After Stage 8, a hardcoded final
mapping handles the numeric-id deprecation window. The cutover is specified in
[Stage 8](lld-phase-b-display-view.md#stage-8----cutover).
