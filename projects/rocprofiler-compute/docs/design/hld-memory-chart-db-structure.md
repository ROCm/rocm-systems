# HLD: Memory chart layout in the analysis database

## System Context

- `rocprof-compute analyze` draws a memory chart (panel 3) in the terminal: blocks such as
  Compute Units, caches and HBM, and arrows between them, each showing a panel 3 metric.
  Each architecture family has a layout file that states the blocks, how they nest, and
  which metric each block and arrow shows.
- `analyze --output-format db` writes an SQLite analysis database. Metric definitions are
  per workload; each has a dotted `metric_id` ("3.1.24") that is unique within the
  workload. Per-workload data that a reader takes as a whole is stored as JSON columns on
  the workload row (`sys_info_extdata`, `profiling_config_extdata`, `roofline_bench_extdata`).
- ROCm Optiq is a separate C++ application that reads only the database. Its current
  development line already draws the memory chart from a JSON layout stored in a workload
  column, `memory_chart_extdata`, and resolves metrics by `metric_id`. When the column is
  absent it falls back to layouts built into Optiq.

## Problem statement

- Optiq's built-in layouts reference metrics by hard-coded `metric_id`. Metric ids are not
  stable across rocprof-compute versions and architectures: they are positions in the
  panel config, which change whenever metrics are added or removed. A database from a
  different rocprof-compute version can therefore show one metric's value under another
  metric's label. Optiq disabled the feature in its release for this reason.
- Metric names are not a usable key either: 19 to 43 names repeat across panels on each
  architecture (for example "LDS Utilization" is both 3.4.5 and 9.4.0 on gfx1250).

## Requirements

### Functional

1. Store each workload's memory chart structure in the database, for that workload's
   architecture, so a reader can draw a chart with the same blocks, nesting, order and
   arrows as rocprof-compute.
2. Link each block metric and arrow to a metric of the same workload by a key that
   rocprof-compute resolves when it writes the database, so readers never match names.
3. Keep the chart when `--block` leaves out some of its metrics; those metrics have no
   values.
4. Store nothing for an architecture without a layout, so readers can fall back.

### Non-functional

- Readers that don't know the new data keep working (additive schema change).
- Follow the database's existing conventions.
- Work with Optiq's existing reader without waiting for Optiq changes.

## Design

The workload table gets one JSON column, `memory_chart_extdata`, holding the layout of the
workload's architecture in Optiq's layout format:

- `version`, `blocks` (each with `id`, `title`, `column`, `row`, `order`, `content` and
  nested `children`) and `arrows` (`from`, `to`, `direction`, `metric`, `title`,
  `category`).
- Each `metric` is the `metric_id` of that metric in the same workload, computed from the
  workload's panel 3 config when the database is written.
- Extra keys carry what rocprof-compute's layouts state beyond Optiq's format today:
  `note`, `host` (the block an above/below block is attached to), arrow `group`,
  `description` and `scope`. Optiq's parser ignores unknown keys; Optiq's published
  layout schema doesn't allow them yet, nor the `neutral` and `bw` categories.
- Positions and directions are derived by the same layout loader that the terminal chart
  uses, so both draw from one resolution of the layout.

### Decisions

| Decision | Why | Alternative considered |
|---|---|---|
| One JSON column, not new tables | The layout is read whole by a renderer, which is what the `*_extdata` columns are for; Optiq already reads this column. | Three normalized tables (blocks, block metrics, arrows): first self-references and multi-references in the schema, no unique keys or view following existing patterns, and a new reader in Optiq. |
| `metric_id` as the metric key | Unique within a workload and what Optiq resolves; writing it at analyze time makes the mapping exact for that database. | `metric_uuid`: exact too, but Optiq's reader would need changes. Metric names: not unique. |
| Optiq's format plus extra keys | Optiq's parser reads it today; the extras are available when Optiq adopts them. | rocprof-compute's own layout format: Optiq would need a new parser first. |
| Categories as rocprof-compute uses them | One vocabulary with the terminal chart; Optiq colors unknown categories with its default. | Mapping `neutral` and `bw` onto Optiq's `misc`. |
| Ids from the full panel config | `--block` filtering keeps the chart; filtered metrics just have no values. | Writing no chart: Optiq would fall back to its hard-coded ids. |
| Minor schema version bump (2.4.0) | The change only adds a column, like 2.3.0. | — |

### Out of scope

- Membw stall annotations: the database doesn't store membw results.
- Display rules (units, precision, sizes): readers have their own.
- CSV export: the column is not part of any view.

## Implementation phases

1. Store the layout column with `metric_id` references (this change).
2. Optiq reads the column on its release line and adopts the extra keys.

## Validation, security and debuggability

- Unit tests: for every layout, block ids and metric references match the formats Optiq
  reads, and every reference is the panel 3 `metric_id` of its own metric; nesting, rows
  and order match the layout; the column is NULL for an architecture without a layout, or
  with a panel 3 config that lacks or repeats a metric the layout shows.
- Real data: on a gfx950 workload every reference resolves to a value in the kernel metric
  view; a `--block`-filtered database stores the full layout without values.
- A layout that can't be stored is logged at debug level.

## Open questions

- When Optiq adds `neutral` and `bw` and the extra keys to its schema, and uses them. Its
  existing `note` key is an authoring note, so the meaning needs agreeing.
- Optiq's built-in gfx950 layout draws connector arrows without metrics (MALL to UMC to
  HBM); rocprof-compute's layouts have no such arrows, so those blocks stand apart.
- Whether the layout format `version` should change together with rocprof-compute's
  layout file format, or only when the stored format changes.
