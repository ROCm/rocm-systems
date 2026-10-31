# HLD: Memory chart layout in the analysis database

## System Context

- `rocprof-compute analyze` draws a memory chart (panel 3) in the terminal. The chart has
  boxes, such as Compute Units, the caches, and HBM, with arrows between them. Each box
  and arrow shows a panel 3 metric.
- A JSON layout file for each GPU family (`src/memory_chart/layouts/`) states everything
  needed to draw the chart: where each box goes, which boxes sit inside others, the
  metric each box and arrow shows, which way each arrow points, and how each part is
  drawn. `src/memory_chart/README.md` documents the format.
- `analyze --output-format db` writes an SQLite analysis database. Each workload has its
  own metric definitions. Data that a reader uses all at once is stored as JSON in
  columns of the workload table, such as `sys_info_extdata` and
  `profiling_config_extdata`.
- ROCm Optiq is a separate application that reads only the database. It draws the memory
  chart from layouts built into Optiq, which find metrics by fixed `metric_id` values.

## Problem statement

- Optiq's built-in layouts use fixed `metric_id` values, such as `3.1.24`. These ids come
  from each metric's position in the panel configuration, so they change between
  rocprof-compute versions and GPU families. A database from a different
  rocprof-compute version can show one metric's value under another metric's label, so
  Optiq turned the feature off in its release.
- Optiq also has to keep its own copy of every layout in step with rocprof-compute.

## Requirements

### Functional

1. Store each workload's memory chart layout in the database, so a reader draws the
   same boxes, nesting, order, and arrows as rocprof-compute from the database alone.
2. The stored layout is the layout file itself. There is one memory chart format, and
   readers need no converter or parsing script.
3. Each metric in the layout is found by a reference that stays correct across
   rocprof-compute versions.
4. Keep the full chart when `--block` leaves out some of its metrics. Those metrics have
   no values.

### Non-functional

- Readers that don't know about the new column keep working; the change only adds a
  column.
- Follow the existing conventions of the database.

## Design

The workload table gets one JSON column, `memory_chart_layout_extdata`. It holds the
layout file of the workload's GPU family, unchanged, or null when the family has no
layout.

- **Metrics are found by name within panel 3.** Each `metric` in the layout is the name
  of a metric in the workload's panel 3 configuration. Names are unique within panel 3
  on every GPU family, and a unit test keeps them unique. A reader finds the metric's
  definition, values, and unit by that name among the workload's metrics whose
  `metric_id` starts with `3.`.
- **Everything is stated in the file.** Each block has its `column`, its `order`, its
  `position` (in the grid, above, or below), and its `parent` or `host`. Each arrow has
  its `direction`. Drawing hints are also in the file: a block's `style` (`panel`, or
  `list` for the compact compute side), whether a metric has a progress `bar`, and an
  arrow's `style` (`inline` or `stacked`). The terminal chart reads the same fields, so
  both charts always match.
- **No format version.** The database schema version covers the column.

### Decisions

| Decision | Why | Alternative considered |
|---|---|---|
| Store the layout file unchanged | One format for the terminal chart, the database, and Optiq. Nothing to convert, and nothing to keep in step. | Converting the layout into a separate format for Optiq, which is a second format to maintain and document. |
| Metric names, looked up within panel 3 | Names are unique within panel 3, and a name refers to the same metric across versions. Looking only in panel 3 avoids names that repeat in other panels, such as "LDS Utilization". | `metric_id`, which changes when metrics are added or removed. |
| Placement, direction, and drawing hints stated in the file | A reader draws the chart by reading fields, without working anything out from where an entry sits in the file, from units, or from categories. | Deriving them in each reader, which is what the terminal renderer did before. |
| One JSON column, not new tables | A reader uses the whole layout at once, which is what the JSON columns of the workload table are for. | Tables for blocks, block metrics, and arrows. These add table links the schema doesn't use anywhere else. |
| Named `memory_chart_layout_extdata` | Says what it holds, and follows the `*_extdata` convention. | `memory_chart_extdata`, the name Optiq's development branch reads in its older format, which an older Optiq build would misread. |
| Null when a GPU family has no layout | The column also covers a layout file that was skipped as broken. Readers then use their own fallback. | Failing the database output. |
| No format version | The database schema version already covers the column. | A `version` field in the layout. |
| Schema version 2.4.0 | The change only adds a column. | — |

### Out of scope

- Memory bandwidth stall notes: the database doesn't store memory bandwidth analysis
  results. Only rocprof-compute reads the layout's `stall_level`.
- How to show values (units, decimal places, colors, box sizes, where an arrow sits
  along its boxes): each reader decides.
- CSV output: the column isn't part of any view.

## Implementation phases

1. The layout format states placement, arrow direction, and drawing hints
   (#12548).
2. Store the layout file in the database (this change).
3. Optiq draws the chart from the column, and finds metrics by name within panel 3.

## Validation, security and debuggability

- Unit tests check that the stored column equals the layout file, and that a GPU family
  without a layout stores null.
- The layout tests check every layout against its panel 3 configuration: the layout
  shows every panel 3 metric and no other, names are unique within panel 3, and the
  drawing hints fit each metric's unit.
- On a real gfx950 workload, the stored layout equals the file, and every layout metric
  finds a panel 3 definition in the database.

## Open questions

- Optiq needs to read the new column and the layout format documented in
  `src/memory_chart/README.md`, and to find metrics by name within panel 3.
- With a panel 3 configuration from `--config-dir`, a layout metric may have no
  definition. Readers should show it as missing.
