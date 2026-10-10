# HLD: Memory chart rendering specification in the analysis database

## System Context

- `rocprof-compute analyze` draws a memory chart (panel 3) in the terminal. The chart has
  boxes, such as Compute Units, the caches, and HBM, with arrows between them. Each box
  and arrow shows a panel 3 metric. A layout file for each GPU family says which boxes
  the chart has, which boxes sit inside others, and which metric each box and arrow shows.
- `analyze --output-format db` writes an SQLite analysis database. Each workload has its
  own metric definitions, and each metric has a `metric_id`, such as `3.1.24`, that is
  unique within the workload. Data that a reader uses all at once is stored as JSON in
  columns of the workload table, such as `sys_info_extdata` and
  `profiling_config_extdata`.
- ROCm Optiq is a separate application that reads only the database. Its development
  branch already draws the memory chart from a JSON layout stored in a workload column
  named `memory_chart_extdata`, and finds each metric by its `metric_id`. When that column
  is missing, Optiq uses layouts built into Optiq.

## Problem statement

- Optiq's built-in layouts use fixed `metric_id` values. These ids change between
  rocprof-compute versions and GPU families, because they come from each metric's
  position in the panel configuration. A database from a different rocprof-compute
  version can show one metric's value under another metric's label, so Optiq turned the
  feature off in its release.
- Metric names can't be used instead, because they aren't unique. Each GPU family has 19
  to 43 names used in more than one panel. For example, "LDS Utilization" is both 3.4.5
  and 9.4.0 on gfx1250.

## Requirements

### Functional

1. Store how to draw each workload's memory chart in the database, so a reader can draw
   the same boxes, nesting, order, and arrows as rocprof-compute.
2. Link each metric in the chart to a metric of the same workload by an id that
   rocprof-compute works out when it writes the database, so readers never match names.
3. Keep the full chart when `--block` leaves out some of its metrics. Those metrics have
   no values.
4. Store the chart for every workload, whether or not its chart metrics have values.

### Non-functional

- Readers that don't know about the new column keep working; the change only adds a
  column.
- Follow the existing conventions of the database.
- Use the layout format that Optiq already reads; only the column name is new to Optiq.

## Design

The workload table gets one JSON column, `memory_chart_render_extdata`, that describes
how to draw the workload's memory chart, in Optiq's layout format:

- `version`, `blocks`, and `arrows`. Each block has an `id`, a `title`, a `column`, a
  `row`, an `order`, the metrics it shows (`content`), and the blocks inside it
  (`children`). Each arrow has `from`, `to`, `direction`, `metric`, `title`, and
  `category`.
- Each `metric` is the `metric_id` of that metric in the same workload. rocprof-compute
  works it out from the workload's panel 3 configuration when it writes the database. It
  is `null` when that configuration, for example one from `--config-dir`, has no single
  metric of that name.
- Some keys are not in Optiq's format yet: `note`, `host` (the block that a block above or
  below is attached to), arrow `group`, `description`, and `scope`. Optiq ignores keys it
  doesn't know. Optiq's published format doesn't list these keys yet, nor the `neutral`
  and `bw` categories.
- The positions and arrow directions come from the same code that draws the terminal
  chart, so both charts always match.

### Decisions

| Decision | Why | Alternative considered |
|---|---|---|
| One JSON column, not new tables | A reader uses the whole layout at once, which is what the JSON columns of the workload table are for. Optiq already reads a layout from a column. | Three new tables for blocks, block metrics, and arrows. This would add table links the schema doesn't use anywhere else, and Optiq would need a new reader. |
| `metric_id` links each metric | It is unique within a workload, and Optiq already finds metrics by it. Writing it when the database is written makes it correct for that database. | `metric_uuid`, which is also correct, but Optiq's reader would need changes. Metric names, which aren't unique. |
| Optiq's format with extra keys | Optiq reads it today, and can use the extra keys later. | rocprof-compute's own layout format, which Optiq would need a new reader for. |
| The same categories as the terminal chart | Both charts use the same names. Optiq draws categories it doesn't know in its default color. | Mapping `neutral` and `bw` to Optiq's `misc`. |
| Named `memory_chart_render_extdata` | Makes clear that it describes how to draw the chart, and holds no metric values. | `memory_chart_extdata`, the name that Optiq's development branch reads today. |
| Always stored, with `null` for metrics without an id | Readers need the chart even when its metrics have no values or ids. | Leaving the column empty when the panel configuration doesn't match the layout. Readers would then use their own layouts. |
| Ids from the full panel configuration | The chart stays complete with `--block`; metrics left out just have no values. | Storing no chart, so Optiq would use its fixed ids. |
| Schema version 2.4.0 | The change only adds a column, like version 2.3.0. | — |

### Out of scope

- Memory bandwidth stall notes: the database doesn't store memory bandwidth analysis
  results.
- How to show values (units, decimal places, box sizes): each reader decides.
- CSV output: the column isn't part of any view.

## Implementation phases

1. Store how to draw the chart, with `metric_id` links (this change).
2. Optiq reads the column by its new name in its release, and uses the extra keys.

## Validation, security and debuggability

- Unit tests check the following for every layout:
  - Block ids and metric links use the formats Optiq reads.
  - Each link is the panel 3 `metric_id` of its own metric.
  - Nesting, rows, and order match the layout.
  - A metric that a panel 3 configuration doesn't have, or has more than once, is stored
    as `null`, and no other metric is.
- On a real gfx950 workload, every link finds a value in the kernel metric view. A
  database written with `--block` stores the full chart without values.

## Open questions

- Optiq's development branch reads `memory_chart_extdata`; it needs to read
  `memory_chart_render_extdata` instead.
- When Optiq adds `neutral`, `bw`, and the extra keys to its format. Optiq already has a
  `note` key with a different meaning (a note for layout authors), so the two teams need
  to agree on it.
- Optiq's built-in gfx950 layout has arrows without metrics, from MALL to UMC to HBM.
  rocprof-compute's layouts don't have these arrows, so those blocks aren't connected.
- Whether `version` should change with rocprof-compute's layout file format, or only
  when the stored format changes.
