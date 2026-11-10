# Memory chart layouts

The memory chart (panel 3 of `analyze`) is drawn from a layout file in
[`layouts/`](layouts/), one per GPU family. The layout file says everything about the
chart except the metric values: which boxes it has, where each box goes, which metric
each box and arrow shows, which way each arrow points, and how each part is drawn.

This README is the contract for any tool that draws the chart, such as ROCm Optiq.

## Example

A small layout with four columns, a box above the grid, and a box below it. JSON has
no comments; they are here only to explain.

```jsonc
{
  "version": 1,                            // the format version of this file
  "archs": ["gfx950"],                     // the GPUs this layout is for
  "description": "CU, VL1D, L2, and Data Fabric, with xGMI and PCIe.",
  "scope": {"labels": ["GPU", "Fabric"],   // two labels in a bar above the chart;
            "split": "data_fabric"},       // "Fabric" starts at this box's column
  "blocks": [
    {"id": "cu", "title": "Compute Units", "column": 0, "row": 0, "order": 0,
     "style": "list", "content": [
      {"metric_name": "VGPR", "metric_id": "3.1.1", "title": "vGPRs",
       "category": "util", "bar": false}
    ]},
    {"id": "vl1d", "title": "VL1D", "column": 1, "row": 0, "order": 0,
     "style": "panel", "content": [
      {"metric_name": "VL1 Hit", "metric_id": "3.1.17", "title": "Hit",
       "category": "hit", "bar": true}
    ]},
    {"id": "l2", "title": "L2", "column": 2, "row": 0, "order": 0,
     "style": "panel", "content": [
      {"metric_name": "L2 Hit", "metric_id": "3.1.28", "title": "Hit",
       "category": "hit", "bar": true}
    ]},
    {"id": "data_fabric", "title": "Data Fabric", "column": 3, "row": 0, "order": 0,
     "style": "panel"},
    {"id": "xgmi", "title": "xGMI", "column": 3, "row": -1,   // above Data Fabric
     "style": "panel"},
    {"id": "pcie", "title": "PCIe", "column": 3, "row": 1,    // below Data Fabric
     "style": "panel"}
  ],
  "arrows": [
    {"from": "cu", "to": "vl1d", "direction": "backward", "metric_name": "Flat Read",
     "metric_id": "3.1.6", "title": "Read", "category": "read", "style": "inline",
     "group": "Non-buffer Request"},
    {"from": "cu", "to": "vl1d", "direction": "forward", "metric_name": "Flat Write",
     "metric_id": "3.1.7", "title": "Write", "category": "write", "style": "inline",
     "group": "Non-buffer Request"},
    {"from": "cu", "to": "vl1d", "direction": "backward", "metric_name": "Buffer Read",
     "metric_id": "3.1.9", "title": "Read", "category": "read", "style": "inline",
     "group": "Buffer Request"},
    {"from": "cu", "to": "vl1d", "direction": "forward", "metric_name": "Buffer Write",
     "metric_id": "3.1.10", "title": "Write", "category": "write", "style": "inline",
     "group": "Buffer Request"},
    {"from": "vl1d", "to": "l2", "direction": "backward",
     "metric_name": "VL1_L2 Read BW", "metric_id": "3.1.19", "title": "Read BW",
     "category": "read", "style": "stacked"},
    {"from": "l2", "to": "data_fabric", "direction": "backward",
     "metric_name": "L2-Fabric Read BW", "metric_id": "3.1.29", "title": "Read BW",
     "category": "read", "style": "stacked"},
    {"from": "data_fabric", "to": "xgmi", "direction": "backward",
     "metric_name": "xGMI Read BW", "metric_id": "3.1.35", "title": "Read BW",
     "category": "read", "style": "inline"},
    {"from": "data_fabric", "to": "pcie", "direction": "backward",
     "metric_name": "PCIe Read BW", "metric_id": "3.1.38", "title": "Read BW",
     "category": "read", "style": "inline"}
  ]
}
```

### How the terminal draws it

`analyze` draws the example like this, here with placeholder values:

```text
                                                                                                  ╭────────────╮
                                                                                                  │ xGMI       │   row -1
                                                                                                  ╰────────────╯
                                                                                                     ||  Read BW  123.456 GB/s
|---------------------------------------------- GPU ----------------------------------------------|--- Fabric ----|   scope

╭─ Compute Units ─╮                   ╭──── VL1D ─────╮             ╭───── L2 ──────╮             ╭─ Data Fabric ─╮   row 0
│ vGPRs 12.5      │Non-buffer Request │ Hit 42.0%     │             │ Hit 42.0%     │             │               │
│                 │Read  : 1.23e+03   │ ████░░░░░░    │             │ ████░░░░░░    │             │               │
│                 │<----------------- │               │             │               │             │               │
│                 │Write : 1.23e+03   │               │Read BW      │               │Read BW      │               │
│                 │-----------------> │               │123.456 GB/s │               │123.456 GB/s │               │
│                 │Buffer Request     │               │<----------- │               │<----------- │               │
│                 │Read  : 1.23e+03   │               │             │               │             │               │
│                 │<----------------- │               │             │               │             │               │
│                 │Write : 1.23e+03   │               │             │               │             │               │
│                 │-----------------> │               │             │               │             │               │
╰─────────────────╯                   ╰───────────────╯             ╰───────────────╯             ╰───────────────╯

                                                                                                     ||  Read BW  123.456 GB/s
                                                                                                  ╭────────────╮
                                                                                                  │ PCIe       │   row 1
                                                                                                  ╰────────────╯
```

How each field shows up in the picture:

- **`row`:** xGMI (row -1) is above the grid, the boxes of row 0 form the grid, and
  PCIe (row 1) is below it. Each box outside the grid lines up with its `column`.
- **`scope`:** the dashed bar above the grid. "GPU" spans columns 0 to 2, and "Fabric"
  starts at the column of `data_fabric`.
- **`group`:** the headings "Non-buffer Request" and "Buffer Request" over the CU to
  VL1D arrows.
- **Arrow `style`:** `inline` arrows show `Read  : 1.23e+03` above the arrow line;
  `stacked` arrows show the title, then the value, then the arrow line, as between VL1D
  and L2. Arrows to a box outside the grid are drawn as a `||` connector.
- **`direction`:** `<---` for `backward`, `--->` for `forward`, and `<--->` for `both`.
- **`bar`:** the `████░░░░░░` under each hit rate.
- **Block `style`:** Compute Units is a `list`, a compact list with every value in one
  color; the other boxes are `panel`s.

## Fields

### Top level

All fields are required.

| Field | Meaning |
|---|---|
| `version` | The format version of the file, now `1`; see [Versions](#versions). |
| `archs` | The GPUs the layout is for, as analysis-config directory names, such as `gfx950` or `gfx115x`. |
| `description` | One line about what the chart shows. |
| `scope` | `labels`: the two labels of the bar above the chart, such as "GPU (XCD)" and "Fabric / Memory". `split`: the id of the box whose column starts the second label. |
| `blocks` | Every box of the chart, in any order. |
| `arrows` | The arrows between boxes, in the order they are drawn. |

### Blocks

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | A name for the box, unique in the layout. |
| `title` | yes | The text at the top of the box. |
| `column` | yes | The column, counting from 0 on the left. A box inside another box has that box's column. |
| `row` | for boxes not inside another box | `0` for the grid. Negative rows are above it, with `-1` next to the grid, and positive rows are below it, with `1` next to the grid. |
| `order` | for boxes in row 0 | The place among the boxes it is stacked with, counting from 0 at the top: the other top-level boxes of its column in row 0, or the other boxes inside the same parent. |
| `style` | yes | `panel` for a box with a border. `list` for a compact list, used by the compute box on the left. Boxes outside row 0 are `panel`. |
| `parent` | no | The id of the box this box is drawn inside. |
| `content` | no | The metrics shown in the box, top to bottom. |
| `stall_level` | no | `GL1`, `GL2`, or `EA`. `analyze` lists the memory bandwidth stalls of that level in the box. |

A box with boxes inside it shows only its title and those boxes; its metrics and arrows
belong to the inner boxes. Boxes outside row 0 are small labelled boxes, such as xGMI
or PCIe, and show only their title.

### Metrics in a box

Each entry in `content` has all of these fields.

| Field | Meaning |
|---|---|
| `metric_name` | The metric's name; see [Metric values](#metric-values). |
| `metric_id` | The metric's id, such as `3.1.24`; see [Metric values](#metric-values). |
| `title` | The label shown before the value. |
| `category` | What the metric measures, which sets its color; see [Categories](#categories). |
| `bar` | `true` to draw a bar under the value, filled to the value out of 100. Only percent metrics have a bar, and never in a `list` box. |

### Arrows

| Field | Required | Meaning |
|---|---|---|
| `from` | yes | The id of the box the arrow starts at. |
| `to` | yes | The id of a box in the next column of row 0, or of the box next to the `from` box on the side away from row 0, such as xGMI above Data Fabric. |
| `direction` | yes | `forward` points at the `to` box, `backward` points at the `from` box, and `both` points at both. |
| `metric_name` | yes | The metric's name; see [Metric values](#metric-values). |
| `metric_id` | yes | The metric's id; see [Metric values](#metric-values). |
| `title` | yes | The label shown with the value. |
| `category` | yes | `read`, `write`, `atomic`, or `neutral`; sets the color. |
| `style` | yes | `inline` draws one line, `<title> : <value>`, such as `Read : 119`. `stacked` draws the title, then the value, then the arrow. Arrows to a box outside row 0 are `inline`. |
| `group` | no | A heading over consecutive arrows between the same two boxes, such as `Buffer Request` and `Non-buffer Request`, which tell apart arrows with the same title. |

### Categories

| Category | Used on | Meaning |
|---|---|---|
| `read`, `write`, `atomic` | arrows and metrics | Read, write, or atomic traffic. |
| `neutral` | arrows and metrics | Anything without a color of its own, such as mixed traffic or a coalescing rate. |
| `util`, `hit`, `stall` | metrics | Utilization, hit rate, or stall rate. |
| `bw` | metrics | Bandwidth shown inside a box. |

### Fields a tool may leave out

A tool must place boxes, metrics, and arrows as the layout says, but it may skip these
display hints: `scope`, `group`, block and arrow `style`, `bar`, and `stall_level`.
`analyze` fills `stall_level` only from its memory bandwidth analysis, which the
analysis database doesn't store.

## What a tool can rely on

`analyze` skips a layout file that breaks any of these rules, so a stored layout
always follows them:

- `version` is `1`.
- Box ids are unique, and every `parent`, `from`, and `to` names a box in the layout.
  `scope.split` names a top-level box of row 0 in a column after the first.
- Columns are numbered 0, 1, 2, ... with no gaps, and every column has a box in row 0.
- In each stack of boxes in row 0, `order` is 0, 1, 2, ... with no gaps.
- Outside row 0 there is at most one box per row and column, and rows have no gaps
  toward 0: a box in row -2 has a box in row -1 below it.
- An arrow between two boxes of row 0 goes to the next column. An arrow to a box
  outside row 0 starts at the box next to it toward row 0: for row -1, the top box of
  its column; for row 1, the bottom box; otherwise the box in the row between. No
  arrow starts or ends at a box that has boxes inside it.

## Versions

`version` changes only when a change to the format could make an existing reader draw
a layout wrongly, for example a renamed or removed field or a field whose meaning
changes. A reader should skip a layout with a version it doesn't know, the way
`analyze` does. A layout with no `version` has the earlier format, with `metric`,
`position`, and `host` fields, which analysis databases from earlier rocprof-compute
builds store; skip it too.

## Metric values

Each metric has a name and an id within the memory chart panel (panel 3). Names are
unique within panel 3, and can repeat in other panels.

- `metric_name` is the same in every rocprof-compute version that has the metric. Use
  it to compare workloads profiled by different versions.
- `metric_id` is the metric's place in panel 3, such as `3.1.24`. An id can refer to
  another metric in another rocprof-compute version, but within one analysis database
  it always matches the layout stored with the workload.

In the analysis database (`analyze --output-format db`), each workload's layout file is
stored unchanged in the `memory_chart_layout_extdata` column of `compute_workload`. It
is null when the workload's GPU has no layout. Each metric of a workload has a row in
`compute_metric_definition`, with the unit. The value the terminal chart shows is in
`compute_workload_metric_value`, in the row for that definition with `value_name`
`Value`. `compute_kernel_metric_value` has the same value for each kernel.

For example, the value of the VL1D hit rate of workload 1:

```sql
SELECT d.name, d.unit, v.value
FROM compute_metric_definition AS d
JOIN compute_workload_metric_value AS v ON v.metric_uuid = d.metric_uuid
WHERE d.workload_id = 1 AND d.metric_id = '3.1.17' AND v.value_name = 'Value';
```

Values are raw, in the unit of the definition, such as `Bytes/s` or `Percent`. A metric
with no row, for example when `--block` leaves out panel 3, has no value; show it as
missing.

## Adding an architecture

1. Copy the layout of the closest architecture, such as `gfx950.json`, and set `archs`.
2. Change its boxes and arrows until it shows every metric of the architecture's
   `0300_memory_chart.yaml`, and no other. Leave out `metric_id`; the next step fills
   it in.
3. Run `./tools/memory_chart_layout_format.py`. It sets every `metric_id` from the
   panel config and rewrites the layout in the standard style. The pre-commit hook runs
   it whenever a layout or a memory chart panel config changes.
4. Look at it without a GPU: `./tools/memory_chart_preview.py <arch>`.
5. Run `python -m pytest tests/unit/memory_chart`. The tests also check that each
   `metric_id` matches the panel config, that each arrow's `style` and each `bar` fit
   the metric's unit, and list any unit that [`units.py`](units.py) cannot show.

No Python changes are needed.
