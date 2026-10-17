# Memory chart layouts

The memory chart (panel 3 of `analyze`) is drawn from a layout file in
[`layouts/`](layouts/), one per GPU family. The layout file says everything about the
chart except the metric values: which boxes it has, where each box goes, which metric
each box and arrow shows, which way each arrow points, and how each part is drawn.

This README is the contract for any tool that draws the chart, such as ROCm Optiq.

## Example

A small layout with three columns. JSON has no comments; they are here only to explain.

```jsonc
{
  "archs": ["gfx950"],                     // the GPUs this layout is for
  "description": "CU, L2, and Data Fabric.",
  "scope": {"labels": ["GPU", "Fabric"],   // two labels in a bar above the chart;
            "split": "data_fabric"},       // "Fabric" starts at this box's column
  "blocks": [
    {"id": "cu", "title": "Compute Units", "column": 0, "order": 0,
     "position": "grid", "style": "list", "content": [
      {"metric": "VGPR", "title": "vGPRs", "category": "util", "bar": false}
    ]},
    {"id": "l2", "title": "L2", "column": 1, "order": 0,
     "position": "grid", "style": "panel", "stall_level": "GL2", "content": [
      {"metric": "L2 Hit", "title": "Hit", "category": "hit", "bar": true}
    ]},
    {"id": "data_fabric", "title": "Data Fabric", "column": 2, "order": 0,
     "position": "grid", "style": "panel"},
    {"id": "pcie", "title": "PCIe", "column": 2, "order": 0,   // a box drawn
     "position": "below", "style": "panel", "host": "data_fabric"}  // under Data Fabric
  ],
  "arrows": [
    {"from": "cu", "to": "l2", "direction": "backward", "metric": "Flat Read",
     "title": "Read", "category": "read", "style": "inline", "group": "Request"},
    {"from": "l2", "to": "data_fabric", "direction": "backward",
     "metric": "L2-Fabric Read BW", "title": "Read BW", "category": "read",
     "style": "stacked"},
    {"from": "data_fabric", "to": "pcie", "direction": "backward",
     "metric": "PCIe Read BW", "title": "Read BW", "category": "read",
     "style": "inline"}
  ]
}
```

## Fields

### Top level

All fields are required.

| Field | Meaning |
|---|---|
| `archs` | The GPUs the layout is for, as analysis-config directory names, such as `gfx950` or `gfx115x`. |
| `description` | One line about what the chart shows. |
| `scope` | `labels`: the two labels of the bar above the chart. `split`: the id of the box whose column starts the second label. |
| `blocks` | Every box of the chart, in any order. |
| `arrows` | The arrows between boxes, in the order they are drawn. |

### Blocks

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | A name for the box, unique in the layout. |
| `title` | yes | The text at the top of the box. |
| `column` | yes | The column, counting from 0 on the left. A box inside another box, or above or below one, has that box's column. |
| `order` | yes | The place among the boxes it is stacked with, counting from 0. Boxes are stacked with the other top-level boxes of their column, with the other boxes inside the same parent, or with the other boxes on the same side of the same host. |
| `position` | yes | `grid` for a box in a column or inside another box. `above` or `below` for a small labelled box outside the columns, such as xGMI or PCIe. |
| `style` | yes | `panel` for a box with a border. `list` for a compact list, used by the compute box on the left. Boxes above or below the columns are `panel`. |
| `parent` | no | The id of the box this box is drawn inside. |
| `host` | for `above` and `below` boxes | The id of the top-level box in a column that this box is drawn above or below. |
| `content` | no | The metrics shown in the box, top to bottom. |
| `stall_level` | no | `GL1`, `GL2`, or `EA`. `analyze` lists the memory bandwidth stalls of that level in the box. |

A box with boxes inside it shows only its title and those boxes; its metrics and arrows
belong to the inner boxes. Boxes above or below the columns show only their title.

### Metrics in a box

Each entry in `content` has all of these fields.

| Field | Meaning |
|---|---|
| `metric` | The metric name; see [Metric values](#metric-values). |
| `title` | The label shown before the value. |
| `category` | What the metric measures, which sets its color; see [Categories](#categories). |
| `bar` | `true` to draw a bar under the value, filled to the value out of 100. Only percent metrics have a bar, and never in a `list` box. |

### Arrows

| Field | Required | Meaning |
|---|---|---|
| `from` | yes | The id of a box in a column. |
| `to` | yes | The id of a box in the next column to the right, or of a box drawn above or below the `from` box. |
| `direction` | yes | `forward` points at the `to` box, `backward` points at the `from` box, and `both` points at both. |
| `metric` | yes | The metric name; see [Metric values](#metric-values). |
| `title` | yes | The label shown with the value. |
| `category` | yes | `read`, `write`, `atomic`, or `neutral`; sets the color. |
| `style` | yes | `inline` draws one line, `<title> : <value>`, such as `Read : 119`. `stacked` draws the title, then the value, then the arrow. Arrows to a box above or below the columns are `inline`. |
| `group` | no | A heading shared by consecutive arrows, such as `Buffer Request`. |

### Categories

| Category | Used on | Meaning |
|---|---|---|
| `read`, `write`, `atomic` | arrows and metrics | Read, write, or atomic traffic. |
| `neutral` | arrows and metrics | Mixed traffic, or a value with no color of its own. |
| `util`, `hit`, `stall` | metrics | Utilization, hit rate, or stall rate. |
| `bw` | metrics | Bandwidth. |

## What a tool can rely on

`analyze` skips a layout file that breaks any of these rules, so a stored layout
always follows them:

- Box ids are unique, and every `parent`, `host`, `from`, and `to` names a box in the
  layout. `scope.split` names a top-level box in a column after the first.
- Columns are numbered 0, 1, 2, ... with no gaps, and every column has a box.
- In each stack of boxes, `order` is 0, 1, 2, ... with no gaps.
- An arrow starts at a box in a column. It ends at a box in the next column, or at a
  box drawn above or below its start box. It never starts or ends at a box that has
  boxes inside it.

## Metric values

Each `metric` is the name of a metric in the memory chart panel (panel 3). Names are
unique within panel 3. They can repeat in other panels, so look only in panel 3.

In the analysis database (`analyze --output-format db`), each workload's layout file is
stored unchanged in the `memory_chart_layout_extdata` column of `compute_workload`. It
is null when the workload's GPU has no layout. Each metric of a workload has a row in
`compute_metric_definition`. For a layout metric, use the workload's row whose
`name` is the metric name and whose `metric_id` starts with `3.`. That row has the unit.
The value the terminal chart shows is in `compute_workload_metric_value`, in the row
for that definition with `value_name` `Value`. `compute_kernel_metric_value` has the
same value for each kernel.

Values are raw, in the unit of the definition, such as `Bytes/s` or `Percent`. A metric
with no row, for example when `--block` leaves out panel 3, has no value; show it as
missing.

## Adding an architecture

1. Copy the layout of the closest architecture, such as `gfx950.json`, and set `archs`.
2. Change its boxes and arrows until it shows every metric of the architecture's
   `0300_memory_chart.yaml`, and no other.
3. Look at it without a GPU: `./tools/memory_chart_preview.py <arch>`.
4. Run `python -m pytest tests/unit/memory_chart`. The tests also check that each
   arrow's `style` and each `bar` fit the metric's unit, and list any unit that
   [`units.py`](units.py) cannot show.

No Python changes are needed. The pre-commit hook runs
`./tools/memory_chart_layout_format.py`, which rewrites layouts in the standard style.
