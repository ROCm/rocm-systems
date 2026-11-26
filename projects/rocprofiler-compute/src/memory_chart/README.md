# Memory chart layouts

`analyze` draws the memory chart (panel 3) from a layout file in
[`layouts/`](layouts/). There is one file per GPU family. The file lists the blocks of
the chart, where each block goes, which metric each block and arrow shows, and how to
draw them. The analysis supplies the values.

Other tools that draw the chart, such as ROCm Optiq, read the same format from the
analysis database.

## Example

This layout has three columns, one block above the grid, and one block below it. JSON
has no comments; the comments here only explain the fields.

```jsonc
{
  "archs": ["gfx950"],                    // GPUs that use this layout
  "description": "CU, L2, and fabric, with xGMI and PCIe.",
  "scope": {"labels": ["GPU", "Fabric"],  // labels of the bar above the grid
            "split": "fabric"},           // "Fabric" starts at this block's column
  "blocks": [
    {"id": "cu", "title": "Compute Units", "column": 0, "row": 0, "order": 0,
     "style": "list", "content": [
      {"metric_id": "3.1.0", "title": "vGPRs", "category": "util", "bar": false}
    ]},
    {"id": "l2", "title": "L2", "column": 1, "row": 0, "order": 0,
     "style": "panel", "content": [
      {"metric_id": "3.1.1", "title": "Hit", "category": "hit", "bar": true}
    ]},
    {"id": "fabric", "title": "Fabric", "column": 2, "row": 0, "order": 0,
     "style": "panel"},
    {"id": "xgmi", "title": "xGMI", "column": 2, "row": -1, "style": "panel"},
    {"id": "pcie", "title": "PCIe", "column": 2, "row": 1, "style": "panel"}
  ],
  "arrows": [
    {"from": "cu", "to": "l2", "direction": "backward", "metric_id": "3.1.2",
     "title": "Read", "category": "read", "style": "inline"},
    {"from": "l2", "to": "fabric", "direction": "backward", "metric_id": "3.1.3",
     "title": "Read BW", "category": "read", "style": "stacked"},
    {"from": "fabric", "to": "xgmi", "direction": "backward", "metric_id": "3.1.4",
     "title": "Read BW", "category": "read", "style": "inline"},
    {"from": "fabric", "to": "pcie", "direction": "backward", "metric_id": "3.1.5",
     "title": "Read BW", "category": "read", "style": "inline"}
  ]
}
```

The terminal chart looks roughly like this sketch. The ids and values are made up.

```text
                                                       ╭────────────╮
                                                       │ xGMI       │   row -1
                                                       ╰────────────╯
                                                          ||  Read BW  1.0 GB/s
|------------------------ GPU -------------------------|- Fabric -|

╭─ Compute Units ─╮            ╭──── L2 ────╮          ╭─ Fabric ─╮
│ vGPRs 12.5      │Read : 100  │ Hit 42.0%  │Read BW   │          │   row 0
│                 │<---------- │ ████░░░░░░ │1.0 GB/s  │          │
│                 │            │            │<-------- │          │
╰─────────────────╯            ╰────────────╯          ╰──────────╯

                                                          ||  Read BW  1.0 GB/s
                                                       ╭────────────╮
                                                       │ PCIe       │   row 1
                                                       ╰────────────╯
```

## Fields

### Top level

All fields are required.

| Field | Meaning |
|---|---|
| `archs` | GPUs that use the layout, as analysis config folder names such as `gfx950` or `gfx115x`. |
| `description` | One line that says what the chart shows. |
| `scope` | `labels` holds the two labels of the bar above the grid. `split` is the id of the block where the second label starts. |
| `blocks` | All blocks of the chart, in any order. |
| `arrows` | The arrows between blocks, in drawing order. |

### Blocks

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Name of the block, unique in the layout. |
| `title` | yes | Text at the top of the block. |
| `column` | yes | Column number, starting at 0 on the left. A nested block uses the column of its parent. |
| `row` | yes, but not in nested blocks | `0` puts the block in the grid, `-1` above the grid, and `1` below it. |
| `order` | yes, but not in rows -1 and 1 | Place in its stack, starting at 0 at the top. A stack is the top-level blocks of one column, or the blocks inside one parent. |
| `style` | yes | `panel` draws a box with a border. `list` draws a compact list, as in the compute block. |
| `parent` | no | Id of the block that holds this block. |
| `content` | no | Metrics shown in the block, from top to bottom. |
| `stall_level` | no | `GL1`, `GL2`, or `EA`. `analyze` lists the memory bandwidth stalls of that level in the block. |

A block that holds other blocks shows only its title. Blocks in rows -1 and 1 use the
`panel` style and show only their title.

### Metrics in a block

All fields are required.

| Field | Meaning |
|---|---|
| `metric_id` | Id of the metric in panel 3 of the analysis config, such as `3.1.24`. |
| `title` | Label shown before the value. |
| `category` | Kind of metric. It sets the color; see [Categories](#categories). |
| `bar` | `true` draws a bar under the value. Use it only for percentages, and never in a `list` block. |

### Arrows

| Field | Required | Meaning |
|---|---|---|
| `from` | yes | Id of the block where the arrow starts. |
| `to` | yes | Id of the block where the arrow ends. |
| `direction` | yes | `forward` points at `to`, `backward` points at `from`, and `both` points at both. |
| `metric_id` | yes | Id of the metric in panel 3 of the analysis config. |
| `title` | yes | Label shown with the value. |
| `category` | yes | `read`, `write`, `atomic`, or `neutral`. It sets the color. |
| `style` | yes | `inline` draws one line, such as `Read : 100`. `stacked` draws the title, the value, and the arrow on separate lines. |
| `group` | no | Heading over a run of arrows between the same two blocks, such as `Buffer Request`. |

### Categories

| Category | Used on | Meaning |
|---|---|---|
| `read`, `write`, `atomic` | arrows and metrics | Read, write, or atomic traffic. |
| `neutral` | arrows and metrics | Anything else, such as mixed traffic. |
| `util`, `hit`, `stall` | metrics | Utilization, hit rate, or stall rate. |
| `bw` | metrics | Bandwidth shown inside a block. |

## Rules

`analyze` skips a layout that breaks these rules:

- Block ids are unique, and `parent`, `from`, and `to` name blocks in the layout.
- `scope.split` names a top-level block in row 0, after the first column.
- Columns are numbered 0, 1, 2, and so on with no gaps. So are the orders in each stack.
- Rows -1 and 1 each hold at most one block, in a column of the grid.
- An arrow in row 0 goes to the next column. An arrow to row -1 starts at the top block
  of that column, and an arrow to row 1 starts at the bottom block. Both are `inline`.
- No arrow starts or ends at a block that holds other blocks.

## Versioning

`LAYOUT_VERSION` in [`loader.py`](loader.py) is the version of this format. `analyze`
stores it with each layout in the analysis database. It follows semantic versioning.
Bump it in the same change that changes the format:

- MAJOR: a field is renamed or removed, or its meaning changes.
- MINOR: an optional field is added.
- PATCH: a fix that changes no field.

## Adding an architecture

1. Copy the layout of the closest architecture, such as `gfx950.json`, and set `archs`.
2. Edit its blocks and arrows until it shows every metric of the architecture's
   `0300_memory_chart.yaml`, and no other. Set each `metric_id` by hand.
3. Run `./tools/memory_chart_layout_format.py` to write the file in the standard style.
   The pre-commit hook checks the style.
4. Preview the chart without a GPU: `./tools/memory_chart_preview.py <arch>`.
5. Run `python -m pytest tests/unit/memory_chart`. The tests check the ids against the
   panel config, and check that each arrow `style` and `bar` fit the metric's unit.

No Python changes are needed. When a change to a `0300_memory_chart.yaml` adds,
removes, or moves a metric, update the `metric_id` values of its layout.
