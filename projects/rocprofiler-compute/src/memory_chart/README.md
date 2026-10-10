# Memory chart layouts

`analyze` draws the memory chart panel from a JSON layout file in
[`layouts/`](layouts/). There is one file per architecture family. A layout lists the
blocks of the chart, how they nest, and which memory chart panel metric each block and
arrow shows. [`loader.py`](loader.py) checks the file and works out where each block
goes and which way each arrow points. [`render.py`](render.py) draws it.

## Adding an architecture

1. Copy the layout of the closest architecture, for example `gfx950.json`. Set `archs`
   to the analysis-config directory names it serves.
2. Edit its blocks and arrows until the chart shows every metric of the architecture's
   `0300_memory_chart.yaml`, and nothing else.
3. Check it without a GPU: `./tools/memory_chart_preview.py path/to/layout.json`.
4. Run `python -m pytest tests/unit/memory_chart`.

`analyze` and the tests find a layout by its `archs`, so no Python changes are needed.
The one exception is a metric whose unit has no display rule in [`units.py`](units.py);
the tests list any such unit. The pre-commit hook runs
`./tools/memory_chart_layout_format.py`, which rewrites layouts in the standard style.

## Example

A small layout with three columns. JSON has no comments; they are added here only to
explain each part.

```jsonc
{
  "archs": ["gfx950"],                     // analysis-config directories it serves
  "description": "CU, L2, and Data Fabric.",
  "scope": {"labels": ["GPU", "Fabric"],   // the two labels of the scope bar
            "split": "data_fabric"},       // the "Fabric" label starts above this block
  "columns": [
    [                                      // column 0: the compute side
      {"id": "cu", "title": "Compute Units",
        "metrics": [
          {"metric": "VGPR", "title": "vGPRs", "category": "util"}
        ]
      }
    ],
    [                                      // column 1
      {"id": "l2", "title": "L2", "stall_level": "GL2",
        "metrics": [
          {"metric": "L2 Hit", "title": "Hit", "category": "hit"}
        ]
      }
    ],
    [                                      // column 2
      {"id": "data_fabric", "title": "Data Fabric",
        "below": [                         // a labelled box drawn under this block
          {"id": "pcie", "title": "PCIe"}
        ]
      }
    ]
  ],
  "arrows": [
    {"from": "cu", "to": "l2", "metric": "Flat Read", "title": "Read",
     "category": "read", "group": "Request"},
    {"from": "l2", "to": "data_fabric", "metric": "L2-Fabric Read BW",
     "title": "Read BW", "category": "read"},
    {"from": "data_fabric", "to": "pcie", "metric": "PCIe Read BW",
     "title": "Read BW", "category": "read"}
  ]
}
```

## Fields

The top level of a layout has these fields. All are required.

| Field | Meaning |
|---|---|
| `archs` | The analysis-config directory names the layout serves. |
| `description` | One line on what the chart shows. |
| `scope` | `labels` is the two labels of the bar above the chart. `split` is the block where the second label starts. It must not be in the first column. |
| `columns` | The columns from left to right. Each column is a list of blocks from top to bottom. The first column is the compute side, drawn as a compact list. |
| `arrows` | The metrics drawn between blocks. |

### Blocks

Every block has an `id`, which must be unique in the layout, and a `title`, which is
shown on the chart. A block can also have:

- `metrics`: the metrics shown inside the block, as `{"metric", "title", "category"}`
  entries. `metric` is the name of the metric in the memory chart panel config, and
  `title` is its label on the chart.
- `note`: a short fixed label, such as the memory type.
- `children`: blocks drawn inside this one. Metrics and arrows go on the children, not on
  this block.
- `above`, `below`: labelled boxes drawn above or below this block, such as xGMI and
  PCIe. They have only an `id` and a `title`.
- `stall_level`: the memory bandwidth analysis level (`GL1`, `GL2`, or `EA`) whose active
  stalls are listed in this block.

### Arrows

An arrow has `from`, `to`, `metric`, `title`, and `category`, and an optional `group`.

- `from` is a block in a column. `to` is a block in the next column, or a box attached
  to `from` with `above` or `below`. Arrows to an attached box are drawn as `||`
  connectors.
- An arrow with children at either end is an error. Point it at one of the children.
- `group` is a heading drawn over consecutive arrows that share it, such as
  `Buffer Request`.
- An arrow whose metric is a count, such as `(Requests + $normUnit)`, is drawn as a
  request (`Read : 119`). Other arrows show a label, a value, and an arrow line.

### Categories

A category sets the color of a metric or arrow, and the direction of an arrow.

| Category | Used on | Drawn as |
|---|---|---|
| `read` | arrows, metrics | Read color. The arrow points back to the requester (`<--`). |
| `write` | arrows, metrics | Write color. The arrow points forward (`-->`). |
| `atomic` | arrows, metrics | Atomic color. The arrow points both ways (`<->`). |
| `neutral` | arrows, metrics | Default color. The arrow points both ways. Use it for mixed traffic or values outside the legend. |
| `util`, `hit`, `stall` | metrics | Utilization, hit, or stall color. |
| `bw` | metrics | Bandwidth color. |

Outside the first column, a percent metric in `util`, `hit`, `stall`, or `neutral` also
gets a progress bar.
