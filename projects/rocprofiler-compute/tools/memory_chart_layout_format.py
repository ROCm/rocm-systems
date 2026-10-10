#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check memory chart layouts and rewrite them in the canonical style.

Each file is first checked by the layout loader. It is then written with keys in
the loader's order, one block, metric, or arrow per line, nested blocks indented
under their parent, and a blank line between arrows of different block pairs.

Usage:
    tools/memory_chart_layout_format.py           # rewrite every layout
    tools/memory_chart_layout_format.py --check   # list files that need it
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_chart.loader import (
    ARROW_KEYS,
    BLOCK_KEYS,
    ITEM_KEYS,
    LAYOUT_KEYS,
    SCOPE_KEYS,
    Keys,
    layout_files,
    load_layout,
)


def _inline(obj: dict[str, Any], keys: Keys) -> str:
    return ", ".join(
        f"{json.dumps(k)}: {json.dumps(obj[k])}" for k in keys.order if k in obj
    )


def _block(block: dict[str, Any], indent: str) -> list[str]:
    # Scalar keys go on the block's first line; list keys each get their own lines
    lists = [(k, block[k]) for k in BLOCK_KEYS.order if isinstance(block.get(k), list)]
    lists = [(key, entries) for key, entries in lists if entries]
    header = _inline(
        {k: v for k, v in block.items() if not isinstance(v, list)}, BLOCK_KEYS
    )
    if not lists:
        return [f"{indent}{{{header}}}"]
    lines = [f"{indent}{{{header},"]
    for n, (key, entries) in enumerate(lists):
        lines.append(f"{indent}  {json.dumps(key)}: [")
        for i, entry in enumerate(entries):
            sep = "," if i < len(entries) - 1 else ""
            if key == "metrics":  # metric entries, not blocks
                lines.append(f"{indent}    {{{_inline(entry, ITEM_KEYS)}}}{sep}")
            else:
                sub = _block(entry, indent + "    ")
                sub[-1] += sep
                lines.extend(sub)
        lines.append(f"{indent}  ]{',' if n < len(lists) - 1 else ''}")
    lines.append(f"{indent}}}")
    return lines


def _columns(columns: list[list[dict[str, Any]]]) -> list[str]:
    lines = ['  "columns": [']
    for c, column in enumerate(columns):
        lines.append("    [")
        for b, block in enumerate(column):
            sub = _block(block, "      ")
            sub[-1] += "," if b < len(column) - 1 else ""
            lines.extend(sub)
        lines.append(f"    ]{',' if c < len(columns) - 1 else ''}")
    lines.append("  ],")
    return lines


def _arrows(arrows: list[dict[str, Any]]) -> list[str]:
    lines = ['  "arrows": [']
    for i, arrow in enumerate(arrows):
        pair = (arrow["from"], arrow["to"])
        if i and pair != (arrows[i - 1]["from"], arrows[i - 1]["to"]):
            lines.append("")
        sep = "," if i < len(arrows) - 1 else ""
        lines.append(f"    {{{_inline(arrow, ARROW_KEYS)}}}{sep}")
    lines.append("  ]")
    return lines


def format_layout(data: dict[str, Any]) -> str:
    """The canonical text of a layout."""
    lines = ["{"]
    for key in LAYOUT_KEYS.order:
        if key == "columns":
            lines += _columns(data["columns"])
        elif key == "arrows":
            lines += _arrows(data["arrows"])
        elif key == "scope":
            lines.append(f'  "scope": {{{_inline(data["scope"], SCOPE_KEYS)}}},')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(data[key])},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report")
    args = parser.parse_args()
    stale = []
    for path in layout_files():
        load_layout(path)
        text = path.read_text(encoding="utf-8")
        canonical = format_layout(json.loads(text))
        if text != canonical:
            stale.append(path.name)
            if not args.check:
                path.write_text(canonical, encoding="utf-8")
    for name in stale:
        print(f"{'needs formatting' if args.check else 'formatted'}: {name}")
    return 1 if stale and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
