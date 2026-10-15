#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check memory chart layouts and rewrite them in the canonical style.

The loader checks each file first. The file is then written with keys in the
loader's order, blocks in drawing order (each block followed by its nested and
attached blocks), one block, metric, or arrow per line, and a blank line between
columns and between arrows of different block pairs.

Usage:
    tools/memory_chart_layout_format.py           # rewrite every layout
    tools/memory_chart_layout_format.py --check   # report unformatted layouts and
                                                  # exit 1 without changing them
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
    LayoutError,
    layout_files,
    layout_from_json,
    load_layout,
)


def _inline(obj: dict[str, Any], keys: Keys) -> str:
    return ", ".join(
        f"{json.dumps(k)}: {json.dumps(obj[k])}" for k in keys.order if k in obj
    )


def _block(block: dict[str, Any]) -> list[str]:
    header = _inline({k: v for k, v in block.items() if k != "content"}, BLOCK_KEYS)
    content = block.get("content") or []
    if not content:
        return [f"    {{{header}}}"]
    lines = [f'    {{{header}, "content": [']
    for i, item in enumerate(content):
        sep = "," if i < len(content) - 1 else ""
        lines.append(f"      {{{_inline(item, ITEM_KEYS)}}}{sep}")
    lines.append("    ]}")
    return lines


def _blocks(data: dict[str, Any]) -> list[str]:
    by_id = {block["id"]: block for block in data["blocks"]}
    ordered = list(layout_from_json(Path("layout"), data).blocks())
    lines = ['  "blocks": [']
    for i, block in enumerate(ordered):
        if i and block.position == "grid" and not block.parent:
            if block.column != ordered[i - 1].column:
                lines.append("")
        sub = _block(by_id[block.id])
        sub[-1] += "," if i < len(ordered) - 1 else ""
        lines.extend(sub)
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
        if key == "blocks":
            lines += _blocks(data)
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="report unformatted layouts and exit 1 without changing them",
    )
    args = parser.parse_args()
    stale = []
    for path in layout_files():
        try:
            load_layout(path)
        except LayoutError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
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
