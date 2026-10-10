#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Draw a memory chart layout in the terminal, without a GPU or a profile.

Values are placeholders that fit each unit. Units come from the memory chart panel
config of the layout's first architecture.

Usage:
    tools/memory_chart_preview.py gfx950
    tools/memory_chart_preview.py src/memory_chart/layouts/gfx1250.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_chart.loader import Layouts, PanelConfigs, load_layout
from memory_chart.render import MemChart
from memory_chart.units import PLAIN, display_unit, panel_units

# Values as shown on the chart, before the unit's scale is undone
_PLACEHOLDERS = {"percent": 42.0, "bandwidth": 123.456, "count": 1234, "value": 12.5}


def placeholder_values(metrics: set[str], units: dict[str, str]) -> dict[str, float]:
    """A plausible value for each metric, by unit."""
    shown = {m: display_unit(units.get(m)) or PLAIN for m in metrics}
    return {m: _PLACEHOLDERS[unit.kind] / unit.scale for m, unit in shown.items()}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("layout", help="an architecture (gfx950) or a layout file")
    args = parser.parse_args(argv)

    path = Path(args.layout)
    layout = (
        load_layout(path.resolve())
        if path.suffix == ".json"
        else Layouts.for_arch(args.layout)
    )
    if layout is None:
        parser.error(f"no layout for {args.layout!r}")

    panel = PanelConfigs.for_arch(layout.archs[0])
    if panel is None:
        print(f"warning: no memory chart panel for {layout.archs[0]}; showing no units")
    units = panel_units(panel) if panel else {}

    values = placeholder_values(layout.metrics(), units)
    title = f"{layout.path.name} ({', '.join(layout.archs)}): {layout.description}"
    print(MemChart(layout, values, units).render(title), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
