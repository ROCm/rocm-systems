#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Draw a memory chart layout in the terminal, without a GPU or a profile.

Values are placeholders that fit each unit. Units come from the memory chart panel
config of the layout's first architecture.

Usage:
    tools/memory_chart_preview.py gfx950
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_chart.loader import Layouts, PanelConfigs
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
    parser.add_argument("arch", help="a GPU architecture, such as gfx950")
    args = parser.parse_args(argv)

    layout = Layouts.for_arch(args.arch)
    if layout is None:
        parser.error(f"no layout for {args.arch!r}")

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
