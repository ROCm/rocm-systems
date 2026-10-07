#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Draw a memory chart layout in the terminal, without a GPU or a profile.

Values are placeholders that fit each unit, or come from a JSON file of
{"metric name": value}. Units come from the panel 300 config of the layout's
first architecture, or from --config.

Usage:
    tools/memory_chart_preview.py gfx950
    tools/memory_chart_preview.py src/memory_chart/layouts/gfx1250.json
    tools/memory_chart_preview.py new.json --config path/to/0300_memory_chart.yaml
    tools/memory_chart_preview.py gfx942 --values values.json
    tools/memory_chart_preview.py gfx942 --empty
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import config  # noqa: E402
from memory_chart.loader import Layouts, load_layout  # noqa: E402
from memory_chart.mem_chart import MemChart  # noqa: E402
from memory_chart.units import PLAIN, display_unit, panel_units  # noqa: E402

# Values as shown on the chart, before the unit's scale is undone
_PLACEHOLDERS = {"percent": 42.0, "bandwidth": 123.456, "count": 1234, "value": 12.5}


def placeholder_values(metrics: set[str], units: dict[str, str]) -> dict[str, float]:
    """A plausible value for each metric, by unit."""
    shown = {m: display_unit(units.get(m)) or PLAIN for m in metrics}
    return {m: _PLACEHOLDERS[unit.kind] / unit.scale for m, unit in shown.items()}


def _shipped_units(arch: str) -> dict[str, str]:
    """Units from the shipped panel 300 config of an analysis-config arch."""
    configs = config.rocprof_compute_home / "rocprof_compute_soc" / "analysis_configs"
    path = configs / arch / "0300_memory_chart.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        return panel_units(yaml.safe_load(stream)["Panel Config"])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("layout", help="an architecture (gfx950) or a layout file")
    parser.add_argument("--config", type=Path, help="panel 300 YAML to read units from")
    parser.add_argument("--values", type=Path, help='JSON file of {"metric": value}')
    parser.add_argument("--empty", action="store_true", help="show every value as N/A")
    args = parser.parse_args(argv)

    path = Path(args.layout)
    layout = (
        load_layout(path.resolve())
        if path.suffix == ".json"
        else Layouts.for_arch(args.layout)
    )
    if layout is None:
        parser.error(f"no layout for {args.layout!r}")

    if args.config:
        with args.config.open(encoding="utf-8") as stream:
            units = panel_units(yaml.safe_load(stream)["Panel Config"])
    else:
        units = _shipped_units(layout.archs[0])
    if not units:
        print(f"warning: no panel 300 config for {layout.archs[0]}; showing no units")

    if args.empty:
        values = {}
    elif args.values:
        values = json.loads(args.values.read_text(encoding="utf-8"))
    else:
        values = placeholder_values(layout.metrics(), units)

    title = f"{layout.path.name} ({', '.join(layout.archs)}): {layout.description}"
    print(MemChart(layout, values, units).render(title), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
