#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

"""
parse_config_template.py

Parse panel configuration based on YAML files for an architecture and, optionally,
generate a lightweight template describing panel IDs, titles, aliases, and
data-source ordering.

Usage
-----
Generate a template from an architecture directory:

    python tools/config_management/parse_config_template.py \
        analysis_configs/gfx950 \
        analysis_configs/config_template.yaml \
        --latest-arch gfx950

Inspect an architecture (no template written):

    python tools/config_management/parse_config_template.py \
        analysis_configs/gfx950
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_management import utils_ruamel as cm_utils  # noqa: E402


def normalize_panel_id(panel_id: Optional[int]) -> Optional[int]:
    """Normalize panel ID by dividing by 100 if needed."""
    if panel_id is None:
        return None
    return panel_id // 100 if panel_id >= 100 else panel_id


def normalize_table_id(table_id: Optional[int]) -> Optional[int]:
    """Normalize table ID using modulo 100."""
    if table_id is None:
        return None
    return table_id % 100


def parse_panel_config(yaml_file: Path) -> Optional[dict[str, Any]]:
    """
    Parse a single panel YAML file and extract template-relevant info.

    Returns a dict with:
      - file: panel filename (without leading numeric prefix)
      - panel_id: normalized panel id (id // 100 when >= 100)
      - panel_title: Panel Config.title
      - panel_alias: Panel Config.alias (optional)
      - data_sources: ordered list of
            {type: <key>, id: <normalized table id>, title: <title>}
    or None if the file does not contain a valid Panel Config or fails basic checks.
    """
    data = cm_utils.load_yaml(yaml_file)
    panel_config = data.get("Panel Config")
    if not isinstance(panel_config, dict):
        print(f"WARNING: {yaml_file} has no valid 'Panel Config' mapping, skipping.")
        return None

    # Enforce presence of core panel-level keys
    missing_keys: list[str] = []
    for key in ("id", "title", "data source", "metrics_description"):
        if key not in panel_config:
            missing_keys.append(key)

    if missing_keys:
        missing_str = ", ".join(missing_keys)
        print(
            f"ERROR: {yaml_file} is missing required Panel Config keys: {missing_str}"
        )
        return None

    filename = (
        yaml_file.name.split("_", 1)[1] if "_" in yaml_file.name else yaml_file.name
    )

    raw_panel_id = panel_config.get("id")
    if not isinstance(raw_panel_id, int):
        print(
            f"ERROR: {yaml_file} has non-integer or missing Panel Config.id "
            f"({raw_panel_id!r})"
        )
        return None

    panel_id = normalize_panel_id(raw_panel_id)

    # Extract and normalize data sources
    data_sources: list[dict[str, Any]] = []
    ds_list = panel_config.get("data source", [])
    if not isinstance(ds_list, list):
        print(
            f"ERROR: {yaml_file} has non-list 'data source' field "
            f"({type(ds_list).__name__})"
        )
        return None

    for ds in ds_list:
        if not isinstance(ds, dict):
            print(f"WARNING: {yaml_file} has non-dict data source entry: {ds!r}")
            continue
        for key, value in ds.items():
            if isinstance(value, dict) and "id" in value and "title" in value:
                norm_id = normalize_table_id(value["id"])
                data_sources.append({
                    "type": key,
                    "id": norm_id,
                    "title": value["title"],
                })

    return {
        "file": filename,
        "panel_id": panel_id,
        "panel_title": panel_config.get("title"),
        "panel_alias": panel_config.get("alias"),
        "data_sources": data_sources,
    }


def build_template_from_directory(
    directory: Path,
    existing_panels_by_id: Optional[dict[int, dict]],
) -> list[dict]:
    panels: list[dict] = []
    errors = 0

    for yaml_file in sorted(directory.glob("*.yaml")):
        info = parse_panel_config(yaml_file)
        if info is None:
            errors += 1
            continue

        panel_id = info.get("panel_id")

        if (
            existing_panels_by_id
            and panel_id is not None
            and panel_id in existing_panels_by_id
        ):
            old_panel = existing_panels_by_id[panel_id]

            # Preserve panel_alias unless explicitly set by panel YAML
            if info.get("panel_alias") is None and "panel_alias" in old_panel:
                info["panel_alias"] = old_panel["panel_alias"]

        panels.append(info)

    # Deterministic ordering for stable templates
    panels.sort(key=lambda p: (p["panel_id"], p["file"]))

    if errors:
        print(
            f"\nEncountered {errors} panel file(s) with structural errors "
            "while building template."
        )

    return panels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse panel YAML files for an architecture and optionally generate "
            "a config_template-style YAML describing panel IDs and data sources."
        )
    )
    parser.add_argument("directory", help="Directory containing panel YAML files")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output YAML file (optional). If omitted, only a summary is printed.",
    )
    parser.add_argument(
        "--latest-arch",
        help=(
            "Specify this architecture as latest (adds 'latest_arch' metadata "
            "to the generated template). Only used when an output file is given."
        ),
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: '{args.directory}' is not a valid directory")
        sys.exit(1)

    existing_template = None
    if args.output and Path(args.output).exists():
        existing_template = cm_utils.load_yaml(Path(args.output))

    existing_panels_by_id = {}
    if existing_template:
        for p in existing_template.get("panels", []):
            pid = p.get("panel_id")
            if pid is not None:
                existing_panels_by_id[pid] = p

    panels = build_template_from_directory(
        directory,
        existing_panels_by_id=existing_panels_by_id if args.output else None,
    )

    if not panels:
        print("No valid panel YAML files found; nothing to do.")
        sys.exit(1)

    # Always show a human-readable summary.
    print(f"Found {len(panels)} panel(s) in {directory}:")
    for panel in panels:
        print(f"\nFile: {panel['file']}")
        print(f"Panel ID: {panel['panel_id']}")
        print(f"Panel Title: {panel['panel_title']}")
        if panel["panel_alias"]:
            print(f"Panel Alias: {panel['panel_alias']}")
        print(f"\nData Sources ({len(panel['data_sources'])}):")
        for ds in panel["data_sources"]:
            print(f"  - {ds['type']}: {ds['id']} - {ds['title']}")

    # Optionally write a template YAML.
    if args.output:
        output_data: Any = {"panels": panels}
        if args.latest_arch:
            output_data = {"latest_arch": args.latest_arch, "panels": panels}
        cm_utils.save_yaml(output_data, Path(args.output))
        print(f"\nTemplate saved to: {args.output}")


if __name__ == "__main__":
    main()
