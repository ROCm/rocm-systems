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
Parse panel configuration based on YAML files for an architecture.
Usage: python parse_yaml.py <directory_path> [output_file.yaml]
"""

import sys
from pathlib import Path

import yaml


def parse_panel_config(yaml_file):
    """Parse a single YAML file and extract panel and data source info."""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    if not data or "Panel Config" not in data:
        return None

    panel_config = data["Panel Config"]

    # Strip panel ID prefix from filename
    filename = (
        yaml_file.name.split("_", 1)[1] if "_" in yaml_file.name else yaml_file.name
    )

    # Normalize panel ID (divide by 100 if >= 100)
    panel_id = panel_config.get("id")
    if panel_id and panel_id >= 100:
        panel_id = panel_id // 100

    # Parse data sources
    data_sources = [
        {"type": key, "id": value["id"] % 100, "title": value["title"]}
        for ds in panel_config.get("data source", [])
        for key, value in ds.items()
        if isinstance(value, dict) and "id" in value and "title" in value
    ]

    return {
        "file": filename,
        "panel_id": panel_id,
        "panel_title": panel_config.get("title"),
        "data_sources": data_sources,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_yaml.py <directory_path> [output_file.yaml]")
        sys.exit(1)

    directory = Path(sys.argv[1])

    if not directory.is_dir():
        print(f"Error: '{sys.argv[1]}' is not a valid directory")
        sys.exit(1)

    # Parse all YAML files
    results = [
        parsed
        for yaml_file in sorted(directory.glob("*.yaml"))
        if (parsed := parse_panel_config(yaml_file))
    ]

    if not results:
        print("No valid panel configurations found.")
        sys.exit(1)

    # Display results
    for panel in results:
        print(f"\n{'=' * 80}")
        print(f"File: {panel['file']}")
        print(f"Panel ID: {panel['panel_id']}")
        print(f"Panel Title: {panel['panel_title']}")
        print(f"\nData Sources ({len(panel['data_sources'])}):")
        for ds in panel["data_sources"]:
            print(f"  - {ds['type']}: {ds['id']} - {ds['title']}")

    # Save if output file specified
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            yaml.dump(results, f, sort_keys=False, default_flow_style=False)
        print(f"\nResults saved to: {sys.argv[2]}")


if __name__ == "__main__":
    main()
