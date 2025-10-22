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
Validate panel YAML files against base template ordering.
Checks that panel configs match expected structure, IDs, titles, and data source order.

Usage:
    python verify_against_config_template.py <analysis_configs_dir> <template_yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml


def normalize_panel_id(panel_id: int) -> int:
    """Normalize panel ID by dividing by 100."""
    return panel_id // 100 if panel_id and panel_id >= 100 else panel_id


def normalize_table_id(table_id: int) -> Optional[int]:
    """Normalize table ID using modulo 100."""
    return table_id % 100 if table_id else None


def load_template(template_file: Path) -> dict[int, dict]:
    """Load template and create lookup by normalized panel ID."""
    with open(template_file) as f:
        data = yaml.safe_load(f) or {}

    panels = data.get("panels", [])
    lookup: dict[int, dict] = {}
    for panel in panels:
        pid = normalize_panel_id(panel["panel_id"])
        lookup[pid] = {
            "panel_title": panel["panel_title"],
            "panel_alias": panel.get("panel_alias"),
            "data_sources": [
                {"type": ds["type"], "id": ds["id"], "title": ds["title"]}
                for ds in panel.get("data_sources", [])
            ],
        }
    return lookup


def extract_panel_info(yaml_file: Path) -> Optional[dict]:
    """Extract panel config info from YAML file."""
    with open(yaml_file) as f:
        data = yaml.safe_load(f) or {}
    if "Panel Config" not in data:
        return None

    panel_config = data["Panel Config"]
    data_sources = []
    for ds in panel_config.get("data source", []):
        for key, value in ds.items():
            if isinstance(value, dict) and "id" in value and "title" in value:
                data_sources.append({
                    "type": key,
                    "id": normalize_table_id(value["id"]),
                    "title": value["title"],
                })

    return {
        "panel_id": normalize_panel_id(panel_config.get("id")),
        "panel_title": panel_config.get("title"),
        "data_sources": data_sources,
    }


def validate_panel(
    yaml_file: Path, panel_info: dict, template: dict[int, dict], stats: dict
) -> None:
    """Validate a single panel YAML against template."""
    panel_id = panel_info["panel_id"]
    file_path = f"{yaml_file.parent.name}/{yaml_file.name}"

    if panel_id not in template:
        print(f"WARNING [{file_path}]: Panel ID {panel_id} not found in template")
        stats["warnings"] += 1
        return

    expected = template[panel_id]
    errors: list[str] = []
    warnings: list[str] = []

    if panel_info["panel_title"] != expected["panel_title"]:
        errors.append(
            f"Panel title mismatch: expected '{expected['panel_title']}', "
            f"got '{panel_info['panel_title']}'"
        )

    if len(panel_info["data_sources"]) != len(expected["data_sources"]):
        errors.append(
            f"Data source count mismatch: expected {len(expected['data_sources'])}, "
            f"got {len(panel_info['data_sources'])}"
        )

    for i, actual_ds in enumerate(panel_info["data_sources"]):
        matching_idx = next(
            (
                j
                for j, exp_ds in enumerate(expected["data_sources"])
                if actual_ds["id"] == exp_ds["id"]
                and actual_ds["title"] == exp_ds["title"]
                and actual_ds["type"] == exp_ds["type"]
            ),
            None,
        )
        if matching_idx is None:
            errors.append(
                f"Data source {i + 1}: No matching entry in template for "
                f"{actual_ds['type']} id={actual_ds['id']} title='{actual_ds['title']}'"
            )
        elif matching_idx != i:
            warnings.append(
                f"Data source {i + 1}: Order mismatch - appears at position {i + 1} "
                f"but expected at position {matching_idx + 1}"
            )

    if errors:
        print(f"ERROR [{file_path}]:")
        for error in errors:
            print(f"  - {error}")
        stats["errors"] += len(errors)
        stats["failed_files"] += 1
    elif warnings:
        print(f"WARNING [{file_path}]:")
        for warning in warnings:
            print(f"  - {warning}")
        stats["warnings"] += len(warnings)
        stats["passed_files"] += 1
    else:
        print(f"PASS [{file_path}]")
        stats["passed_files"] += 1


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python verify_against_config_template.py "
            "<analysis_configs_dir> <template_yaml>"
        )
        sys.exit(1)

    configs_dir = Path(sys.argv[1])
    template_file = Path(sys.argv[2])

    if not configs_dir.is_dir():
        print(f"Error: {configs_dir} is not a directory")
        sys.exit(1)
    if not template_file.is_file():
        print(f"Error: {template_file} is not a file")
        sys.exit(1)

    print(f"Loading template from {template_file}")
    template = load_template(template_file)
    print(f"Template loaded: {len(template)} panels\n")

    stats = {
        "total_files": 0,
        "passed_files": 0,
        "failed_files": 0,
        "errors": 0,
        "warnings": 0,
    }

    for arch_dir in sorted(configs_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        print(f"{'=' * 80}\nValidating architecture: {arch_dir.name}\n{'=' * 80}")
        for yaml_file in sorted(arch_dir.glob("*.yaml")):
            stats["total_files"] += 1
            panel_info = extract_panel_info(yaml_file)
            if panel_info:
                validate_panel(yaml_file, panel_info, template, stats)
            else:
                print(f"ERROR [{arch_dir.name}/{yaml_file.name}]: Invalid panel config")
                stats["errors"] += 1
                stats["failed_files"] += 1
        print()

    print(f"{'=' * 80}\nVALIDATION SUMMARY\n{'=' * 80}")
    print(f"Total files checked: {stats['total_files']}")
    print(f"Passed: {stats['passed_files']}")
    print(f"Failed: {stats['failed_files']}")
    print(f"Total errors: {stats['errors']}")
    print(f"Total warnings: {stats['warnings']}")

    if stats["failed_files"] > 0:
        print("\nValidation FAILED")
        sys.exit(1)
    elif stats["warnings"] > 0:
        print("\nValidation PASSED with warnings")
    else:
        print("\nValidation PASSED")


if __name__ == "__main__":
    main()
