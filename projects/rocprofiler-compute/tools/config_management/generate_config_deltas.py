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
Analysis Config Differentiation Script
Generates differences from curr arch directory to prev arch directory.
Output shows what needs to change in prev arch to match curr arch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_management import utils_ruamel as cm_utils  # noqa: E402
from ruamel.yaml.comments import CommentedMap  # noqa: E402


def load_yaml_roundtrip(path: Path) -> Any:
    return cm_utils.load_yaml(path, round_trip=True)


def diff_metric_fields(base_fields, new_fields) -> Optional[CommentedMap]:
    out = CommentedMap()

    for key in new_fields:
        if key not in base_fields or base_fields[key] != new_fields[key]:
            # Preserve the original value with comments
            out[key] = new_fields[key]

    return out if out else None


def descriptions_equal(base_desc, new_desc) -> bool:
    """Check if two descriptions are equal by comparing their string representation."""
    return str(base_desc) == str(new_desc)


def diff_metric_table(
    base_table, new_table
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (additions, modifications, deletions) tuple.
    """
    addition_metrics: list[dict] = []
    modification_metrics: list[dict] = []
    deletion_metrics: list[dict] = []

    base_metrics = base_table.get("metric", {})
    new_metrics = new_table.get("metric", {})

    # Metrics deleted
    for metric in base_metrics:
        if metric not in new_metrics:
            deletion_metrics.append({metric: None})

    # Metrics added or modified
    for metric in new_metrics:
        if metric not in base_metrics:
            # Entire metric is new - preserve original with comments
            addition_metrics.append({metric: new_metrics[metric]})
        else:
            # Field-level diff
            changes = diff_metric_fields(base_metrics[metric], new_metrics[metric])
            if changes:
                modification_metrics.append({metric: changes})

    return addition_metrics, modification_metrics, deletion_metrics


def diff_descriptions(
    base_md, new_md
) -> tuple[Optional[CommentedMap], Optional[CommentedMap], Optional[CommentedMap]]:
    """
    Returns (additions, modifications, deletions) tuple.
    """
    additions = CommentedMap()
    modifications = CommentedMap()
    deletions = CommentedMap()

    # Deletions
    for key in base_md:
        if key not in new_md:
            deletions[key] = None

    # Additions and modifications
    for key in new_md:
        if key not in base_md:
            # New description - preserve original node
            additions[key] = new_md[key]
        else:
            # Check if modified
            if not descriptions_equal(base_md[key], new_md[key]):
                # Preserve original node to maintain style
                modifications[key] = new_md[key]

    return (
        additions if additions else None,
        modifications if modifications else None,
        deletions if deletions else None,
    )


def extract_metric_tables(data_sources) -> list[Any]:
    out = []
    for ds in data_sources:
        if "metric_table" in ds:
            mt = ds["metric_table"]
            table_id = mt.get("id")
            if table_id is not None:
                out.append((table_id, mt))
    return out


def diff_panel(base_config, new_config) -> Optional[dict[str, list[Any]]]:
    """
    Produce delta for a single panel.
    Returns dicts under keys:
       'Addition', 'Deletion', 'Modification'
    or None if no diffs.
    """
    out = {"Addition": [], "Deletion": [], "Modification": []}
    panel_id = base_config["Panel Config"]["id"]

    # Table-level diffs
    base_tables = extract_metric_tables(
        base_config["Panel Config"].get("data source", [])
    )
    new_tables = extract_metric_tables(
        new_config["Panel Config"].get("data source", [])
    )

    # Indexing by table ID to preserve order
    base_by_id = {tid: table for (tid, table) in base_tables}
    new_by_id = {tid: table for (tid, table) in new_tables}

    # Table deletions
    for tid in base_by_id:
        if tid not in new_by_id:
            out["Deletion"].append({
                "Panel Config": {"id": panel_id},
                "metric_tables": [{"metric_table": {"id": tid}}],
            })

    # Table additions + modifications
    for tid in new_by_id:
        if tid not in base_by_id:
            # Entire table is added - preserve original
            out["Addition"].append({
                "Panel Config": {"id": panel_id},
                "metric_tables": [{"metric_table": new_by_id[tid]}],
            })
        else:
            # Check metric-level diffs
            additions, modifications, deletions = diff_metric_table(
                base_by_id[tid], new_by_id[tid]
            )

            if deletions:
                out["Deletion"].append({
                    "Panel Config": {"id": panel_id},
                    "metric_table": {"id": tid, "metrics": deletions},
                })

            if additions:
                out["Addition"].append({
                    "Panel Config": {"id": panel_id},
                    "metric_table": {"id": tid, "metrics": additions},
                })

            if modifications:
                out["Modification"].append({
                    "Panel Config": {"id": panel_id},
                    "metric_table": {"id": tid, "metrics": modifications},
                })

    # Description diffs
    base_md = base_config["Panel Config"].get("metrics_description", {})
    new_md = new_config["Panel Config"].get("metrics_description", {})
    desc_additions, desc_modifications, desc_deletions = diff_descriptions(
        base_md, new_md
    )

    if desc_deletions:
        out["Deletion"].append({
            "Panel Config": {"id": panel_id},
            "metric_descriptions": desc_deletions,
        })

    if desc_additions:
        out["Addition"].append({
            "Panel Config": {"id": panel_id},
            "metric_descriptions": desc_additions,
        })

    if desc_modifications:
        out["Modification"].append({
            "Panel Config": {"id": panel_id},
            "metric_descriptions": desc_modifications,
        })

    # Clean empties
    if not out["Addition"]:
        del out["Addition"]
    if not out["Deletion"]:
        del out["Deletion"]
    if not out["Modification"]:
        del out["Modification"]

    return out if out else None


def generate_arch_delta(base_dir: Path, new_dir: Path) -> CommentedMap:
    """
    Compare all YAML files panel-by-panel.
    """
    out = CommentedMap()
    out["Addition"] = []
    out["Deletion"] = []
    out["Modification"] = []

    base_files = sorted(base_dir.glob("*.yaml"))
    for base_file in base_files:
        new_file = new_dir / base_file.name
        if not new_file.exists():
            continue

        base_config = load_yaml_roundtrip(base_file)
        new_config = load_yaml_roundtrip(new_file)

        diff = diff_panel(base_config, new_config)
        if not diff:
            continue

        if "Addition" in diff:
            out["Addition"].extend(diff["Addition"])
        if "Deletion" in diff:
            out["Deletion"].extend(diff["Deletion"])
        if "Modification" in diff:
            out["Modification"].extend(diff["Modification"])

    # Strip empty categories
    if not out["Addition"]:
        del out["Addition"]
    if not out["Deletion"]:
        del out["Deletion"]
    if not out["Modification"]:
        del out["Modification"]

    return out


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python generate_config_deltas.py <base_arch_dir> <new_arch_dir> <output_delta_yaml>"  # noqa: E501
        )
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    new_dir = Path(sys.argv[2])
    out_file = Path(sys.argv[3])

    delta = generate_arch_delta(base_dir, new_dir)

    cm_utils.save_yaml(delta, out_file)
    print(f"Delta generated at: {out_file}")


if __name__ == "__main__":
    main()
