#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Validate memory chart layout JSON files.

Runs three checks:
  1. Schema validation (required fields, types, referential integrity)
  2. Metric alignment (JSON metrics vs YAML analysis configs)
  3. Manifest hash integrity (tools/memory_chart_manifest.json)

Usage:
    python tools/validate_layout_schema.py
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LAYOUTS_DIR = _PROJECT_ROOT / "src" / "memory_chart" / "layouts"
_CONFIGS_DIR = _PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
_MANIFEST_PATH = _PROJECT_ROOT / "tools" / "memory_chart_manifest.json"
_MEMORY_CHART_YAML = "0300_memory_chart.yaml"

CURRENT_SCHEMA_VERSION = 2

VALID_DIRECTIONS = frozenset({"forward", "backward", "both"})
VALID_POSITIONS = frozenset({"grid", "above", "below"})
VALID_CATEGORIES = frozenset({
    "read",
    "write",
    "atomic",
    "hit",
    "util",
    "stall",
    "bw",
    "info",
})

ARCH_TO_LAYOUT: dict[str, str] = {
    "gfx908": "gfx90x",
    "gfx90a": "gfx90x",
    "gfx940": "gfx94x",
    "gfx942": "gfx94x",
    "gfx950": "gfx950",
    "gfx115x": "gfx115x",
    "gfx1250": "gfx1250",
}


# -------------------------------------------------------------------
# Schema validation
# -------------------------------------------------------------------


def validate_layout(data: dict[str, Any]) -> list[str]:
    """Validate a single layout dict. Returns errors (empty = valid)."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Root must be a JSON object"]

    if data.get("schema_version") != CURRENT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {CURRENT_SCHEMA_VERSION}, "
            f"got {data.get('schema_version')!r}"
        )

    if not isinstance(data.get("arch"), str) or not data["arch"]:
        errors.append("arch must be a non-empty string")

    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        errors.append("blocks must be a list")
        return errors

    block_ids = validate_blocks(blocks, errors)
    validate_arrows(data.get("arrows", []), block_ids, errors)
    return errors


def validate_blocks(
    blocks: list[Any],
    errors: list[str],
) -> set[str]:
    """Validate block entries. Returns the set of valid block IDs."""
    block_ids: set[str] = set()

    for i, block in enumerate(blocks):
        prefix = f"blocks[{i}]"
        if not isinstance(block, dict):
            errors.append(f"{prefix}: must be an object")
            continue

        bid = block.get("id")
        if not isinstance(bid, str) or not bid:
            errors.append(f"{prefix}: id must be a non-empty string")
        elif bid in block_ids:
            errors.append(f"{prefix}: duplicate block id {bid!r}")
        else:
            block_ids.add(bid)

        if not isinstance(block.get("title"), str):
            errors.append(f"{prefix}: title must be a string")
        if not isinstance(block.get("column"), int):
            errors.append(f"{prefix}: column must be an integer")
        if "order" in block and not isinstance(block["order"], int):
            errors.append(f"{prefix}: order must be an integer")

        position = block.get("position", "grid")
        if position not in VALID_POSITIONS:
            errors.append(
                f"{prefix}: position must be one of "
                f"{sorted(VALID_POSITIONS)}, got {position!r}"
            )

        for child in block.get("children", []):
            if not isinstance(child, str):
                errors.append(f"{prefix}: children entries must be strings")

        validate_content(block.get("content", []), prefix, errors)

    for i, block in enumerate(blocks):
        for child in block.get("children", []):
            if child not in block_ids:
                errors.append(f"blocks[{i}]: child {child!r} not found in block ids")

    return block_ids


def validate_arrows(
    arrows: Any,  # noqa: ANN401
    block_ids: set[str],
    errors: list[str],
) -> None:
    """Validate arrow entries against known block IDs."""
    if not isinstance(arrows, list):
        errors.append("arrows must be a list")
        return

    for i, arrow in enumerate(arrows):
        prefix = f"arrows[{i}]"
        if not isinstance(arrow, dict):
            errors.append(f"{prefix}: must be an object")
            continue

        for field in ("from", "to"):
            ref = arrow.get(field)
            if not isinstance(ref, str):
                errors.append(f"{prefix}: {field} must be a string")
            elif ref not in block_ids:
                errors.append(f"{prefix}: {field} {ref!r} not found in block ids")

        direction = arrow.get("direction")
        if direction not in VALID_DIRECTIONS:
            errors.append(
                f"{prefix}: direction must be one of "
                f"{sorted(VALID_DIRECTIONS)}, "
                f"got {direction!r}"
            )

        if not isinstance(arrow.get("metric"), str):
            errors.append(f"{prefix}: metric must be a string")
        if not isinstance(arrow.get("title"), str):
            errors.append(f"{prefix}: title must be a string")

        category = arrow.get("category")
        if category not in VALID_CATEGORIES:
            errors.append(
                f"{prefix}: category must be one of "
                f"{sorted(VALID_CATEGORIES)}, "
                f"got {category!r}"
            )


def validate_content(
    content: list[Any],
    parent_prefix: str,
    errors: list[str],
) -> None:
    """Validate block content metric items."""
    if not isinstance(content, list):
        errors.append(f"{parent_prefix}: content must be a list")
        return
    for j, item in enumerate(content):
        prefix = f"{parent_prefix}.content[{j}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        if not isinstance(item.get("metric"), str):
            errors.append(f"{prefix}: metric must be a string")
        if not isinstance(item.get("title"), str):
            errors.append(f"{prefix}: title must be a string")
        category = item.get("category")
        if category not in VALID_CATEGORIES:
            errors.append(
                f"{prefix}: category must be one of "
                f"{sorted(VALID_CATEGORIES)}, "
                f"got {category!r}"
            )


# -------------------------------------------------------------------
# Metric alignment
# -------------------------------------------------------------------


def layout_metric_names(data: dict[str, Any]) -> set[str]:
    """Collect all metric names referenced in a layout."""
    names: set[str] = set()
    for block in data.get("blocks", []):
        for item in block.get("content", []):
            names.add(item["metric"])
    for arrow in data.get("arrows", []):
        names.add(arrow["metric"])
    return names


def yaml_metric_names(arch: str) -> set[str]:
    """Extract metric names from an arch's 0300_memory_chart.yaml."""
    yaml_path = _CONFIGS_DIR / arch / _MEMORY_CHART_YAML
    if not yaml_path.exists():
        return set()
    with yaml_path.open(encoding="utf-8") as fobj:
        config = yaml.safe_load(fobj)
    names: set[str] = set()
    panel = config.get("Panel Config", {})
    for source in panel.get("data source", []):
        metric_block = source.get("metric_table", {}).get("metric", {})
        if isinstance(metric_block, dict):
            names.update(metric_block.keys())
    return names


def check_metric_alignment() -> list[str]:
    """Verify JSON layout metrics match YAML configs."""
    errors: list[str] = []
    for arch, layout_name in ARCH_TO_LAYOUT.items():
        layout_path = _LAYOUTS_DIR / f"{layout_name}.json"
        if not layout_path.exists():
            errors.append(f"{arch}: layout {layout_name}.json missing")
            continue
        with layout_path.open(encoding="utf-8") as fobj:
            layout = json.load(fobj)
        json_metrics = layout_metric_names(layout)
        yaml_metrics = yaml_metric_names(arch)
        if not yaml_metrics:
            continue
        missing_in_yaml = json_metrics - yaml_metrics
        if missing_in_yaml:
            errors.append(
                f"{arch}: JSON metrics not in YAML: {sorted(missing_in_yaml)}"
            )
        missing_in_json = yaml_metrics - json_metrics
        if missing_in_json:
            errors.append(
                f"{arch}: YAML metrics not in JSON: {sorted(missing_in_json)}"
            )
    return errors


# -------------------------------------------------------------------
# Manifest hash integrity
# -------------------------------------------------------------------


def compute_file_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_manifest() -> list[str]:
    """Verify manifest hashes match actual layout files."""
    errors: list[str] = []
    if not _MANIFEST_PATH.exists():
        errors.append(
            "tools/memory_chart_manifest.json not found. "
            "Run: python tools/generate_layout_manifest.py"
        )
        return errors

    with _MANIFEST_PATH.open(encoding="utf-8") as fobj:
        manifest = json.load(fobj)

    stored_files = manifest.get("files", {})
    actual_names = {
        p.name for p in _LAYOUTS_DIR.glob("*.json") if p.name != "manifest.json"
    }

    for name in sorted(actual_names - set(stored_files)):
        errors.append(f"Layout {name} not in manifest")
    for name in sorted(set(stored_files) - actual_names):
        errors.append(f"Manifest lists {name} but file missing")
    for name in sorted(actual_names & set(stored_files)):
        actual_md5 = compute_file_md5(_LAYOUTS_DIR / name)
        if actual_md5 != stored_files[name]:
            errors.append(
                f"{name}: md5 mismatch "
                f"(manifest={stored_files[name]}, "
                f"actual={actual_md5})"
            )

    if errors:
        errors.append("Run: python tools/generate_layout_manifest.py")
    return errors


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------


def main() -> int:
    """Run all layout validation checks."""
    layout_files = sorted(
        p for p in _LAYOUTS_DIR.glob("*.json") if p.name != "manifest.json"
    )

    if not layout_files:
        print(f"No layout files found in {_LAYOUTS_DIR}")
        return 2

    all_errors: list[str] = []

    # 1. Schema validation
    for layout_path in layout_files:
        with layout_path.open(encoding="utf-8") as fobj:
            data = json.load(fobj)
        errors = validate_layout(data)
        if errors:
            all_errors.append(f"\n[Schema] {layout_path.name}:")
            for error in errors:
                all_errors.append(f"  - {error}")

    # 2. Metric alignment
    metric_errors = check_metric_alignment()
    if metric_errors:
        all_errors.append("\n[Metric alignment]:")
        for error in metric_errors:
            all_errors.append(f"  - {error}")

    # 3. Manifest integrity
    manifest_errors = check_manifest()
    if manifest_errors:
        all_errors.append("\n[Manifest]:")
        for error in manifest_errors:
            all_errors.append(f"  - {error}")

    if all_errors:
        print("LAYOUT VALIDATION ERRORS:")
        for line in all_errors:
            print(line)
        return 1

    print(
        f"All {len(layout_files)} layout files pass validation "
        f"(schema, metric alignment, manifest)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
