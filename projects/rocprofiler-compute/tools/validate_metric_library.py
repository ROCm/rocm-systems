#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: validate the metric library.

For every arch in src/rocprof_compute_soc/metric_library/, check that:
  1. each file loads, and a child file inherits from its parent arch's file;
  2. metrics and variables have the required fields and types;
  3. metric ids look like "{category}.{subcategory}_{name}", name hardware rather than
     a view, and are unique on the arch;
  4. units use one spelling.

test_metric_library_equivalence checks that the library matches analysis_configs.
"""

import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.metric_library import (  # noqa: E402
    METRIC_LIBRARY_PARENT_ARCH,
    VARIABLES_FILENAME,
    arch_files,
    default_metric_library_dir,
)
from utils.yaml_inherit_loader import (  # noqa: E402
    INHERIT_KEY,
    InheritanceError,
    InheritLoader,
    load_inherited_yaml,
)
from vendored import yaml  # noqa: E402

# "{category}.{subcategory}_{name}": lowercase words joined by underscores.
METRIC_ID_RE = re.compile(r"^[a-z][a-z0-9]*\.[a-z0-9]+(_[a-z0-9]+)*$")
# Categories named after panels rather than hardware.
VIEW_CATEGORIES = ("system", "memchart", "roofline", "wavefront")

# Alias -> canonical unit. A display scale such as Bytes/s vs GB/s is not stored here.
UNIT_ALIASES = {
    "Gbps": "GB/s",
    "Bytes/s": "GB/s",
    "GFLOPs": "GFLOP/s",
    "GIOPs": "GIOP/s",
    "(Req + $normUnit)": "(Requests + $normUnit)",
    "Instr/cycle": "Instructions per Cycle",
    "Waves/cycle": "Wavefronts per Cycle",
    "Wave": "Wavefronts",
    "Waves": "Wavefronts",
}

METRIC_REQUIRED_FIELDS = ("name", "formula", "unit", "description")
METRIC_OPTIONAL_FIELDS = ("peak", "coll_level")
VARIABLE_FIELDS = ("formula",)


def validate(metric_dir: Path) -> list[str]:
    """Validate every arch in a metric library tree. Return error messages."""
    errors: list[str] = []
    for arch in METRIC_LIBRARY_PARENT_ARCH:
        if (metric_dir / arch).is_dir():
            errors.extend(validate_arch(metric_dir, arch))
    errors.extend(
        f"[{path.name}] is not a known metric library arch"
        for path in sorted(metric_dir.iterdir())
        if path.is_dir() and path.name not in METRIC_LIBRARY_PARENT_ARCH
    )
    return errors


def validate_arch(metric_dir: Path, arch: str) -> list[str]:
    """Validate one arch's library files."""
    errors: list[str] = []
    metric_ids: set[str] = set()
    for name, path in arch_files(metric_dir, arch).items():
        source = f"{path.parent.name}/{name}"
        if path.parent.name == arch:
            errors.extend(_inherit_errors(metric_dir, arch, path, source))
        try:
            document = load_inherited_yaml(path)
        except (InheritanceError, yaml.YAMLError) as error:
            errors.append(f"[{source}] {error}")
            continue
        if name == VARIABLES_FILENAME:
            errors.extend(_variables_errors(document, source))
        else:
            errors.extend(_metrics_errors(document, source, metric_ids))
    return errors


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate(default_metric_library_dir())
    if errors:
        print("Metric library validation failed:\n")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Metric library validation passed.")
    return 0


# ---------------------------------------------------------------------------
# Library files
# ---------------------------------------------------------------------------


def _inherit_errors(metric_dir: Path, arch: str, path: Path, source: str) -> list[str]:
    parent = METRIC_LIBRARY_PARENT_ARCH[arch]
    parent_path = arch_files(metric_dir, parent).get(path.name) if parent else None
    return _parent_errors(path, _declared_parent(path), parent_path, source)


def _declared_parent(path: Path) -> object:
    """Return the file's "inherit" value, or None if it does not load."""
    try:
        with open(path, encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=InheritLoader)
    except (InheritanceError, yaml.YAMLError):
        return None
    return raw.get(INHERIT_KEY) if isinstance(raw, dict) else None


def _parent_errors(
    path: Path, named: object, parent_path: Optional[Path], source: str
) -> list[str]:
    if parent_path is None:
        return [] if named is None else [f"[{source}] has no parent file to inherit"]
    if named is None or (path.parent / named).resolve() != parent_path.resolve():
        expected = f"{parent_path.parent.name}/{parent_path.name}"
        return [f"[{source}] must inherit from {expected}"]
    return []


def _variables_errors(document: dict[str, Any], source: str) -> list[str]:
    errors = []
    for name, entry in document.items():
        errors.extend(_field_errors(entry, VARIABLE_FIELDS, (), source, str(name)))
        if isinstance(entry, dict):
            errors.extend(_text_errors(entry, VARIABLE_FIELDS, source, str(name)))
    return errors


def _metrics_errors(
    document: dict[str, Any], source: str, metric_ids: set[str]
) -> list[str]:
    errors = []
    for metric_id, entry in document.items():
        item = str(metric_id)
        if not METRIC_ID_RE.match(item):
            errors.append(
                f"[{source}] {item}: the id must look like "
                "'{category}.{subcategory}_{name}'"
            )
        if item.split(".")[0] in VIEW_CATEGORIES:
            errors.append(
                f"[{source}] {item}: the id names a view; use a hardware category"
            )
        if item in metric_ids:
            errors.append(f"[{source}] {item}: the id is defined twice on the arch")
        metric_ids.add(item)
        errors.extend(_metric_errors(entry, source, item))
    return errors


def _metric_errors(entry: object, source: str, item: str) -> list[str]:
    errors = _field_errors(
        entry, METRIC_REQUIRED_FIELDS, METRIC_OPTIONAL_FIELDS, source, item
    )
    if errors:
        return errors
    errors.extend(_text_errors(entry, ("name", "formula"), source, item))
    for field in ("unit", "description"):
        if not isinstance(entry[field], str):
            errors.append(f"[{source}] {item}: '{field}' must be a string")
    if entry["unit"] in UNIT_ALIASES:
        errors.append(
            f"[{source}] {item}: 'unit' {entry['unit']!r} is an alias; "
            f"use {UNIT_ALIASES[entry['unit']]!r}"
        )
    peak = entry.get("peak")
    if isinstance(peak, bool) or not isinstance(peak, (str, int, float, type(None))):
        errors.append(f"[{source}] {item}: 'peak' must be a string or a number")
    if not isinstance(entry.get("coll_level", ""), str):
        errors.append(f"[{source}] {item}: 'coll_level' must be a string")
    return errors


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _field_errors(
    entry: object,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    source: str,
    item: str,
) -> list[str]:
    if not isinstance(entry, dict):
        return [f"[{source}] {item}: must be a mapping"]
    errors = [
        f"[{source}] {item}: '{field}' is required"
        for field in required
        if field not in entry
    ]
    errors.extend(
        f"[{source}] {item}: '{field}' is not a known field"
        for field in entry
        if field not in required and field not in optional
    )
    return errors


def _text_errors(
    entry: dict[str, Any], fields: tuple[str, ...], source: str, item: str
) -> list[str]:
    return [
        f"[{source}] {item}: '{field}' must be a non-empty string"
        for field in fields
        if not isinstance(entry.get(field), str) or not entry[field].strip()
    ]


if __name__ == "__main__":
    sys.exit(main())
