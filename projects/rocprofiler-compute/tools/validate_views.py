#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: validate the analysis views.

For every arch of the metric library, check that:
  1. its view files have known fields, archs, render types, CLI styles and display
     forms; every display form is shown and every view file is read by an arch;
  2. every metric a view shows is in the arch's metric library, with a display unit
     of its unit and the peak the row asks for;
  3. the views give the analysis_configs panel configs exactly: each panel, table, row
     and column, in order, and the descriptions; each panel's -b alias is the config
     template's; and every metric the arch shows has one legacy_id, its row's position.
Check 3 is temporary: it goes with the panel files (AIPROFCOMP-821 PR 12).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from utils.analysis_views import (
    DISPLAY_FORMS,
    default_views_dir,
    iter_view_rows,
    panel_aliases,
    read_view_files,
    row_metric_ids,
    select_arch_views,
    view_metric_errors,
    view_schema_errors,
    views_to_panel_configs,
)
from utils.metric_library import (
    METRIC_LIBRARY_PARENT_ARCH,
    default_metric_library_dir,
    load_metric_library,
)
from utils.utils_common import get_arch_panel_id_to_alias, load_panel_configs

# Panel config keys whose entries are in display order; the order of other keys does
# not matter, nor does the order of a panel's descriptions.
ORDERED_KEYS = ("header", "metric")


def validate(
    views_dir: Path, metric_dir: Path, analysis_dir: Optional[Path] = None
) -> List[str]:
    """Validate every arch's views. Return error messages.

    With analysis_dir, also check each arch's views against its analysis configs.
    """
    errors: List[str] = []
    read_paths: Set[Path] = set()
    shown_forms: Set[str] = set()
    for arch in METRIC_LIBRARY_PARENT_ARCH:
        documents = read_view_files(arch, views_dir)
        read_paths.update(documents)
        schema_errors = view_schema_errors({
            _source(path, views_dir): document for path, document in documents.items()
        })
        if schema_errors:
            errors.extend(schema_errors)
            continue
        views = select_arch_views(list(documents.values()), arch)
        shown_forms.update(_display_forms(views))
        metrics = load_metric_library(metric_dir, arch).metrics
        arch_errors = view_metric_errors(views, metrics)
        if not arch_errors and analysis_dir is not None:
            arch_errors = equivalence_errors(views, metrics, analysis_dir, arch)
        errors.extend(f"[{arch}] {error}" for error in arch_errors)
    errors.extend(
        f"[{_source(path, views_dir)}] is read by no arch"
        for path in sorted(set(Path(views_dir).rglob("*.yaml")) - read_paths)
    )
    if not errors:
        errors.extend(
            f"DISPLAY_FORMS lists {name}, but no view shows it"
            for name in DISPLAY_FORMS
            if name not in shown_forms
        )
    return errors


def equivalence_errors(
    views: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    analysis_dir: Path,
    arch: str,
) -> List[str]:
    """Check one arch's views against its analysis configs. Temporary; see above."""
    expected = load_panel_configs([str(analysis_dir / arch)])
    actual = views_to_panel_configs(views, metrics)
    errors = _differences(
        {str(panel_id): panel for panel_id, panel in expected.items()},
        {str(panel_id): panel for panel_id, panel in actual.items()},
        "panel",
        ordered=False,
    )
    errors.extend(_alias_errors(views, arch))
    errors.extend(_legacy_id_errors(views))
    return errors


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate(
        default_views_dir(), default_metric_library_dir(), config.analysis_configs_dir
    )
    if errors:
        print("Analysis view validation failed:\n")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Analysis view validation passed.")
    return 0


def _source(path: Path, views_dir: Path) -> str:
    """Return a view file's path in the views tree, as used in error messages."""
    return str(Path(path).relative_to(views_dir))


def _display_forms(views: List[Dict[str, Any]]) -> Set[str]:
    return {row["display"] for _, row in iter_view_rows(views) if "display" in row}


def _differences(expected: Any, actual: Any, path: str, ordered: bool) -> List[str]:
    """Return where actual differs from expected; ordered says whether key order counts.

    Values compare by their JSON text, so 1, "1" and True all differ.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        if (list if ordered else set)(expected) != (list if ordered else set)(actual):
            return [f"{path}: panel has {list(expected)}, views give {list(actual)}"]
        return [
            difference
            for key in expected
            for difference in _differences(
                expected[key],
                actual[key],
                f"{path}/{key}",
                key in ORDERED_KEYS or path.endswith("/metric"),
            )
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: panel has {len(expected)} items, views {len(actual)}"]
        return [
            difference
            for index, (want, got) in enumerate(zip(expected, actual))
            for difference in _differences(want, got, f"{path}/{index}", ordered)
        ]
    if json.dumps(expected) == json.dumps(actual):
        return []
    return [f"{path}: panel has {expected!r}, views give {actual!r}"]


def _alias_errors(views: List[Dict[str, Any]], arch: str) -> List[str]:
    template_aliases = get_arch_panel_id_to_alias(arch)
    return [
        f"panel {panel_number}: -b alias {alias!r}, config template has "
        f"{template_aliases.get(panel_number)!r}"
        for panel_number, alias in panel_aliases(views).items()
        if template_aliases.get(panel_number) != alias
    ]


def _legacy_id_errors(views: List[Dict[str, Any]]) -> List[str]:
    """Check that each stored legacy_id is its row's position, one per metric."""
    errors = []
    stored: Dict[str, List[str]] = {}
    for position, row in iter_view_rows(views):
        for metric_id in row_metric_ids(row):
            stored.setdefault(metric_id, [])
        if "legacy_id" not in row:
            continue
        if row["legacy_id"] != position:
            errors.append(f"{position}: legacy_id is {row['legacy_id']}")
        for metric_id in row_metric_ids(row):
            stored[metric_id].append(row["legacy_id"])
    errors.extend(
        f"{metric_id}: has legacy ids {legacy_ids}, not one"
        for metric_id, legacy_ids in stored.items()
        if len(legacy_ids) != 1
    )
    return errors


if __name__ == "__main__":
    sys.exit(main())
