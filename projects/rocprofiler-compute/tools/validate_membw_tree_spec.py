#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: validate membw tree spec metric names.

Checks:
  1. Every metric referenced in the tree spec exists in the membw tables of the views.
  2. No tree-referenced metric formula contains $denom.
"""

import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from membw_analysis.guidance import _METRIC_THRESHOLD_RE
from membw_analysis.models import MEMBW_TABLE_IDS
from membw_analysis.tree_spec import (
    collect_metric_keys,
    load_tree_spec,
    membw_analysis_dir,
)
from utils.analysis_views import arch_panel_configs
from utils.utils_common import metric_tables_by_id


def load_view_metrics(arch: str) -> Dict[str, str]:
    """Return {metric_name: formula} from the membw tables of arch's views."""
    tables = metric_tables_by_id(arch_panel_configs(arch))
    return {
        name: str(row.get("value", ""))
        for table_id in MEMBW_TABLE_IDS
        for name, row in (tables.get(table_id, {}).get("metric") or {}).items()
    }


def validate() -> list[str]:
    """Run validation across all tree spec files. Return error messages."""
    errors: list[str] = []

    for spec_path in sorted(membw_analysis_dir().glob("*_membw_tree_spec.yaml")):
        arch = spec_path.stem.replace("_membw_tree_spec", "")
        try:
            tree_spec = load_tree_spec(arch)
        except SystemExit:
            errors.append(f"[{arch}] tree spec failed to load")
            continue

        print(
            f"[{arch}] schema hash: {tree_spec.schema_hash}",
            file=sys.stderr,
        )

        tree_keys = collect_metric_keys(tree_spec)
        view_metrics = load_view_metrics(arch)

        if not view_metrics:
            errors.append(
                f"[{arch}] no view metrics found for tables {MEMBW_TABLE_IDS}"
            )
            continue

        tree_only = sorted(tree_keys - frozenset(view_metrics))
        for metric in tree_only:
            errors.append(
                f"[{arch}] tree spec references metric not in the views: {metric!r}"
            )

        # Tree metrics must use hardware cycle denominators, not $denom
        for metric in sorted(tree_keys & frozenset(view_metrics)):
            formula = view_metrics[metric]
            if "$denom" in formula:
                errors.append(
                    f"[{arch}] tree metric {metric!r} uses $denom "
                    f"in its formula -- must use hardware cycle denominators"
                )

        for gid, template in tree_spec.guidance_templates.items():
            for match in _METRIC_THRESHOLD_RE.finditer(template):
                kind, key = match.group(1), match.group(2)
                if kind == "metric" and key not in view_metrics:
                    errors.append(
                        f"[{arch}] guidance {gid!r}: {{metric:{key}}} not in the views"
                    )
                if kind == "threshold" and key not in tree_spec.thresholds:
                    errors.append(
                        f"[{arch}] guidance {gid!r}: "
                        f"{{threshold:{key}}} not in thresholds"
                    )

        view_only = sorted(frozenset(view_metrics) - tree_keys)
        if view_only:
            print(
                f"[{arch}] info: {len(view_only)} view metrics not used by tree "
                "spec (OK)",
                file=sys.stderr,
            )

    return errors


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate()
    if errors:
        print("membw tree spec validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("membw tree spec validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
