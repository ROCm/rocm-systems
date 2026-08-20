#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: validate membw tree spec metric names.

Checks:
  1. Every metric referenced in the tree spec exists in the analysis config.
  2. No tree-referenced metric formula contains $denom.
"""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
TREE_SPEC_DIR = PROJECT_ROOT / "src" / "membw" / "tree_spec"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from membw.models import MEMBW_TABLE_IDS  # noqa: E402
from membw.tree_spec import collect_metric_keys, load_tree_spec  # noqa: E402
from utils.utils_common import canonical_config_arch  # noqa: E402


def load_analysis_metrics(arch: str) -> dict[str, str]:
    """Return {metric_name: formula} from membw-relevant tables."""
    config_arch = canonical_config_arch(arch) or arch
    arch_dir = ANALYSIS_DIR / config_arch
    if not arch_dir.is_dir():
        return {}

    result: dict[str, str] = {}
    for config_path in sorted(arch_dir.glob("*.yaml")):
        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        panel = (data or {}).get("Panel Config", {})
        for source in panel.get("data source", []):
            metric_table = source.get("metric_table", {})
            table_id = metric_table.get("id")
            if table_id not in MEMBW_TABLE_IDS:
                continue
            metrics = metric_table.get("metric", {})
            if not metrics:
                continue
            for name, definition in metrics.items():
                if isinstance(definition, dict):
                    result[name] = str(definition.get("value", ""))
                else:
                    result[name] = str(definition)
    return result


def validate() -> list[str]:
    """Run validation across all tree spec files. Return error messages."""
    errors: list[str] = []

    for spec_path in sorted(TREE_SPEC_DIR.glob("*_membw_tree_spec.yaml")):
        arch = spec_path.stem.replace("_membw_tree_spec", "")
        try:
            tree_spec = load_tree_spec(arch)
        except SystemExit:
            errors.append(f"[{arch}] tree spec failed to load")
            continue

        tree_keys = collect_metric_keys(tree_spec)
        analysis_metrics = load_analysis_metrics(arch)

        if not analysis_metrics:
            errors.append(
                f"[{arch}] no analysis config metrics found "
                f"for tables {MEMBW_TABLE_IDS}"
            )
            continue

        tree_only = sorted(tree_keys - frozenset(analysis_metrics))
        for metric in tree_only:
            errors.append(
                f"[{arch}] tree spec references metric "
                f"not in analysis config: {metric!r}"
            )

        # Tree metrics must use hardware cycle denominators, not $denom
        for metric in sorted(tree_keys & frozenset(analysis_metrics)):
            formula = analysis_metrics[metric]
            if "$denom" in formula:
                errors.append(
                    f"[{arch}] tree metric {metric!r} uses $denom "
                    f"in its formula -- must use hardware cycle denominators"
                )

        config_only = sorted(frozenset(analysis_metrics) - tree_keys)
        if config_only:
            print(
                f"[{arch}] info: {len(config_only)} analysis config "
                f"metrics not used by tree spec (OK)",
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
