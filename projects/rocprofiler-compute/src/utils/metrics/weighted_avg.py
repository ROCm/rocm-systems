# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""WEIGHTED_AVG composite metrics (AIPROFCOMP-865 Phase 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.metrics.collectable import (
    COLLECTABLE_EXPR_CACHE_ATTR,
    WEIGHTED_AVG_ATTR,
    CompositeDef,
    CompositeKind,
    apply_composite_metrics,
    cache_collectable_expressions,
)

__all__ = [
    "WEIGHTED_AVG_ATTR",
    "apply_weighted_avg_metrics",
    "cache_weighted_avg_sub_expressions",
    "evaluate_weighted_avg_parent",
]
from utils.metrics.expression import parse_weighted_avg_submetrics
from vendored import yaml

WEIGHTED_AVG_FIELD_NAMES = frozenset({"avg", "average"})
WEIGHTED_AVG_SUB_EXPR_ATTR = COLLECTABLE_EXPR_CACHE_ATTR
apply_weighted_avg_metrics = apply_composite_metrics
cache_weighted_avg_sub_expressions = cache_collectable_expressions


def evaluate_weighted_avg_parent(
    submetric_names: list[str],
    weight_meta: dict[str, Any],
    df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> float | str:
    from utils.metrics.collectable import _evaluate_weighted_composite

    composite = CompositeDef(
        metric_id="",
        kind=CompositeKind.WEIGHTED_AVG,
        refs=submetric_names,
        weight_meta=weight_meta,
    )
    return _evaluate_weighted_composite(
        composite, df, raw_pmc_df, sys_vars, empirical_peaks
    )


def scan_weighted_avg_parents(
    config_arch_path: Path,
) -> list[tuple[str, str, list[str]]]:
    """Return (yaml file, metric key, submetric names) for WEIGHTED_AVG parents."""
    if not config_arch_path.is_dir():
        return []

    found: list[tuple[str, str, list[str]]] = []
    for ypath in sorted(config_arch_path.glob("*.yaml")):
        try:
            with open(ypath, encoding="utf-8") as stream:
                doc = yaml.safe_load(stream)
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if not isinstance(doc, dict):
            continue
        panel_cfg = doc.get("Panel Config")
        if not isinstance(panel_cfg, dict):
            continue
        sources = panel_cfg.get("data source")
        if not isinstance(sources, list):
            continue
        for section in sources:
            if not isinstance(section, dict):
                continue
            metric_table = section.get("metric_table")
            if not isinstance(metric_table, dict):
                continue
            metrics = metric_table.get("metric")
            if not isinstance(metrics, dict):
                continue
            for metric_key, body in metrics.items():
                if not isinstance(body, dict):
                    continue
                avg_formula = body.get("avg")
                if not isinstance(avg_formula, str):
                    continue
                subs = parse_weighted_avg_submetrics(avg_formula)
                if subs and isinstance(body.get("_weighted_avg"), dict):
                    found.append((ypath.name, metric_key, subs))
    return found


def format_weighted_avg_inspector_section(
    config_arch_path: Path,
) -> str:
    """Text block for counter_grouping_inspector (Milestone C hint)."""
    parents = scan_weighted_avg_parents(config_arch_path)
    lines = ["WEIGHTED_AVG parent metrics (analyze-only composites):"]
    if not parents:
        lines.append("  (none in analysis_configs for this arch)")
        return "\n".join(lines) + "\n\n"
    for file_name, metric_key, subs in parents:
        lines.append(f"  - {file_name}: {metric_key} -> submetrics {subs}")
    lines.append("  Verify each submetric id is single-bucket in the plan above.")
    return "\n".join(lines) + "\n\n"
