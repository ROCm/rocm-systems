# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""WEIGHTED_AVG composite metrics (AIPROFCOMP-865 Phase 2)."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from utils.metrics.collectable import (
    COLLECTABLE_EXPR_CACHE_ATTR,
    WEIGHTED_AVG_ATTR,
    CompositeDef,
    CompositeKind,
    _evaluate_weighted_composite,
    apply_composite_metrics,
    cache_collectable_expressions,
)
from utils.metrics.expression import parse_weighted_avg_submetrics
from vendored import yaml

if TYPE_CHECKING:
    from utils.metrics.pass_provenance import PassLayout

__all__ = [
    "WEIGHTED_AVG_ATTR",
    "apply_weighted_avg_metrics",
    "cache_weighted_avg_sub_expressions",
    "evaluate_weighted_avg_parent",
]

WEIGHTED_AVG_FIELD_NAMES = frozenset({"avg", "average"})
WEIGHTED_AVG_SUB_EXPR_ATTR = COLLECTABLE_EXPR_CACHE_ATTR
apply_weighted_avg_metrics = apply_composite_metrics
cache_weighted_avg_sub_expressions = cache_collectable_expressions


def evaluate_weighted_avg_parent(
    submetric_names: List[str],
    weight_meta: Dict[str, Any],
    df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
    pass_layout: Optional["PassLayout"] = None,
) -> Union[float, str]:
    composite = CompositeDef(
        metric_id="",
        kind=CompositeKind.WEIGHTED_AVG,
        refs=submetric_names,
        weight_meta=weight_meta,
    )
    return _evaluate_weighted_composite(
        composite,
        df,
        raw_pmc_df,
        sys_vars,
        empirical_peaks,
        pass_layout=pass_layout,
    )


def scan_weighted_avg_parents(
    config_arch_path: Path,
) -> List[Tuple[str, str, List[str]]]:
    """Return (yaml file, metric key, submetric names) for WEIGHTED_AVG parents."""
    if not config_arch_path.is_dir():
        return []

    found: List[Tuple[str, str, List[str]]] = []
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
