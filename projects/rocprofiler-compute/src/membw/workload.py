# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load membw metrics from a profiled workload directory."""

from pathlib import Path
from typing import Optional

import pandas as pd

import config
from membw.metric_extract import MetricExtractionResult, extract_membw_metrics
from membw.models import MEMBW_TABLE_IDS
from membw.tree_spec import collect_metric_keys, load_tree_spec
from utils import file_io, schema
from utils.logger import console_log, console_warning
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import build_metric_value_string
from utils.parser import build_dfs
from utils.utils_common import canonical_config_arch


def load_membw_metrics(
    workload_dir: Path,
    arch: str,
) -> MetricExtractionResult:
    """Load and evaluate membw metrics from a workload directory."""
    sys_info = _load_sys_info(workload_dir)
    arch_configs = _build_membw_arch_configs(arch, sys_info)
    raw_pmc_df = file_io.create_df_pmc(str(workload_dir), kernel_verbose=0, verbose=0)

    if raw_pmc_df.empty:
        _report_missing_pmc(workload_dir)

    membw_dfs, membw_types, membw_exprs = _filter_membw_tables(arch_configs)

    build_metric_value_string(membw_dfs, membw_types, "per_kernel")

    eval_metric(
        dfs=membw_dfs,
        dfs_type=membw_types,
        dfs_expressions=membw_exprs,
        sys_info=sys_info,
        empirical_peaks_df=pd.DataFrame(),
        raw_pmc_df=raw_pmc_df,
        debug=False,
    )

    tree_spec = load_tree_spec(arch)
    metric_keys = collect_metric_keys(tree_spec)
    extraction = extract_membw_metrics(membw_dfs, metric_keys)

    _log_metric_availability(metric_keys, extraction.values)

    return extraction


# --- Private helpers ---


def _filter_membw_tables(
    arch_configs: schema.ArchConfig,
) -> tuple[
    dict[int, pd.DataFrame],
    dict[int, str],
    dict[int, list[str]],
]:
    """Extract only the membw-relevant tables from the full ArchConfig."""
    dfs = {
        tid: arch_configs.dfs[tid] for tid in MEMBW_TABLE_IDS if tid in arch_configs.dfs
    }
    dfs_type = {
        tid: arch_configs.dfs_type[tid] for tid in dfs if tid in arch_configs.dfs_type
    }
    dfs_expressions = {
        tid: arch_configs.dfs_expressions[tid]
        for tid in dfs
        if tid in arch_configs.dfs_expressions
    }
    return dfs, dfs_type, dfs_expressions


def _report_missing_pmc(workload_dir: Path) -> None:
    """Warn when pmc_perf.csv is missing or empty."""
    has_gz = any(workload_dir.glob("results_pmc_perf_*.csv.gz"))
    if has_gz:
        console_warning(
            "membw",
            f"No pmc_perf.csv in {workload_dir}. "
            "Raw results_pmc_perf_*.csv.gz files exist -- "
            "run 'rocprof-compute analyze' first to join them.",
        )
    else:
        console_warning(
            "membw",
            f"No profiling data in {workload_dir}. "
            "Run 'rocprof-compute profile' first.",
        )


def _log_metric_availability(
    metric_keys: frozenset[str],
    metric_values: dict[str, Optional[float]],
) -> None:
    """Log which tree-spec metrics evaluated successfully vs failed."""
    unavailable = sorted(k for k, v in metric_values.items() if v is None)
    if not unavailable:
        console_log(
            "membw",
            f"All {len(metric_keys)} tree-spec metrics evaluated.",
        )
        return

    console_warning(
        "membw",
        f"{len(unavailable)} of {len(metric_keys)} tree-spec metrics "
        f"could not be evaluated "
        f"(counters not in profiling sets):",
    )
    for metric_name in unavailable:
        console_warning("membw", f"  - {metric_name}")


def _load_sys_info(workload_dir: Path) -> pd.Series:
    """Load sysinfo.csv and return the first row as a Series."""
    sysinfo_path = workload_dir / "sysinfo.csv"
    df = pd.read_csv(str(sysinfo_path))
    return df.iloc[0]


def _build_membw_arch_configs(
    arch: str,
    sys_info: pd.Series,
) -> schema.ArchConfig:
    """Build ArchConfig with membw panel DataFrames."""
    config_arch = canonical_config_arch(arch) or arch
    analysis_dir = str(
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "analysis_configs"
        / config_arch
    )
    arch_config = schema.ArchConfig()
    arch_config.panel_configs = file_io.load_panel_configs([analysis_dir])
    build_dfs(
        arch_configs=arch_config,
        filter_metrics=None,
        sys_info=sys_info,
        profiling_config={},
        arch=arch,
        membw_analysis=True,
    )
    return arch_config
