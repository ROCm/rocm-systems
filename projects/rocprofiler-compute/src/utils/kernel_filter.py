# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Kernel selection filtering for analyze mode."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from utils.logger import console_error, console_log, console_warning
from utils.pattern_matching import fnmatch_glob_matches
from utils.utils_analysis import (
    decode_marker_name,
    process_ml_api_trace_output,
    write_ml_api_trace_consolidated_csv,
)

ML_API_ANALYSIS_CLI_OPTIONS = {
    "torch": {
        "filter_attr": "torch_operator",
        "list_attr": "list_torch_operators",
        "label": "PyTorch",
    },
    "triton": {
        "filter_attr": "triton_operator",
        "list_attr": "list_triton_operators",
        "label": "Triton",
    },
}


@dataclass(frozen=True)
class KernelSelectionRequest:
    """A raw, unresolved ``-k`` selection.

    ``indices`` are integer row positions in the Top Stats table, which only
    exists after the workload's kernels are known, so the request is carried
    unresolved until :func:`resolve_kernel_filter` can map indices to names.
    """

    indices: list[int]


@dataclass(frozen=True)
class KernelFilter:
    """A resolved kernel selection, expressed as a set of kernel names.

    Names are the single selection currency for ``-k``, GUI selection, and
    operator filters alike.
    """

    names: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_active(self) -> bool:
        return bool(self.names)

    def describe(self) -> str:
        """Return a stable short label for filenames/logging."""
        return "_".join(sorted(self.names))


def resolve_kernel_filter(
    request: Optional[KernelSelectionRequest],
    kernel_top_df: Optional[pd.DataFrame],
) -> KernelFilter:
    """Resolve a selection request into a :class:`KernelFilter`.

    Integer indices are validated against and mapped through ``kernel_top_df``.
    Returns an inactive filter when ``request`` is empty.
    """
    if request is None or not request.indices:
        return KernelFilter()

    if kernel_top_df is None:
        console_error(
            "Kernel top stats table not loaded. Ensure create_df_kernel_top_stats() "
            "is called before resolving an index-based kernel filter."
        )

    num_kernels = len(kernel_top_df["Kernel_Name"])
    for index in request.indices:
        if not 0 <= index < num_kernels:
            console_error(
                f"{index} is an invalid kernel id. "
                f"Please enter an id between 0-{num_kernels - 1}"
            )

    return KernelFilter(
        names=frozenset(
            str(kernel_top_df.iloc[index]["Kernel_Name"]).strip()
            for index in request.indices
        )
    )


def apply_workload_filters(
    df: pd.DataFrame,
    kernel_filter: KernelFilter,
    filter_gpu_ids: Optional[Iterable[object]] = None,
    filter_dispatch_ids: Optional[Iterable[object]] = None,
    validate_dispatch_ids: bool = False,
) -> pd.DataFrame:
    """Apply analyze filters to a raw PMC-like dataframe in one shared order."""
    filtered_df = df.copy()

    gpu_ids = _normalize_strings(filter_gpu_ids)
    if gpu_ids and "GPU_ID" in filtered_df.columns:
        filtered_df = filtered_df.loc[filtered_df["GPU_ID"].astype(str).isin(gpu_ids)]

    if kernel_filter.is_active:
        filtered_df = apply_kernel_filter_to_df(filtered_df, kernel_filter)

    dispatch_filters = _normalize_strings(filter_dispatch_ids)
    if dispatch_filters and "Dispatch_ID" in filtered_df.columns:
        filtered_df = _apply_dispatch_id_filters(
            filtered_df, dispatch_filters, validate_dispatch_ids
        )

    return filtered_df


def apply_kernel_filter_to_df(
    df: pd.DataFrame,
    kernel_filter: KernelFilter,
) -> pd.DataFrame:
    """Filter rows to the resolved kernel names."""
    if not kernel_filter.is_active or "Kernel_Name" not in df.columns:
        return df

    stripped_names = df["Kernel_Name"].apply(_strip_name)
    return df.loc[stripped_names.isin(kernel_filter.names)]


def parse_operator_patterns(args: argparse.Namespace, attr: str) -> list[str]:
    """Extract and flatten operator glob patterns from ``args.<attr>``."""
    raw = getattr(args, attr, None)
    if raw is None:
        return []
    pattern_list: list[str] = []
    for operator_arg in raw:
        pattern_list.extend(
            pattern.strip()
            for pattern in str(operator_arg).split(",")
            if pattern.strip()
        )
    if not pattern_list:
        pattern_list = ["**"]
    return pattern_list


def filter_by_backend(consolidated_df: pd.DataFrame, backend: str) -> pd.DataFrame:
    """Return rows attributed to ``backend``."""
    if "Backend" in consolidated_df.columns:
        return consolidated_df[consolidated_df["Backend"] == backend].copy()
    if backend == "torch":
        return consolidated_df.copy()
    return consolidated_df.iloc[0:0].copy()


def load_consolidated_ml_api_trace(workload_path: str) -> pd.DataFrame:
    """Load the cached consolidated trace, generating it on first use."""
    ml_api_trace_dir = Path(workload_path) / "ml_api_trace"
    consolidated_path = ml_api_trace_dir / "consolidated.csv"

    if consolidated_path.exists():
        console_log(
            "ml api trace",
            f"Loaded cached {consolidated_path}. "
            "Delete ml_api_trace/ directory to force regeneration from raw traces.",
        )
        return pd.read_csv(consolidated_path)

    consolidated_df, ml_api_trace_path = process_ml_api_trace_output(workload_path)
    if not consolidated_df.empty:
        write_ml_api_trace_consolidated_csv(consolidated_df, ml_api_trace_path)
    return consolidated_df


def build_operator_filter(
    args: argparse.Namespace,
    workload: object,
    workload_path: str,
    backend: str,
    exit_on_no_match: bool = True,
) -> Optional[tuple[KernelFilter, pd.DataFrame]]:
    """Return the resolved operator filter and its scoped trace rows."""
    cli = ML_API_ANALYSIS_CLI_OPTIONS[backend]
    label = cli["label"]

    consolidated_df = load_consolidated_ml_api_trace(workload_path)
    if consolidated_df.empty:
        console_warning(
            "ml api trace",
            f"No {label} operator data found in this workload. "
            f"Proceeding without {label} operator filter.",
        )
        return None

    consolidated_df = filter_by_backend(consolidated_df, backend)
    if consolidated_df.empty:
        console_warning(
            "ml api trace",
            f"No {label} operator data found in this workload. "
            f"Proceeding without {label} operator filter.",
        )
        return None

    pattern_list = parse_operator_patterns(args, cli["filter_attr"])
    all_operators = consolidated_df["Operator_Name"].dropna().unique()
    matched_names = [
        str(op).strip()
        for op in all_operators
        if any(
            fnmatch_glob_matches(p.strip(), candidate)
            for candidate in {
                str(op).strip(),
                decode_marker_name(str(op).strip()),
            }
            for p in pattern_list
        )
    ]

    if not matched_names:
        console_warning(
            "ml api trace",
            f"No {label} operators matched the pattern(s): {pattern_list}",
        )
        if exit_on_no_match:
            sys.exit(0)
        return None

    matched_df = consolidated_df[
        consolidated_df["Operator_Name"].isin(matched_names)
    ].copy()

    base_filter = getattr(workload, "kernel_filter", KernelFilter())
    if base_filter.names:
        matched_df = matched_df[
            matched_df["Kernel_Name"].dropna().str.strip().isin(base_filter.names)
        ].copy()
        if matched_df.empty:
            return KernelFilter(), matched_df

    names = frozenset(matched_df["Kernel_Name"].dropna().str.strip().unique())
    return KernelFilter(names=names), matched_df


def _normalize_ints(values: Iterable[object]) -> frozenset[int]:
    normalized: set[int] = set()
    for value in values:
        if pd.isna(value):
            continue
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return frozenset(normalized)


def _normalize_strings(values: Optional[Iterable[object]]) -> list[str]:
    if not values:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _strip_name(name: object) -> object:
    return name.strip() if isinstance(name, str) else name


def _apply_dispatch_id_filters(
    df: pd.DataFrame,
    dispatch_filters: list[str],
    validate_dispatch_ids: bool,
) -> pd.DataFrame:
    first_filter = dispatch_filters[0]
    if first_filter.startswith(">"):
        match = re.match(r">\s*(\d+)", first_filter)
        if match:
            return df.loc[_numeric_series(df["Dispatch_ID"]) > int(match.group(1))]
        return df

    selected_dispatches = _normalize_ints(dispatch_filters)
    if validate_dispatch_ids:
        available_dispatches = set(
            _numeric_series(df["Dispatch_ID"]).dropna().astype(int)
        )
        for dispatch_id in selected_dispatches:
            if dispatch_id not in available_dispatches:
                console_error("analysis", f"{dispatch_id} is an invalid dispatch id.")

    return df.loc[_numeric_series(df["Dispatch_ID"]).isin(selected_dispatches)]


def mark_selected_kernels(
    kernel_top_df: pd.DataFrame,
    kernel_filter: KernelFilter,
) -> None:
    """Set a ``Selected`` column ("*") on rows selected by ``kernel_filter``."""
    stripped_names = kernel_top_df["Kernel_Name"].apply(
        lambda name: name.strip() if isinstance(name, str) else name
    )
    kernel_top_df["Selected"] = stripped_names.isin(kernel_filter.names).map({
        True: "*",
        False: "",
    })
