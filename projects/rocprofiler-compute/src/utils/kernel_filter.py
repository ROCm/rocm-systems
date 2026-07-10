# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Kernel selection filtering for analyze mode."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

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
    """A raw, unresolved kernel selection.

    ``mode`` disambiguates the values so no downstream code has to guess:
    ``"indices"`` are integer row positions in the Top Stats table (CLI ``-k``);
    ``"names"`` are kernel-name strings (GUI or operator selection).
    """

    mode: Literal["indices", "names"]
    values: list[int] | list[str]


@dataclass(frozen=True)
class KernelFilter:
    """A resolved kernel selection.

    Kernel names remain the compatibility layer for ``-k`` and GUI selection.
    Operator filters may additionally carry identity columns so dispatches from
    unrelated operators with the same generated kernel name are not included.
    """

    names: frozenset[str] = field(default_factory=frozenset)
    correlation_ids: frozenset[int] = field(default_factory=frozenset)
    dispatch_ids: frozenset[int] = field(default_factory=frozenset)

    @property
    def is_active(self) -> bool:
        return bool(self.names or self.correlation_ids or self.dispatch_ids)

    @property
    def is_identity_scoped(self) -> bool:
        return bool(self.correlation_ids or self.dispatch_ids)

    def describe(self) -> str:
        """Return a stable short label for filenames/logging."""
        if self.names:
            return "_".join(sorted(self.names))
        if self.dispatch_ids:
            return "dispatch_" + "_".join(str(i) for i in sorted(self.dispatch_ids))
        if self.correlation_ids:
            return "corr_" + "_".join(str(i) for i in sorted(self.correlation_ids))
        return ""


def resolve_kernel_filter(
    request: Optional[KernelSelectionRequest],
    kernel_top_df: Optional[pd.DataFrame],
) -> KernelFilter:
    """Resolve a selection request into a :class:`KernelFilter`.

    Integer indices are validated against and mapped through ``kernel_top_df``.
    Returns an inactive filter when ``request`` is empty.
    """
    if request is None or not request.values:
        return KernelFilter()

    if request.mode == "names":
        return KernelFilter(names=_normalize_names(request.values))

    if kernel_top_df is None:
        console_error(
            "Kernel top stats table not loaded. Ensure create_df_kernel_top_stats() "
            "is called before resolving an index-based kernel filter."
        )

    num_kernels = len(kernel_top_df["Kernel_Name"])
    for index in request.values:
        if not 0 <= index < num_kernels:
            console_error(
                f"{index} is an invalid kernel id. "
                f"Please enter an id between 0-{num_kernels - 1}"
            )

    return KernelFilter(
        names=frozenset(
            str(kernel_top_df.iloc[index]["Kernel_Name"]).strip()
            for index in request.values
        )
    )


def merge_kernel_filters(left: KernelFilter, right: KernelFilter) -> KernelFilter:
    """Intersect active filter dimensions where both sides constrain them."""
    if not left.is_active:
        return right
    if not right.is_active:
        return left
    return KernelFilter(
        names=_intersect_or_take(left.names, right.names),
        correlation_ids=_intersect_or_take(left.correlation_ids, right.correlation_ids),
        dispatch_ids=_intersect_or_take(left.dispatch_ids, right.dispatch_ids),
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
    """Filter rows by resolved kernel identity and/or name constraints."""
    if not kernel_filter.is_active:
        return df

    filtered_df = df
    if kernel_filter.correlation_ids and "Correlation_ID" in filtered_df.columns:
        filtered_df = filtered_df.loc[
            _numeric_series(filtered_df["Correlation_ID"]).isin(
                kernel_filter.correlation_ids
            )
        ]

    if kernel_filter.dispatch_ids and "Dispatch_ID" in filtered_df.columns:
        filtered_df = filtered_df.loc[
            _numeric_series(filtered_df["Dispatch_ID"]).isin(kernel_filter.dispatch_ids)
        ]

    if kernel_filter.names and "Kernel_Name" in filtered_df.columns:
        stripped_names = filtered_df["Kernel_Name"].apply(_strip_name)
        filtered_df = filtered_df.loc[stripped_names.isin(kernel_filter.names)]

    return filtered_df


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
    """Load cached consolidated trace, regenerating when identity columns are stale."""
    ml_api_trace_dir = Path(workload_path) / "ml_api_trace"
    consolidated_path = ml_api_trace_dir / "consolidated.csv"

    if consolidated_path.exists():
        consolidated_df = pd.read_csv(consolidated_path)
        if _has_identity_columns(consolidated_df) or not _raw_trace_files_exist(
            workload_path
        ):
            console_log(
                "ml api trace",
                f"Loaded cached {consolidated_path}. "
                "Delete ml_api_trace/ directory to force regeneration from raw traces.",
            )
            return consolidated_df

        console_log(
            "ml api trace",
            f"Cached {consolidated_path} has an old schema; regenerating it.",
        )

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

    raw_pmc = getattr(workload, "raw_pmc", pd.DataFrame())
    return _kernel_filter_from_matched_df(matched_df, raw_pmc), matched_df


def _normalize_names(values: Iterable[object]) -> frozenset[str]:
    return frozenset(str(name).strip() for name in values if str(name).strip())


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


def _intersect_or_take(left: frozenset, right: frozenset) -> frozenset:
    if left and right:
        return left & right
    return left or right


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


def _kernel_filter_from_matched_df(
    matched_df: pd.DataFrame,
    raw_pmc: pd.DataFrame,
) -> KernelFilter:
    names = frozenset(matched_df["Kernel_Name"].dropna().str.strip().unique())
    correlation_ids = _column_int_set(matched_df, "Correlation_ID")
    dispatch_ids = _derive_dispatch_ids(matched_df, raw_pmc, correlation_ids)
    return KernelFilter(
        names=names,
        correlation_ids=correlation_ids,
        dispatch_ids=dispatch_ids,
    )


def _derive_dispatch_ids(
    matched_df: pd.DataFrame,
    raw_pmc: pd.DataFrame,
    correlation_ids: frozenset[int],
) -> frozenset[int]:
    if raw_pmc.empty or "Dispatch_ID" not in raw_pmc.columns:
        return frozenset()

    if correlation_ids and "Correlation_ID" in raw_pmc.columns:
        selected = raw_pmc[
            pd.to_numeric(raw_pmc["Correlation_ID"], errors="coerce")
            .astype("Int64")
            .isin(correlation_ids)
        ]
        return _column_int_set(selected, "Dispatch_ID")

    timestamp_columns = {
        "Kernel_Name",
        "Start_Timestamp_kernel",
        "End_Timestamp_kernel",
    }
    if not timestamp_columns.issubset(matched_df.columns):
        return frozenset()
    if not {"Kernel_Name", "Start_Timestamp", "End_Timestamp"}.issubset(
        raw_pmc.columns
    ):
        return frozenset()

    trace_keys = matched_df[
        ["Kernel_Name", "Start_Timestamp_kernel", "End_Timestamp_kernel"]
    ].drop_duplicates()
    trace_keys = trace_keys.rename(
        columns={
            "Start_Timestamp_kernel": "Start_Timestamp",
            "End_Timestamp_kernel": "End_Timestamp",
        }
    )
    trace_keys["Kernel_Name"] = trace_keys["Kernel_Name"].astype(str).str.strip()
    raw_keys = raw_pmc.copy()
    raw_keys["Kernel_Name"] = raw_keys["Kernel_Name"].astype(str).str.strip()
    merged = raw_keys.merge(
        trace_keys,
        on=["Kernel_Name", "Start_Timestamp", "End_Timestamp"],
        how="inner",
    )
    return _column_int_set(merged, "Dispatch_ID")


def _column_int_set(df: pd.DataFrame, column: str) -> frozenset[int]:
    if column not in df.columns:
        return frozenset()
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return frozenset(int(value) for value in values)


def _has_identity_columns(df: pd.DataFrame) -> bool:
    return bool({"Correlation_ID", "Dispatch_ID"}.intersection(df.columns))


def _raw_trace_files_exist(workload_path: str) -> bool:
    return any(Path(workload_path).glob("**/ml_api_trace*_marker_api_trace.csv"))


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
