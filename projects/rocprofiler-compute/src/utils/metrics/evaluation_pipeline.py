# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""High-level metric-evaluation pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from utils.logger import console_error, console_warning
from utils.mi_gpu_spec import mi_gpu_specs

# Present for every arch; a missing or zero value warns.
REQUIRED_SYS_VARS: tuple[tuple[str, type], ...] = (
    ("se_per_gpu", int),
    ("pipes_per_gpu", int),
    ("cu_per_gpu", int),
    ("simd_per_cu", int),
    ("sqc_per_gpu", int),
    ("lds_banks_per_cu", int),
    ("cur_sclk", float),
    ("cur_mclk", float),
    ("max_mclk", float),
    ("max_sclk", float),
    ("max_waves_per_cu", int),
    ("wave_size", int),
    ("total_l2_chan", int),
)
# Arch-specific; silently skipped when the value is absent.
OPTIONAL_SYS_VARS: tuple[tuple[str, type], ...] = (
    ("num_memory_channels", float),
    ("num_gl1c", int),
)


def create_empirical_peaks_dict(
    empirical_peaks_df: pd.DataFrame,
    prefix: str = "ammolite__",
) -> dict[str, float]:
    """Create empirical peaks dictionary, one entry per roofline.csv column.

    The peaks come from the first row. The keys are prefix plus the column name
    and "_empirical_peak".
    """
    if empirical_peaks_df.empty:
        # Without roofline.csv, every ceiling is NaN.
        return {
            f"{prefix}{name}_empirical_peak": np.nan
            for name in mi_gpu_specs.get_all_roofline_columns()
        }

    peak_data_row = empirical_peaks_df.iloc[0]
    return {
        f"{prefix}{col}_empirical_peak": peak_data_row[col]
        for col in empirical_peaks_df.columns
    }


def type_sys_vars(
    sys_info: Mapping[str, Any],
    missing_as_null: bool,
) -> dict[str, int | float]:
    """Type the sys.info values that metric expressions read, by bare name.

    A missing required value warns. It becomes 0 for the CLI, and NaN (NULL) when
    missing_as_null is set, so an expression that uses it evaluates to NULL. A zero
    stays 0 and warns too. num_xcd defaults to 1 because RDNA is single-die.
    """
    typed: dict[str, Any] = {}

    for var_name, var_type in REQUIRED_SYS_VARS:
        number = _cast_sys_value(sys_info.get(var_name), var_type)
        if number is None or number == 0:
            console_warning(
                f"{var_name} is not available in sysinfo.csv, please provide the "
                "correct value using --specs-correction"
            )
        if number is None:
            number = np.nan if missing_as_null else var_type(0)
        typed[var_name] = number

    for var_name, var_type in OPTIONAL_SYS_VARS:
        number = _cast_sys_value(sys_info.get(var_name), var_type)
        if number is not None:
            typed[var_name] = number

    # num_xcd is a CDNA-only concept; RDNA is single-die, so default to 1.
    num_xcd = _cast_sys_value(sys_info.get("num_xcd"), int)
    typed["num_xcd"] = 1 if num_xcd is None else num_xcd

    return typed


def check_gui_active_counter(
    sys_info: pd.Series | Mapping[str, Any],
    raw_pmc_df: pd.DataFrame,
) -> None:
    """Halt the run when a non-roofline workload reports zero GPU activity."""
    # Only non-roofline runs carry the GRBM counters that prove activity.
    if sys_info.get("ip_blocks") == "roofline":
        return
    gui_active_counter = next(
        (
            counter
            for counter in ("GRBM_GUI_ACTIVE_sum", "GRBM_GUI_ACTIVE")
            if counter in raw_pmc_df.columns
        ),
        None,
    )
    if gui_active_counter is not None and (raw_pmc_df[gui_active_counter] == 0).any():
        console_warning(f"Detected {gui_active_counter} == 0")
        console_error("Halting execution for warning above.")


def is_pct_of_peak_flagged(flag: object) -> bool:
    """Read a Percent of Peak flag, whether it is still a bool or already a string.

    The metric config's pct_of_peak is a bool, but build_metric_value_string turns
    it into "True" or "" before evaluation. Only "True" counts as flagged.
    """
    return str(flag).strip() == "True"


def _cast_sys_value(raw_value: Any, var_type: type) -> Optional[int | float]:  # noqa ANN401
    """Cast one sys.info value to var_type; None when it is missing or not a number."""
    if raw_value is None:
        return None
    number = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(number):
        return None
    return var_type(number)
