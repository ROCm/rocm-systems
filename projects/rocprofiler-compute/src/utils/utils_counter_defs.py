# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared counter definitions for HW performance counter parsing.

Imported by both the main runtime code and lightweight tooling such as
``tools/validate_sets_metric_ids.py``.
"""

import re

from utils.logger import console_error

# ---------------------------------------------------------------------------
# Regex patterns for HW counter and variable extraction
# ---------------------------------------------------------------------------

# HW counter name must start with an IP block prefix, followed by an
# underscore-separated suffix of uppercase letters/digits, optionally
# ending with '[' or an aggregation suffix (_sum, _avr, _max, _min).
# IP block prefixes for PMC-style names (aligned with analysis YAML / tooling).
HW_COUNTER_BLK = (
    r"(?:CHA|CHC|CPC|CPF|GCEA|GC_CANE|GC_EA_SE|GCR|"
    r"GL1A|GL1C|GL2A|GL2C|GLARBA|GLARBC|GRBM|SDMA|SPI|"
    r"SQ|SQC|SP|SQG|TA|TD|TCP|TCC|TX|UTCL1)"
)
HW_COUNTER_SFX = r"_[0-9A-Z_]*[0-9A-Z](?:\[|_sum|_avr|_max|_min)*"
HW_COUNTER_RE = re.compile(HW_COUNTER_BLK + HW_COUNTER_SFX)

# Captures the variable name after '$'.
VARIABLE_RE = re.compile(r"\$([0-9A-Za-z_]*[0-9A-Za-z])")
AMMOLITE_VAR_RE = re.compile(r"ammolite__([0-9A-Za-z_]+)")

# ---------------------------------------------------------------------------
# Built-in variable and denominator definitions
# ---------------------------------------------------------------------------

# per_kernel denom column, injected at pmc load with 1 per dispatch so
# SUM($denom) == N and Avg is the mean per dispatch.
UNIT_COUNTER = "Dispatch_Unit"

SUPPORTED_DENOM: dict[str, str] = {
    "per_wave": "SQ_WAVES",
    "per_cycle": "$GRBM_GUI_ACTIVE_PER_XCD",
    "per_second": "((End_Timestamp - Start_Timestamp) / 1000000000)",
    # A vector, not a scalar, so SUM($denom) == N.
    "per_kernel": UNIT_COUNTER,
}


def get_build_in_vars(gpu_series: str) -> dict[str, str]:
    """Return the architecture-specific built-in variables for *gpu_series*.

    Args:
        gpu_series: GPU series string from mi_gpu_specs.get_gpu_series()

    Returns:
        Dictionary mapping built-in variable names to their formulas.

    Exits via console_error if *gpu_series* is falsy or not recognized.
    """
    if not gpu_series:
        console_error(
            "Cannot resolve built-in variables: no GPU series provided "
            "(unknown GPU arch?)."
        )

    build_in_vars: dict[str, dict[str, str]] = {
        "cdna": {
            "GRBM_GUI_ACTIVE_PER_XCD": "(GRBM_GUI_ACTIVE / $num_xcd)",
            "GRBM_COUNT_PER_XCD": "(GRBM_COUNT / $num_xcd)",
            "GRBM_SPI_BUSY_PER_XCD": "(GRBM_SPI_BUSY / $num_xcd)",
            "numActiveCUs": (
                "TO_INT(MIN(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) / "
                "SUM($GRBM_GUI_ACTIVE_PER_XCD), 0) / $max_waves_per_cu * 8 + "
                "MIN(MOD(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) / "
                "SUM($GRBM_GUI_ACTIVE_PER_XCD), 0), "
                "$max_waves_per_cu), 8), $cu_per_gpu))"
            ),
            "kernelBusyCycles": (
                "ROUND(AVG((((End_Timestamp - Start_Timestamp) / 1000) * "
                "$max_sclk)), 0)"
            ),
            "hbmBandwidth": "($max_mclk / 1000 * 32 * $num_memory_channels)",
        },
        "rdna35": {
            "GRBM_GUI_ACTIVE_PER_XCD": "(GRBM_GUI_ACTIVE / $num_xcd)",
            "GRBM_COUNT_PER_XCD": "(GRBM_COUNT / $num_xcd)",
            "GRBM_SPI_BUSY_PER_XCD": "(GRBM_SPI_BUSY / $num_xcd)",
        },
    }

    if gpu_series.startswith("MI"):
        return build_in_vars["cdna"]
    elif gpu_series.startswith("RDNA"):
        return build_in_vars["rdna35"]
    else:
        console_error(
            f"Unknown GPU series '{gpu_series}': cannot determine built-in variables."
        )
        return {}


# ---------------------------------------------------------------------------
# Block remapping — SQC and SP counters belong to the SQ IP block
# ---------------------------------------------------------------------------

BLOCK_REMAP: dict[str, str] = {"SQC": "SQ", "SP": "SQ"}


# ---------------------------------------------------------------------------
# Stateless counter-parsing helpers
# ---------------------------------------------------------------------------


def parse_counters_text(text: str) -> tuple[set[str], set[str]]:
    """Extract HW counter names and ``$variable`` names from formula *text*.

    Returns ``(hw_counters, variables)`` where *variables* are the names
    without the leading ``$``.  Matches that look like variables are removed
    from the HW counter set.
    """
    hw_counter_matches = set(HW_COUNTER_RE.findall(text))
    variable_matches = set(VARIABLE_RE.findall(text))
    # variable matches cannot be counters
    hw_counter_matches -= variable_matches
    return hw_counter_matches, variable_matches


def extract_counters_and_variables(
    text: str, gpu_series: str
) -> tuple[set[str], set[str]]:
    """Return (hw_counters, builtin_vars) referenced by text, with transitive
    resolution. Recognizes both $var and ammolite__var forms.
    """
    hw, variables = parse_counters_text(text)
    variables.update(AMMOLITE_VAR_RE.findall(text))

    for formula in SUPPORTED_DENOM.values():
        hw_d, var_d = parse_counters_text(formula)
        hw.update(hw_d)
        variables.update(var_d)

    build_in_vars = get_build_in_vars(gpu_series)
    builtin_vars: set[str] = set()
    seen: set[str] = set()
    while variables - seen:
        new_vars: set[str] = set()
        for var in variables - seen:
            seen.add(var)
            if var in build_in_vars:
                builtin_vars.add(var)
                hw_v, var_v = parse_counters_text(build_in_vars[var])
                hw.update(hw_v)
                new_vars.update(var_v)
        variables.update(new_vars)

    return hw, builtin_vars


def counter_to_block(counter: str) -> str:
    """Map a counter name to its IP block, applying :data:`BLOCK_REMAP`."""
    block = counter.split("_")[0]
    return BLOCK_REMAP.get(block, block)
