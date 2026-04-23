# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared counter definitions for HW performance counter parsing.

This module is intentionally dependency-free (only ``re``) so that it can be
imported by both the main runtime code and lightweight tooling such as
``tools/validate_sets_metric_ids.py``.
"""

import re

# ---------------------------------------------------------------------------
# Regex patterns for HW counter and variable extraction
# ---------------------------------------------------------------------------

# HW counter name must start with an IP block prefix, followed by an
# underscore-separated suffix of uppercase letters/digits, optionally
# ending with '[' or an aggregation suffix (_sum, _avr, _max, _min).
HW_COUNTER_BLK = (
    r"(?:SQ|SQC|SP|TA|TD|TCP|TCC|GL1A|GL1C|GL2A|GL2C|"
    r"CPC|CPF|SPI|GCEA|GRBM)"
)
HW_COUNTER_SFX = r"_[0-9A-Z_]*[0-9A-Z](?:\[|_sum|_avr|_max|_min)*"
HW_COUNTER_RE = re.compile(HW_COUNTER_BLK + HW_COUNTER_SFX)

# Captures the variable name after '$'.
VARIABLE_RE = re.compile(r"\$([0-9A-Za-z_]*[0-9A-Za-z])")

# ---------------------------------------------------------------------------
# Built-in variable and denominator definitions
# ---------------------------------------------------------------------------

SUPPORTED_DENOM: dict[str, str] = {
    "per_wave": "SQ_WAVES",
    "per_cycle": "$GRBM_GUI_ACTIVE_PER_XCD",
    "per_second": "((End_Timestamp - Start_Timestamp) / 1000000000)",
    "per_kernel": "1",
}

BUILD_IN_VARS: dict[str, str] = {
    "GRBM_GUI_ACTIVE_PER_XCD": "(GRBM_GUI_ACTIVE / $num_xcd)",
    "GRBM_COUNT_PER_XCD": "(GRBM_COUNT / $num_xcd)",
    "GRBM_SPI_BUSY_PER_XCD": "(GRBM_SPI_BUSY / $num_xcd)",
    "numActiveCUs": (
        "TO_INT(MIN(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) / "
        "SUM($GRBM_GUI_ACTIVE_PER_XCD), 0) / $max_waves_per_cu * 8 + "
        "MIN(MOD(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) / "
        "SUM($GRBM_GUI_ACTIVE_PER_XCD), 0), $max_waves_per_cu), 8), $cu_per_gpu))"
    ),
    "kernelBusyCycles": (
        "ROUND(AVG((((End_Timestamp - Start_Timestamp) / 1000) * $max_sclk)), 0)"
    ),
    "hbmBandwidth": "($max_mclk / 1000 * 32 * $num_hbm_channels)",
}

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


def extract_counters(text: str) -> set[str]:
    """Return the full set of HW counters referenced by *text*.

    Resolves ``$variable`` references and supported denominators
    recursively so that all transitive counter dependencies are included.
    """
    hw, variables = parse_counters_text(text)

    # Include counters from all supported denominators
    for formula in SUPPORTED_DENOM.values():
        hw_d, var_d = parse_counters_text(formula)
        hw.update(hw_d)
        variables.update(var_d)

    # Recursively resolve built-in variables
    seen: set[str] = set()
    while variables - seen:
        new_vars: set[str] = set()
        for var in variables - seen:
            seen.add(var)
            if var in BUILD_IN_VARS:
                hw_v, var_v = parse_counters_text(BUILD_IN_VARS[var])
                hw.update(hw_v)
                new_vars.update(var_v)
        variables.update(new_vars)

    return hw


def counter_to_block(counter: str) -> str:
    """Map a counter name to its IP block, applying :data:`BLOCK_REMAP`."""
    block = counter.split("_")[0]
    return BLOCK_REMAP.get(block, block)
