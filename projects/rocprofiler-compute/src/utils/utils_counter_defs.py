# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared counter definitions for HW performance counter parsing.

Imported by both the main runtime code and lightweight tooling such as
``tools/validate_sets_metric_ids.py``.
"""

import re
from typing import Any, Dict, List, Set, Tuple

import config
from utils.logger import console_error
from utils.utils_common import canonical_config_arch
from vendored import yaml

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
# Collectables and denominator definitions
# ---------------------------------------------------------------------------

# Per-arch files of named formulas, written $name in metric formulas
COLLECTABLES_DIR = config.analysis_configs_dir / "collectables"

# per_kernel denom column, injected at pmc load with 1 per dispatch so
# SUM($denom) == N and Avg is the mean per dispatch.
UNIT_COUNTER = "Dispatch_Unit"

SUPPORTED_DENOM: Dict[str, str] = {
    "per_wave": "SQ_WAVES",
    "per_cycle": "$GRBM_GUI_ACTIVE_PER_XCD",
    "per_second": "((End_Timestamp - Start_Timestamp) / 1000000000)",
    # A vector, not a scalar, so SUM($denom) == N.
    "per_kernel": UNIT_COUNTER,
}


# ---------------------------------------------------------------------------
# Block remapping: SQC and SP counters belong to the SQ IP block
# ---------------------------------------------------------------------------

BLOCK_REMAP: Dict[str, str] = {"SQC": "SQ", "SP": "SQ"}


# ---------------------------------------------------------------------------
# Stateless counter-parsing helpers
# ---------------------------------------------------------------------------


def metric_value_keys(table: Dict[str, Any]) -> Set[str]:
    """Metric keys whose values are columns of a metric_table: the header's
    keys except metric and expr."""
    return set(table["header"]) - {"metric", "expr"}


def metric_expressions(table: Dict[str, Any], metric: Dict[str, Any]) -> List[str]:
    """Return the expressions of one metric of a metric_table.

    simple_box tables keep their one expression in expr; other tables use
    metric_value_keys. Empty and "None" values are left out.
    """
    if table.get("cli_style") == "simple_box":
        keys = {"expr"}
    else:
        keys = metric_value_keys(table)
    return [
        value
        for key, value in metric.items()
        if key in keys and isinstance(value, str) and value and value != "None"
    ]


def parse_counters_text(text: str) -> Tuple[Set[str], Set[str]]:
    """Extract HW counter names and $variable names from formula text.

    Returns (hw_counters, variables) where variables are the names without
    the leading $. Matches that look like variables are removed from the HW
    counter set.
    """
    hw_counter_matches = set(HW_COUNTER_RE.findall(text))
    variable_matches = set(VARIABLE_RE.findall(text))
    # variable matches cannot be counters
    hw_counter_matches -= variable_matches
    return hw_counter_matches, variable_matches


def extract_counters_and_variables(
    text: str, collectables: Dict[str, str], include_supported_denom: bool = True
) -> Tuple[Set[str], Set[str]]:
    """Return (hw_counters, collectable names) referenced by text, with the
    counters of the collectables it names. Recognizes both $var and
    ammolite__var forms.

    Profiling must collect the SUPPORTED_DENOM counters too, because every
    metric is implicitly divided by its unit denominator. Pass
    include_supported_denom=False to get only the counters a formula names
    itself, which is what deciding perfmon bucket membership needs.
    """
    hw, variables = parse_counters_text(text)
    variables.update(AMMOLITE_VAR_RE.findall(text))

    if include_supported_denom:
        for formula in SUPPORTED_DENOM.values():
            hw_d, var_d = parse_counters_text(formula)
            hw.update(hw_d)
            variables.update(var_d)

    collectable_names: set[str] = set()
    seen: set[str] = set()
    while variables - seen:
        new_vars: set[str] = set()
        for var in variables - seen:
            seen.add(var)
            if var in collectables:
                collectable_names.add(var)
                hw_v, var_v = parse_counters_text(collectables[var])
                hw.update(hw_v)
                new_vars.update(var_v)
        variables.update(new_vars)

    return hw, collectable_names


def counter_to_block(counter: str) -> str:
    """Map a counter name to its IP block, applying the BLOCK_REMAP table."""
    if counter.startswith("GC_CANE_"):
        return "GC_CANE"
    if counter.startswith("GC_EA_SE_"):
        return "GC_EA_SE"
    block = counter.split("_")[0]
    if block == "TX":
        return "TCP"
    return BLOCK_REMAP.get(block, block)


def _dependency_order(formulas: Dict[str, str]) -> Dict[str, str]:
    """Return formulas ordered so each collectable comes after those it uses."""
    ordered: Dict[str, str] = {}
    pending = dict(formulas)
    while pending:
        ready = {
            name: formula
            for name, formula in pending.items()
            if not parse_counters_text(formula)[1] & pending.keys()
        }
        if not ready:
            console_error(f"Collectables use each other in a loop: {sorted(pending)}")
        ordered.update(ready)
        pending = {name: pending[name] for name in pending.keys() - ready.keys()}
    return ordered


class Collectables:
    """The collectables of each arch: named formulas that metric formulas use
    as $name. Each arch's file is read once, and its collectables are kept so
    that each one comes after the collectables it uses."""

    _formulas_by_arch: Dict[str, Dict[str, str]] = {}

    @classmethod
    def of_arch(cls, gpu_arch: str) -> Dict[str, str]:
        """Return the formula of each collectable of gpu_arch, by name."""
        config_arch = canonical_config_arch(gpu_arch)
        if config_arch not in cls._formulas_by_arch:
            path = COLLECTABLES_DIR / f"{config_arch}_collectables.yaml"
            if not path.is_file():
                console_error(f"No collectables file for {gpu_arch}: {path}")
            cls._formulas_by_arch[config_arch] = _dependency_order(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            )
        return cls._formulas_by_arch[config_arch]
