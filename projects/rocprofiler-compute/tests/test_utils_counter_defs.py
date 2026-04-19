# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.utils_counter_defs - hardware-independent.

Test function names describe intent; per-function docstrings would be
redundant. ``missing-function-docstring`` is disabled at the module level.
"""

# pylint: disable=missing-function-docstring
# ruff: noqa: S101  # pytest idiomatically uses plain `assert` statements.

from __future__ import annotations

import pytest

from utils.utils_counter_defs import (
    AGGREGATOR_SUFFIXES,
    BLOCK_NAMES,
    BLOCK_REMAP,
    SYNTHETIC_COUNTERS,
    counter_to_block,
    extract_counter_tokens,
    extract_counters,
    extract_variable_names,
    parse_counters_text,
    remap_block,
    resolve_variables,
    strip_channel,
    strip_suffix,
)

# ---------------------------------------------------------------------------
# Block set sourced from mi_gpu_spec.yaml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        "SQ",
        "TA",
        "TD",
        "TCP",
        "TCC",
        "CPC",
        "CPF",
        "SPI",
        "GRBM",
        "GDS",  # regression: missing from legacy regex
        "GCEA",
        "GL1A",
        "GL1C",
        "GL2A",
        "GL2C",
    ],
)
def test_block_names_covers_all_perfmon_blocks(block: str) -> None:
    assert block in BLOCK_NAMES, (
        f"{block} should be discovered from mi_gpu_spec.yaml perfmon_config"
    )


def test_block_names_excludes_non_block_metadata() -> None:
    # TCC_channels is an integer-valued metadata entry but not a block name.
    assert "TCC_channels" not in BLOCK_NAMES


# ---------------------------------------------------------------------------
# extract_counter_tokens
# ---------------------------------------------------------------------------


_EXTRACT_CASES = [
    pytest.param(
        "AVG(((100 * CPF_CPF_STAT_BUSY) / (CPF_CPF_STAT_BUSY + CPF_CPF_STAT_IDLE)))",
        {"CPF_CPF_STAT_BUSY", "CPF_CPF_STAT_IDLE"},
        id="simple-avg-formula",
    ),
    pytest.param(
        "TO_INT(MIN(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) / $GRBM_GUI_ACTIVE_PER_XCD, 0)))",
        {"SQ_BUSY_CU_CYCLES"},
        id="variable-body-not-counted-as-counter",
    ),
    pytest.param(
        "(TCC_HIT[0] + TCC_HIT[1] + TCC_MISS[0])",
        {"TCC_HIT[", "TCC_MISS["},
        id="tcc-channel-template-preserves-bracket-marker",
    ),
    pytest.param(
        "TCC_EA_RDREQ_sum + SQ_WAVES",
        {"TCC_EA_RDREQ_sum", "SQ_WAVES"},
        id="aggregator-suffix-preserved",
    ),
    pytest.param(
        "SQC_ICACHE_HITS + SP_INSTS + SQ_WAVES",
        {"SQC_ICACHE_HITS", "SP_INSTS", "SQ_WAVES"},
        id="aliased-blocks-SQC-and-SP-still-extract",
    ),
    pytest.param(
        "GDS_BUSY + 100",
        {"GDS_BUSY"},
        id="GDS-counters-extracted-regression-fix",
    ),
    pytest.param(
        "AVG(100) + ROUND(SUM(End_Timestamp - Start_Timestamp)) + None",
        set(),
        id="formula-keywords-not-misclassified",
    ),
    pytest.param(
        "",
        set(),
        id="empty-string",
    ),
]


@pytest.mark.parametrize(("text", "expected"), _EXTRACT_CASES)
def test_extract_counter_tokens(text: str, expected: set[str]) -> None:
    assert extract_counter_tokens(text) == expected


# ---------------------------------------------------------------------------
# extract_variable_names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SQ_WAVES / $GRBM_GUI_ACTIVE_PER_XCD", {"GRBM_GUI_ACTIVE_PER_XCD"}),
        ("$max_mclk + $num_hbm_channels", {"max_mclk", "num_hbm_channels"}),
        ("no dollar sign here", set()),
        ("$TrailingDollar$", {"TrailingDollar"}),
    ],
)
def test_extract_variable_names(text: str, expected: set[str]) -> None:
    assert extract_variable_names(text) == expected


# ---------------------------------------------------------------------------
# strip_suffix / strip_channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("TCC_EA_RDREQ_sum", "TCC_EA_RDREQ"),
        ("SQ_WAVES_avr", "SQ_WAVES"),
        ("SQ_WAVES_max", "SQ_WAVES"),
        ("SQ_WAVES_min", "SQ_WAVES"),
        ("SQ_WAVES", "SQ_WAVES"),  # no suffix
    ],
)
def test_strip_suffix(token: str, expected: str) -> None:
    assert strip_suffix(token) == expected


def test_aggregator_suffix_tuple_is_stable() -> None:
    # Tests depend on the exact suffix set; if this changes the helper
    # semantics shift and dependent tests must be reviewed.
    assert AGGREGATOR_SUFFIXES == ("_sum", "_avr", "_max", "_min")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("TCC_HIT[0]", "TCC_HIT"),
        ("TCC_HIT[15]", "TCC_HIT"),
        ("TCC_HIT[", "TCC_HIT"),  # template marker from extract_counter_tokens
        ("SQ_WAVES", "SQ_WAVES"),
    ],
)
def test_strip_channel(token: str, expected: str) -> None:
    assert strip_channel(token) == expected


# ---------------------------------------------------------------------------
# remap_block / counter_to_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("counter", "expected"),
    [
        ("SQC_WAVES", "SQ"),  # SQC aliased to SQ
        ("SP_INSTS", "SQ"),  # SP aliased to SQ
        ("SQ_WAVES", "SQ"),  # identity
        ("TCC_HIT", "TCC"),
        ("GRBM_GUI_ACTIVE", "GRBM"),
        ("GDS_BUSY", "GDS"),
    ],
)
def test_remap_block(counter: str, expected: str) -> None:
    assert remap_block(counter) == expected


def test_counter_to_block_matches_remap_block() -> None:
    # Public legacy alias must stay consistent with the internal helper.
    for c in ("SQC_WAVES", "SP_INSTS", "SQ_WAVES", "TCC_HIT", "GDS_BUSY"):
        assert counter_to_block(c) == remap_block(c)


def test_block_remap_constants_match_documented_aliases() -> None:
    assert BLOCK_REMAP == {"SQC": "SQ", "SP": "SQ"}


# ---------------------------------------------------------------------------
# resolve_variables
# ---------------------------------------------------------------------------


def test_resolve_variables_follows_build_in_vars() -> None:
    # Formula references a $var whose body mentions another counter.
    build_in_vars = {
        "GRBM_GUI_ACTIVE_PER_XCD": "(GRBM_GUI_ACTIVE / $num_xcd)",
    }
    out = resolve_variables(
        "SQ_WAVES / $GRBM_GUI_ACTIVE_PER_XCD",
        build_in_vars,
    )
    assert out == {"SQ_WAVES", "GRBM_GUI_ACTIVE"}


def test_resolve_variables_folds_in_extra_texts() -> None:
    build_in_vars: dict[str, str] = {}
    # extra_texts simulates SUPPORTED_DENOM formulas.
    out = resolve_variables(
        "SQ_WAVES",
        build_in_vars,
        extra_texts=["(End_Timestamp - Start_Timestamp) / 1000000000"],
    )
    # extra text adds no counters here (Start_Timestamp is a keyword), but
    # the call must still succeed and include SQ_WAVES.
    assert "SQ_WAVES" in out


def test_resolve_variables_filters_synthetic_counters() -> None:
    # SQ_ACCUM_PREV_HIRES is injected elsewhere; the parser must drop it.
    out = resolve_variables("SQ_WAVES + SQ_ACCUM_PREV_HIRES", {})
    assert "SQ_ACCUM_PREV_HIRES" not in out
    assert "SQ_WAVES" in out
    assert "SQ_ACCUM_PREV_HIRES" in SYNTHETIC_COUNTERS


def test_resolve_variables_unknown_variable_is_ignored() -> None:
    # A $var with no BUILD_IN_VARS entry must not raise - it is simply not
    # expanded. This matches legacy parse_counters behaviour.
    out = resolve_variables("SQ_WAVES / $unknown_var", {})
    assert out == {"SQ_WAVES"}


# ---------------------------------------------------------------------------
# Legacy-compatible top-level API (kept for existing callers)
# ---------------------------------------------------------------------------


def test_parse_counters_text_returns_pair() -> None:
    hw, variables = parse_counters_text(
        "SQ_WAVES / $GRBM_GUI_ACTIVE_PER_XCD + CPF_CPF_STAT_BUSY"
    )
    assert hw == {"SQ_WAVES", "CPF_CPF_STAT_BUSY"}
    assert variables == {"GRBM_GUI_ACTIVE_PER_XCD"}


def test_extract_counters_resolves_supported_denom() -> None:
    # Using $GRBM_GUI_ACTIVE_PER_XCD as a denom pulls GRBM_GUI_ACTIVE in via
    # BUILD_IN_VARS; extract_counters must see it transitively.
    out = extract_counters("SQ_WAVES / $GRBM_GUI_ACTIVE_PER_XCD")
    assert "SQ_WAVES" in out
    assert "GRBM_GUI_ACTIVE" in out
    assert "SQ_ACCUM_PREV_HIRES" not in out
