# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metric_listing.py."""

import pytest

import utils.metric_listing as metric_listing

METRIC_IDS = [
    "0",
    "2",
    "2.1",
    "2.1.0",
    "2.1.1",
    "4",
    "4.1",
    "4.1.0",
    "18",
    "18.1",
    "18.1.0",
    "18.2",
    "21",
    "30",
    "30.1",
    "30.1.0",
]
ALIAS_TO_BLOCK_ID = {"sol": "2", "l2_per_channel": "18"}
BLOCK_ALIASES = {block_id: alias for alias, block_id in ALIAS_TO_BLOCK_ID.items()}
LONG_DESCRIPTION = " ".join(["Counts the cycles the VALU was busy."] * 6)
METRIC_DETAILS = {
    "0": ("Top Stats", "", ""),
    "2": ("System Speed-of-Light", "", ""),
    "2.1": ("System Speed-of-Light", "", ""),
    "2.1.0": ("VALU Utilization", "Percent", LONG_DESCRIPTION),
    "2.1.1": ("VALU Instructions", "(Req + $normUnit)", "Short."),
    "18": ("L2 Cache (per Channel)", "", ""),
    "18.1": ("Aggregate Stats", "", ""),
    "18.1.0": ("L2 Cache Hit Rate", "Percent", ""),
    "18.2": ("L2 Cache Hit Rate (Percent)", "", ""),
}


@pytest.mark.parametrize(
    ("selection", "expected_ids"),
    [
        ([], ["0", "2", "4", "18", "21", "30"]),
        (["all"], METRIC_IDS),
        (["sol", "18.1.0"], ["2", "2.1", "2.1.0", "2.1.1", "18", "18.1", "18.1.0"]),
    ],
)
def test_select_listed_ids(selection, expected_ids):
    """No selection lists blocks; ids and aliases keep their parents and children."""
    listed_ids = metric_listing.select_listed_ids(
        METRIC_IDS, selection, ALIAS_TO_BLOCK_ID
    )
    assert listed_ids == expected_ids


@pytest.mark.parametrize("selection", [["bogus"], ["9"], ["2.1.5"], ["all", "sol"]])
def test_select_listed_ids_rejects_invalid_selection(selection):
    with pytest.raises(SystemExit):
        metric_listing.select_listed_ids(METRIC_IDS, selection, ALIAS_TO_BLOCK_ID)


@pytest.mark.parametrize(
    ("profiling_config", "has_roofline_data", "expected_blocks"),
    [
        ({"filter_blocks": ["2"]}, True, ["0", "2"]),
        ({"filter_blocks": []}, False, ["0", "2", "18"]),
        (
            {"filter_blocks": [], "membw_analysis": True},
            True,
            ["0", "2", "4", "18", "30"],
        ),
        ({"filter_blocks": ["21"]}, False, ["0", "21"]),
    ],
)
def test_keep_profiled_ids(profiling_config, has_roofline_data, expected_blocks):
    """Blocks follow the profiling config; Roofline needs roofline data, block 30
    --membw-analysis, and PC Sampling an explicit request."""
    kept_ids = metric_listing.keep_profiled_ids(
        METRIC_IDS, profiling_config, has_roofline_data, "gfx950"
    )
    assert [metric_id for metric_id in kept_ids if "." not in metric_id] == (
        expected_blocks
    )


def test_format_metric_tables(monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")
    listed_ids = ["0", *(key for key in METRIC_DETAILS if key != "0")]

    output = metric_listing.format_metric_tables(
        METRIC_DETAILS, listed_ids, BLOCK_ALIASES, {"18.2"}
    )
    lines = output.splitlines()

    assert "2. System Speed-of-Light (alias: sol)" in lines
    assert "0. Top Stats" in lines
    assert lines[lines.index("0. Top Stats") + 1] == "No metrics to list."
    assert "Per-channel metrics are not listed: 18.2" in lines
    assert "Req per normUnit" in output
    # Both tables share column widths and fill the terminal width
    grid_lines = [line for line in lines if line.startswith(tuple("╒╞├╘│"))]
    assert {len(line) for line in grid_lines} == {100}
    # The long description wraps onto several lines of one row
    row_start = next(i for i, line in enumerate(lines) if "│ 2.1.0 " in line)
    assert lines[row_start + 1].startswith("│           │ ")
