# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/utils/tty.py."""

import argparse

import pandas as pd
import pytest

from utils.tty import convert_time_columns, format_table_output, has_time_data
from utils.utils_common import is_gfx115x

TIME_UNITS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


def make_args() -> argparse.Namespace:
    """Minimal args for the plain-table render path."""
    return argparse.Namespace(decimal=2, view=None, normal_unit="per_wave")


def _sample_time_data() -> pd.DataFrame:
    """Metric table mixing a time row, a cycle row, and a count row."""
    return pd.DataFrame({
        "Metric_ID": ["7.2.0", "7.2.1", "7.2.2"],
        "Metric": ["Kernel Time", "Kernel Time (Cycles)", "Non-Time Metric"],
        "Avg": [3446.64, 64499.39, 1000.0],
        "Min": [1769.25, 17269.25, 500.0],
        "Max": [12532.12, 337030.50, 2000.0],
        "Unit": ["ns", "Cycle", "Count"],
    })


def _original_ns_values() -> dict[str, float]:
    """Original nanosecond values for the time row of _sample_time_data."""
    return {"Avg": 3446.64, "Min": 1769.25, "Max": 12532.12}


def test_format_table_output_suppresses_empty_column() -> None:
    """A non-PC-sampling table with an all-'N/A' column is suppressed."""
    df = pd.DataFrame({"Metric": ["a", "b"], "Value": ["N/A", "N/A"]})
    content = format_table_output(
        make_args(),
        {"id": 1101, "title": "Some Table"},
        df,
        "metric_table",
        runs={"only": object()},
    )
    assert content == ""


def test_format_table_output_keeps_pc_sampling_table_21_1() -> None:
    """PC sampling table 21.1 is shown even with an all-'N/A' source column."""
    df = pd.DataFrame({
        "source_line": ["N/A", "N/A"],
        "instruction": ["v_mov", "v_add"],
        "count": [3, 1],
    })
    content = format_table_output(
        make_args(),
        {"id": 2101, "title": "PC Sampling"},
        df,
        "pc_sampling_table",
        runs={"only": object()},
    )
    assert content != ""
    assert "v_mov" in content


def test_has_time_data_detection() -> None:
    """has_time_data is True only when a 'ns' Unit column is present."""
    assert has_time_data(_sample_time_data())

    no_time_data = pd.DataFrame({
        "Metric": ["Non-Time Metric"],
        "Avg": [1000.0],
        "Unit": ["Count"],
    })
    assert not has_time_data(no_time_data)

    no_unit_column = pd.DataFrame({"Metric": ["Some Metric"], "Avg": [1000.0]})
    assert not has_time_data(no_unit_column)


def test_default_unit_is_nanoseconds() -> None:
    """The fixture's time row defaults to nanoseconds."""
    sample_time_data = _sample_time_data()
    time_rows = sample_time_data["Unit"].str.lower().str.contains("ns", na=False)
    assert time_rows.any()
    assert sample_time_data.loc[0, "Unit"] == "ns"


def test_conversion_to_seconds() -> None:
    """Converting to seconds divides the time row and leaves others untouched."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "s")

    assert converted_df.loc[0, "Unit"] == "s"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["s"], abs=1e-10
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"] / TIME_UNITS["s"], abs=1e-10
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"] / TIME_UNITS["s"], abs=1e-10
    )
    assert converted_df.loc[1, "Unit"] == "Cycle"
    assert converted_df.loc[2, "Unit"] == "Count"


def test_conversion_to_milliseconds() -> None:
    """Converting to milliseconds divides the time row by 10^6."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "ms")

    assert converted_df.loc[0, "Unit"] == "ms"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["ms"], abs=1e-6
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"] / TIME_UNITS["ms"], abs=1e-6
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"] / TIME_UNITS["ms"], abs=1e-6
    )


def test_conversion_to_microseconds() -> None:
    """Converting to microseconds divides the time row by 10^3."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "us")

    assert converted_df.loc[0, "Unit"] == "us"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["us"], abs=1e-3
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"] / TIME_UNITS["us"], abs=1e-3
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"] / TIME_UNITS["us"], abs=1e-3
    )


def test_conversion_to_nanoseconds() -> None:
    """Converting to nanoseconds leaves the time row unchanged."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "ns")

    assert converted_df.loc[0, "Unit"] == "ns"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"], abs=1e-10
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"], abs=1e-10
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"], abs=1e-10
    )


def test_non_time_rows_unchanged() -> None:
    """Cycle and Count rows keep their unit and value after conversion."""
    converted_df = convert_time_columns(_sample_time_data(), "ms")

    assert converted_df.loc[1, "Unit"] == "Cycle"
    assert converted_df.loc[2, "Unit"] == "Count"
    assert converted_df.loc[1, "Avg"] == 64499.39
    assert converted_df.loc[2, "Avg"] == 1000.0


def test_invalid_time_unit_is_noop() -> None:
    """An unrecognised target unit leaves the frame unchanged."""
    sample_time_data = _sample_time_data()
    original_df = sample_time_data.copy()
    converted_df = convert_time_columns(sample_time_data, "invalid_unit")
    pd.testing.assert_frame_equal(converted_df, original_df)


def test_missing_unit_column_is_noop() -> None:
    """A frame with no Unit column is returned unchanged."""
    df_no_unit = pd.DataFrame({"Metric": ["Test Metric"], "Avg": [1000.0]})
    converted_df = convert_time_columns(df_no_unit, "ms")
    pd.testing.assert_frame_equal(converted_df, df_no_unit)


def test_conversion_with_missing_columns() -> None:
    """Conversion works when Min/Max columns are absent."""
    original_ns_values = _original_ns_values()
    df_partial = _sample_time_data()[["Metric_ID", "Metric", "Avg", "Unit"]].copy()
    converted_df = convert_time_columns(df_partial, "ms")

    assert converted_df.loc[0, "Unit"] == "ms"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["ms"], abs=1e-6
    )


def test_mathematical_correctness_all_units() -> None:
    """Every supported unit divides the time row by the correct factor."""
    original_ns_values = _original_ns_values()
    for target_unit, divisor in TIME_UNITS.items():
        converted_df = convert_time_columns(_sample_time_data(), target_unit)

        assert converted_df.loc[0, "Avg"] == pytest.approx(
            original_ns_values["Avg"] / divisor, abs=1e-10
        )
        assert converted_df.loc[0, "Min"] == pytest.approx(
            original_ns_values["Min"] / divisor, abs=1e-10
        )
        assert converted_df.loc[0, "Max"] == pytest.approx(
            original_ns_values["Max"] / divisor, abs=1e-10
        )
        assert converted_df.loc[0, "Unit"] == target_unit


def test_integration_conversion_flow() -> None:
    """has_time_data gates convert_time_columns in the show_all flow."""
    args = argparse.Namespace(time_unit="ms", decimal=2)

    sample_df = pd.DataFrame({
        "Metric_ID": ["7.2.0"],
        "Metric": ["Kernel Time"],
        "Avg": [3446640.0],
        "Min": [1769250.0],
        "Max": [12532120.0],
        "Unit": ["ns"],
    })

    if has_time_data(sample_df):
        converted_df = convert_time_columns(sample_df, args.time_unit)
    else:
        converted_df = sample_df

    assert converted_df.loc[0, "Unit"] == "ms"
    assert converted_df.loc[0, "Avg"] == pytest.approx(3.44664, abs=1e-5)
    assert converted_df.loc[0, "Min"] == pytest.approx(1.76925, abs=1e-5)
    assert converted_df.loc[0, "Max"] == pytest.approx(12.53212, abs=1e-5)


def test_show_all_with_time_unit_conversion() -> None:
    """Mixed-case 'Ns' unit converts correctly across every target unit."""
    test_data = pd.DataFrame({
        "Metric_ID": ["7.2.0"],
        "Metric": ["Kernel Time"],
        "Avg": [3446.64],
        "Min": [1769.25],
        "Max": [12532.12],
        "Unit": ["Ns"],
    })

    for time_unit in ["s", "ms", "us", "ns"]:
        converted_df = convert_time_columns(test_data, time_unit)
        assert converted_df.loc[0, "Unit"] == time_unit
        assert converted_df.loc[0, "Avg"] == pytest.approx(
            3446.64 / TIME_UNITS[time_unit], abs=1e-10
        )


def test_edge_cases_and_error_handling() -> None:
    """Empty, NaN, and mixed-case unit frames convert without error."""
    empty_df = pd.DataFrame()
    assert convert_time_columns(empty_df, "ms").empty

    nan_df = pd.DataFrame({
        "Avg": [float("nan"), 1000.0],
        "Unit": ["ns", "Count"],
    })
    result = convert_time_columns(nan_df, "ms")
    assert result.loc[0, "Unit"] == "ms"

    mixed_case_df = pd.DataFrame({
        "Avg": [1000.0, 2000.0],
        "Unit": ["ns", "NS"],
    })
    result = convert_time_columns(mixed_case_df, "ms")
    assert result.loc[0, "Unit"] == "ms"
    assert result.loc[1, "Unit"] == "ms"


@pytest.mark.parametrize(
    "gpu_arch",
    [
        pytest.param("gfx1151", id="rdna35"),
        pytest.param("gfx942", id="cdna"),
    ],
)
def test_format_table_output_dispatches_memory_chart_renderer(
    monkeypatch: pytest.MonkeyPatch,
    gpu_arch: str,
) -> None:
    """Memory Chart output uses the architecture renderer and shared heading."""
    calls: dict[str, dict] = {}

    def record(name: str, return_value: str):
        def stub(mem_data: dict, *, chart_title: str) -> str:
            calls[name] = {
                "mem_data": mem_data,
                "chart_title": chart_title,
            }
            return return_value

        return stub

    monkeypatch.setattr(
        "utils.tty.mem_chart_gfx11.plot_mem_chart",
        record("gfx11", "rendered RDNA3.5 memory chart"),
    )
    monkeypatch.setattr(
        "utils.tty.mem_chart_gfx9.plot_mem_chart",
        record("gfx9", "rendered CDNA memory chart"),
    )
    df = pd.DataFrame({"Metric": ["Metric A"], "Value": [1]})

    content = format_table_output(
        make_args(),
        {
            "id": 701,
            "title": "Memory Chart",
            "cli_style": "mem_chart",
        },
        df,
        "metric_table",
        runs={"only": object()},
        gpu_arch=gpu_arch,
    )

    expected = "gfx11" if is_gfx115x(gpu_arch) else "gfx9"
    unexpected = "gfx9" if is_gfx115x(gpu_arch) else "gfx11"
    assert calls[expected] == {
        "mem_data": {"Metric A": 1},
        "chart_title": "7. Memory Chart (Normalization: per_wave)",
    }
    assert unexpected not in calls
    return_value = (
        "rendered RDNA3.5 memory chart"
        if is_gfx115x(gpu_arch)
        else "rendered CDNA memory chart"
    )
    assert content == f"{return_value}\n"
