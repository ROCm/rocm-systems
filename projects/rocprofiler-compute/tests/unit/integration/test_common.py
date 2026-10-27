# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CPU regressions for integration counter comparisons and common selectors."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rocprof_compute_soc.soc_base import OmniSoC_Base
from tests.integration.common import (
    DETERMINISTIC_ACCURACY_BLOCKS,
    DETERMINISTIC_COUNTER_PATTERNS,
    STOCHASTIC_ACCURACY_BLOCKS,
    STOCHASTIC_COUNTER_PATTERNS,
    are_deterministic_counters_equal,
    are_stochastic_counters_similar,
)
from utils.utils_common import (
    build_metric_list,
    canonical_config_arch,
    load_panel_configs,
)

COUNTER_COLUMNS = (
    "Kernel_Name",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Counter_Name",
    "Counter_Value",
)


def counter_frame(counter_name, values):
    return pd.DataFrame(
        [("jacobi", 1024, 256, 1296, counter_name, value) for value in values],
        columns=COUNTER_COLUMNS,
    )


@pytest.mark.parametrize(
    "counter_name",
    [
        "SQ_WAVES",
        "SQ_INSTS_ALL_sum",
        "SPI_CSN_WAVE",
        "SPI_CS0_NUM_THREADGROUPS",
        "SPI_CSN_WAVE_sum",
        "SPI_CSN_NUM_THREADGROUPS_sum",
    ],
)
def test_deterministic_equal_counters_compare_real_groups(counter_name):
    baseline = counter_frame(counter_name, [0, 0, 0])

    counters_equal, reason = are_deterministic_counters_equal(
        [baseline.copy(), baseline.copy()], baseline
    )

    assert counters_equal, reason
    assert reason == "Compared 1 deterministic counter groups"


@pytest.mark.parametrize(
    "counter_name", ["SQ_WAVES", "SPI_CSN_WAVE_sum", "SPI_CSN_NUM_THREADGROUPS_sum"]
)
def test_deterministic_mismatch_returns_failure(counter_name):
    baseline = counter_frame(counter_name, [100, 100])
    measured = counter_frame(counter_name, [101, 101])

    counters_equal, reason = are_deterministic_counters_equal([measured], baseline)

    assert not counters_equal
    assert counter_name in reason


@pytest.mark.parametrize("counter_name", ["GRBM_GUI_ACTIVE", "SPI_CSN_WAVE_CYCLES"])
def test_deterministic_unrelated_counters_do_not_pass(counter_name):
    baseline = counter_frame(counter_name, [100, 100])

    counters_equal, reason = are_deterministic_counters_equal([baseline], baseline)

    assert not counters_equal
    assert reason == "No deterministic counter groups were compared"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "invalid"])
def test_deterministic_invalid_samples_do_not_pass(value):
    baseline = counter_frame("SQ_WAVES", [value, value])

    counters_equal, reason = are_deterministic_counters_equal([baseline], baseline)

    assert not counters_equal
    assert "invalid counter samples" in reason


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
@pytest.mark.parametrize("missing", ["baseline", "test", "test-list", "group-label"])
def test_missing_comparison_inputs_do_not_pass(kind, missing):
    counter_name = "SQ_WAVES" if kind == "deterministic" else "TCP_TCC_READ_REQ_sum"
    baseline = counter_frame(counter_name, [100, 100])
    measured = baseline.copy()
    test_frames = [measured]
    if missing == "baseline":
        baseline = baseline.iloc[:0]
    elif missing == "test":
        test_frames = [measured.iloc[:0]]
    elif missing == "test-list":
        test_frames = []
    else:
        measured["Kernel_Name"] = None

    if kind == "deterministic":
        counters_equal, _ = are_deterministic_counters_equal(test_frames, baseline)
        assert not counters_equal
    else:
        assert not are_stochastic_counters_similar(test_frames, baseline)


@pytest.mark.parametrize(
    ("baseline_values", "test_values", "expected"),
    [
        pytest.param([0, 0, 0], [0, 0, 0], True, id="constant-zero-match"),
        pytest.param([100, 100, 100], [100, 100, 100], True, id="constant-match"),
        pytest.param([100, 100, 100], [200, 200, 200], False, id="constant-mismatch"),
        pytest.param([99, 100, 101], [100, 100, 100], True, id="within-bounds"),
        pytest.param([99, 100, 101], [199, 200, 201], False, id="outside-bounds"),
        pytest.param([100], [100], False, id="too-few-samples"),
        pytest.param([100, 100], [float("nan"), 100], False, id="nan-sample"),
        pytest.param([100, 100], [float("inf"), 100], False, id="infinite-sample"),
        pytest.param([100, 100], ["invalid", 100], False, id="nonnumeric-sample"),
    ],
)
def test_stochastic_samples_require_real_comparisons(
    baseline_values, test_values, expected
):
    baseline = counter_frame("TCP_TCC_READ_REQ_sum", baseline_values)
    measured = counter_frame("TCP_TCC_READ_REQ_sum", test_values)

    assert are_stochastic_counters_similar([measured], baseline) is expected


def test_stochastic_unrelated_counters_do_not_pass():
    baseline = counter_frame("SQ_WAVES", [100, 100])

    assert not are_stochastic_counters_similar([baseline], baseline)


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
def test_missing_launch_configuration_does_not_pass(kind):
    counter_name = "SQ_WAVES" if kind == "deterministic" else "TCP_TCC_READ_REQ_sum"
    small_launch = counter_frame(counter_name, [100, 100])
    large_launch = counter_frame(counter_name, [200, 200])
    large_launch["Grid_Size"] = 2048
    baseline = pd.concat([small_launch, large_launch])

    if kind == "deterministic":
        counters_equal, reason = are_deterministic_counters_equal(
            [small_launch], baseline
        )
        assert not counters_equal
        assert "Group keys do not match" in reason
    else:
        assert not are_stochastic_counters_similar([small_launch], baseline)


@pytest.mark.parametrize(
    ("gpu_arch", "gpu_series"),
    [
        ("gfx942", "MI300"),
        ("gfx950", "MI350"),
        ("gfx1151", "RDNA3.5"),
        ("gfx1250", "GFX1250_SERIES"),
    ],
)
@pytest.mark.parametrize(
    ("blocks", "counter_patterns", "panel_titles"),
    [
        (
            DETERMINISTIC_ACCURACY_BLOCKS,
            DETERMINISTIC_COUNTER_PATTERNS,
            ("System Speed-of-Light", "Roofline", "Workgroup Manager (SPI)"),
        ),
        (
            STOCHASTIC_ACCURACY_BLOCKS,
            STOCHASTIC_COUNTER_PATTERNS,
            ("Memory Chart", "Roofline"),
        ),
    ],
    ids=["deterministic", "stochastic"],
)
def test_accuracy_blocks_resolve_to_relevant_counters_on_each_architecture(
    gpu_arch, gpu_series, blocks, counter_patterns, panel_titles
):
    config_root = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "rocprof_compute_soc"
        / "analysis_configs"
    )
    panels = load_panel_configs([str(config_root / canonical_config_arch(gpu_arch))])
    metric_list = build_metric_list(panels, None)
    assert tuple(metric_list[block] for block in blocks) == panel_titles

    args = Namespace(
        config_dir=str(config_root),
        filter_blocks=list(blocks),
        set_selected=None,
        roof_only=False,
    )
    machine_specs = SimpleNamespace(
        gpu_arch=gpu_arch, gpu_series=gpu_series, l2_banks=1
    )
    soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch(gpu_arch)
    requested_counters, selected_blocks = soc.detect_counters()

    assert selected_blocks == list(blocks)
    counter_names = pd.Series(sorted(requested_counters))
    assert counter_names.str.match("|".join(counter_patterns)).any()
    if blocks == DETERMINISTIC_ACCURACY_BLOCKS:
        assert counter_names.str.startswith("SQ_INSTS_").any()
        assert counter_names.str.match(r"SPI_CS.*(?:WAVE|NUM_THREADGROUPS)").any()
