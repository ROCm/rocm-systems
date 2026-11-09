# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Navi48 SoC setup and WMMA collection diagnostics without GPU hardware."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

import config
from rocprof_compute_soc import soc_gfx1201
from rocprof_compute_soc.soc_base import OmniSoC_Base, flat_counters_in_perfmon_file
from utils.mi_gpu_spec import mi_gpu_specs


@pytest.fixture
def navi48_soc(monkeypatch, tmp_path):
    monkeypatch.setattr(OmniSoC_Base, "populate_mspec", lambda self: None)
    args = Namespace(
        output_directory=str(tmp_path),
        no_roof=True,
        filter_blocks=[],
        config_dir=config.analysis_configs_dir,
        membw_analysis=False,
        roof_only=False,
        set_selected=None,
    )
    return soc_gfx1201.gfx1201_soc(args, SimpleNamespace())


def test_gfx1201_soc_initializes_with_current_base_api(navi48_soc):
    assert navi48_soc.get_arch() == "gfx1201"
    assert navi48_soc._mspec.l2_banks == 32
    assert navi48_soc._mspec.lds_banks_per_cu == 32
    assert navi48_soc._mspec.pipes_per_gpu == 2


def test_gfx1201_profiling_setup_returns_selected_blocks(navi48_soc, monkeypatch):
    monkeypatch.setattr(navi48_soc, "perfmon_filter", lambda: ["2", "3"])
    assert navi48_soc.profiling_setup() == ["2", "3"]


@pytest.mark.parametrize(
    "requested, collected, should_warn",
    [
        pytest.param(
            {"SQ_WMMA_VALU_INSTS"},
            {"SQ_WMMA_VALU_INSTS_sum"},
            False,
            id="sum-alias-collected",
        ),
        pytest.param(
            {"SQ_WMMA_VALU_INSTS_sum"},
            {"SQ_WMMA_VALU_INSTS"},
            False,
            id="raw-alias-collected",
        ),
        pytest.param({"SQ_WMMA_VALU_INSTS_sum"}, set(), False, id="no-kernel-data"),
        pytest.param(
            {"SQ_WMMA_VALU_INSTS_sum"},
            {"UNRELATED_COUNTER"},
            False,
            id="no-requested-counter-data",
        ),
        pytest.param(
            {"SQ_INSTS_VALU_sum"}, {"SQ_INSTS_VALU_sum"}, False, id="wmma-not-requested"
        ),
        pytest.param(
            {"SQ_WMMA_VALU_INSTS_sum", "SQ_WAVES_sum"},
            {"SQ_WAVES_sum"},
            True,
            id="wmma-missing-from-populated-pass",
        ),
    ],
)
def test_counter_collection_diagnoses_only_missing_requested_wmma(
    navi48_soc, monkeypatch, requested, collected, should_warn
):
    warnings = []
    monkeypatch.setattr(soc_gfx1201, "console_warning", warnings.append)
    navi48_soc.validate_counter_collection(requested, collected)
    assert bool(warnings) is should_warn
    if should_warn:
        assert len(warnings) == 1
        assert "SQ event 233" in warnings[0]
        assert "collected other counters" in warnings[0]


@pytest.mark.parametrize(
    "filter_blocks, expected_passes",
    [
        pytest.param([], (1, 2), id="full-profile"),
        pytest.param(["17.1.0"], (1, 1), id="gpu-busy-only"),
    ],
)
def test_gpu_busy_retains_fei_pass_placement(
    navi48_soc, filter_blocks, expected_passes
):
    """Protect the current allocation while Fei's pairing confirmation is pending."""
    navi48_soc.get_args().filter_blocks = filter_blocks
    navi48_soc._mspec.gpu_arch = "gfx1201"
    navi48_soc._mspec.gpu_series = mi_gpu_specs.get_gpu_series("gfx1201")
    counters, _ = navi48_soc.detect_counters()
    counter_files, _, _ = navi48_soc._allocate_perfmon_counter_files(counters)
    placements = {
        counter: int(counter_file.name) + 1
        for counter_file in counter_files
        for counter in flat_counters_in_perfmon_file(counter_file)
    }
    assert (placements["GRBM_COUNT"], placements["GRBM_GUI_ACTIVE"]) == expected_passes
