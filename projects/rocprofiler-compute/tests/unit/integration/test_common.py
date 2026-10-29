# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CPU regressions for profiling counter selections."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from config import analysis_configs_dir
from rocprof_compute_soc.soc_base import OmniSoC_Base
from tests.integration.common import (
    LIVE_ATTACH_SET,
    MEMORY_CHART_COUNTERS,
    SPI_BUSY_COUNTERS,
    metric_table_counters,
)
from utils.mi_gpu_spec import mi_gpu_specs


@pytest.mark.parametrize("gpu_arch", sorted(mi_gpu_specs.get_gpu_series_dict()))
def test_live_attach_set_fits_one_pass_on_each_architecture(gpu_arch):
    args = Namespace(
        config_dir=str(analysis_configs_dir),
        filter_blocks=[],
        set_selected=LIVE_ATTACH_SET,
        roof_only=False,
        membw_analysis=False,
    )
    machine_specs = SimpleNamespace(
        gpu_arch=gpu_arch,
        gpu_series=mi_gpu_specs.get_gpu_series(gpu_arch),
        l2_banks=1,
    )
    soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch(gpu_arch)
    soc.set_perfmon_config(mi_gpu_specs.get_perfmon_config(gpu_arch))
    requested_counters, selected_blocks = soc.detect_counters()
    assert selected_blocks
    assert requested_counters

    counter_files, _, accumulator_count = soc._allocate_perfmon_counter_files(
        requested_counters
    )
    assert len(counter_files) == 1, f"{gpu_arch}: live attachment needs one pass"
    assert accumulator_count == 0, f"{gpu_arch}: live attachment set uses accumulators"


@pytest.mark.parametrize("gpu_arch", sorted(mi_gpu_specs.get_gpu_series_dict()))
@pytest.mark.parametrize(
    "blocks",
    [
        pytest.param(["3"], id="memory-chart"),
        pytest.param(["3", "6"], id="multiple-blocks"),
        pytest.param(["6.1"], id="subsection"),
    ],
)
def test_block_filters_select_distinct_counters_on_each_architecture(gpu_arch, blocks):
    tables = metric_table_counters(gpu_arch, 6)
    workgroup_counters = set().union(*tables.values())
    memory_counters = set().union(*metric_table_counters(gpu_arch, 3).values())
    if blocks == ["3"]:
        expected_counters = memory_counters
        assert expected_counters & MEMORY_CHART_COUNTERS
        assert (workgroup_counters - memory_counters) & SPI_BUSY_COUNTERS
    elif blocks == ["3", "6"]:
        expected_counters = workgroup_counters | memory_counters
        assert (workgroup_counters - memory_counters) & SPI_BUSY_COUNTERS
        assert (memory_counters - workgroup_counters) & MEMORY_CHART_COUNTERS
    else:
        expected_counters = tables[601]
        assert expected_counters & SPI_BUSY_COUNTERS
        assert workgroup_counters - expected_counters

    args = Namespace(
        config_dir=str(analysis_configs_dir),
        filter_blocks=blocks,
        set_selected=None,
        roof_only=False,
        membw_analysis=False,
    )
    machine_specs = SimpleNamespace(
        gpu_arch=gpu_arch,
        gpu_series=mi_gpu_specs.get_gpu_series(gpu_arch),
        l2_banks=1,
    )
    soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch(gpu_arch)
    requested_counters, selected_blocks = soc.detect_counters()
    assert requested_counters == expected_counters
    assert selected_blocks == blocks
