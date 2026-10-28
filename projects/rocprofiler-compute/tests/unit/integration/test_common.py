# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CPU regressions for live-attach counter selection."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from config import analysis_configs_dir
from rocprof_compute_soc.soc_base import OmniSoC_Base
from tests.integration.common import LIVE_ATTACH_SET
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
