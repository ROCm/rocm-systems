# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Profile-mode set/--list-sets integration tests.

Extracted verbatim from tests/test_profile_general.py during the test-suite
refactor; no logic changes.
"""

import csv
import inspect
import os
import re
import socket
import subprocess
import sqlite3
import sys
import time
from pathlib import Path

import common
import numpy as np
import pandas as pd
import pytest
import yaml

from _profile_helpers import (
    AVAILABLE_SETS,
    CSVS,
    DEFAULT_ABS_DIFF,
    DEFAULT_REL_DIFF,
    GPU_ARCH,
    GPU_MODEL,
    MAX_REOCCURING_COUNT,
    METRIC_THRESHOLDS,
    MockMachineSpecs,
    MockProfiler,
    MockSoc,
    ROOF_ONLY_FILES,
    SLURM_RANK_VAR,
    SLURM_SIZE_VAR,
    are_deterministic_counters_equal,
    are_stochastic_counters_similar,
    arch_config,
    attach_detach_interval_msec_no_delay,
    attach_detach_interval_msec_with_delay,
    baseline_compare_metric,
    clear_rank_env,
    config,
    counter_compare,
    get_available_sets_for_arch,
    gpu_arch,
    is_rdna35_halo_soc,
    mock_generate_machine_specs,
    mock_load_soc_specs,
    num_devices,
    num_kernels,
    skip_unsupported_roofline_soc,
    soc,
    validate,
)

@pytest.mark.sets_func
class TestSetsIntegration:
    # Ensure single pass for auto-discovered sets from YAML for the current GPU arch.
    @pytest.mark.parametrize("set_name", AVAILABLE_SETS, ids=lambda s: s)
    def test_set_profiling(
        self, binary_handler_profile_rocprof_compute, set_name, request
    ):
        """Each set_option runs successfully and produces a single PMC file."""
        options = ["--set", set_name]
        workload_dir = common.get_output_dir(param_id=set_name)

        returncode = binary_handler_profile_rocprof_compute(
            config, workload_dir, options, check_success=True, roof=False
        )

        assert returncode == 0
        assert common.get_num_pmc_file(workload_dir) == 1
        common.clean_output_dir(config["cleanup"], workload_dir)

    @pytest.mark.parametrize(
        "set_name",
        [
            pytest.param("nonexistent_set", id="nonexistent"),
            pytest.param("x" * 1024, id="very_long_name"),
            pytest.param("mem_thruput; rm -rf /", id="shell_metachar"),
        ],
    )
    def test_invalid_set_rejected(
        self, binary_handler_profile_rocprof_compute, set_name, request
    ):
        """Invalid or adversarial set names are rejected with exit code 1."""
        options = ["--set", set_name]
        workload_dir = common.get_output_dir(
            param_id=f"invalid_{request.node.callspec.id}"
        )

        returncode = binary_handler_profile_rocprof_compute(
            config, workload_dir, options, check_success=False, roof=False
        )

        assert returncode == 1
        common.clean_output_dir(config["cleanup"], workload_dir)

    def test_set_and_block_mutual_exclusion(
        self, binary_handler_profile_rocprof_compute
    ):
        options = ["--set", "compute_thruput_util", "--block", "12"]
        workload_dir = common.get_output_dir()

        returncode = binary_handler_profile_rocprof_compute(
            config, workload_dir, options, check_success=False, roof=False
        )

        assert returncode == 1
        common.clean_output_dir(config["cleanup"], workload_dir)

    def test_list_sets_functionality(self, binary_handler_profile_rocprof_compute):
        options = ["--list-sets"]
        workload_dir = common.get_output_dir()

        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=False,
            roof=False,
        )
        # workload dir should not exist
        assert not Path(workload_dir).exists()
        common.clean_output_dir(config["cleanup"], workload_dir)
