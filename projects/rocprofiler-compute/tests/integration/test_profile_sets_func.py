# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for the profile --set option."""

from pathlib import Path

import common
import pytest

from tests.integration import common as integration_common
from tests.integration.common import config

# Parametrization is resolved at collection, before fixtures exist, so this is the
# one place the GPU has to be probed at import rather than through the soc fixture.
AVAILABLE_SETS = integration_common.get_available_sets_for_arch(
    integration_common.gpu_soc()[0]
)


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
        assert integration_common.get_num_pmc_file(workload_dir) == 1
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
