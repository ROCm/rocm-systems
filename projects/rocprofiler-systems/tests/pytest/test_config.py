# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
General configuration file tests.
"""

from __future__ import annotations
import pytest
from pathlib import Path
import shutil
from conftest import RocprofsysTest

pytestmark = [pytest.mark.rocprof_config, pytest.mark.ci_enable]


# =============================================================================
# Config fixtures
# =============================================================================


@pytest.fixture
def config_target(rocprof_config) -> str:
    """Get the target executable for config tests."""
    target_name = "parallel-overhead"
    try:
        rocprof_config.get_target_executable(target_name)
    except FileNotFoundError:
        # Fall back to system ls command
        target_name = shutil.which("ls") or "ls"
    return target_name


# =============================================================================
# Configuration file tests
# =============================================================================


class TestConfig(RocprofsysTest):
    """Tests for configuration file tests."""

    def test_invalid(self, config_target, create_config_file):
        """Test that invalid config file causes failure."""
        # Write invalid configuration file to test output directory
        config_env = {
            "ROCPROFSYS_CONFIG_FILE": "",
            "FOOBAR": "ON",
        }
        config_file = create_config_file(config_env, "invalid.cfg", skip_filter=True)

        env = {"ROCPROFSYS_CONFIG_FILE": str(config_file)}

        result = self.run_test(
            "runtime_instrument",
            target=config_target,
            env=env,
            timeout=400,  # In xdist, it can take much longer
            fail_on_pass=True,  # Expected to fail
        )

        self.assert_regex(
            result,
            pass_regex=[r"Unknown setting 'FOOBAR' \(value = 'ON'\)"],
            use_abort_fail_regex=False,
        )

    def test_missing(self, test_output_dir: Path, config_target: str):
        """Test that missing config file causes failure."""
        # Use a path to a config file that doesn't exist
        missing_config = test_output_dir / "missing.cfg"

        env = {"ROCPROFSYS_CONFIG_FILE": str(missing_config)}

        result = self.run_test(
            "runtime_instrument",
            target=config_target,
            env=env,
            timeout=120,
            fail_on_pass=True,  # Expected to fail
        )

        self.assert_regex(
            result,
            pass_regex=[r"Error reading configuration file"],
            use_abort_fail_regex=False,
        )
