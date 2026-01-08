# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
General configuration file tests.
"""

import pytest
from pathlib import Path

import shutil

from rocprofsys import RocprofsysConfig


# ============================================================================
# Helper functions
# ============================================================================


def write_invalid_config_file(output_dir: Path) -> Path:
    """Write an invalid configuration file."""
    config_path = output_dir / "invalid.cfg"
    config_path.write_text(
        """\
ROCPROFSYS_CONFIG_FILE =
FOOBAR = ON
"""
    )
    return config_path


# =============================================================================
# Config fixtures
# =============================================================================


@pytest.fixture
def config_target(rocprof_config: RocprofsysConfig) -> str:
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


class TestConfig:
    """Tests for configuration file tests."""

    def test_invalid_config(
        self,
        test_output_dir: Path,
        config_target: str,
        run_test,
        assert_regex,
    ):
        """Test that invalid config file causes failure."""
        # Write invalid configuration file to test output directory
        config_file = write_invalid_config_file(test_output_dir)

        env = {"ROCPROFSYS_CONFIG_FILE": str(config_file)}

        result = run_test(
            "runtime_instrument",
            target=config_target,
            env=env,
            timeout=300,  # In xdist, it can take much longer
            fail_on_pass=True,  # Expected to fail
        )

        assert_regex(
            result,
            pass_regex=[r"Unknown setting 'FOOBAR' \(value = 'ON'\)"],
            use_abort_fail_regex=False,
        )

    def test_missing_config(
        self,
        test_output_dir: Path,
        config_target: str,
        run_test,
        assert_regex,
    ):
        """Test that missing config file causes failure."""
        # Use a path to a config file that doesn't exist
        missing_config = test_output_dir / "missing.cfg"

        env = {"ROCPROFSYS_CONFIG_FILE": str(missing_config)}

        result = run_test(
            "runtime_instrument",
            target=config_target,
            env=env,
            timeout=120,
            fail_on_pass=True,  # Expected to fail
        )

        assert_regex(
            result,
            pass_regex=[r"Error reading configuration file"],
            use_abort_fail_regex=False,
        )
