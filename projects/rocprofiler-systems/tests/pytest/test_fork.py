# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Fork tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.fork]

# =============================================================================
# Fork fixtures
# =============================================================================


@pytest.fixture
def fork_env() -> dict[str, str]:
    """Environment variables for fork tests."""
    return {
        "ROCPROFSYS_SAMPLING_FREQ": "250",
        "ROCPROFSYS_SAMPLING_REALTIME": "ON",
    }


# =============================================================================
# Fork tests
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    [
        "baseline",
        "sampling",
        "binary_rewrite",
        "sys_run",
        pytest.param("runtime_instrument", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("fork-example", id="example"),
        pytest.param("hipMallocConcurrencyMproc", marks=pytest.mark.gpu),
    ],
)
class TestFork(RocprofsysTest):
    REWRITE_ARGS = ["-e", "-v", "2", "--print-instrumented", "modules", "-i", "16"]
    RUNTIME_ARGS = ["-e", "-v", "1", "--label", "file", "-i", "16"]

    def test(self, mode, target, fork_env):
        if target == "hipMallocConcurrencyMproc":
            pass_regex = ["Validation PASSED|fork.. called on PID"]
        else:
            pass_regex = ["fork.. called on PID"]

        result = self.run_test(
            mode,
            target,
            env=fork_env,
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
            check_target_arch=True,
        )

        if mode != "baseline":
            self.assert_regex(result, pass_regex=pass_regex)
