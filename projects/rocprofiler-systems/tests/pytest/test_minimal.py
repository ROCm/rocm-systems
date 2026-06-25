# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
These tests exercise minimal examples to verify targeted behavior of the profiler.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [pytest.mark.minimal]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rocpd_env() -> dict[str, str]:
    return {}


@pytest.fixture
def recursion_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "minimal"
    return [rules_dir / "recursion-rules.json"]


# =============================================================================
# Tests
# =============================================================================


class TestMinimal(RocprofsysTest):
    """Test minimal examples."""

    RECURSION_DEPTH = 100

    @pytest.mark.rocpd("rocpd_env")
    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument"])
    def test_recursion(self, mode, rocpd_env, recursion_rules):
        """
        Ensure that recursion traces are properly present in both
        perfetto and rocpd.
        """
        env = rocpd_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "minimal-recursion",
            env=env,
            binary_rewrite_args=["--min-instructions", "0"],
            runtime_instrument_args=["--min-instructions", "0"],
            run_args=[str(self.RECURSION_DEPTH)],
        )
        self.assert_regex(result)

        # Look for the last line | recurse | 1 | <depth + 1> |
        deepest_depth = self.RECURSION_DEPTH + 1
        self.assert_perfetto(
            result,
            subtest_name="Perfetto Recursion Validation",
            categories=["host"],
            pass_regex=[rf"\|_recurse\s+\|\s+1\s+\|\s+{deepest_depth}\s+\|"],
        )

        self.assert_rocpd(
            result,
            subtest_name="ROCpd Recursion Validation",
            rules_files=recursion_rules,
        )

        # Every 'recurse' frame carries a source_object trace-arg
        # with the value 'minimal-recursion'
        self.assert_perfetto(
            result,
            subtest_name="Perfetto Debug Validation",
            key_names=["source_object"],
            key_counts=[deepest_depth],
            pass_regex=[
                r"key\s*::\s*debug\.source_object",
                r"string_value\s*::\s*minimal-recursion",
            ],
        )
