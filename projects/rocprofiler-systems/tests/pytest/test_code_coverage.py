# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Code coverage tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.code_coverage]


# =============================================================================
# Code coverage tests
# =============================================================================


@pytest.mark.xdist_group(name="python_code_coverage")
class TestCodeCoverage(RocprofsysTest):
    def get_rewrite_args(self, type) -> list[str]:
        ret = ["-e", "-v", "2", "--min-instructions=4", "-E", "^std::"]
        if "base" in type:
            ret.extend(["--coverage", "function"])
        else:
            ret.extend(["--coverage", "basic_block"])
        if "hybrid" not in type:
            ret.extend(["-M", "coverage"])
        return ret

    def get_runtime_args(self, type) -> list[str]:
        ret = [
            "-e",
            "-v",
            "1",
            "--min-instructions=4",
            "-E",
            "^std::",
            "--label",
            "file",
            "line",
            "return",
            "args",
            "--module-restrict",
            "code.coverage",
        ]
        if "base" in type:
            ret.extend(["--coverage", "function"])
        elif "hybrid" in type:
            ret.extend(["--coverage", "basic_block"])
        if "hybrid" not in type:
            ret.extend(["-M", "coverage"])
        return ret

    def get_pass_regex(self, type) -> list[str]:
        if "base" in type:
            return ["code coverage :: 66.67%"]
        return ["function coverage :: 66.67%"]

    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument"])
    @pytest.mark.parametrize(
        "type", ["base", "base_hybrid", "basic_blocks", "basic_blocks_hybrid"]
    )
    def test(self, mode, type, get_test_num_threads, collect_output_path):
        run_args = ["10", str(get_test_num_threads), "1000"]
        result = self.run_test(
            mode,
            "code-coverage",
            run_args=run_args,
            rewrite_args=self.get_rewrite_args(type),
            runtime_args=self.get_runtime_args(type),
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=self.get_pass_regex(type),
        )

        # Store the output path for the test_python test below
        key = None
        if type == "basic_blocks" and mode == "binary_rewrite":
            key = "basic_blocks_coverage_brw"
        elif type == "basic_blocks" and mode == "runtime_instrument":
            key = "basic_blocks_hybrid_coverage_ri"
        if key:
            collect_output_path.store(key, result.output_dir)

    @pytest.mark.python_versions
    def test_python(self, python_version, collect_output_path):
        # Get the coverage paths from the previously ran tests
        brw_coverage_path = collect_output_path.get("basic_blocks_coverage_brw")
        ri_coverage_path = collect_output_path.get("basic_blocks_hybrid_coverage_ri")
        if not brw_coverage_path or not ri_coverage_path:
            pytest.skip("coverage paths not found")

        run_args = [
            "-i",
            brw_coverage_path / "coverage.json",
            ri_coverage_path / "coverage.json",
            "-o",
            self.test_output_dir / "coverage.json",
        ]
        result = self.run_test(
            "python",
            target="code-coverage.py",
            run_args=run_args,
            python_version=python_version,
            timeout=120,
        )
        self.assert_regex(result)
