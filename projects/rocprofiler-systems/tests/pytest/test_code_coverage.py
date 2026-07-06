# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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


@pytest.mark.class_name("code-coverage")
class TestCodeCoverage(RocprofsysTest):
    def get_binary_rewrite_args(self, type) -> list[str]:
        ret = ["-e", "-v", "2", "--min-instructions=4", "-E", "^std::"]
        if "base" in type:
            ret.extend(["--coverage", "function"])
        else:
            ret.extend(["--coverage", "basic_block"])
        if "hybrid" not in type:
            ret.extend(["-M", "coverage"])
        return ret

    def get_runtime_instrument_args(self, type) -> list[str]:
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

    @pytest.mark.preserve("coverage.json")
    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument"])
    @pytest.mark.parametrize(
        "type", ["base", "base_hybrid", "basic_blocks", "basic_blocks_hybrid"]
    )
    def test(self, mode, type, get_test_num_threads):
        run_args = ["10", str(get_test_num_threads), "1000"]
        result = self.run_test(
            mode,
            "code-coverage",
            run_args=run_args,
            binary_rewrite_args=self.get_binary_rewrite_args(type),
            runtime_instrument_args=self.get_runtime_instrument_args(type),
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=self.get_pass_regex(type),
        )

        if (type == "basic_blocks" and mode == "binary_rewrite") or (
            type == "basic_blocks_hybrid" and mode == "runtime_instrument"
        ):
            self.assert_file_exists(
                result.output_dir / "coverage.json",
                subtest_name="Coverage JSON file existence validation",
            )

    @pytest.mark.timeout(120)
    @pytest.mark.depends_on(
        "code-coverage-basic-blocks-binary-rewrite",
        "code-coverage-basic-blocks-hybrid-runtime-instrument",
    )
    @pytest.mark.python_versions
    def test_python(self, python_version, test_output_base):
        # Get the coverage paths from the previously ran tests
        brw_coverage_path = (
            test_output_base
            / "code-coverage-basic-blocks-binary-rewrite"
            / "coverage.json"
        )
        ri_coverage_path = (
            test_output_base
            / "code-coverage-basic-blocks-hybrid-runtime-instrument"
            / "coverage.json"
        )
        missing = []
        if not brw_coverage_path.exists():
            missing.append("code-coverage-basic-blocks-binary-rewrite/coverage.json")
        if not ri_coverage_path.exists():
            missing.append(
                "code-coverage-basic-blocks-hybrid-runtime-instrument/coverage.json"
            )
        if missing:
            pytest.skip(f"Missing output from dependency tests: {', '.join(missing)}")

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
        )
        self.assert_regex(result)
