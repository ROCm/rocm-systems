# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for the unified ``rocsys`` CLI entry point."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from conftest import RocprofsysTest
from rocprofsys.config import RocprofsysConfig

pytestmark = [pytest.mark.rocsys, pytest.mark.rocprof_binary]


def _ls_command() -> list[str]:
    """Return argv for ``ls``, handling the Red Hat coreutils wrapper."""
    if os.path.exists("/usr/bin/coreutils"):
        return ["coreutils", "--coreutils-prog=ls"]
    ls_cmd = shutil.which("ls")
    if not ls_cmd:
        pytest.skip("ls command not found")
    return [ls_cmd]


def _sibling_exists(config: RocprofsysConfig, name: str) -> bool:
    return (config.rocprofsys_bin_dir / name).is_file()


FORWARD_HELP_CASES = [
    pytest.param(
        "profile",
        "rocprof-sys-sample",
        [r"QUICK START", r"Usage:"],
        False,
        id="profile",
    ),
    pytest.param(
        "trace",
        "rocprof-sys-run",
        [r"QUICK START", r"Usage:"],
        False,
        id="trace",
    ),
    pytest.param(
        "instrument",
        "rocprof-sys-instrument",
        [r"rocprof-sys-instrument"],
        False,
        id="instrument",
    ),
    pytest.param(
        "causal",
        "rocprof-sys-causal",
        [r"Usage|usage|causal"],
        False,
        id="causal",
    ),
    pytest.param(
        "avail",
        "rocprof-sys-avail",
        [r"rocprof-sys-avail"],
        False,
        id="avail",
    ),
    pytest.param(
        "python",
        "rocprof-sys-python",
        [r"usage|Usage"],
        True,
        id="python",
    ),
    pytest.param(
        "attach",
        "rocprof-sys-attach",
        [r"Usage|usage|pid|attach"],
        True,
        id="attach",
    ),
]


@pytest.mark.class_name("rocsys")
class TestRocsys(RocprofsysTest):
    """Tests for the ``rocsys`` dispatcher binary."""

    target = "rocsys"

    @pytest.mark.timeout(30)
    def test_top_level_help(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"Usage:",
                r"profile",
                r"trace",
                r"instrument",
                r"causal",
                r"avail",
                r"python",
                r"attach",
                r"rocsys -- \./app",
            ],
        )

    @pytest.mark.timeout(15)
    def test_version(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--version"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"rocsys version "])

    @pytest.mark.timeout(15)
    def test_unknown_subcommand(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["not-a-subcommand"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"unknown subcommand",
                r"rocsys --help",
            ],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    def test_profile_missing_app(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["profile"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"missing application argument"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(45)
    @pytest.mark.parametrize(
        "subcommand, sibling, pass_regex, optional", FORWARD_HELP_CASES
    )
    def test_subcommand_help_forwards(
        self,
        rocprof_config: RocprofsysConfig,
        subcommand: str,
        sibling: str,
        pass_regex: list[str],
        optional: bool,
    ) -> None:
        if not _sibling_exists(rocprof_config, sibling):
            if optional:
                pytest.skip(f"{sibling} not built")
            pytest.fail(f"required sibling binary {sibling} not found")

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[subcommand, "--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(120)
    @pytest.mark.sampling
    def test_profile_ls_produces_report(self, rocprof_config: RocprofsysConfig) -> None:
        dl_lib = rocprof_config.rocprofsys_lib_dir / "librocprof-sys-dl.so"
        if not dl_lib.is_file():
            pytest.skip("librocprof-sys-dl.so not built")
        ls_cmd = _ls_command()
        result = self.run_test(
            "baseline",
            target=self.target,
            no_base_env=True,
            env={
                "ROCPROFSYS_SAMPLING_DELAY": "0",
                "ROCPROFSYS_USE_AMD_SMI": "OFF",
                "ROCPROFSYS_TIME_OUTPUT": "OFF",
                "ROCPROFSYS_FILE_OUTPUT": "ON",
            },
            run_args=["profile", "--", *ls_cmd],
            fail_on_not_found=True,
        )
        assert result.success, result.test_output
        report_files = result.rocpd_files + result.timemory_files
        metadata = list(Path(result.output_dir).glob("**/metadata*.json"))
        assert report_files or metadata, (
            f"expected a sampling report under {result.output_dir}, "
            f"contents={list(result.output_dir.rglob('*'))}"
        )

    @pytest.mark.timeout(120)
    @pytest.mark.sampling
    def test_implicit_profile_ls_produces_report(
        self, rocprof_config: RocprofsysConfig
    ) -> None:
        dl_lib = rocprof_config.rocprofsys_lib_dir / "librocprof-sys-dl.so"
        if not dl_lib.is_file():
            pytest.skip("librocprof-sys-dl.so not built")
        ls_cmd = _ls_command()
        result = self.run_test(
            "baseline",
            target=self.target,
            no_base_env=True,
            env={
                "ROCPROFSYS_SAMPLING_DELAY": "0",
                "ROCPROFSYS_USE_AMD_SMI": "OFF",
                "ROCPROFSYS_TIME_OUTPUT": "OFF",
                "ROCPROFSYS_FILE_OUTPUT": "ON",
            },
            run_args=["--", *ls_cmd],
            fail_on_not_found=True,
        )
        assert result.success, result.test_output
        report_files = result.rocpd_files + result.timemory_files
        metadata = list(Path(result.output_dir).glob("**/metadata*.json"))
        assert report_files or metadata, (
            f"expected a sampling report under {result.output_dir}, "
            f"contents={list(result.output_dir.rglob('*'))}"
        )
