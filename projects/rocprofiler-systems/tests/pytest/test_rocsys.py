# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for the unified ``rocsys`` CLI entry point."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import RocprofsysTest
from rocprofsys.commands import get_ls_command
from rocprofsys.config import RocprofsysConfig

pytestmark = [pytest.mark.rocsys, pytest.mark.rocprof_binary]


def _sibling_exists(config: RocprofsysConfig, name: str) -> bool:
    return (config.rocprofsys_bin_dir / name).is_file()


# Phase 2 option names. They must not be flags of any legacy tool.
_PHASE2_OPTIONS = ("--dry-run", "--traces")

# Hand-parsed from the rocprof-sys-attach usage text. `--pid` is accepted but
# not printed. Keep in sync with k_attach_long_options in test_help.cpp.
_ATTACH_LONG_OPTIONS = frozenset(
    {"--detach-after", "--format", "--help", "--output"}
)

# Hand-parsed from source/python/rocprofsys/__main__.py. Keep in sync with
# k_python_long_options in source/bin/rocsys/tests/test_help.cpp.
_PYTHON_LONG_OPTIONS = frozenset(
    {
        "--annotate-trace",
        "--builtin",
        "--config",
        "--full-filepath",
        "--function-exclude",
        "--function-include",
        "--function-restrict",
        "--help",
        "--label",
        "--module-exclude",
        "--module-include",
        "--module-restrict",
        "--setup",
        "--trace-c",
        "--verbosity",
    }
)

_LONG_OPTION = re.compile(r"(?<![\w-])(--[A-Za-z][\w-]*)")

# Live --help sources. run and sample use timemory's full help.
_LIVE_HELP = (
    ("rocprof-sys-run", ["--help=all"]),
    ("rocprof-sys-sample", ["--help=all"]),
    ("rocprof-sys-instrument", ["--help"]),
    ("rocprof-sys-causal", ["--help"]),
    ("rocprof-sys-avail", ["--help"]),
)


def _long_options(text: str) -> set[str]:
    return set(_LONG_OPTION.findall(text))


FORWARD_HELP_CASES = [
    pytest.param(
        "profile",
        "rocprof-sys-run",
        [r"rocsys options for profile:", r"--dry-run", r"QUICK START", r"Usage:"],
        False,
        id="profile",
    ),
    pytest.param(
        "instrument",
        "rocprof-sys-instrument",
        [r"rocsys options for instrument:", r"--dry-run", r"rocprof-sys-instrument"],
        False,
        id="instrument",
    ),
    pytest.param(
        "rewrite",
        "rocprof-sys-instrument",
        [
            r"rocsys options for rewrite:",
            r"--dry-run",
            r"rocsys -- \./app\.inst",
            r"rocprof-sys-instrument",
        ],
        False,
        id="rewrite",
    ),
    pytest.param(
        "causal",
        "rocprof-sys-causal",
        [r"rocsys options for causal:", r"--dry-run", r"Usage|usage|causal"],
        False,
        id="causal",
    ),
    pytest.param(
        "avail",
        "rocprof-sys-avail",
        [r"rocsys options for avail:", r"--dry-run", r"rocprof-sys-avail"],
        False,
        id="avail",
    ),
    pytest.param(
        "python",
        "rocprof-sys-python",
        [r"rocsys options for python:", r"--dry-run", r"usage|Usage"],
        True,
        id="python",
    ),
    pytest.param(
        "attach",
        "rocprof-sys-attach",
        [r"rocsys options for attach:", r"--dry-run", r"Usage|usage|pid|attach"],
        True,
        id="attach",
    ),
]

DRY_RUN_CASES = [
    pytest.param(
        ["--dry-run", "--", "/no/such/rocsys-app"],
        "profile",
        r"/no/such/rocsys-app",
        id="implicit-profile",
    ),
    pytest.param(
        ["profile", "--dry-run", "--verbose", "--", "/no/such/rocsys-app"],
        "profile",
        r"--verbose",
        id="profile-keeps-legacy-flags",
    ),
    pytest.param(
        ["instrument", "--dry-run", "--", "/no/such/rocsys-app"],
        "instrument",
        r"rocprof-sys-instrument -- /no/such/rocsys-app",
        id="instrument",
    ),
    pytest.param(
        ["rewrite", "--dry-run", "--", "/no/such/rocsys-app"],
        "rewrite",
        r"rocprof-sys-instrument -o -- /no/such/rocsys-app",
        id="rewrite",
    ),
    pytest.param(
        ["causal", "--dry-run", "--", "/no/such/rocsys-app"],
        "causal",
        r"rocprof-sys-causal -- /no/such/rocsys-app",
        id="causal",
    ),
    pytest.param(
        ["avail", "--dry-run"],
        "avail",
        r"rocprof-sys-avail",
        id="avail",
    ),
    pytest.param(
        ["python", "--dry-run", "--", "script.py"],
        "python",
        r"rocprof-sys-python -- script.py",
        id="python",
    ),
    pytest.param(
        ["attach", "--dry-run", "-p", "1"],
        "attach",
        r"rocprof-sys-attach -p 1",
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
                r"rocsys \[subcommand\] \[options\] -- <app> \[app-args\]",
                r"profile",
                r"instrument",
                r"rewrite",
                r"causal",
                r"avail",
                r"python",
                r"attach",
                r"rocsys -- \./app",
                r"rocsys profile -- \./app",
                r"Common options:",
                r"--dry-run",
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
        self.assert_regex(
            result,
            pass_regex=[
                (
                    r"rocsys v\d+\.\d+\.\d+ \(rev: [^,]+(?:, [^,]+){0,2}, "
                    r"compiler: .+, rocm: v.+\)"
                ),
            ],
        )

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
                r"error: unknown subcommand",
                r"Usage: rocsys \[subcommand\] \[options\] -- <app> \[app-args\]",
                r"hint: run 'rocsys --help'",
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
            pass_regex=[
                r"error: missing application argument",
                r"Usage: rocsys \[subcommand\] \[options\] -- <app> \[app-args\]",
                r"hint:",
            ],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    @pytest.mark.parametrize("verb", ["sample", "trace"])
    def test_sample_and_trace_unknown(self, verb: str) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[verb, "--", "/bin/ls"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                rf"error: unknown subcommand '{verb}'",
                r"Usage: rocsys \[subcommand\] \[options\] -- <app> \[app-args\]",
                r"hint:",
            ],
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

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("run_args, verb, argv_regex", DRY_RUN_CASES)
    def test_dry_run_prints_the_plan_and_does_not_launch(
        self, run_args: list[str], verb: str, argv_regex: str
    ) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=run_args,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                rf"verb: {verb}",
                argv_regex,
                r"rocsys overrides; legacy flags are applied later by the tool",
            ],
            fail_regex=[r"--dry-run"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(30)
    def test_dry_run_with_help_still_forwards_help(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["profile", "--dry-run", "--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"rocsys options for profile:", r"QUICK START"],
            fail_regex=[r"rocsys overrides"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    def test_dry_run_rejects_a_value(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--dry-run=yes", "--", "/bin/true"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--dry-run does not take a value"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("binary, args", _LIVE_HELP)
    def test_legacy_help_does_not_use_phase2_option_names(
        self, binary: str, args: list[str]
    ) -> None:
        result = self.run_test(
            "baseline",
            target=binary,
            run_args=args,
            fail_on_not_found=True,
        )
        found = _long_options(result.test_output)
        overlap = found.intersection(_PHASE2_OPTIONS)
        assert not overlap, f"{binary} help contains rocsys options {sorted(overlap)}"

    @pytest.mark.timeout(30)
    def test_attach_help_matches_the_checked_in_flags(
        self, rocprof_config: RocprofsysConfig
    ) -> None:
        if not _sibling_exists(rocprof_config, "rocprof-sys-attach"):
            pytest.skip("rocprof-sys-attach not built")
        result = self.run_test(
            "baseline",
            target="rocprof-sys-attach",
            run_args=["--help"],
            fail_on_not_found=True,
        )
        found = _long_options(result.test_output)
        assert found == _ATTACH_LONG_OPTIONS, sorted(found.symmetric_difference(
            _ATTACH_LONG_OPTIONS
        ))
        assert not found.intersection(_PHASE2_OPTIONS)

    @pytest.mark.timeout(30)
    def test_python_help_matches_the_checked_in_flags(
        self, rocprof_config: RocprofsysConfig
    ) -> None:
        if not _sibling_exists(rocprof_config, "rocprof-sys-python"):
            pytest.skip("rocprof-sys-python not built")
        result = self.run_test(
            "baseline",
            target="rocprof-sys-python",
            run_args=["--help"],
            fail_on_not_found=True,
        )
        found = _long_options(result.test_output)
        assert found == _PYTHON_LONG_OPTIONS, sorted(found.symmetric_difference(
            _PYTHON_LONG_OPTIONS
        ))
        assert not found.intersection(_PHASE2_OPTIONS)

    @pytest.mark.timeout(120)
    @pytest.mark.sys_run
    def test_profile_ls_produces_report(self, rocprof_config: RocprofsysConfig) -> None:
        dl_lib = rocprof_config.rocprofsys_lib_dir / "librocprof-sys-dl.so"
        if not dl_lib.is_file():
            pytest.skip("librocprof-sys-dl.so not built")
        ls_name, ls_args = get_ls_command()
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
            run_args=["profile", "--", ls_name, *ls_args],
            fail_on_not_found=True,
        )
        assert result.success, result.test_output
        report_files = result.rocpd_files + result.timemory_files
        metadata = list(Path(result.output_dir).glob("**/metadata*.json"))
        assert report_files or metadata, (
            f"expected a trace report under {result.output_dir}, "
            f"contents={list(result.output_dir.rglob('*'))}"
        )

    @pytest.mark.timeout(120)
    @pytest.mark.sys_run
    def test_implicit_default_ls_produces_report(
        self, rocprof_config: RocprofsysConfig
    ) -> None:
        dl_lib = rocprof_config.rocprofsys_lib_dir / "librocprof-sys-dl.so"
        if not dl_lib.is_file():
            pytest.skip("librocprof-sys-dl.so not built")
        ls_name, ls_args = get_ls_command()
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
            run_args=["--", ls_name, *ls_args],
            fail_on_not_found=True,
        )
        assert result.success, result.test_output
        report_files = result.rocpd_files + result.timemory_files
        metadata = list(Path(result.output_dir).glob("**/metadata*.json"))
        assert report_files or metadata, (
            f"expected a trace report under {result.output_dir}, "
            f"contents={list(result.output_dir.rglob('*'))}"
        )
