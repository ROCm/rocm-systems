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


QUIET_PROFILE_ENV = {
    "ROCPROFSYS_SAMPLING_DELAY": "0",
    "ROCPROFSYS_USE_AMD_SMI": "OFF",
    "ROCPROFSYS_TIME_OUTPUT": "OFF",
    "ROCPROFSYS_FILE_OUTPUT": "ON",
}


ROCSYS_HELP_CASES = [
    pytest.param("profile", "rocprof-sys-run", id="profile"),
    pytest.param("instrument", "rocprof-sys-instrument", id="instrument"),
    pytest.param("python", "rocprof-sys-python", id="python"),
    pytest.param("attach", "rocprof-sys-attach", id="attach"),
]

ROCSYS_OPTION_REGEX = [
    r"(?m)^  --traces +APIs and runtimes to trace",
    r"(?m)^  --output-path +Directory for output files",
    r"(?m)^  --log-level +Log level",
]

TOOL_HELP_CASES = [
    pytest.param("rewrite", "rocprof-sys-instrument", id="rewrite"),
    pytest.param("avail", "rocprof-sys-avail", id="avail"),
    pytest.param("causal", "rocprof-sys-causal", id="causal"),
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
                r"rocsys options:",
                *ROCSYS_OPTION_REGEX,
                r"'rocsys <option> --help' for one rocsys option in full",
                r"rocsys -- \./app",
                r"rocsys profile -- \./app",
                r"rocsys --traces hip mpi -- \./app",
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

    @pytest.mark.timeout(15)
    def test_traces_rejects_an_unknown_name(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--traces", "bogus", "--", "/bin/true"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"error: invalid value 'bogus' for parameter 'traces'",
                r"hint: run 'rocsys --help' for the rocsys options\.",
            ],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    def test_equals_form_takes_names_separated_by_commas(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--traces=hip", "mpi", "--", "echo", "the application ran"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"error: unexpected argument 'mpi'; did you mean '--traces=hip,mpi'\?",
                r"hint: run 'rocsys --help' for the rocsys options\.",
            ],
            fail_regex=[r"the application ran"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    def test_an_application_needs_the_separator(self, tmp_path: Path) -> None:
        script = tmp_path / "script.py"
        script.write_text('print("the script ran")\n')
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["python", str(script), "--log-level", "info"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                (
                    r"error: put '--' before the application: rocsys python \[options\] "
                    rf"-- {re.escape(str(script))} \[app-args\]"
                ),
                r"hint: run 'rocsys --help'",
            ],
            fail_regex=[r"the script ran"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    @pytest.mark.parametrize(
        "run_args, command",
        [
            pytest.param(
                ["python", "-v", "./script.py", "--log-level", "info"],
                "rocsys python",
                id="python",
            ),
            pytest.param(
                ["rewrite", "-o", "out.inst", "./transpose"],
                "rocsys rewrite",
                id="rewrite",
            ),
            pytest.param(["--traces", "hip", "./app"], "rocsys", id="profile"),
        ],
    )
    def test_a_command_without_the_separator_is_refused(
        self, run_args: list[str], command: str
    ) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=run_args,
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                (
                    rf"error: put '--' before the application: {command} \[options\] "
                    r"-- <app> \[app-args\]"
                ),
                r"hint: run 'rocsys --help'",
            ],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    def test_attach_leaves_a_missing_pid_to_the_tool(
        self, rocprof_config: RocprofsysConfig
    ) -> None:
        if rocprof_config.rocprofsys_attach is None:
            pytest.skip("rocprof-sys-attach not built")
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["attach", "1234"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"Unknown option '1234'",
                r"Usage: rocprof-sys-attach -p <pid>",
            ],
            fail_regex=[r"put '--' before the application"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    @pytest.mark.parametrize(
        "run_args",
        [
            pytest.param(["attach", "--log-level", "info", "--", "-p", "1"], id="attach"),
            pytest.param(["avail", "--", "gpu"], id="avail"),
        ],
    )
    def test_a_subcommand_without_an_application_refuses_the_separator(
        self, run_args: list[str]
    ) -> None:
        subcommand = run_args[0]
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=run_args,
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                rf"error: rocsys {subcommand} runs no application, so it takes no '--'",
                rf"hint: run 'rocsys {subcommand} --help' for its options\.",
            ],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(45)
    @pytest.mark.parametrize(
        "run_args, option",
        [
            pytest.param(["avail", "--log-level", "debug"], "log-level", id="avail"),
            pytest.param(
                ["causal", "--traces", "hip", "--", "/bin/true"], "traces", id="causal"
            ),
            pytest.param(
                ["rewrite", "--traces", "hip", "--", "/bin/true"], "traces", id="rewrite"
            ),
        ],
    )
    def test_rocsys_options_reach_a_tool_that_takes_none(
        self, run_args: list[str], option: str
    ) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=run_args,
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[rf"Unrecognized command line option '{option}'"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(120)
    @pytest.mark.sys_run
    def test_options_reach_the_tool_environment(
        self, rocprof_config: RocprofsysConfig, tmp_path: Path
    ) -> None:
        if not (rocprof_config.rocprofsys_lib_dir / "librocprof-sys-dl.so").is_file():
            pytest.skip("librocprof-sys-dl.so not built")
        output = tmp_path / "rocsys-output"
        result = self.run_test(
            "baseline",
            target=self.target,
            no_base_env=True,
            env=QUIET_PROFILE_ENV,
            run_args=[
                "--traces",
                "hip,",
                "mpi",
                "--output-path",
                str(output),
                "--log-level",
                "info",
                "--",
                "printenv",
                "ROCPROFSYS_ROCM_DOMAINS",
                "ROCPROFSYS_USE_MPIP",
                "ROCPROFSYS_OUTPUT_PATH",
                "ROCPROFSYS_LOG_LEVEL",
            ],
            fail_on_not_found=True,
        )
        assert result.success, result.test_output
        self.assert_regex(
            result,
            pass_regex=[
                r"(?m)^hip_api$",
                r"(?m)^true$",
                rf"(?m)^{re.escape(str(output))}$",
                r"(?m)^info$",
            ],
        )

    @pytest.mark.timeout(120)
    @pytest.mark.sys_run
    def test_tool_options_reach_the_tool(self, rocprof_config: RocprofsysConfig) -> None:
        if not (rocprof_config.rocprofsys_lib_dir / "librocprof-sys-dl.so").is_file():
            pytest.skip("librocprof-sys-dl.so not built")
        result = self.run_test(
            "baseline",
            target=self.target,
            no_base_env=True,
            env=QUIET_PROFILE_ENV,
            run_args=["-T", "--traces", "hip", "--", "printenv", "ROCPROFSYS_TRACE"],
            fail_on_not_found=True,
        )
        assert result.success, result.test_output
        self.assert_regex(result, pass_regex=[r"(?m)^true$"])

    @pytest.mark.timeout(15)
    @pytest.mark.parametrize("subcommand, tool", ROCSYS_HELP_CASES)
    def test_subcommand_help_lists_rocsys_options(
        self, subcommand: str, tool: str
    ) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[subcommand, "--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                rf"(?m)^  rocsys {subcommand} \[options\]$",
                *ROCSYS_OPTION_REGEX,
                rf"Run '{re.escape(tool)} --help' to list them\.",
            ],
        )

    @pytest.mark.timeout(45)
    def test_help_topic_reaches_the_tool(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["profile", "--help=tracing"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"rocprof-sys-run --help=tracing", r"\[TRACING OPTIONS\]"],
        )

    @pytest.mark.timeout(15)
    def test_help_topic_prints_a_rocsys_category_in_full(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--help=tracing"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"(?m)^tracing$",
                r"(?m)^  --traces <value>$",
                r"choices: rccl mpi openmp ucx openshmem hip-api hip hsa-api hsa",
            ],
            fail_regex=[r"--output-path", r"--log-level"],
        )

    @pytest.mark.timeout(15)
    def test_option_help_prints_one_option_in_full(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["log-level", "--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"(?m)^  --log-level <value>$",
                r"choices: trace debug info warn warning error err critical off",
                r"env: ROCPROFSYS_LOG_LEVEL",
            ],
            fail_regex=[r"--traces", r"--output-path"],
        )

    @pytest.mark.timeout(15)
    def test_unknown_help_topic_fails(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--help=bogus"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"error: unknown help topic 'bogus'",
                r"hint: run 'rocsys --help' for the categories\.",
            ],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(15)
    @pytest.mark.parametrize("subcommand, tool", TOOL_HELP_CASES)
    def test_subcommand_help_names_only_the_tool(
        self, subcommand: str, tool: str
    ) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[subcommand, "--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                rf"(?m)^  rocsys {subcommand} \[options\]$",
                rf"Run '{re.escape(tool)} --help' to list them\.",
            ],
            fail_regex=[r"--traces", r"--output-path", r"--log-level"],
        )

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
            env=QUIET_PROFILE_ENV,
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
            env=QUIET_PROFILE_ENV,
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
