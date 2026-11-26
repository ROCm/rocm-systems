"""Tests for the FAILED report a harness sends when its setup fails.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import logging
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import rccl_ci_utils
import test_jax_collective
import test_madengine
import test_pytorch_c10d
from rccl_ci_utils import SetupFailureReport

log = logging.getLogger(__name__)

HarnessMain = Callable[[SetupFailureReport], None]

RUN_ENV = {
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "ROCm/rocm-systems",
    "GITHUB_RUN_ID": "123",
}


def script_path(module: types.ModuleType) -> str:
    """Return the file an imported harness module was loaded from."""
    assert module.__file__ is not None
    return module.__file__


class ReportCapture(unittest.TestCase):
    """Patch the report channels and record what each one is sent."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.summary_file = self.tmp / "step_summary.md"
        env = dict(RUN_ENV, GITHUB_STEP_SUMMARY=str(self.summary_file))
        env_patcher = mock.patch.dict(os.environ, env)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        self.email = self.patch_channel("send_email_report")
        self.webhook = self.patch_channel("send_teams_webhook")

    def patch_channel(self, name: str) -> mock.Mock:
        patcher = mock.patch.object(rccl_ci_utils, name)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def summary(self) -> str:
        return self.summary_file.read_text() if self.summary_file.exists() else ""

    def assert_reported(self, subject: str, step: str) -> None:
        summary = self.summary()
        self.assertIn(f"{subject}: FAILED before any tests ran", summary)
        self.assertIn(f"Failed step: {step}", summary)
        self.assertIn("Run: https://github.com/ROCm/rocm-systems/actions/runs/123", summary)
        for channel, target in ((self.email, "dev@example.com"),
                                (self.webhook, "https://webhook.example")):
            channel.assert_called_once()
            report, to, status = channel.call_args.args
            self.assertEqual((to, status), (target, "FAILED"))
            self.assertEqual(channel.call_args.kwargs, {"subject_prefix": subject})
            self.assertIn(f"Failed step: {step}", report)

    def assert_not_reported(self) -> None:
        self.assertEqual(self.summary(), "")
        self.email.assert_not_called()
        self.webhook.assert_not_called()


class SetupFailureReportTest(ReportCapture):
    def run_main(self, main: HarnessMain, **notify: str) -> None:
        report = SetupFailureReport("RCCL Test")
        report.notify(notify.get("email", "dev@example.com"),
                      notify.get("webhook", "https://webhook.example"))
        report.run(lambda: main(report))

    def test_sys_exit_in_a_step_reports_step_and_logged_errors(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Discover RCCL library")
            log.error("librccl.so not found in /build")
            sys.exit(1)

        with self.assertRaises(SystemExit) as raised:
            self.run_main(main)
        self.assertEqual(raised.exception.code, 1)
        self.assert_reported("RCCL Test", "Discover RCCL library")
        self.assertIn("Cause:       exited with status 1", self.summary())
        self.assertIn("  librccl.so not found in /build", self.summary())

    def test_exception_in_a_step_reports_it_and_is_reraised(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Install madengine")
            subprocess.run(["false"], check=True)

        with self.assertRaises(subprocess.CalledProcessError):
            self.run_main(main)
        self.assert_reported("RCCL Test", "Install madengine")
        self.assertIn("Cause:       CalledProcessError: Command '['false']'", self.summary())

    def test_names_the_last_step_started(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Discover RCCL library")
            report.step("Clone test sources")
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.assert_reported("RCCL Test", "Clone test sources")

    def test_failure_after_tests_started_is_left_to_the_harness(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Clone test sources")
            report.tests_started()
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.assert_not_reported()

    def test_success_and_early_return_send_nothing(self) -> None:
        def exit_zero(report: SetupFailureReport) -> None:
            report.step("Discover RCCL library")
            sys.exit(0)

        def early_return(report: SetupFailureReport) -> None:  # --discover-only
            report.step("Discover RCCL library")

        with self.assertRaises(SystemExit):
            self.run_main(exit_zero)
        self.run_main(early_return)
        self.assert_not_reported()

    def test_failure_before_the_first_step_sends_nothing(self) -> None:
        def main(report: SetupFailureReport) -> None:
            sys.exit(2)  # e.g. argparse, before notify targets are known

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.assert_not_reported()

    def test_cancellation_is_not_reported_as_a_failure(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Clone test sources")
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_main(main)
        self.assert_not_reported()

    def test_without_notify_targets_only_the_summary_is_written(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Discover RCCL library")
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main, email="", webhook="")
        self.assertIn("Failed step: Discover RCCL library", self.summary())
        self.email.assert_not_called()
        self.webhook.assert_not_called()

    def test_a_reporting_error_does_not_replace_the_original_failure(self) -> None:
        self.webhook.side_effect = RuntimeError("webhook down")

        def main(report: SetupFailureReport) -> None:
            report.step("Discover RCCL library")
            sys.exit(3)

        with self.assertRaises(SystemExit) as raised:
            self.run_main(main)
        self.assertEqual(raised.exception.code, 3)

    def test_a_failure_the_harness_reported_itself_is_not_reported_again(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Resolve A/B baseline")
            report.failure_reported()
            sys.exit(1)

        with self.assertRaises(SystemExit) as raised:
            self.run_main(main)
        self.assertEqual(raised.exception.code, 1)
        self.assert_not_reported()

    def test_only_errors_logged_during_this_run_are_reported(self) -> None:
        # Harnesses build their report at import, long before run().
        report = SetupFailureReport("RCCL Test")
        log.error("from an earlier run")

        def main() -> None:
            report.step("Discover RCCL library")
            sys.exit(1)

        with self.assertRaises(SystemExit):
            report.run(main)
        self.assertNotIn("from an earlier run", self.summary())

    def test_a_failed_summary_write_still_sends_email_and_teams(self) -> None:
        def main(report: SetupFailureReport) -> None:
            report.step("Discover RCCL library")
            sys.exit(1)

        with mock.patch.object(rccl_ci_utils, "write_github_summary",
                               side_effect=OSError("summary file not writable")):
            with self.assertRaises(SystemExit):
                self.run_main(main)
        self.email.assert_called_once()
        self.webhook.assert_called_once()

    def test_a_failed_email_still_sends_teams(self) -> None:
        self.email.side_effect = RuntimeError("SMTP down")

        def main(report: SetupFailureReport) -> None:
            report.step("Discover RCCL library")
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.webhook.assert_called_once()

    def test_an_unformattable_error_message_goes_to_handle_error(self) -> None:
        # Fed to the handler directly: pytest's own log capture re-raises
        # formatting errors, so going through log.error() would test pytest.
        handler = rccl_ci_utils._ErrorLog()
        record = logging.LogRecord("harness", logging.ERROR, __file__, 1,
                                   "%d libraries found", ("no",), None)
        with mock.patch.object(handler, "handleError") as handle_error:
            handler.handle(record)
        handle_error.assert_called_once_with(record)
        self.assertEqual(handler.messages, [])


class HarnessSetupFailureTest(ReportCapture):
    """A forced setup failure in each harness's real main() sends a report."""

    def run_harness(self, module: types.ModuleType, subject: str, argv: list[str]) -> None:
        report = SetupFailureReport(subject)
        with mock.patch.object(module, "setup_report", report), \
                mock.patch.object(sys, "argv", [script_path(module)] + argv):
            with self.assertRaises(SystemExit) as raised:
                report.run(module.main)
        self.assertNotEqual(raised.exception.code, 0)

    def empty_artifact_dir(self) -> str:
        artifact_dir = self.tmp / "build"
        artifact_dir.mkdir(exist_ok=True)
        return str(artifact_dir)

    def test_jax_collective(self) -> None:
        self.run_harness(test_jax_collective, test_jax_collective.REPORT_SUBJECT, [
            "--artifact-dir", self.empty_artifact_dir(),
            "--jax-src", str(self.tmp / "jax-src"),
            "--notify-email", "dev@example.com",
            "--notify-webhook", "https://webhook.example",
        ])
        self.assert_reported("RCCL JAX Collective Test", "Discover RCCL library")
        self.assertIn("librccl.so not found in", self.summary())

    def test_pytorch_c10d(self) -> None:
        self.run_harness(test_pytorch_c10d, test_pytorch_c10d.REPORT_SUBJECT, [
            "--artifact-dir", self.empty_artifact_dir(),
            "--pytorch-src", str(self.tmp / "pytorch-src"),
            "--notify-email", "dev@example.com",
            "--notify-webhook", "https://webhook.example",
        ])
        self.assert_reported("RCCL PyTorch c10d Test", "Discover RCCL library")

    def madengine_argv(self, nodes: int | None = None) -> tuple[str, list[str]]:
        workload = next(iter(test_madengine.WORKLOAD_CONFIGS))
        required = test_madengine.WORKLOAD_CONFIGS[workload].get("required_nodes")
        return workload, [
            "--artifact-dir", self.empty_artifact_dir(),
            "--baseline-artifact-dir", self.empty_artifact_dir(),
            "--workload", workload,
            "--cluster", next(iter(test_madengine.CLUSTER_CONFIGS)),
            "--nodes", str(nodes or required or 2),
            "--work-dir", str(self.tmp / "work"),
            "--notify-email", "dev@example.com",
            "--teams-webhook", "https://webhook.example",
        ]

    def test_madengine(self) -> None:
        workload, argv = self.madengine_argv()
        self.run_harness(test_madengine, "RCCL MADEngine", argv)
        self.assert_reported(f"RCCL MADEngine {workload}", "Discover RCCL libraries")

    def test_madengine_fail_before_run_reports_once(self) -> None:
        workload, argv = self.madengine_argv()
        unreadable = {"md5": "", "version": "", "size": 0}
        with mock.patch.object(test_madengine, "find_rccl_library",
                               return_value=self.tmp / "librccl.so"), \
                mock.patch.object(test_madengine, "get_rccl_commit", return_value="abc1234"), \
                mock.patch.object(test_madengine, "get_rccl_fingerprint",
                                  return_value=unreadable), \
                mock.patch.object(test_madengine, "send_report") as send_report:
            self.run_harness(test_madengine, "RCCL MADEngine", argv)
        send_report.assert_called_once()
        self.assertIn("not run: Could not read the RCCL version", self.summary())
        self.assertNotIn("FAILED before any tests ran", self.summary())
        self.email.assert_not_called()
        self.webhook.assert_not_called()

    def test_madengine_argument_error(self) -> None:
        workload, argv = self.madengine_argv(nodes=1)
        with mock.patch("sys.stderr"):  # argparse prints usage
            self.run_harness(test_madengine, "RCCL MADEngine", argv)
        self.assert_reported(f"RCCL MADEngine {workload}", "Check arguments")

    def test_scripts_report_when_run_as_scripts(self) -> None:
        # Covers the __main__ entry points; no notify targets, so only the
        # GitHub summary is written.
        artifact_dir = self.empty_artifact_dir()
        _, madengine_argv = self.madengine_argv()
        cases = {
            test_jax_collective: (["--artifact-dir", artifact_dir,
                                   "--jax-src", str(self.tmp / "jax-src")],
                                  "Discover RCCL library"),
            test_pytorch_c10d: (["--artifact-dir", artifact_dir,
                                 "--pytorch-src", str(self.tmp / "pytorch-src")],
                                "Discover RCCL library"),
            test_madengine: (madengine_argv[:madengine_argv.index("--notify-email")],
                             "Discover RCCL libraries"),
        }
        for module, (argv, step) in cases.items():
            with self.subTest(script=Path(script_path(module)).name):
                self.summary_file.unlink(missing_ok=True)
                result = subprocess.run(
                    [sys.executable, script_path(module)] + argv,
                    capture_output=True, text=True, env=os.environ.copy(),
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"Failed step: {step}", self.summary())



class FakeProcess:
    """A test process that starts, prints nothing, and exits 1."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.stdout: Iterator[str] = iter([])
        self.returncode = 1
        self.pid = 0

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


class HarnessTestLaunchTest(ReportCapture):
    """Setup covers launching the tests; a failing test run is not a setup failure."""

    def patch_all(self, module: types.ModuleType, names: tuple[str, ...],
                  **returns: Any) -> None:
        for name in names:
            patcher = mock.patch.object(module, name, return_value=returns.get(name))
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, module: types.ModuleType, subject: str, argv: list[str],
                 expected: type[BaseException] = SystemExit) -> SetupFailureReport:
        report = SetupFailureReport(subject)
        with mock.patch.object(module, "setup_report", report), \
                mock.patch.object(sys, "argv", [script_path(module)] + argv):
            with self.assertRaises(expected):
                report.run(module.main)
        return report

    def patch_jax_setup(self) -> None:
        self.patch_all(
            test_jax_collective,
            ("find_rccl_library", "find_lib_dirs", "set_github_output",
             "populate_rocm_lib_dir", "setup_ld_library_path", "verify_rccl_override",
             "setup_kpack_device_code", "setup_xla_environment",
             "clone_jax_test_sources", "print_environment_info",
             "generate_summary_report"),
            find_rccl_library=self.tmp / "librccl.so", find_lib_dirs=[],
            generate_summary_report="RCCL JAX Collective Test Report: FAILED",
        )

    def jax_argv(self, results_log: Path) -> list[str]:
        return ["--artifact-dir", str(self.tmp), "--jax-src", str(self.tmp),
                "--results-log", str(results_log)]

    def test_jax_failing_tests_are_left_to_the_harness_report(self) -> None:
        self.patch_jax_setup()
        with mock.patch.object(test_jax_collective.subprocess, "Popen", FakeProcess):
            report = self.run_main(test_jax_collective, "RCCL JAX Collective Test",
                                   self.jax_argv(self.tmp / "results.log"))
        self.assertTrue(report.setup_done)
        self.assertIn("RCCL JAX Collective Test Report: FAILED", self.summary())
        self.assertNotIn("FAILED before any tests ran", self.summary())

    def test_jax_failure_to_launch_pytest_is_reported(self) -> None:
        self.patch_jax_setup()
        self.run_main(test_jax_collective, "RCCL JAX Collective Test",
                      self.jax_argv(self.tmp / "missing" / "results.log"),
                      expected=FileNotFoundError)
        self.assertIn("Failed step: Start pytest", self.summary())

    def test_jax_popen_failure_is_reported(self) -> None:
        # Fails at Popen itself, so tests_started() must sit after Popen.
        self.patch_jax_setup()
        with mock.patch.object(test_jax_collective.subprocess, "Popen",
                               side_effect=FileNotFoundError("pytest")):
            self.run_main(test_jax_collective, "RCCL JAX Collective Test",
                          self.jax_argv(self.tmp / "results.log"),
                          expected=FileNotFoundError)
        self.assertIn("Failed step: Start pytest", self.summary())

    def patch_pytorch_setup(self) -> None:
        self.patch_all(
            test_pytorch_c10d,
            ("find_rccl_library", "find_rocm_lib_dir", "set_github_output",
             "find_pip_sdk_lib_dirs", "reconcile_soname_versions",
             "quarantine_rocm_sysdeps", "override_bundled_rccl",
             "override_bundled_hip_runtime", "override_bundled_hsa_runtime",
             "setup_kpack_device_code", "setup_ld_library_path",
             "clone_pytorch_test_sources", "patch_missing_torch_modules",
             "print_environment_info", "generate_summary_report"),
            find_rccl_library=self.tmp / "librccl.so", find_pip_sdk_lib_dirs=[],
            generate_summary_report="RCCL PyTorch c10d Test Report: FAILED",
        )
        patcher = mock.patch.object(test_pytorch_c10d.tempfile, "mkdtemp",
                                    return_value=str(self.tmp))
        patcher.start()
        self.addCleanup(patcher.stop)

    def pytorch_argv(self, results_log: Path) -> list[str]:
        return ["--artifact-dir", str(self.tmp), "--pytorch-src", str(self.tmp),
                "--results-log", str(results_log)]

    def test_pytorch_failing_tests_are_left_to_the_harness_report(self) -> None:
        self.patch_pytorch_setup()
        with mock.patch.object(test_pytorch_c10d.subprocess, "Popen", FakeProcess):
            report = self.run_main(test_pytorch_c10d, "RCCL PyTorch c10d Test",
                                   self.pytorch_argv(self.tmp / "results.log"))
        self.assertTrue(report.setup_done)
        self.assertIn("RCCL PyTorch c10d Test Report: FAILED", self.summary())
        self.assertNotIn("FAILED before any tests ran", self.summary())

    def test_pytorch_failure_to_launch_pytest_is_reported(self) -> None:
        self.patch_pytorch_setup()
        self.run_main(test_pytorch_c10d, "RCCL PyTorch c10d Test",
                      self.pytorch_argv(self.tmp / "missing" / "results.log"),
                      expected=FileNotFoundError)
        self.assertIn("Failed step: Start pytest", self.summary())

    def test_pytorch_popen_failure_is_reported(self) -> None:
        # Fails at Popen itself, so tests_started() must sit after Popen.
        self.patch_pytorch_setup()
        with mock.patch.object(test_pytorch_c10d.subprocess, "Popen",
                               side_effect=FileNotFoundError("python")):
            self.run_main(test_pytorch_c10d, "RCCL PyTorch c10d Test",
                          self.pytorch_argv(self.tmp / "results.log"),
                          expected=FileNotFoundError)
        self.assertIn("Failed step: Start pytest", self.summary())

    def madengine_phases(self) -> list[tuple[str, Path]]:
        return [(phase, test_madengine.prepare_phase_dir(self.tmp, phase))
                for phase in (test_madengine.BASELINE, test_madengine.CANDIDATE)]

    def run_allocation(self, report: SetupFailureReport) -> dict[str, Any]:
        workload = next(iter(test_madengine.WORKLOAD_CONFIGS))
        with mock.patch.object(test_madengine, "setup_report", report):
            return test_madengine.run_ab_in_one_allocation(
                self.madengine_phases(), self.tmp, 2, workload,
                test_madengine.WORKLOAD_CONFIGS[workload],
                next(iter(test_madengine.CLUSTER_CONFIGS.values())), 1)

    def test_madengine_tests_start_once_salloc_launches(self) -> None:
        report = SetupFailureReport("RCCL MADEngine")
        with mock.patch.object(test_madengine.subprocess, "run"), \
                mock.patch.object(test_madengine.subprocess, "Popen", FakeProcess):
            self.run_allocation(report)
        self.assertTrue(report.setup_done)

    def test_madengine_allocation_preparation_is_setup(self) -> None:
        report = SetupFailureReport("RCCL MADEngine")
        with mock.patch.dict(os.environ, {"MADENGINE_SLURM_EXCLUDE": "node1; rm -rf"}), \
                mock.patch.object(test_madengine.subprocess, "Popen") as popen:
            with self.assertRaises(RuntimeError):
                self.run_allocation(report)
        popen.assert_not_called()
        self.assertFalse(report.setup_done)


    def test_madengine_salloc_launch_failure_is_setup(self) -> None:
        report = SetupFailureReport("RCCL MADEngine")
        with mock.patch.object(test_madengine.subprocess, "run"), \
                mock.patch.object(test_madengine.subprocess, "Popen",
                                  side_effect=FileNotFoundError("salloc")):
            with self.assertRaises(FileNotFoundError):
                self.run_allocation(report)
        self.assertFalse(report.setup_done)


if __name__ == "__main__":
    unittest.main()
