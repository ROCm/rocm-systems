"""Tests for the FAILED report a harness sends when its setup fails.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import rccl_ci_utils
import test_jax_collective
import test_madengine
import test_pytorch_c10d
from rccl_ci_utils import SetupFailureReport

log = logging.getLogger(__name__)

RUN_ENV = {
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "ROCm/rocm-systems",
    "GITHUB_RUN_ID": "123",
}


class ReportCapture(unittest.TestCase):
    """Patch the report channels and record what each one is sent."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.summary_file = self.tmp / "step_summary.md"
        env = dict(RUN_ENV, GITHUB_STEP_SUMMARY=str(self.summary_file))
        for patcher in (
            mock.patch.dict(os.environ, env),
            mock.patch.object(rccl_ci_utils, "send_email_report"),
            mock.patch.object(rccl_ci_utils, "send_teams_webhook"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @property
    def email(self):
        return rccl_ci_utils.send_email_report

    @property
    def webhook(self):
        return rccl_ci_utils.send_teams_webhook

    def summary(self):
        return self.summary_file.read_text() if self.summary_file.exists() else ""

    def assert_reported(self, subject, step):
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

    def assert_not_reported(self):
        self.assertEqual(self.summary(), "")
        self.email.assert_not_called()
        self.webhook.assert_not_called()


class SetupFailureReportTest(ReportCapture):
    def run_main(self, main, **notify):
        report = SetupFailureReport("RCCL Test")
        report.notify(notify.get("email", "dev@example.com"),
                      notify.get("webhook", "https://webhook.example"))
        report.run(lambda: main(report))

    def test_sys_exit_in_a_step_reports_step_and_logged_errors(self):
        def main(report):
            report.step("Discover RCCL library")
            log.error("librccl.so not found in /build")
            sys.exit(1)

        with self.assertRaises(SystemExit) as raised:
            self.run_main(main)
        self.assertEqual(raised.exception.code, 1)
        self.assert_reported("RCCL Test", "Discover RCCL library")
        self.assertIn("Cause:       exited with status 1", self.summary())
        self.assertIn("  librccl.so not found in /build", self.summary())

    def test_exception_in_a_step_reports_it_and_is_reraised(self):
        def main(report):
            report.step("Install madengine")
            subprocess.run(["false"], check=True)

        with self.assertRaises(subprocess.CalledProcessError):
            self.run_main(main)
        self.assert_reported("RCCL Test", "Install madengine")
        self.assertIn("Cause:       CalledProcessError: Command '['false']'", self.summary())

    def test_names_the_last_step_started(self):
        def main(report):
            report.step("Discover RCCL library")
            report.step("Clone test sources")
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.assert_reported("RCCL Test", "Clone test sources")

    def test_failure_after_tests_started_is_left_to_the_harness(self):
        def main(report):
            report.step("Clone test sources")
            report.tests_started()
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.assert_not_reported()

    def test_success_and_early_return_send_nothing(self):
        def exit_zero(report):
            report.step("Discover RCCL library")
            sys.exit(0)

        def early_return(report):  # --discover-only
            report.step("Discover RCCL library")

        with self.assertRaises(SystemExit):
            self.run_main(exit_zero)
        self.run_main(early_return)
        self.assert_not_reported()

    def test_failure_before_the_first_step_sends_nothing(self):
        def main(report):
            sys.exit(2)  # e.g. argparse, before notify targets are known

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.assert_not_reported()

    def test_cancellation_is_not_reported_as_a_failure(self):
        def main(report):
            report.step("Clone test sources")
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_main(main)
        self.assert_not_reported()

    def test_without_notify_targets_only_the_summary_is_written(self):
        def main(report):
            report.step("Discover RCCL library")
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main, email="", webhook="")
        self.assertIn("Failed step: Discover RCCL library", self.summary())
        self.email.assert_not_called()
        self.webhook.assert_not_called()

    def test_a_reporting_error_does_not_replace_the_original_failure(self):
        self.webhook.side_effect = RuntimeError("webhook down")

        def main(report):
            report.step("Discover RCCL library")
            sys.exit(3)

        with self.assertRaises(SystemExit) as raised:
            self.run_main(main)
        self.assertEqual(raised.exception.code, 3)

    def test_only_errors_logged_during_this_run_are_reported(self):
        log.error("from an earlier run")

        def main(report):
            report.step("Discover RCCL library")
            sys.exit(1)

        with self.assertRaises(SystemExit):
            self.run_main(main)
        self.assertNotIn("from an earlier run", self.summary())


class HarnessSetupFailureTest(ReportCapture):
    """A forced setup failure in each harness's real main() sends a report."""

    def run_harness(self, module, subject, argv):
        report = SetupFailureReport(subject)
        with mock.patch.object(module, "setup_report", report), \
                mock.patch.object(sys, "argv", [module.__file__] + argv):
            with self.assertRaises(SystemExit) as raised:
                report.run(module.main)
        self.assertNotEqual(raised.exception.code, 0)

    def empty_artifact_dir(self):
        artifact_dir = self.tmp / "build"
        artifact_dir.mkdir(exist_ok=True)
        return str(artifact_dir)

    def test_jax_collective(self):
        self.run_harness(test_jax_collective, test_jax_collective.REPORT_SUBJECT, [
            "--artifact-dir", self.empty_artifact_dir(),
            "--jax-src", str(self.tmp / "jax-src"),
            "--notify-email", "dev@example.com",
            "--notify-webhook", "https://webhook.example",
        ])
        self.assert_reported("RCCL JAX Collective Test", "Discover RCCL library")
        self.assertIn("librccl.so not found in", self.summary())

    def test_pytorch_c10d(self):
        self.run_harness(test_pytorch_c10d, test_pytorch_c10d.REPORT_SUBJECT, [
            "--artifact-dir", self.empty_artifact_dir(),
            "--pytorch-src", str(self.tmp / "pytorch-src"),
            "--notify-email", "dev@example.com",
            "--notify-webhook", "https://webhook.example",
        ])
        self.assert_reported("RCCL PyTorch c10d Test", "Discover RCCL library")

    def madengine_argv(self, nodes=None):
        workload = next(iter(test_madengine.WORKLOAD_CONFIGS))
        required = test_madengine.WORKLOAD_CONFIGS[workload].get("required_nodes")
        return workload, [
            "--artifact-dir", self.empty_artifact_dir(),
            "--workload", workload,
            "--cluster", next(iter(test_madengine.CLUSTER_CONFIGS)),
            "--nodes", str(nodes or required or 2),
            "--work-dir", str(self.tmp / "work"),
            "--notify-email", "dev@example.com",
            "--teams-webhook", "https://webhook.example",
        ]

    def test_madengine(self):
        workload, argv = self.madengine_argv()
        self.run_harness(test_madengine, "RCCL MADEngine", argv)
        self.assert_reported(f"RCCL MADEngine {workload}", "Discover RCCL library")

    def test_madengine_argument_error(self):
        workload, argv = self.madengine_argv(nodes=1)
        with mock.patch("sys.stderr"):  # argparse prints usage
            self.run_harness(test_madengine, "RCCL MADEngine", argv)
        self.assert_reported(f"RCCL MADEngine {workload}", "Check arguments")

    def test_scripts_report_when_run_as_scripts(self):
        # Covers the __main__ entry points; no notify targets, so only the
        # GitHub summary is written.
        artifact_dir = self.empty_artifact_dir()
        _, madengine_argv = self.madengine_argv()
        cases = {
            test_jax_collective: ["--artifact-dir", artifact_dir,
                                  "--jax-src", str(self.tmp / "jax-src")],
            test_pytorch_c10d: ["--artifact-dir", artifact_dir,
                                "--pytorch-src", str(self.tmp / "pytorch-src")],
            test_madengine: madengine_argv[:madengine_argv.index("--notify-email")],
        }
        for module, argv in cases.items():
            with self.subTest(script=Path(module.__file__).name):
                self.summary_file.unlink(missing_ok=True)
                result = subprocess.run(
                    [sys.executable, module.__file__] + argv,
                    capture_output=True, text=True, env=os.environ.copy(),
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Failed step: Discover RCCL library", self.summary())


if __name__ == "__main__":
    unittest.main()
