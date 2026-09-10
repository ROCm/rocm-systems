#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdlib-only unit tests for ``dme_integration.metrics``."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Allow running directly (``python3 dme_integration/test_metrics.py``) as well
# as via unittest discovery from ``projects/amdsmi/tests``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dme_integration import metrics


class ExposedMetricNamesTest(unittest.TestCase):
    def test_normal_timestamp_and_special_values(self):
        body = (
            "# HELP gpu_edge_temperature Edge temperature\n"
            "# TYPE gpu_edge_temperature gauge\n"
            'gpu_edge_temperature{gpu="0"} 42.5\n'
            "gpu_power_usage 100 1620000000000\n"
            "gpu_nan_metric NaN\n"
            "gpu_posinf_metric +Inf\n"
            "gpu_neginf_metric -Inf\n"
        )
        self.assertEqual(
            metrics._exposed_metric_names(body),
            {
                "gpu_edge_temperature",
                "gpu_power_usage",
                "gpu_nan_metric",
                "gpu_posinf_metric",
                "gpu_neginf_metric",
            },
        )

    def test_help_type_only_body_is_empty(self):
        body = "# HELP gpu_edge_temperature Edge temperature\n# TYPE gpu_edge_temperature gauge\n"
        self.assertEqual(metrics._exposed_metric_names(body), set())


class GpuAgentCrashedTest(unittest.TestCase):
    def _write_log(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def test_stack_smashing_detected_is_crash(self):
        log = self._write_log("some output\n*** stack smashing detected ***: terminated\n")
        self.assertTrue(metrics._gpu_agent_crashed(log))

    def test_segmentation_fault_is_crash(self):
        log = self._write_log("Segmentation fault (core dumped)\n")
        self.assertTrue(metrics._gpu_agent_crashed(log))

    def test_connection_aborted_by_peer_is_not_crash(self):
        # F-5: benign gRPC line must NOT be treated as a crash.
        log = self._write_log("W: Connection Aborted by peer\ninfo: serving metrics\n")
        self.assertFalse(metrics._gpu_agent_crashed(log))

    def test_missing_log_file_is_not_crash(self):
        self.assertFalse(metrics._gpu_agent_crashed(Path("/nonexistent/gpu-agent.log")))


_VALID_BODY_WITH_REQUIRED = (
    "# HELP gpu_edge_temperature Edge temperature\n"
    "# TYPE gpu_edge_temperature gauge\n"
    "gpu_edge_temperature 42.5\n"
)

_VALID_BODY_MISSING_REQUIRED = (
    "# HELP gpu_other_metric Other\n# TYPE gpu_other_metric gauge\ngpu_other_metric 1\n"
)


class VerifyTest(unittest.TestCase):
    def _verify(self, **overrides):
        kwargs = dict(
            url="http://localhost:5000/metrics",
            required_metrics=("gpu_edge_temperature",),
            max_retries=1,
            retry_delay=0,
            request_timeout=1.0,
        )
        kwargs.update(overrides)
        return metrics.verify(**kwargs)

    def test_unreachable_endpoint_exits(self):
        with mock.patch.object(metrics, "_fetch", return_value=(0, "")):
            with self.assertRaises(SystemExit):
                self._verify()

    def test_all_required_metrics_present_returns(self):
        with mock.patch.object(metrics, "_fetch", return_value=(200, _VALID_BODY_WITH_REQUIRED)):
            self.assertIsNone(self._verify())

    def _make_pid_file(self, pid: int = 1234) -> Path:
        with tempfile.NamedTemporaryFile("w", suffix=".pid", delete=False) as f:
            f.write(str(pid))
            pid_file = Path(f.name)
        self.addCleanup(pid_file.unlink, missing_ok=True)
        return pid_file

    def test_missing_metrics_gpu_agent_alive_exits(self):
        pid_file = self._make_pid_file()
        with (
            mock.patch.object(metrics, "_fetch", return_value=(200, _VALID_BODY_MISSING_REQUIRED)),
            mock.patch.object(metrics, "_read_pid_file", return_value=1234),
            mock.patch.object(metrics, "_process_alive", return_value=True),
        ):
            with self.assertRaises(SystemExit):
                self._verify(gpu_agent_pid_file=pid_file)

    def test_missing_metrics_gpu_agent_started_then_died_soft_passes(self):
        # Agent started (PID file present) then died: soft-pass (upstream ABI
        # skew, not a PR regression); self-disables once the agent stays up.
        pid_file = self._make_pid_file()
        with (
            mock.patch.object(metrics, "_fetch", return_value=(200, _VALID_BODY_MISSING_REQUIRED)),
            mock.patch.object(metrics, "_read_pid_file", return_value=1234),
            mock.patch.object(metrics, "_process_alive", return_value=False),
        ):
            self.assertIsNone(self._verify(gpu_agent_pid_file=pid_file))

    def test_missing_metrics_no_pid_file_hard_fails(self):
        # No PID file means the agent never started -- a real failure, not
        # upstream skew -- so missing metrics must hard-fail, not soft-pass.
        missing = Path(tempfile.gettempdir()) / "dme-nonexistent-agent.pid"
        if missing.exists():
            missing.unlink()
        with mock.patch.object(metrics, "_fetch", return_value=(200, _VALID_BODY_MISSING_REQUIRED)):
            with self.assertRaises(SystemExit):
                self._verify(gpu_agent_pid_file=missing)

    def test_empty_body_exits_not_assertion_error(self):
        # F-4: HTTP 200 with an empty body on every retry must raise SystemExit,
        # not AssertionError from the format check.
        with mock.patch.object(metrics, "_fetch", return_value=(200, "")):
            with self.assertRaises(SystemExit):
                self._verify()


if __name__ == "__main__":
    unittest.main()
