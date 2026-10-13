#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdlib-only unit tests for ``dynolog_integration.run_example`` verification."""

import sys
import unittest
from pathlib import Path

# Allow running directly as well as via unittest discovery from
# ``projects/rdc/tests``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynolog_integration import run_example

# A representative healthy run: started, discovered entities, collected metrics.
_HEALTHY_LOG = """\
Initializing RDC example, ROCM version: 70200
RDC watch config: update_interval=1000ms, max_keep_age=10s, max_keep_samples=100
Collecting 30 metrics:
  metric RDC_FI_GPU_UTIL (id=500)
Monitoring 8 entities (physical GPUs + partitions):
  entity index=0
entity 0: collected 30 metrics
  entity 0 RDC_FI_GPU_UTIL (id=500)=42
entity 1: collected 30 metrics
"""

# Fatal due to a capability/field gap on this arch (acceptable on a GPU box).
_CAPABILITY_FATAL_LOG = """\
Initializing RDC example, ROCM version: 70200
Collecting 30 metrics:
RDC example failed: rdc_field_watch failed: RDC_ST_NOT_SUPPORTED
"""

# Fatal due to a genuine setup failure (a real RDC regression).
_INIT_FATAL_LOG = """\
Initializing RDC example, ROCM version: 70200
Collecting 30 metrics:
RDC example failed: rdc_start_embedded failed: RDC_ST_FAIL_LOAD_MODULE
"""

# Started and discovered GPUs but never read a metric.
_NO_METRICS_LOG = """\
Initializing RDC example, ROCM version: 70200
Monitoring 8 entities (physical GPUs + partitions):
entity 0: collected 0 metrics
"""

# Started, then hung in RdcWrapper setup: no entity count, no fatal, no exit.
_HUNG_LOG = """\
Initializing RDC example, ROCM version: 70200
Collecting 30 metrics:
  metric RDC_FI_GPU_UTIL (id=500)
"""


class EvaluateTest(unittest.TestCase):
    def test_healthy_run_passes(self):
        ok, soft, reason = run_example.evaluate(_HEALTHY_LOG, gpu_available=True, self_exited=False)
        self.assertTrue(ok)
        self.assertFalse(soft)
        self.assertIn("30 metrics", reason)

    def test_never_started_hard_fails(self):
        for gpu in (True, False):
            ok, soft, _ = run_example.evaluate(
                "garbage output\n", gpu_available=gpu, self_exited=False
            )
            self.assertFalse(ok)
            self.assertFalse(soft)

    def test_capability_fatal_soft_passes_on_gpu(self):
        ok, soft, _ = run_example.evaluate(
            _CAPABILITY_FATAL_LOG, gpu_available=True, self_exited=True
        )
        self.assertTrue(ok)
        self.assertTrue(soft)

    def test_genuine_fatal_hard_fails_on_gpu(self):
        ok, soft, reason = run_example.evaluate(
            _INIT_FATAL_LOG, gpu_available=True, self_exited=True
        )
        self.assertFalse(ok)
        self.assertFalse(soft)
        self.assertIn("fatal error", reason)

    def test_any_fatal_soft_passes_without_gpu(self):
        ok, soft, _ = run_example.evaluate(_INIT_FATAL_LOG, gpu_available=False, self_exited=True)
        self.assertTrue(ok)
        self.assertTrue(soft)

    def test_self_exit_after_healthy_hard_fails(self):
        # Designed to loop forever; a self-exit after a healthy prefix is a crash.
        ok, soft, reason = run_example.evaluate(_HEALTHY_LOG, gpu_available=True, self_exited=True)
        self.assertFalse(ok)
        self.assertFalse(soft)
        self.assertIn("exited on its own", reason)

    def test_no_metrics_soft_passes_by_default_on_gpu(self):
        ok, soft, reason = run_example.evaluate(
            _NO_METRICS_LOG, gpu_available=True, self_exited=False
        )
        self.assertTrue(ok)
        self.assertTrue(soft)
        self.assertIn("no metrics", reason)

    def test_no_metrics_hard_fails_under_strict(self):
        ok, soft, _ = run_example.evaluate(
            _NO_METRICS_LOG, gpu_available=True, self_exited=False, strict=True
        )
        self.assertFalse(ok)
        self.assertFalse(soft)

    def test_no_metrics_soft_passes_without_gpu(self):
        ok, soft, _ = run_example.evaluate(_NO_METRICS_LOG, gpu_available=False, self_exited=False)
        self.assertTrue(ok)
        self.assertTrue(soft)

    def test_hang_hard_fails_on_gpu(self):
        # No entity count and no fatal means setup never returned; no GPU arch
        # explains that, so it must not be excused as field availability.
        ok, soft, reason = run_example.evaluate(_HUNG_LOG, gpu_available=True, self_exited=False)
        self.assertFalse(ok)
        self.assertFalse(soft)
        self.assertIn("never reported an entity count", reason)

    def test_hang_soft_passes_without_gpu(self):
        ok, soft, _ = run_example.evaluate(_HUNG_LOG, gpu_available=False, self_exited=False)
        self.assertTrue(ok)
        self.assertTrue(soft)


class ParseHelpersTest(unittest.TestCase):
    def test_monitored_entities_reads_last(self):
        self.assertEqual(run_example._monitored_entities(_HEALTHY_LOG), 8)

    def test_monitored_entities_none_when_absent(self):
        self.assertIsNone(run_example._monitored_entities("no entities here"))

    def test_max_metrics_collected(self):
        self.assertEqual(run_example._max_metrics_collected(_HEALTHY_LOG), 30)

    def test_max_metrics_zero_when_absent(self):
        self.assertEqual(run_example._max_metrics_collected("nothing"), 0)


if __name__ == "__main__":
    unittest.main()
