"""Tests for metrics_prove_completion() in test_madengine.py.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm


def row(status="pass", metric=100.0):
    return {"status": status, "metric_value": metric}


def results(**overrides):
    out = {tm.BASELINE: [row()], tm.CANDIDATE: [row()]}
    out.update(overrides)
    return out


class MetricsProveCompletion(unittest.TestCase):
    def test_megatron_rows_that_passed_with_metrics_prove_it(self):
        self.assertTrue(tm.metrics_prove_completion("megatron", results()))

    def test_maxtext_rows_never_prove_it(self):
        # A crashed MaxText run still leaves passing rows with metrics.
        self.assertFalse(tm.metrics_prove_completion("maxtext", results()))

    def test_a_phase_without_rows_does_not(self):
        self.assertFalse(tm.metrics_prove_completion("megatron", results(candidate=[])))

    def test_a_failed_row_does_not(self):
        self.assertFalse(
            tm.metrics_prove_completion("megatron", results(baseline=[row(status="fail")]))
        )

    def test_a_row_without_a_metric_does_not(self):
        self.assertFalse(
            tm.metrics_prove_completion("megatron", results(candidate=[row(metric=None)]))
        )

    def test_no_results_do_not(self):
        self.assertFalse(tm.metrics_prove_completion("megatron", {}))


if __name__ == "__main__":
    unittest.main()
