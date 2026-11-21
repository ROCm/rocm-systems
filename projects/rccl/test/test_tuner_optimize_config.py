# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Guards for plugins/tuner/example/scripts/optimize_config.py.

A NaN measurement must never win a size range (every comparison against NaN
is false, so it would stick as the first "best"), nor may a negative value or a
row missing the selected metric (which used to default to 0). NCCL PR #2453.
Plain unittest, python3 only: no GPU, no build.
"""

import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

RCCL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RCCL_ROOT / "plugins" / "tuner" / "example" / "scripts"))

from optimize_config import ConfigOptimizer  # noqa: E402

FIELDS = ["collective", "size_bytes", "algorithm", "protocol", "channels",
          "nodes", "ranks", "pipeOps", "regBuff"]


class _Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "measurements.csv"

    def best(self, rows, metric, columns=None):
        """Write rows of (algorithm, value) and return the winning algorithms."""
        columns = [metric] if columns is None else columns
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS + columns)
            for algorithm, value in rows:
                writer.writerow(["allreduce", 1024, algorithm, "simple", 4, 1, 8, 1, 0]
                                + [value] * len(columns))
        optimizer = ConfigOptimizer(metric)
        with contextlib.redirect_stdout(io.StringIO()):
            data = optimizer.load_data(str(self.path))
            configs = optimizer.optimize_configurations(data)
        return [config.split(",")[3] for config in configs]


class OptimizeConfigControl(_Base):
    def test_valid_measurements_choose_best(self):
        rows = [("nvls", "7"), ("tree", "3"), ("ring", "5")]
        self.assertEqual(self.best(rows, "latency_us"), ["tree"])
        self.assertEqual(self.best(rows, "bandwidth_gbps"), ["nvls"])


class OptimizeConfigInvalidMetric(_Base):
    def test_nonfinite_measurement_never_wins(self):
        for metric, expected in [("latency_us", "tree"), ("bandwidth_gbps", "ring")]:
            for invalid in ["nan", "inf" if metric == "bandwidth_gbps" else "-inf"]:
                with self.subTest(metric=metric, invalid=invalid):
                    rows = [("nvls", invalid), ("tree", "3"), ("ring", "5")]
                    self.assertEqual(self.best(rows, metric), [expected])

    def test_negative_latency_never_wins(self):
        rows = [("nvls", "-1"), ("tree", "3"), ("ring", "5")]
        self.assertEqual(self.best(rows, "latency_us"), ["tree"])

    def test_missing_selected_metric_is_skipped(self):
        # Only bandwidth_gbps is measured; optimizing latency_us has no data.
        rows = [("tree", "3"), ("ring", "5")]
        self.assertEqual(self.best(rows, "latency_us", ["bandwidth_gbps"]), [])


if __name__ == "__main__":
    unittest.main()
