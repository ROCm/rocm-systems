"""Tests for the per-workload metric aliases collect_phase_results() accepts.

The MaxText extractor writes `tok_per_s_per_gpu` and `TFLOPS_per_gpu`, already
averaged over the trailing steps. Megatron writes a per-iteration
`tok_per_s_per_gpu` alongside an averaged `tok_per_s_per_gpu_avg`, and the
global table accepts only the `_avg` names. So the un-suffixed names are
widened per workload, never globally: in the global table they would silently
score Llama-3.1-70B on its final iteration instead of its average.

Both halves of that are load-bearing and neither is visible in a passing run:
drop the widening and a finished 405B run finds no comparison row and fails
closed; move it into the global table and the 70B number changes meaning
without anything failing at all.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm

MAXTEXT = "llama-3.1-405b-maxtext"
MEGATRON = "llama-3.1-70b-training"


def rows(metric_name, tflops_name):
    """One precision's worth of madengine long-format output."""
    return [
        {"metric": metric_name, "performance": "506.8", "training_precision": "fp8",
         "status": "SUCCESS"},
        {"metric": tflops_name, "performance": "1279.2", "training_precision": "fp8",
         "status": "SUCCESS"},
    ]


class MetricAliases(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, payload):
        (self.run_dir / "perf_entry_super.json").write_text(json.dumps(payload))

    def collect(self, workload):
        cfg = tm.WORKLOAD_CONFIGS[workload]
        return tm.collect_phase_results(
            self.run_dir,
            cfg["metric_key"],
            extra_aliases=cfg.get("metric_aliases"),
            extra_tflops_aliases=cfg.get("tflops_aliases"),
        )

    def test_maxtext_accepts_the_extractor_names(self):
        self.write(rows("tok_per_s_per_gpu", "TFLOPS_per_gpu"))
        results = self.collect(MAXTEXT)
        self.assertEqual(len(results), 1, "405B found no comparison row")
        self.assertAlmostEqual(results[0]["metric_value"], 506.8)
        self.assertAlmostEqual(results[0]["tflops_avg"], 1279.2)
        self.assertEqual(results[0]["precision"], "fp8")

    def test_megatron_rejects_the_un_suffixed_names(self):
        # The same rows under the workload that must NOT accept them: taking
        # them would score this workload on its final iteration.
        self.write(rows("tok_per_s_per_gpu", "TFLOPS_per_gpu"))
        self.assertEqual(self.collect(MEGATRON), [])

    def test_megatron_still_accepts_its_own_averaged_names(self):
        self.write(rows("tok_per_s_per_gpu_avg", "TFLOPS_per_gpu_avg"))
        results = self.collect(MEGATRON)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["metric_value"], 506.8)
        self.assertAlmostEqual(results[0]["tflops_avg"], 1279.2)

    def test_widening_is_not_global(self):
        # Guards the direction the comment warns about: a later edit that
        # moves the alias into the global table would pass every test above.
        for key, accepted in tm._METRIC_ALIASES.items():
            self.assertNotIn("tok_per_s_per_gpu", accepted,
                             f"un-suffixed throughput leaked into _METRIC_ALIASES[{key!r}]")
        self.assertNotIn("TFLOPS_per_gpu", tm._TFLOPS_METRICS)

    def test_maxtext_config_carries_both_aliases(self):
        cfg = tm.WORKLOAD_CONFIGS[MAXTEXT]
        self.assertIn("tok_per_s_per_gpu", cfg["metric_aliases"])
        self.assertIn("TFLOPS_per_gpu", cfg["tflops_aliases"])


if __name__ == "__main__":
    unittest.main()
