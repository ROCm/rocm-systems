"""Tests for scoring training on the median of per-iteration throughput.

Run 45125244674 compared two builds of the same librccl: FP8 70B scored
-5.3% on Megatron's running mean while its iteration median was +0.2%,
because 18 of 48 candidate iterations stalled by 2-5 s. These tests pin the
pieces that make the median the gate: per-precision parsing of the node logs,
the same statistic on both sides, and the mean kept for comparison.

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


def iteration_line(i, total, tps, avg):
    return (f"\x1b[32m[20261123 10:06:00]\x1b[0m[rank-15/16] iteration {i:>8}/{total:>8} | "
            f"consumed samples: {i * 64} | elapsed time per iteration (ms): 11900.0/11900.0 | "
            f"throughput per GPU (TFLOP/s/GPU): 1238.0/1230.0 | "
            f"tokens per GPU (tokens/s/GPU): {tps:.1f}/{avg:.1f} | learning rate: 1.0E-05\n")


def node_log(runs, total=12):
    """A last-rank node log: one header and its iterations per precision."""
    out = []
    for precision, values in runs:
        out.append(f"Running: Llama-3.1-70B - {precision} - pretrain\n")
        # Iterations 1-2 print no tokens/s, so the values start at 3.
        out.append("iteration        1/      12 | elapsed time per iteration (ms): 34000.0 |"
                   " throughput per GPU (TFLOP/s/GPU): 430.0 |\n")
        for i, tps in enumerate(values, start=3):
            mean = sum(values[: i - 2]) / (i - 2)
            out.append(iteration_line(i, total, tps, mean))
    return "".join(out)


STEADY = [2750.0] * 10
STALLED = [2750.0, 2000.0, 2750.0, 2100.0, 2750.0, 2750.0, 2050.0, 2750.0, 2750.0, 2750.0]


class IterationParsing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        (self.run_dir / "slurm_output").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        (self.run_dir / "slurm_output" / name).write_text(text)

    def test_values_are_split_by_precision_header(self):
        self.write("job_node_1.out", node_log([("BF16", [1650.0] * 10), ("FP8", STALLED)]))
        got = tm.parse_iteration_throughput(self.run_dir)
        self.assertEqual(got["BF16"], [1650.0] * 10)
        self.assertEqual(got["FP8"], STALLED)

    def test_the_node_with_most_values_wins(self):
        # Node 0 prints the header but not the iterations of the last rank.
        self.write("job_node_0.out", node_log([("FP8", STEADY[:2])]))
        self.write("job_node_1.out", node_log([("FP8", STEADY)]))
        self.assertEqual(len(tm.parse_iteration_throughput(self.run_dir)["FP8"]), 10)

    def test_logs_without_headers_yield_nothing(self):
        self.write("job_node_0.out", iteration_line(3, 12, 2750.0, 2750.0))
        self.assertEqual(tm.parse_iteration_throughput(self.run_dir), {})

    def test_stalls_move_the_mean_but_not_the_median(self):
        stats = tm.iteration_stats(STALLED)
        self.assertEqual(stats["median"], 2750.0)
        self.assertEqual(stats["iterations"], 10)
        self.assertEqual(stats["stalled"], 3)
        self.assertEqual(tm.iteration_stats(STEADY)["stalled"], 0)

    def test_collect_attaches_the_iterations_to_the_structured_row(self):
        self.write("job_node_1.out", node_log([("FP8", STALLED)]))
        (self.run_dir / "perf_entry_super.json").write_text(json.dumps([
            {"metric": "tok_per_s_per_gpu_avg", "performance": "2540.0",
             "training_precision": "FP8", "status": "SUCCESS"},
        ]))
        cfg = tm.WORKLOAD_CONFIGS["llama-3.1-70b-training"]
        (row,) = tm.collect_phase_results(self.run_dir, cfg["metric_key"])
        self.assertAlmostEqual(row["metric_value"], 2540.0)
        self.assertEqual(row["median"], 2750.0)
        self.assertEqual(row["iteration_values"], STALLED)

    def test_too_few_iterations_keep_the_mean(self):
        self.write("job_node_1.out", node_log([("FP8", STEADY[: tm.MIN_GATE_ITERATIONS - 1])]))
        (self.run_dir / "perf_entry_super.json").write_text(json.dumps([
            {"metric": "tok_per_s_per_gpu_avg", "performance": "2750.0",
             "training_precision": "FP8", "status": "SUCCESS"},
        ]))
        (row,) = tm.collect_phase_results(self.run_dir, "tokens_per_second_per_gpu")
        self.assertNotIn("median", row)


def phase_row(mean, values=None):
    row = {"precision": "FP8", "metric_value": mean, "status": "pass", "tflops_avg": None}
    if values:
        row["iteration_values"] = values
        row.update(tm.iteration_stats(values))
    return row


class GateOnMedian(unittest.TestCase):
    def test_stalled_candidate_passes_on_the_median(self):
        base = phase_row(2725.4, STEADY)
        cand = phase_row(2580.3, STALLED)
        (c,) = tm.pair_by_precision([base], [cand])
        self.assertEqual(c["metric"], "median")
        self.assertAlmostEqual(c["ratio"], 1.0)
        self.assertAlmostEqual(c["ratio_avg"], 2580.3 / 2725.4)
        reg, msg = tm.check_ab_regression(*c["gate_values"], "training", metric=c["metric"])
        self.assertFalse(reg)
        self.assertIn("(median)", msg)
        # The mean alone would have failed this pair.
        self.assertTrue(tm.check_ab_regression(2725.4, 2580.3, "training")[0])

    def test_a_slowdown_of_every_iteration_still_fails(self):
        base = phase_row(2750.0, STEADY)
        cand = phase_row(2612.5, [v * 0.95 for v in STEADY])
        (c,) = tm.pair_by_precision([base], [cand])
        self.assertTrue(tm.check_ab_regression(*c["gate_values"], "training")[0])

    def test_one_side_without_iterations_scores_both_on_the_mean(self):
        (c,) = tm.pair_by_precision([phase_row(2725.4)], [phase_row(2580.3, STEADY)])
        self.assertEqual(c["metric"], "avg")
        self.assertEqual(c["gate_values"], (2725.4, 2580.3))
        self.assertAlmostEqual(c["ratio"], c["ratio_avg"])

    def test_summary_keeps_the_mean_beside_the_median(self):
        comparisons = tm.pair_by_precision(
            [phase_row(2725.4, STEADY)], [phase_row(2580.3, STALLED)])
        report = tm.generate_summary_report(
            "llama-3.1-70b-training", "2N", 0, "[FP8] No regression", "", "ruby",
            comparisons=comparisons)
        self.assertIn("of the iteration medians; of the means 0.9468 (-5.3%)", report)
        self.assertIn("median 2750.0, mean 2580.3, 3/10 iterations stalled", report)


if __name__ == "__main__":
    unittest.main()
