#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Unit tests for the perf measurement helpers and the two perf cost models.
#
# These helpers decide whether a perf test passes or fails, so a bug in them is worse than a bug in
# the thing being measured: it either hides a real regression or fails CI on a healthy build.
# Neither shows up as a test failure anywhere else, because the perf tests need a GPU and do not run
# on most machines. Everything here is pure Python and runs anywhere.

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import perf_stats


def _load(name: str, path: Path):
    """Load a module from an explicit path.

    The two suites each have their own perf_cost_model.py. They are different models with the same
    module name, so importing both by name would silently give whichever landed in sys.modules
    first, and the second suite's tests would then be checking the wrong model.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kr_cost_model = _load(
    "kr_perf_cost_model", _HERE.parent / "kernel-replay-perf" / "perf_cost_model.py"
)
qh_cost_model = _load(
    "qh_perf_cost_model", _HERE.parent / "queue-hooks-perf" / "perf_cost_model.py"
)


class MarkerParsing(unittest.TestCase):
    """The workload prints one key=value marker line; everything downstream reads it."""

    def test_parses_every_field(self):
        out = "[kr-perf] ballast_mb=64 launches=8 wall_ms=123.5 counter=8"
        self.assertEqual(
            perf_stats.parse_marker(out, "kr-perf"),
            {"ballast_mb": 64, "launches": 8, "wall_ms": 123.5, "counter": 8},
        )

    def test_integers_stay_integers_and_decimals_become_floats(self):
        m = perf_stats.parse_marker("[qh-perf] launches=32 wall_ms=0.5", "qh-perf")
        self.assertIsInstance(m["launches"], int)
        self.assertIsInstance(m["wall_ms"], float)

    def test_tolerates_fields_it_does_not_know(self):
        """Adding a field to the marker must not break the parser.

        The previous implementation was a positional regex over the whole line, so introducing
        warmup= made every perf test fail to parse its own output.
        """
        out = (
            "[kr-perf] ballast_mb=64 launches=8 warmup=1 wall_ms=99.0 counter=8 future=7"
        )
        m = perf_stats.parse_marker(out, "kr-perf")
        self.assertEqual(m["wall_ms"], 99.0)
        self.assertEqual(m["warmup"], 1)
        self.assertEqual(m["future"], 7)

    def test_still_reads_a_marker_missing_the_newer_fields(self):
        m = perf_stats.parse_marker("[kr-perf] launches=8 wall_ms=42.0", "kr-perf")
        self.assertEqual(m["wall_ms"], 42.0)
        self.assertNotIn("warmup", m)

    def test_picks_the_requested_tag_when_several_are_present(self):
        out = "[qh-perf] wall_ms=10.0\n[kr-perf] wall_ms=20.0"
        self.assertEqual(perf_stats.parse_marker(out, "kr-perf")["wall_ms"], 20.0)
        self.assertEqual(perf_stats.parse_marker(out, "qh-perf")["wall_ms"], 10.0)

    def test_ignores_the_pass_line_which_has_no_key_values(self):
        out = "[kr-perf] wall_ms=5.0\n[kr-perf] PASS"
        self.assertEqual(perf_stats.parse_marker(out, "kr-perf")["wall_ms"], 5.0)

    def test_missing_marker_is_an_error_rather_than_an_empty_result(self):
        with self.assertRaises(AssertionError):
            perf_stats.parse_marker("nothing here", "kr-perf")

    def test_marker_for_a_different_tag_does_not_satisfy_the_request(self):
        with self.assertRaises(AssertionError):
            perf_stats.parse_marker("[qh-perf] wall_ms=1.0", "kr-perf")

    def test_finds_the_marker_among_unrelated_output(self):
        out = "loading...\nwarning: something\n[kr-perf] wall_ms=7.5\ndone\n"
        self.assertEqual(perf_stats.parse_marker(out, "kr-perf")["wall_ms"], 7.5)


class RepeatMeasure(unittest.TestCase):
    """Repetition plus median is what makes a shared-runner measurement usable."""

    @staticmethod
    def _sequence(values):
        it = iter(values)
        return lambda: next(it)

    def test_warmup_samples_are_discarded(self):
        # 500.0 is a cold first run; it must not reach the reported median.
        st = perf_stats.repeat_measure(
            self._sequence([500.0, 10.0, 12.0, 11.0]), repeat=3, warmup=1
        )
        self.assertEqual(st["median_ms"], 11.0)
        self.assertEqual(st["max_ms"], 12.0)

    def test_median_of_an_odd_number_of_samples(self):
        st = perf_stats.repeat_measure(
            self._sequence([30.0, 10.0, 20.0]), repeat=3, warmup=0
        )
        self.assertEqual(st["median_ms"], 20.0)

    def test_median_of_an_even_number_of_samples(self):
        st = perf_stats.repeat_measure(
            self._sequence([10.0, 20.0, 30.0, 40.0]), repeat=4, warmup=0
        )
        self.assertEqual(st["median_ms"], 25.0)

    def test_median_ignores_a_single_slow_outlier(self):
        """The reason for using the median rather than the mean."""
        st = perf_stats.repeat_measure(
            self._sequence([10.0, 10.5, 900.0, 10.2, 10.1]), repeat=5, warmup=0
        )
        self.assertLess(st["median_ms"], 11.0)

    def test_single_sample_is_allowed(self):
        st = perf_stats.repeat_measure(self._sequence([42.0]), repeat=1, warmup=0)
        self.assertEqual(st["median_ms"], 42.0)
        self.assertEqual(st["runs"], 1)

    def test_reports_min_max_and_run_count(self):
        st = perf_stats.repeat_measure(
            self._sequence([10.0, 30.0, 20.0]), repeat=3, warmup=0
        )
        self.assertEqual(st["min_ms"], 10.0)
        self.assertEqual(st["max_ms"], 30.0)
        self.assertEqual(st["runs"], 3)

    def test_spread_is_the_max_over_min_ratio(self):
        st = perf_stats.repeat_measure(
            self._sequence([10.0, 20.0, 15.0]), repeat=3, warmup=0
        )
        self.assertAlmostEqual(st["spread"], 2.0)

    def test_spread_of_identical_samples_is_one(self):
        st = perf_stats.repeat_measure(
            self._sequence([12.0, 12.0, 12.0]), repeat=3, warmup=0
        )
        self.assertAlmostEqual(st["spread"], 1.0)

    def test_all_samples_are_retained_for_reporting(self):
        st = perf_stats.repeat_measure(
            self._sequence([10.0, 30.0, 20.0]), repeat=3, warmup=0
        )
        self.assertEqual(sorted(st["samples_ms"]), [10.0, 20.0, 30.0])

    def test_repeat_below_one_is_rejected(self):
        with self.assertRaises(AssertionError):
            perf_stats.repeat_measure(self._sequence([1.0]), repeat=0, warmup=0)

    def test_a_failing_run_propagates_rather_than_being_averaged_away(self):
        def boom():
            raise RuntimeError("workload failed")

        with self.assertRaises(RuntimeError):
            perf_stats.repeat_measure(boom, repeat=3, warmup=0)


class Ceilings(unittest.TestCase):
    """Absolute wall-time ceilings are advisory by default; relative checks gate CI."""

    def setUp(self):
        os.environ.pop("ROCPROFILER_PERF_STRICT_CEILING", None)

    tearDown = setUp

    def test_value_under_the_ceiling_passes(self):
        self.assertTrue(perf_stats.check_ceiling(10.0, 100.0, "under"))

    def test_value_over_the_ceiling_is_only_a_warning_by_default(self):
        self.assertFalse(perf_stats.check_ceiling(200.0, 100.0, "over"))

    def test_value_over_the_ceiling_fails_when_strict(self):
        os.environ["ROCPROFILER_PERF_STRICT_CEILING"] = "1"
        with self.assertRaises(AssertionError):
            perf_stats.check_ceiling(200.0, 100.0, "over")

    def test_value_under_the_ceiling_passes_when_strict(self):
        os.environ["ROCPROFILER_PERF_STRICT_CEILING"] = "1"
        self.assertTrue(perf_stats.check_ceiling(10.0, 100.0, "under"))

    def test_value_exactly_at_the_ceiling_passes(self):
        os.environ["ROCPROFILER_PERF_STRICT_CEILING"] = "1"
        self.assertTrue(perf_stats.check_ceiling(100.0, 100.0, "exact"))

    def test_strict_mode_is_off_unless_asked_for(self):
        self.assertFalse(perf_stats.strict_ceilings())

    def test_strict_mode_accepts_the_usual_truthy_spellings(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            os.environ["ROCPROFILER_PERF_STRICT_CEILING"] = value
            self.assertTrue(perf_stats.strict_ceilings(), value)

    def test_strict_mode_accepts_the_usual_falsy_spellings(self):
        for value in ("0", "false", "no", "off", ""):
            os.environ["ROCPROFILER_PERF_STRICT_CEILING"] = value
            self.assertFalse(perf_stats.strict_ceilings(), value)


class ResultWriting(unittest.TestCase):
    """Results are written only when a destination was actually requested."""

    ENV = "ROCPROFILER_TEST_RESULTS_JSON"

    def setUp(self):
        os.environ.pop(self.ENV, None)

    tearDown = setUp

    def test_unset_destination_writes_nothing(self):
        """Path(os.environ.get(VAR, "")) is PosixPath('.'), which is truthy.

        The earlier code took that branch and wrote results into whatever the current directory
        happened to be, which on CI is the build tree.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertIsNone(perf_stats.write_results(self.ENV, {"a": 1}))
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)

    def test_empty_destination_writes_nothing(self):
        os.environ[self.ENV] = ""
        self.assertIsNone(perf_stats.write_results(self.ENV, {"a": 1}))

    def test_whitespace_destination_writes_nothing(self):
        os.environ[self.ENV] = "   "
        self.assertIsNone(perf_stats.write_results(self.ENV, {"a": 1}))

    def test_writes_json_that_round_trips(self):
        payload = {"scaling_ratio": 1.5, "runs": 3, "label": "P=5"}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "results.json"
            os.environ[self.ENV] = str(dest)
            written = perf_stats.write_results(self.ENV, payload)
            self.assertEqual(written, dest)
            self.assertEqual(json.loads(dest.read_text()), payload)

    def test_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nested" / "deeper" / "results.json"
            os.environ[self.ENV] = str(dest)
            perf_stats.write_results(self.ENV, {"a": 1})
            self.assertTrue(dest.is_file())

    def test_surrounding_whitespace_in_the_path_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "results.json"
            os.environ[self.ENV] = f"  {dest}  "
            self.assertEqual(perf_stats.write_results(self.ENV, {"a": 1}), dest)
            self.assertTrue(dest.is_file())


class KernelReplayCostModel(unittest.TestCase):
    """The replay ceiling follows bytes moved: dispatches x footprint x passes."""

    def test_more_passes_costs_more(self):
        self.assertLess(
            kr_cost_model.model_max_ms(64, 8, 1), kr_cost_model.model_max_ms(64, 8, 5)
        )

    def test_more_dispatches_costs_more(self):
        self.assertLess(
            kr_cost_model.model_max_ms(64, 8, 5), kr_cost_model.model_max_ms(64, 16, 5)
        )

    def test_a_bigger_footprint_costs_more(self):
        self.assertLess(
            kr_cost_model.model_max_ms(64, 8, 5), kr_cost_model.model_max_ms(128, 8, 5)
        )

    def test_the_ceiling_includes_a_fixed_floor(self):
        """Attach and context setup do not scale with the workload, so a tiny workload must
        still be allowed more than zero milliseconds."""
        self.assertGreaterEqual(
            kr_cost_model.model_max_ms(1, 1, 1), kr_cost_model.FIXED_REPLAY_OVERHEAD_MS
        )

    def test_a_slower_assumed_link_raises_the_ceiling(self):
        fast = kr_cost_model.model_max_ms(64, 8, 5, min_gbps=50.0)
        slow = kr_cost_model.model_max_ms(64, 8, 5, min_gbps=4.0)
        self.assertGreater(slow, fast)

    def test_the_ceiling_is_proportional_to_the_margin(self):
        tight = kr_cost_model.model_max_ms(64, 8, 5, margin=1.0)
        loose = kr_cost_model.model_max_ms(64, 8, 5, margin=10.0)
        self.assertGreater(loose, tight)

    def test_pass_scaling_bound_is_linear_times_slack(self):
        self.assertAlmostEqual(kr_cost_model.max_pass_scaling_ratio(1, 5, 2.0), 10.0)
        self.assertAlmostEqual(kr_cost_model.max_pass_scaling_ratio(2, 8, 1.5), 6.0)

    def test_pass_scaling_bound_survives_a_zero_baseline(self):
        self.assertEqual(kr_cost_model.max_pass_scaling_ratio(0, 4, 2.0), 8.0)

    def test_equal_pass_counts_bound_at_the_slack_alone(self):
        self.assertAlmostEqual(kr_cost_model.max_pass_scaling_ratio(4, 4, 2.0), 2.0)


class QueueHooksCostModel(unittest.TestCase):
    """Interposition cost is per dispatch and should stay flat as dispatches grow."""

    def test_more_dispatches_costs_more(self):
        self.assertLess(qh_cost_model.model_max_ms(8), qh_cost_model.model_max_ms(32))

    def test_the_ceiling_includes_a_fixed_floor(self):
        self.assertGreaterEqual(
            qh_cost_model.model_max_ms(0), qh_cost_model.FIXED_OVERHEAD_MS
        )

    def test_the_two_models_are_actually_distinct(self):
        """Guards the loader above: if both names resolved to the same file, every queue-hooks
        assertion here would be re-testing the kernel-replay model."""
        self.assertNotEqual(kr_cost_model.__file__, qh_cost_model.__file__)
        self.assertTrue(hasattr(qh_cost_model, "max_launch_scaling_ratio"))
        self.assertTrue(hasattr(kr_cost_model, "max_pass_scaling_ratio"))

    def test_launch_scaling_bound_is_linear_times_slack(self):
        self.assertAlmostEqual(qh_cost_model.max_launch_scaling_ratio(8, 32, 2.0), 8.0)

    def test_launch_scaling_bound_survives_a_zero_baseline(self):
        self.assertEqual(qh_cost_model.max_launch_scaling_ratio(0, 16, 2.0), 32.0)

    def test_a_higher_per_dispatch_allowance_raises_the_ceiling(self):
        self.assertGreater(
            qh_cost_model.model_max_ms(16, per_dispatch_ms=10.0),
            qh_cost_model.model_max_ms(16, per_dispatch_ms=1.0),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
