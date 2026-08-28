#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Tests for the parts of the kernel replay performance harness that do not need a
GPU: how it builds rocprofv3 command lines, how it measures a child process, and
what it writes out. The harness decides whether a replay configuration looks
fast or slow, so a measurement bug in it is a wrong conclusion rather than a
failed test, and nothing on a machine with a GPU would notice.
"""

import os
import csv
import sys
import argparse
import importlib.util
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BENCHMARK_DIR = os.path.dirname(THIS_DIR)
HARNESS = os.path.join(BENCHMARK_DIR, "scripts", "replay_perf.py")

# a child that holds this much memory is far enough above the noise of an
# interpreter start-up to be recognisable in its peak RSS
CHILD_ALLOCATION_MIB = 256


def load_harness():
    spec = importlib.util.spec_from_file_location("replay_perf", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load_harness()


def python_cmd(statement):
    return [sys.executable, "-c", statement]


class TestCommandConstruction(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            rocprofv3="rocprofv3",
            app_cmd=["./kernel-replay", "1048576"],
            workdir="/tmp/work",
            output_format="rocpd",
        )

    def test_one_pmc_flag_per_group(self):
        args = harness._pmc_args(["GRBM_COUNT", "SQ_INSTS_SALU"])
        self.assertEqual(args.count("--pmc"), 2)
        # the shared counters are repeated in every group
        self.assertEqual(args.count("SQ_WAVES"), 2)

    def test_replay_flag_is_absent_unless_requested(self):
        cmd = harness._profile_cmd(self.args, ["GRBM_COUNT"], "out")
        self.assertNotIn(harness.REPLAY_FLAG, cmd)

    def test_replay_flag_precedes_the_application(self):
        cmd = harness._profile_cmd(self.args, ["GRBM_COUNT"], "out", replay=True)
        self.assertLess(cmd.index(harness.REPLAY_FLAG), cmd.index("--"))

    def test_application_is_last_and_separated(self):
        cmd = harness._profile_cmd(self.args, ["GRBM_COUNT"], "out")
        self.assertEqual(cmd[cmd.index("--") + 1 :], self.args.app_cmd)

    def test_extra_arguments_are_included(self):
        cmd = harness._profile_cmd(
            self.args, ["GRBM_COUNT"], "out", extra=["--kernel-include-regex", "^x$"]
        )
        self.assertIn("--kernel-include-regex", cmd)
        self.assertLess(cmd.index("^x$"), cmd.index("--"))

    def test_output_format_is_passed_through(self):
        self.args.output_format = "json"
        cmd = harness._profile_cmd(self.args, ["GRBM_COUNT"], "out")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")

    def test_output_directory_is_per_measurement(self):
        first = harness._profile_cmd(self.args, ["GRBM_COUNT"], "p1-replay")
        second = harness._profile_cmd(self.args, ["GRBM_COUNT"], "p2-3")
        self.assertIn("/tmp/work/p1-replay", first)
        self.assertIn("/tmp/work/p2-3", second)


class TestRun(unittest.TestCase):
    def test_wall_time_covers_the_child(self):
        wall, _, rc = harness._run(python_cmd("import time; time.sleep(0.25)"))
        self.assertEqual(rc, 0)
        self.assertGreaterEqual(wall, 0.25)

    def test_failure_is_reported_as_a_return_code(self):
        _, _, rc = harness._run(python_cmd("raise SystemExit(3)"))
        self.assertEqual(rc, 3)

    def test_signalled_child_is_reported_negatively(self):
        _, _, rc = harness._run(
            python_cmd("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
        )
        self.assertLess(rc, 0)

    def test_peak_rss_reflects_the_child(self):
        _, rss, rc = harness._run(
            python_cmd(f"x = bytearray({CHILD_ALLOCATION_MIB} * 1024 * 1024); len(x)")
        )
        self.assertEqual(rc, 0)
        self.assertGreater(rss, CHILD_ALLOCATION_MIB * 1024 * 0.5)

    def test_peak_rss_is_not_carried_over_from_an_earlier_child(self):
        # getrusage(RUSAGE_CHILDREN) reports a high water mark that never falls,
        # so a large run would otherwise set the figure for every run after it
        harness._run(
            python_cmd(f"x = bytearray({CHILD_ALLOCATION_MIB} * 1024 * 1024); len(x)")
        )
        _, rss, _ = harness._run(python_cmd("pass"))
        self.assertLess(rss, CHILD_ALLOCATION_MIB * 1024 * 0.5)

    def test_environment_is_passed_through(self):
        _, _, rc = harness._run(
            python_cmd(
                "import os, sys; sys.exit(0 if os.environ.get('KR') == 'x' else 1)"
            ),
            env={"KR": "x"},
        )
        self.assertEqual(rc, 0)


class TestRepeat(unittest.TestCase):
    def setUp(self):
        handle, self.counter = tempfile.mkstemp()
        os.close(handle)

    def tearDown(self):
        os.unlink(self.counter)

    def counting_cmd(self, exit_code=0):
        return python_cmd(
            f"open({self.counter!r}, 'a').write('x'); raise SystemExit({exit_code})"
        )

    def invocations(self):
        with open(self.counter) as ifs:
            return len(ifs.read())

    def test_warmup_runs_are_not_counted(self):
        stats = harness._repeat(self.counting_cmd(), repeats=2, warmup=3)
        self.assertEqual(self.invocations(), 5)
        self.assertEqual(stats["samples"], 2)

    def test_warmup_can_be_skipped(self):
        harness._repeat(self.counting_cmd(), repeats=2, warmup=0)
        self.assertEqual(self.invocations(), 2)

    def test_negative_warmup_is_treated_as_none(self):
        harness._repeat(self.counting_cmd(), repeats=1, warmup=-4)
        self.assertEqual(self.invocations(), 1)

    def test_median_of_the_timed_runs_is_reported(self):
        stats = harness._repeat(python_cmd("pass"), repeats=3, warmup=0)
        self.assertEqual(stats["samples"], 3)
        self.assertEqual(stats["failures"], 0)
        self.assertLessEqual(stats["min_s"], stats["median_s"])
        self.assertLessEqual(stats["median_s"], stats["max_s"])

    def test_a_run_that_always_fails_yields_nothing(self):
        self.assertIsNone(harness._repeat(self.counting_cmd(1), repeats=2, warmup=0))
        self.assertEqual(self.invocations(), 2)


class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "out.csv")
        self.recorder = harness.Recorder(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def stats(self):
        return {
            "median_s": 2.0,
            "min_s": 1.0,
            "max_s": 3.0,
            "peak_child_rss_kib": 1024,
            "failures": 0,
            "samples": 3,
        }

    def test_row_records_the_command_that_produced_it(self):
        row = self.recorder.add(
            "P1",
            "replay",
            harness.MODE_KERNEL_REPLAY,
            "4 groups",
            ["a", "b"],
            self.stats(),
        )
        self.assertEqual(row["command"], "a b")

    def test_failed_measurement_is_recorded_rather_than_dropped(self):
        row = self.recorder.add(
            "P1", "replay", harness.MODE_KERNEL_REPLAY, "4 groups", ["a"], None
        )
        self.assertEqual(row["failures"], "all")
        self.assertEqual(row["median_s"], "")

    def test_written_file_has_a_stable_column_set(self):
        self.recorder.add(
            "P1", "replay", harness.MODE_KERNEL_REPLAY, "4 groups", ["a"], self.stats()
        )
        self.recorder.add("P1", "base", harness.MODE_SINGLE_PASS, "1 group", ["b"], None)
        self.recorder.write()

        with open(self.path, newline="") as ifs:
            rows = list(csv.DictReader(ifs))

        self.assertEqual(list(rows[0]), harness.CSV_FIELDS)
        self.assertEqual([itr["mode"] for itr in rows], ["kernel-replay", "single-pass"])

    def test_nothing_is_written_when_nothing_was_measured(self):
        self.recorder.write()
        self.assertFalse(os.path.exists(self.path))


class TestArguments(unittest.TestCase):
    def base(self, *extra):
        return ["--app", "./kernel-replay", *extra]

    def test_defaults_include_a_warmup(self):
        self.assertEqual(harness.parse_args(self.base()).warmup, 1)

    def test_default_output_format_is_the_one_users_get(self):
        self.assertEqual(harness.parse_args(self.base()).output_format, "rocpd")

    def test_unknown_output_format_is_rejected(self):
        with self.assertRaises(SystemExit):
            harness.parse_args(self.base("--output-format", "parquet"))

    def test_repeats_must_be_positive(self):
        with self.assertRaises(SystemExit):
            harness.parse_args(self.base("--repeats", "0"))

    def test_warmup_cannot_be_negative(self):
        with self.assertRaises(SystemExit):
            harness.parse_args(self.base("--warmup", "-1"))

    def test_every_experiment_runs_by_default(self):
        self.assertEqual(
            harness.parse_args(self.base()).only, sorted(harness.EXPERIMENTS)
        )

    def test_dry_run_needs_neither_rocprofv3_nor_the_application(self):
        argv = self.base("--dry-run", "--rocprofv3", "/nonexistent/rocprofv3")
        self.assertEqual(harness.main(argv), 0)

    def test_missing_rocprofv3_is_an_error(self):
        argv = self.base("--rocprofv3", "/nonexistent/rocprofv3")
        self.assertEqual(harness.main(argv), 2)


class TestModeVocabulary(unittest.TestCase):
    """The CSV and the benchmark database have to describe a run the same way"""

    def test_modes_match_the_database_vocabulary(self):
        runner = os.path.join(BENCHMARK_DIR, "source", "bin", "rocprofv3-benchmark.py")
        spec = importlib.util.spec_from_file_location("rocprofv3_benchmark", runner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for args, mode in (
            (["--pmc", "SQ_WAVES"], harness.MODE_SINGLE_PASS),
            (["--pmc", "A", "--pmc", "B"], harness.MODE_MULTIPLEXED),
            (
                ["--pmc", "A", "--pmc", "B", harness.REPLAY_FLAG],
                harness.MODE_KERNEL_REPLAY,
            ),
        ):
            derived = module.derive_replay_config(args)
            self.assertEqual(derived["counter_collection_mode"], mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
