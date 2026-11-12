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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free unit tests for the rocprofv3 trace analysis script."""

from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_trace.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_module()

US = 1_000
MIB = 1024 * 1024


def make_db(path: Path, kernels, copies=(), regions=(), scratch=0) -> None:
    """Build a minimal database exposing the rocpd views the script reads."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE _kernels (name TEXT, nid INT, pid INT, agent_abs_index INT, agent_type TEXT,
            start INT, end INT, duration INT, vgpr_count INT, accum_vgpr_count INT,
            sgpr_count INT, lds_size INT, scratch_size INT, workgroup_x INT, workgroup_y INT,
            workgroup_z INT, grid_x INT, grid_y INT, grid_z INT);
        CREATE TABLE _copies (name TEXT, nid INT, pid INT, start INT, end INT, duration INT,
            size INT, src_agent_type TEXT, dst_agent_type TEXT, src_agent_abs_index INT,
            dst_agent_abs_index INT);
        CREATE TABLE _regions (category TEXT, name TEXT, pid INT, start INT, end INT, duration INT);
        CREATE TABLE _processes (pid INT, command TEXT, start INT, end INT);
        CREATE TABLE _agents (absolute_index INT, name TEXT, product_name TEXT,
            type_index INT, type TEXT);
        CREATE VIEW kernels AS SELECT * FROM _kernels;
        CREATE VIEW memory_copies AS SELECT * FROM _copies;
        CREATE VIEW regions AS SELECT * FROM _regions;
        CREATE VIEW processes AS SELECT * FROM _processes;
        CREATE VIEW rocpd_info_agent AS SELECT * FROM _agents;
        """)
    conn.execute("INSERT INTO _processes VALUES (42, './app', 0, 10000000)")
    conn.execute("INSERT INTO _agents VALUES (0, 'cpu', 'Xeon', 0, 'CPU')")
    conn.execute(
        "INSERT INTO _agents VALUES (2, 'gfx942', 'AMD Instinct MI300X', 0, 'GPU')"
    )
    for name, start, end in kernels:
        conn.execute(
            "INSERT INTO _kernels VALUES (?, 0, 42, 2, 'GPU', ?, ?, ?, 32, 0, 16, 0, ?, 256, 1, 1, 65536, 1, 1)",
            (name, start, end, end - start, scratch),
        )
    for name, start, end, size, src, dst in copies:
        conn.execute(
            "INSERT INTO _copies VALUES (?, 0, 42, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                start,
                end,
                end - start,
                size,
                src,
                dst,
                0 if src == "CPU" else 2,
                0 if dst == "CPU" else 2,
            ),
        )
    for region in regions:
        category, name, duration = region[:3]
        start = region[3] if len(region) > 3 else 0
        conn.execute(
            "INSERT INTO _regions VALUES (?, ?, 42, ?, ?, ?)",
            (category, name, start, start + duration, duration),
        )
    conn.commit()
    conn.close()


class MergeIntervalsTest(unittest.TestCase):
    def test_overlapping_intervals_are_unioned(self):
        spans = list(analyze.merge_intervals([(0, 10, "a"), (5, 20, "b"), (30, 40, "c")]))
        self.assertEqual(spans, [(0, 20, "a", "b"), (30, 40, "c", "c")])

    def test_busy_time_does_not_double_count(self):
        self.assertEqual(
            analyze.busy_time([(0, 10, "a"), (2, 8, "b"), (20, 25, "c")]), 15
        )


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = analyze.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_breakdown_hot_kernels_and_gaps(self):
        db = self.dir / "run_results.db"
        make_db(
            db,
            kernels=[("big_kernel", 0, 800 * US), ("small_kernel", 900 * US, 1000 * US)],
        )
        report = analyze.build_report([db], top=5)
        self.assertEqual(len(report.timelines), 1)
        timeline = report.timelines[0]
        self.assertEqual(timeline.window_ns, 1000 * US)
        self.assertEqual(timeline.any_busy_ns, 900 * US)
        self.assertEqual(timeline.gaps[0].length_ns, 100 * US)
        self.assertEqual(timeline.gaps[0].before, "big_kernel")
        self.assertEqual(timeline.gaps[0].after, "small_kernel")

        text = analyze.render(report, top=5)
        self.assertIn("AMD Instinct MI300X", text)
        self.assertIn("| 1 | big_kernel | 1 |", text)
        self.assertIn("GPU idle 10%", text)
        self.assertIn("**Hot kernels**", text)

    def test_launch_bound_and_scratch_findings(self):
        db = self.dir / "run_results.db"
        kernels = [("tiny", i * 10 * US, i * 10 * US + 2 * US) for i in range(150)]
        make_db(db, kernels=kernels, scratch=128)
        text = analyze.render(analyze.build_report([db], top=5), top=5)
        self.assertIn("Launch-bound candidate", text)
        self.assertIn("Scratch memory in use", text)

    def test_slow_pageable_copy_and_copy_share(self):
        db = self.dir / "run_results.db"
        make_db(
            db,
            kernels=[("k", 50_000 * US, 51_000 * US)],
            copies=[
                ("MEMORY_COPY_HOST_TO_DEVICE", 0, 40_000 * US, 64 * MIB, "CPU", "GPU")
            ],
            regions=[
                ("HIP_RUNTIME_API_EXT", "hipMemcpy", 45_000 * US),
                ("HIP_RUNTIME_API_EXT", "hipLaunchKernel", 10 * US),
            ],
        )
        report = analyze.build_report([db], top=5)
        text = analyze.render(report, top=5)
        self.assertIn("| Host to device (H2D) | 1 | 64.0 |", text)
        self.assertIn("Slow Host to device (H2D) copies", text)
        self.assertIn("Memory copies occupy", text)
        self.assertIn("Host mostly waits on the GPU", text)

    def test_markers_and_kfd_are_separated_from_api(self):
        db = self.dir / "run_results.db"
        make_db(
            db,
            kernels=[("k", 0, 10 * US)],
            regions=[
                ("MARKER_CORE_RANGE_API", "iteration", 5 * US),
                ("KFD_PAGE_MIGRATE", "migrate", 7 * US),
                ("HIP_RUNTIME_API_EXT", "hipLaunchKernel", 3 * US),
            ],
        )
        report = analyze.build_report([db], top=5)
        self.assertIn("iteration", report.markers)
        self.assertIn("KFD_PAGE_MIGRATE", report.kfd)
        self.assertEqual(list(report.apis), [("HIP_RUNTIME_API_EXT", "hipLaunchKernel")])
        self.assertIn("KFD driver events", analyze.render(report, top=5))

    def test_kfd_time_is_attributed_to_overlapping_api(self):
        db = self.dir / "run_results.db"
        make_db(
            db,
            kernels=[("k", 0, 10 * US)],
            regions=[
                ("HIP_RUNTIME_API_EXT", "hipMemcpy", 100 * US, 0),
                ("HIP_RUNTIME_API_EXT", "hipLaunchKernel", 5 * US, 200 * US),
                ("KFD_PAGE_MIGRATE", "prefetch", 20 * US, 10 * US),
            ],
        )
        report = analyze.build_report([db], top=5)
        self.assertEqual(report.kfd_overlap, {"hipMemcpy": 20 * US})
        self.assertIn(
            "mostly during `hipMemcpy` (100% of KFD time)", analyze.render(report, top=5)
        )

    def test_directory_input_and_multiple_databases(self):
        (self.dir / "rank0").mkdir()
        (self.dir / "rank1").mkdir()
        make_db(self.dir / "rank0" / "a_results.db", kernels=[("k", 0, 10 * US)])
        make_db(self.dir / "rank1" / "b_results.db", kernels=[("k", 0, 30 * US)])
        code, out, _ = self.run_main(str(self.dir))
        self.assertEqual(code, 0)
        self.assertIn("| 1 | k | 2 |", out)
        self.assertIn("Busy share across 2 GPU/process pairs", out)

    def test_rejects_non_rocpd_database(self):
        db = self.dir / "other.db"
        sqlite3.connect(db).execute("CREATE TABLE t (x INT)").connection.close()
        code, _, err = self.run_main(str(db))
        self.assertEqual(code, 1)
        self.assertIn("not a rocprofv3 rocpd database", err)

    def test_missing_input_reports_error(self):
        code, _, err = self.run_main(str(self.dir / "nope.db"))
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)


if __name__ == "__main__":
    unittest.main()
