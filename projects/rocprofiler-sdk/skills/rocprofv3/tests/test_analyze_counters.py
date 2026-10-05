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

"""GPU-free unit tests for the rocprofv3 counter analysis script."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_counters.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_counters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_module()


def make_db(path: Path, rows, product="AMD Instinct MI300X") -> None:
    """rows: (dispatch_id, kernel, counter, value, duration_ns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE _cc (pid INT, dispatch_id INT, kernel_name TEXT, counter_name TEXT,
            value REAL, duration INT, vgpr_count INT, accum_vgpr_count INT, sgpr_count INT,
            lds_block_size INT, scratch_size INT, workgroup_size INT, grid_size INT);
        CREATE TABLE _agents (absolute_index INT, name TEXT, product_name TEXT, extdata TEXT,
            type TEXT);
        CREATE VIEW counters_collection AS SELECT * FROM _cc;
        CREATE VIEW rocpd_info_agent AS SELECT * FROM _agents;
        """)
    conn.execute(
        "INSERT INTO _agents VALUES (2, 'gfx942', ?, ?, 'GPU')",
        (product, json.dumps({"wave_front_size": 64})),
    )
    for dispatch_id, kernel, counter, value, duration in rows:
        conn.execute(
            "INSERT INTO _cc VALUES (7, ?, ?, ?, ?, ?, 12, 0, 32, 0, 0, 256, 1048576)",
            (dispatch_id, kernel, counter, value, duration),
        )
    conn.commit()
    conn.close()


class PeakLookupTest(unittest.TestCase):
    def test_product_names(self):
        self.assertEqual(analyze.resolve_peak("AMD Instinct MI325X", None), 6.0)
        self.assertEqual(analyze.resolve_peak("AMD Instinct MI300X", None), 5.3)
        self.assertIsNone(analyze.resolve_peak("AMD Radeon PRO W7900", None))
        self.assertEqual(analyze.resolve_peak("anything", 2.5), 2.5)


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

    def test_multipass_merge_and_derived_metrics(self):
        duration = 100_000  # 100 us
        make_db(
            self.dir / "pass_1" / "run_results.db",
            [
                (1, "strided", "SQ_WAVES", 1000, duration),
                (1, "strided", "TCC_HIT_sum", 10, duration),
                (1, "strided", "TCC_MISS_sum", 90, duration),
                (1, "strided", "SQ_INSTS_VMEM_RD", 1000, duration),
                (1, "strided", "SQ_INSTS_VMEM_WR", 1000, duration),
                (1, "strided", "SQ_INSTS_VALU", 20000, duration),
            ],
        )
        # 400,000 KiB in 100 us is 4096 GB/s, 77% of the MI300X 5.3 TB/s peak.
        make_db(
            self.dir / "pass_2" / "run_results.db",
            [(1, "strided", "FETCH_SIZE", 400_000, duration)],
        )
        make_db(
            self.dir / "pass_3" / "run_results.db",
            [(1, "strided", "WRITE_SIZE", 0, duration)],
        )

        report = analyze.build_report([self.dir], None)
        kernel = report.kernels["strided"]
        self.assertEqual(kernel.dispatches, 1)
        derived = analyze.derive(kernel, report)
        self.assertAlmostEqual(derived.l2_hit, 0.10)
        self.assertAlmostEqual(derived.dram_gbps, 400_000 * 1024 / duration)
        self.assertGreater(derived.peak_fraction, analyze.HBM_BOUND)
        flags = " ".join(derived.flags)
        self.assertIn("L2 hit rate 10%", flags)
        self.assertIn("memory-bandwidth-bound", flags)
        self.assertIn("coalesced", flags)  # 409.6 MB / 2000 insts >> 1024 B

        text = analyze.render(report, top=5, kernel_filter=None)
        self.assertIn("AMD Instinct MI300X (gfx942)", text)
        self.assertIn("Instructions per wave | VALU 20.0", text)

    def test_rdna_l2_counters(self):
        make_db(
            self.dir / "run_results.db",
            [(1, "k", "GL2C_HIT_sum", 30, 1000), (1, "k", "GL2C_MISS_sum", 70, 1000)],
            product="AMD Radeon PRO W7900",
        )
        report = analyze.build_report([self.dir], None)
        derived = analyze.derive(report.kernels["k"], report)
        self.assertAlmostEqual(derived.l2_hit, 0.30)
        self.assertIsNone(report.peak_tbps)

    def test_percent_over_100_and_low_occupancy(self):
        make_db(
            self.dir / "run_results.db",
            [
                (1, "k", "VALUBusy", 150.0, 1000),
                (1, "k", "OccupancyPercent", 10.0, 1000),
                (1, "k", "LDSBankConflict", 5.0, 1000),
            ],
        )
        report = analyze.build_report([self.dir / "run_results.db"], None)
        flags = " ".join(analyze.derive(report.kernels["k"], report).flags)
        self.assertIn("VALUBusy reads 150%", flags)
        self.assertIn("OccupancyPercent 10%", flags)
        self.assertIn("LDSBankConflict", flags)

    def test_csv_input_with_agent_info(self):
        out_dir = self.dir / "conv"
        out_dir.mkdir()
        with (out_dir / "p_counter_collection_trace.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Dispatch_Id",
                    "Agent_Id",
                    "Process_Id",
                    "Kernel_Name",
                    "Vgpr_Count",
                    "Counter_Name",
                    "Counter_Value",
                    "Start_Timestamp",
                    "End_Timestamp",
                ]
            )
            writer.writerow([1, "Agent 2", 7, "kern", 8, "TCC_HIT_sum", 75, 0, 1000])
            writer.writerow([1, "Agent 2", 7, "kern", 8, "TCC_MISS_sum", 25, 0, 1000])
        with (out_dir / "p_agent_info.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Agent_Type", "Name", "Product_Name", "Wave_Front_Size"])
            writer.writerow(["CPU", "cpu", "Xeon", 0])
            writer.writerow(["GPU", "gfx942", "AMD Instinct MI325X", 64])
        code, out, _ = self.run_main(str(out_dir))
        self.assertEqual(code, 0)
        self.assertIn("AMD Instinct MI325X (gfx942)", out)
        self.assertIn("| L2 hit rate | 75.0% |", out)
        self.assertIn("6 TB/s", out)

    def test_kernel_filter(self):
        make_db(
            self.dir / "run_results.db",
            [(1, "alpha", "SQ_WAVES", 1, 10), (2, "beta", "SQ_WAVES", 1, 10)],
        )
        code, out, _ = self.run_main(str(self.dir), "--kernel", "beta")
        self.assertEqual(code, 0)
        self.assertIn("## beta", out)
        self.assertNotIn("## alpha", out)

    def test_no_counter_records_is_an_error(self):
        make_db(self.dir / "run_results.db", [])
        code, _, err = self.run_main(str(self.dir))
        self.assertEqual(code, 1)
        self.assertIn("no counter records", err)

    def test_trace_only_database_is_rejected(self):
        db = self.dir / "trace.db"
        sqlite3.connect(db).execute("CREATE VIEW kernels AS SELECT 1").connection.close()
        code, _, err = self.run_main(str(db))
        self.assertEqual(code, 1)
        self.assertIn("not a rocprofv3 rocpd database", err)


if __name__ == "__main__":
    unittest.main()
