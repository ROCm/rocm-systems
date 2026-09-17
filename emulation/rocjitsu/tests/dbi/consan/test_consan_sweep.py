#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import consan_sweep as sweep


class SampledSweepTest(unittest.TestCase):
    def test_budget_and_offsets_define_a_fixed_prefix(self):
        self.assertEqual(sweep.schedule(4, 8, [2, 0], [7, 1, 0], 4),
                         [(2, 7), (2, 1), (2, 0), (0, 7)])
        for args in [(4, 8, [4], [0], 1), (4, 8, [0], [8], 1),
                     (4, 8, [0], [0], 0), (4, 8, [0], [0], 4097)]:
            with self.assertRaises(ValueError):
                sweep.schedule(*args)

    def test_catalog_only_references_existing_complete_pairs(self):
        pair = {"code_object": "obj", "first_instruction": "0x10",
                "second_instruction": "0x20", "first_kind": "2", "second_kind": "1",
                "first_bytes": "[0,4)", "second_bytes": "[0,4)"}
        runs = [{"run": i, "evidence": {"examples": [dict(pair)]}} for i in (0, 1)]
        catalog = sweep.diagnostic_catalog(runs)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["observations"],
                         [{"run": 0, "example": 0}, {"run": 1, "example": 0}])
        # Two separate executions with only retained accesses cannot produce a pair.
        self.assertEqual(sweep.diagnostic_catalog([
            {"run": 0, "evidence": {"examples": [], "retained_site": "0x10"}},
            {"run": 1, "evidence": {"examples": [], "retained_site": "0x20"}},
        ]), [])
        for unresolved in ("unavailable", "ambiguous"):
            runs[1]["evidence"]["examples"][0]["second_instruction"] = unresolved
            self.assertEqual(len(sweep.diagnostic_catalog(runs)), 2)

    def test_serial_execution_records_settings_and_does_not_hide_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hook = root / "hook.so"
            hook.write_bytes(b"fake hook")
            program = root / "fixture.py"
            program.write_text('''import os
assert "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE" not in os.environ
assert "RJ_CONSAN_RUNTIME_SAMPLE_OFFSET" not in os.environ
wg = os.environ["RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET"]
cell = os.environ["RJ_CONSAN_CELL_SAMPLE_OFFSET"]
print("ConSan configuration workgroup_offset=" + wg + " cell_offset=" + cell)
print("ConSan auto report plan outcome=complete watchpoint_banks=8")
print("ConSan diagnostic map effective_banks_min=4 effective_banks_max=4")
print("ConSan auto report reader=1 visible=0 conflicts=0 saturated_windows=3 static_mapping_malformed=0")
print("ConSan analysis verdict applicable=true analysis_complete=true static_complete=true dynamic_complete=true")
''')
            output = root / "output"
            with mock.patch.dict(os.environ, {"RJ_CONSAN_RUNTIME_SAMPLE_STRIDE": "1",
                                               "RJ_CONSAN_RUNTIME_SAMPLE_OFFSET": "0"}):
                with redirect_stdout(io.StringIO()):
                    code = sweep.main(["--hook", str(hook), "--output", str(output),
                                       "--workgroup-stride", "4", "--cell-stride", "8",
                                       "--workgroup-offsets", "2,0", "--cell-offsets", "7,1,0",
                                       "--run-budget", "4", "--", sys.executable, str(program)])
            self.assertEqual(code, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["requested_combinations"], 6)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["completed_schedule"])
            self.assertEqual(len(summary["runs"]), 4)
            self.assertEqual(summary["diagnostic_catalog"], [])
            for row, (wg, cell) in zip(summary["runs"], [(2, 7), (2, 1), (2, 0), (0, 7)]):
                self.assertEqual(row["evidence"]["configurations"][0],
                                 {"workgroup_offset": str(wg), "cell_offset": str(cell)})
                self.assertEqual(row["evidence"]["reports"][0]["saturated_windows"], "3")
                self.assertEqual(row["evidence"]["reported_conflict_pairs"], 0)
                self.assertEqual(row["evidence"]["geometry"][1]["effective_banks_min"], "4")

    def test_missing_hook_evidence_stops_campaign_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hook = root / "hook.so"
            hook.write_bytes(b"fake hook")
            output = root / "output"
            with redirect_stdout(io.StringIO()):
                code = sweep.main(["--hook", str(hook), "--output", str(output),
                                   "--cell-offsets", "0,1", "--run-budget", "2", "--",
                                   sys.executable, "-c", "print('workload passed')"])
            self.assertEqual(code, 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["completed_schedule"])
            self.assertTrue(summary["stopped_on_failure"])
            self.assertEqual(len(summary["runs"]), 1)

    def test_timeout_reaps_process_and_retains_partial_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "timeout.log"
            code, timed_out, elapsed = sweep.execute(
                [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(60)"],
                os.environ.copy(), log, 0.2)
            self.assertTrue(timed_out)
            self.assertNotEqual(code, 0)
            self.assertLess(elapsed, 10)
            self.assertIn("started", log.read_text())


if __name__ == "__main__":
    unittest.main()
