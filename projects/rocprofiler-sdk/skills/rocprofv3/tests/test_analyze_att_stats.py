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

"""GPU-free unit tests for the rocprofv3 thread trace stats analysis script."""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_att_stats.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_att_stats", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_module()

HEADER = [
    "CodeObj",
    "Vaddr",
    "Instruction",
    "Hitcount",
    "Latency",
    "Stall",
    "Idle",
    "Source",
]
SRC = "/opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:255 -> /src/app.cpp:{line}"


def write_stats(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerow([2, 1024, "; _Z6kernelPf", 0, 0, 0, 0, "kernel(float*)"])
        for vaddr, text, hits, latency, stall, idle, line in rows:
            writer.writerow(
                [
                    2,
                    vaddr,
                    text,
                    hits,
                    latency,
                    stall,
                    idle,
                    SRC.format(line=line) if line else "",
                ]
            )


MEMORY_BOUND = [
    (1024, "s_load_dwordx2 s[0:1], s[4:5], 0x0", 64, 500, 0, 0, 10),
    (1032, "s_waitcnt lgkmcnt(0)", 64, 300, 300, 0, 10),
    (1040, "global_load_dword v1, v[0:1], off", 64, 4000, 3900, 0, 11),
    (1048, "s_waitcnt vmcnt(0)", 64, 5000, 5000, 0, 11),
    (1056, "v_add_f32_e32 v1, 1.0, v1", 64, 300, 10, 0, 12),
    (1060, "global_store_dword v[0:1], v1, off", 64, 600, 550, 0, 12),
    (1068, "s_endpgm", 64, 64, 0, 9000, 13),
    (1072, "v_cvt_f32_u32_e32 v4, s8", 0, 0, 0, 0, 12),
]


class ClassifyTest(unittest.TestCase):
    def test_instruction_classes(self):
        cases = {
            "s_waitcnt vmcnt(0)": "wait: vector memory (vmcnt)",
            "s_waitcnt lgkmcnt(0)": "wait: scalar memory / LDS (lgkmcnt)",
            "s_waitcnt vmcnt(0) lgkmcnt(0)": "wait: vector memory + scalar/LDS",
            "global_load_dwordx4 v[0:3], v[4:5], off": "VMEM load",
            "buffer_store_dword v1, off, s[0:3], 0": "VMEM store",
            "global_atomic_add_f32 v0, v1, s[0:1]": "VMEM atomic",
            "ds_read_b128 v[0:3], v4": "LDS",
            "s_load_dwordx4 s[0:3], s[4:5], 0x0": "SMEM (scalar load/store)",
            "v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]": "MFMA / WMMA",
            "v_fma_f32 v0, v1, v2, v3": "VALU",
            "s_barrier": "barrier",
            "s_cbranch_execz 12": "branch",
            "s_add_u32 s0, s1, s2": "SALU",
        }
        for text, expected in cases.items():
            self.assertEqual(analyze.classify(text), expected, text)


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

    def test_memory_bound_dispatch(self):
        write_stats(self.dir / "stats_ui_output_agent_1_dispatch_7.csv", MEMORY_BOUND)
        code, out, _ = self.run_main(str(self.dir))
        self.assertEqual(code, 0)
        self.assertIn("`kernel(float*)`", out)
        self.assertIn("Traced waves (hitcount of first instruction): 64", out)
        self.assertIn("| 1 | 0x418 | `s_waitcnt vmcnt(0)` |", out)
        self.assertIn("| 1 | /src/app.cpp:11 |", out)
        self.assertIn("Global memory accounts for", out)
        self.assertIn("Idle before s_endpgm", out)
        self.assertIn("Instructions executed: 7 of 8", out)

    def test_satisfied_wait_in_hot_loop_is_not_memory(self):
        loop = [
            (1024, "global_load_dword v1, v[0:1], off", 64, 200, 150, 0, 10),
            (1032, "s_waitcnt vmcnt(0)", 16384, 400_000, 400_000, 0, 11),
            (1036, "v_fmac_f32_e32 v3, 0x3f7ff972, v4", 16384, 300_000, 20_000, 0, 12),
            (1040, "s_cbranch_scc0 65525", 16384, 100_000, 0, 0, 12),
        ]
        write_stats(self.dir / "stats_loop.csv", loop)
        _, out, _ = self.run_main(str(self.dir / "stats_loop.csv"))
        self.assertIn("almost always already satisfied", out)
        self.assertNotIn("Global memory accounts for", out)
        self.assertIn("Vector ALU account for", out)

    def test_untraced_cu_reports_troubleshooting(self):
        write_stats(self.dir / "stats_x.csv", [(1024, "s_endpgm", 0, 0, 0, 0, 13)])
        code, out, _ = self.run_main(str(self.dir / "stats_x.csv"))
        self.assertEqual(code, 0)
        self.assertIn("No traced instruction executed", out)

    def test_missing_source_suggests_debug_build(self):
        write_stats(
            self.dir / "stats_y.csv", [(1024, "v_add_f32 v0, v1, v2", 8, 100, 0, 0, None)]
        )
        _, out, _ = self.run_main(str(self.dir / "stats_y.csv"))
        self.assertIn("rebuild the application with -g", out)

    def test_kernel_filter_and_bad_input(self):
        write_stats(self.dir / "stats_z.csv", MEMORY_BOUND)
        code, _, err = self.run_main(str(self.dir), "--kernel", "other_kernel")
        self.assertEqual(code, 1)
        self.assertIn("no traced dispatch matched", err)

        bogus = self.dir / "bogus.csv"
        bogus.write_text("a,b\n1,2\n", encoding="utf-8")
        code, _, err = self.run_main(str(bogus))
        self.assertEqual(code, 1)
        self.assertIn("not a thread trace stats CSV", err)


if __name__ == "__main__":
    unittest.main()
