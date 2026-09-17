#!/usr/bin/env python3

from __future__ import annotations

import os
import unittest
from unittest import mock

import consan_hipblaslt_benchmark_workload as workload


class ConSanHipblasltBenchmarkWorkloadTest(unittest.TestCase):
    def test_parses_performance_record_and_exact_kernel_name(self) -> None:
        output = """\
[0]:transA,transB,M,N,K,hipblaslt-Gflops,us,norm_error
    N,N,512,512,512,1234.5,21.25,0
    --Solution index: 42
    --Solution name:  fixture solution
    --kernel name:    Cijk_Ailk_Bljk_F8NH_BH_Bias_AS_SAB_SAV_UserArgs_MT128x128x32
[0]:transA,transB,M,N,K,hipblaslt-Gflops,us,norm_error
    N,N,512,512,512,1334.5,19.5,0
    --Solution index: 42
    --Solution name:  fixture solution
    --kernel name:    Cijk_Ailk_Bljk_F8NH_BH_Bias_AS_SAB_SAV_UserArgs_MT128x128x32
"""
        self.assertEqual(workload._performance_rows(output)[0]["us"], "21.25")
        self.assertEqual(
            workload._positive_floats(workload._performance_rows(output), "us"),
            (21.25, 19.5),
        )
        self.assertEqual(
            workload._nonnegative_floats(
                workload._performance_rows(output), "norm_error"
            ),
            (0.0, 0.0),
        )
        self.assertEqual(
            workload.KERNEL_RE.findall(output),
            [
                "Cijk_Ailk_Bljk_F8NH_BH_Bias_AS_SAB_SAV_UserArgs_MT128x128x32",
                "Cijk_Ailk_Bljk_F8NH_BH_Bias_AS_SAB_SAV_UserArgs_MT128x128x32",
            ],
        )

    def test_yaml_requests_two_identical_verified_operations(self) -> None:
        text = workload._yaml_text(
            workload.Path("/dist/hipblaslt_common.yaml"), 64, 32, 16, 7
        )
        self.assertEqual(text.count("function: matmul"), 2)
        self.assertEqual(text.count("norm_check: 1"), 2)
        self.assertEqual(text.count("use_gpu_timer: true"), 2)
        self.assertIn("M: 64", text)
        self.assertIn("N: 32", text)
        self.assertIn("K: 16", text)
        self.assertIn("iters: 7", text)

    def test_instrumentation_record_is_required_only_for_consan(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(workload._instrumentation_ms("native output"), 0.0)
        output = (
            "[rocjitsu-dbi-hooks] ConSan instrumentation timing " "total_ns=123456789\n"
        )
        with mock.patch.dict(
            os.environ,
            {"HSA_TOOLS_LIB": "/hook", "RJ_CONSAN_MODE": "default"},
            clear=True,
        ):
            self.assertEqual(workload._instrumentation_ms(output), 123.456789)
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                workload._instrumentation_ms("missing")


if __name__ == "__main__":
    unittest.main()
