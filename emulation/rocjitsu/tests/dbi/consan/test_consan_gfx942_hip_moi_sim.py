# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import consan_gfx942_hip_moi_sim as simulator


class Gfx942HipMoiSimulatorTest(unittest.TestCase):
    def test_registry_uses_exact_build_root_and_totals_fourteen_tests(self) -> None:
        self.assertEqual(
            sum(suite.expected_tests for suite in simulator.SUITES),
            simulator.EXPECTED_TESTS,
        )
        self.assertEqual(simulator.EXPECTED_TESTS, 14)
        self.assertEqual(len(simulator.SUITES), 6)
        self.assertEqual(
            {
                suite.id: str(simulator._executable_suffix(suite))
                for suite in simulator.SUITES
            },
            {
                "jakub-matmul": "tests/hip_moi_reference_cdna3_jakub_matmul",
                "mfma-attention": (
                    "tests/hip_moi_instrumented_cdna3_mfma_attention_block_test"
                ),
                "d128-block": (
                    "tests/hip_moi_instrumented_cdna3_d128_attention_block_test"
                ),
                "d128-pressure": (
                    "tests/hip_moi_instrumented_cdna3_d128_attention_pressure_test"
                ),
                "streamk-arrival": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_mfma_streamk_arrival_counter_test"
                ),
                "tree-atomic-or": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_mfma_streamk_tree_atomic_or_test"
                ),
            },
        )

    def test_suite_runs_through_rocjitsu_and_checks_passed_count(self) -> None:
        suite = simulator.SUITE_BY_ID["d128-block"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocjitsu = root / "rocjitsu"
            config = root / "gfx942.json"
            build = root / simulator.BUILD_DIR_NAME
            executable = build / simulator._executable_suffix(suite)
            executable.parent.mkdir(parents=True)
            executable.touch()
            rocjitsu.touch()
            config.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[  PASSED  ] 2 tests.\n",
                stderr="",
            )
            with mock.patch.object(
                simulator.subprocess, "run", return_value=completed
            ) as run:
                result = simulator._run_suite(
                    suite, rocjitsu, config, build, timeout=60.0
                )

        self.assertTrue(result.accepted)
        run.assert_called_once_with(
            [
                str(rocjitsu),
                "--config",
                str(config),
                "--",
                str(executable),
                "--gtest_brief=1",
            ],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )

    def test_suite_rejects_unexpected_test_count(self) -> None:
        suite = simulator.SUITE_BY_ID["d128-pressure"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / simulator.BUILD_DIR_NAME
            executable = build / simulator._executable_suffix(suite)
            executable.parent.mkdir(parents=True)
            executable.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[  PASSED  ] 3 tests.\n",
                stderr="",
            )
            with mock.patch.object(simulator.subprocess, "run", return_value=completed):
                result = simulator._run_suite(
                    suite,
                    root / "rocjitsu",
                    root / "gfx942.json",
                    build,
                    timeout=60.0,
                )

        self.assertFalse(result.accepted)
        self.assertEqual(result.passed_tests, 3)
        self.assertEqual(result.error, "expected 4 tests, observed 3")

    def test_all_suites_run_and_report_the_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocjitsu = root / "rocjitsu"
            config = root / "gfx942.json"
            build = root / simulator.BUILD_DIR_NAME
            rocjitsu.touch()
            config.touch()
            for suite in simulator.SUITES:
                executable = build / simulator._executable_suffix(suite)
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.touch()

            completed = [
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=f"[  PASSED  ] {suite.expected_tests} tests.\n",
                    stderr="",
                )
                for suite in simulator.SUITES
            ]
            output = io.StringIO()
            with (
                mock.patch.object(
                    simulator.subprocess, "run", side_effect=completed
                ) as run,
                contextlib.redirect_stdout(output),
            ):
                result = simulator.main(
                    [
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--hip-moi-build",
                        str(build),
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 6)
        self.assertIn("total: 14/14 tests", output.getvalue())

    def test_single_suite_count_mismatch_fails_the_ctest_entry(self) -> None:
        suite = simulator.SUITE_BY_ID["d128-pressure"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocjitsu = root / "rocjitsu"
            config = root / "gfx942.json"
            build = root / simulator.BUILD_DIR_NAME
            executable = build / simulator._executable_suffix(suite)
            executable.parent.mkdir(parents=True)
            executable.touch()
            rocjitsu.touch()
            config.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[  PASSED  ] 3 tests.\n",
                stderr="",
            )
            output = io.StringIO()
            with (
                mock.patch.object(
                    simulator.subprocess,
                    "run",
                    return_value=completed,
                ) as run,
                contextlib.redirect_stdout(output),
            ):
                result = simulator.main(
                    [
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--hip-moi-build",
                        str(build),
                        "--suite",
                        suite.id,
                    ]
                )

        self.assertEqual(result, 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn(
            "FAIL d128-pressure: expected 4 tests, observed 3",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
