#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that expected GPU traps cannot hide unrelated child failures."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CheckedRunnerTest(unittest.TestCase):
    def run_case(self, code: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            pattern = Path(directory) / "pattern.txt"
            pattern.write_text("CONSAN_EXPECTED_DEVICE_TRAP|HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION")
            return subprocess.run(
                [
                    "cmake",
                    f"-DRJ_EXPECTED_REGEX_FILE={pattern}",
                    "-DRJ_EXPECTED_EXIT=gpu-trap",
                    "-P",
                    str(Path(__file__).with_name("run_checked_test.cmake")),
                    "--",
                    sys.executable,
                    "-c",
                    code,
                ],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_accepts_reported_dispatch_error(self):
        self.assertEqual(self.run_case("print('CONSAN_EXPECTED_DEVICE_TRAP')").returncode, 0)

    def test_accepts_queue_callback_abort(self):
        result = self.run_case(
            "import sys; print('HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION'); sys.exit(134)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_success_without_a_trap(self):
        self.assertNotEqual(self.run_case("print('passed')").returncode, 0)

    def test_rejects_unrelated_exit_even_with_trap_text(self):
        result = self.run_case("import sys; print('CONSAN_EXPECTED_DEVICE_TRAP'); sys.exit(1)")
        self.assertNotEqual(result.returncode, 0)

    def test_rejects_sanitizer_errors_even_with_trap_text(self):
        for diagnostic in ("ERROR: AddressSanitizer", "ERROR: LeakSanitizer", "runtime error:"):
            with self.subTest(diagnostic=diagnostic):
                result = self.run_case(
                    f"print('CONSAN_EXPECTED_DEVICE_TRAP'); print({diagnostic!r})"
                )
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
