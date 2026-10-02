#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check trap exits and the isolated independent-workgroup diagnostic oracle."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CheckedRunnerTest(unittest.TestCase):
    witness_regex_file: Path
    ctest_directory: Path

    def run_case(
        self,
        code: str,
        pattern_text: str = "CONSAN_EXPECTED_DEVICE_TRAP|HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION",
        expected_exit: str = "gpu-trap",
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            pattern = Path(directory) / "pattern.txt"
            pattern.write_text(pattern_text)
            return subprocess.run(
                [
                    "cmake",
                    f"-DRJ_EXPECTED_REGEX_FILE={pattern}",
                    f"-DRJ_EXPECTED_EXIT={expected_exit}",
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
        self.assertEqual(
            self.run_case("print('CONSAN_EXPECTED_DEVICE_TRAP')").returncode, 0
        )

    def test_accepts_queue_callback_abort(self):
        result = self.run_case(
            "import sys; print('HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION'); sys.exit(134)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_success_without_a_trap(self):
        self.assertNotEqual(self.run_case("print('passed')").returncode, 0)

    def test_rejects_unrelated_exit_even_with_trap_text(self):
        result = self.run_case(
            "import sys; print('CONSAN_EXPECTED_DEVICE_TRAP'); sys.exit(1)"
        )
        self.assertNotEqual(result.returncode, 0)

    def test_rejects_sanitizer_errors_even_with_trap_text(self):
        for diagnostic in (
            "ERROR: AddressSanitizer",
            "ERROR: LeakSanitizer",
            "runtime error:",
        ):
            with self.subTest(diagnostic=diagnostic):
                result = self.run_case(
                    f"print('CONSAN_EXPECTED_DEVICE_TRAP'); print({diagnostic!r})"
                )
                self.assertNotEqual(result.returncode, 0)

    def run_output(self, output: str) -> subprocess.CompletedProcess[str]:
        return self.run_case(
            f"print({output!r})", self.witness_regex_file.read_text(), expected_exit="0"
        )

    def test_accepts_origin_conflict_on_one_line(self):
        result = self.run_output(
            "ConSan conflict dispatch=2 workgroup=(0,0,0) owners=0,1"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_origin_access_after_another_workgroup_conflict(self):
        result = self.run_output(
            "ConSan conflict dispatch=2 workgroup=(1,1,1) owners=0,1\n"
            "ConSan access dispatch=2 workgroup=(0,0,0)"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test output did not match", result.stderr)

    def test_rejects_missing_witness_even_when_identity_case_has_a_conflict(self):
        manifest = subprocess.run(
            [
                "ctest",
                "--test-dir",
                str(self.ctest_directory),
                "--show-only=json-v1",
                "-R",
                r"\.ConSanIndependentWorkgroupsWitness\.Incorrect$",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        witness_cases = json.loads(manifest.stdout)["tests"]
        if not witness_cases:
            self.skipTest("HIP witness cases are not enabled in this configuration")
        identity = self.run_output(
            "ConSan conflict dispatch=1 workgroup=(0,0,0) owners=0,1"
        )
        self.assertEqual(identity.returncode, 0, identity.stderr)
        witness_output = (
            "ConSan access dispatch=2 workgroup=(0,0,0)\n"
            "ConSan summary diagnostics=0 conflicts=0"
        )
        for case in witness_cases:
            with self.subTest(case=case["name"]):
                command = case["command"]
                filters = [arg for arg in command if arg.startswith("--gtest_filter=")]
                self.assertEqual(
                    filters,
                    [
                        "--gtest_filter=ConSanDeviceIndependentWorkgroupsTest.WitnessIncorrect"
                    ],
                )
                pattern_files = [
                    arg.removeprefix("-DRJ_EXPECTED_REGEX_FILE=")
                    for arg in command
                    if arg.startswith("-DRJ_EXPECTED_REGEX_FILE=")
                ]
                self.assertEqual(len(pattern_files), 1)
                self.assertEqual(
                    Path(pattern_files[0]).read_text(),
                    self.witness_regex_file.read_text(),
                )
                # Keep the registered checked runner, but replay a report with
                # its diagnostic removed instead of launching the HIP child.
                # The identity process's conflict cannot satisfy this oracle.
                witness = subprocess.run(
                    command[: command.index("--") + 1]
                    + [sys.executable, "-c", f"print({witness_output!r})"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(witness.returncode, 0)
                self.assertIn("test output did not match", witness.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--witness-regex-file", type=Path, required=True)
    parser.add_argument("--ctest-directory", type=Path, required=True)
    arguments, unittest_arguments = parser.parse_known_args()
    CheckedRunnerTest.witness_regex_file = arguments.witness_regex_file
    CheckedRunnerTest.ctest_directory = arguments.ctest_directory
    unittest.main(argv=[sys.argv[0], *unittest_arguments])
