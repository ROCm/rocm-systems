# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rocjitsu_additional_suites as suites


class AdditionalSuitesTest(unittest.TestCase):
    def test_wrapper_success_and_empty_or_skipped_tests_cannot_pass(self):
        for text in ("1/1 Test #1: amdsmi ... Passed", "1: [ PASSED ] 0 tests.",
                     "1: 4 skipped", "1: No tests were found!!!"):
            self.assertFalse(suites.has_executed_tests(text, "amdsmi"))

    def test_positive_execution_evidence(self):
        for suite, text in (
            ("amdsmi", "1: [  PASSED  ] 4 tests."),
            ("hipfile", "1: 1/4 Test #1: real_test .... Passed 0.3 sec"),
            ("aqlprofile", "1: 4 tests run"),
            ("rccl", "1: # Out of bounds values : 0 OK"),
        ):
            self.assertTrue(suites.has_executed_tests(text, suite))
        self.assertFalse(suites.has_executed_tests("# Out of bounds values : 3 FAILED", "rccl"))

    def test_sdk_generated_preload_preserves_simulator(self):
        module = mock.Mock()
        module.get_cmake_config_cmd.return_value = [
            "cmake", "-DROCPROFILER_MEMCHECK_PRELOAD_ENV=LD_PRELOAD=asan.so",
            "-DROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE=asan.so", "-DOTHER=unchanged",
        ]
        command = suites.sdk_config_command(module, "asan.so:librocjitsu.so")
        self.assertIn("-DROCPROFILER_MEMCHECK_PRELOAD_ENV=LD_PRELOAD=asan.so:librocjitsu.so", command)
        self.assertIn("-DROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE=asan.so:librocjitsu.so", command)
        self.assertIn("-DOTHER=unchanged", command)

    def test_topology_specific_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in ("bin/all_reduce_perf", "bin/rocshmem_unit_tests", "bin/rocshmem/CTestTestfile.cmake"):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            for backend, count in (("rocjitsu", "1"), ("native-gfx942", "2")):
                command, scope = suites.suite_command("rccl", root, root, backend)
                self.assertEqual(command[command.index("-g") + 1], count)
                self.assertIn("topology differs", scope)
            command, scope = suites.suite_command("rocshmem", root, root, "rocjitsu")
            self.assertIn("--gtest_filter=EnvVar*", command)
            self.assertIn("no GPU communication", scope)
            command, scope = suites.suite_command("rocshmem", root, root, "native-gfx942")
            self.assertIn("^unit_tests_n4$", command)
            self.assertIn("--no-tests=error", command)

    def test_missing_artifact_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                suites.suite_command("hipfile", Path(tmp), Path(tmp))

    def test_wrong_therock_revision_rejected(self):
        with mock.patch.object(suites.subprocess, "check_output", return_value="wrong\n"):
            with self.assertRaises(RuntimeError):
                suites.pinned_script(Path("unused"), "test_runner.py")


if __name__ == "__main__":
    unittest.main()
