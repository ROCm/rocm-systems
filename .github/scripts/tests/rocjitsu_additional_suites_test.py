# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import json
import os
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
            for path in ("bin/all_reduce_perf", "bin/rocshmem_functional_tests",
                         "bin/rocshmem/CTestTestfile.cmake", "share/rocshmem/test_wrapper.sh"):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            for backend, count in (("rocjitsu", "1"), ("native-gfx942", "2")):
                command, scope = suites.suite_command("rccl", root, root, backend)
                self.assertEqual(command[command.index("-g") + 1], count)
                self.assertIn("topology differs", scope)
            command, scope = suites.suite_command("rocshmem", root, root, "rocjitsu")
            self.assertEqual(command[command.index("-a") + 1], "130")
            self.assertIn("no inter-GPU communication", scope)
            manifest = json.dumps({"tests": [{"name": name} for name in suites.ROCSHMEM_NATIVE_TESTS]})
            with mock.patch.object(suites.subprocess, "check_output", return_value=manifest) as discover:
                command, scope = suites.suite_command("rocshmem", root, root, "native-gfx942")
            self.assertIn("--show-only=json-v1", discover.call_args.args[0])
            self.assertIn("^(init_n2_w1_z1_uuid|p_n2_w1_z1_128B_uuid)$", command)
            self.assertIn("--no-tests=error", command)
            for tests in ([], [{"name": "unit_tests_n4"}], [{"name": suites.ROCSHMEM_NATIVE_TESTS[0]}]):
                with mock.patch.object(suites.subprocess, "check_output", return_value=json.dumps({"tests": tests})):
                    with self.assertRaises(RuntimeError):
                        suites.suite_command("rocshmem", root, root, "native-gfx942")

    def test_rocshmem_requires_actual_selected_test_execution(self):
        marker = "1: ### Creating Test:\tHost_Amo_Self\tB=ipc PE=1 W=1 Z=1 ###\n"
        row = "1: 8              8              1                  0.00                inf                inf\n"
        self.assertTrue(suites.has_executed_tests(marker + row, "rocshmem"))
        for log in ("", marker, row, marker.replace("B=ipc", "B=gda") + row,
                    marker + row.replace("8              1", "8              0")):
            self.assertFalse(suites.has_executed_tests(log, "rocshmem"))
        passed = [f"1: 1/2 Test #1: {name} .... Passed 0.2 sec\n" for name in suites.ROCSHMEM_NATIVE_TESTS]
        self.assertTrue(suites.has_executed_tests("".join(passed), "rocshmem", "native-gfx942"))
        self.assertFalse(suites.has_executed_tests(passed[0], "rocshmem", "native-gfx942"))
        self.assertFalse(suites.has_executed_tests(passed[0] + passed[1].replace("Passed", "***Skipped"),
                                                 "rocshmem", "native-gfx942"))

    def test_missing_artifact_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                suites.suite_command("hipfile", Path(tmp), Path(tmp))

    def test_rocshmem_simulator_environment_and_failed_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "bin/rocshmem_functional_tests").touch()
            argv = ["adapter", "--suite", "rocshmem", "--backend", "rocjitsu",
                    "--rocm-root", tmp, "--log-dir", str(root / "logs"),
                    "--rocjitsu", "simulator", "--config", "gfx942"]
            result = {"passed": False, "log_lines": [
                "1: ### Creating Test: Host_Amo_Self B=ipc PE=1 W=1 Z=1 ###",
                "1: 8 8 1 0.00 inf inf"]}
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(os.environ, {"ROCSHMEM_BACKEND": "gda", "ROCSHMEM_SLR_NP": "8"}), \
                    mock.patch.object(suites.runner, "run_under_rocjitsu", return_value=result) as run:
                self.assertEqual(suites.main(), 1)
                env = run.call_args.args[5]
                self.assertEqual(env["ROCSHMEM_BACKEND"], "ipc")
                self.assertEqual(env["ROCSHMEM_SLR_NP"], "1")
                self.assertEqual(env["ROCSHMEM_TEST_UUID"], "1")
                self.assertEqual(env["ROCSHMEM_HEAP_SIZE"], "67108864")
                self.assertEqual(os.environ["ROCSHMEM_BACKEND"], "gda")
            self.assertEqual((root / "logs/rocshmem.result").read_text(), "failed\n")

    def test_wrong_therock_revision_rejected(self):
        with mock.patch.object(suites.subprocess, "check_output", return_value="wrong\n"):
            with self.assertRaises(RuntimeError):
                suites.pinned_script(Path("unused"), "test_runner.py")


if __name__ == "__main__":
    unittest.main()
