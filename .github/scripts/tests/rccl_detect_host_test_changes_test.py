# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_detect_host_test_changes.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_detect_host_test_changes as detect


class TouchesHostTestsTest(unittest.TestCase):
    def test_rccl_paths_match(self):
        for path in [
            "projects/rccl/src/init.cc",
            "projects/rccl/test/host/CMakeLists.txt",
            "projects/rccl/tools/scripts/test_runner/test_runner.py",
            "projects/rccl/README.md",
        ]:
            with self.subTest(path=path):
                self.assertTrue(detect.touches_host_tests([path]))

    def test_ci_files_governing_the_gate_match(self):
        # A change to any of these can alter the gate's own verdict, so it has
        # to be tested by the PR making it.
        for path in [
            ".github/workflows/rccl-host-unit-tests.yml",
            ".github/scripts/rccl_detect_host_test_changes.py",
            ".github/scripts/rccl_change_sets.py",
            ".github/scripts/ci_utils.py",
        ]:
            with self.subTest(path=path):
                self.assertTrue(detect.touches_host_tests([path]))

    def test_unrelated_paths_do_not_match(self):
        for path in [
            "projects/rccl-tests/src/all_reduce.cu",
            "projects/rccl",
            "projects/hip/src/hip_memory.cpp",
            ".github/workflows/rccl-coco-pr.yml",
            "README.md",
        ]:
            with self.subTest(path=path):
                self.assertFalse(detect.touches_host_tests([path]))


class MainTest(unittest.TestCase):
    """The GITHUB_OUTPUT string is what the workflow actually consumes."""

    def _run_main(self, changed_files):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github_output"
            output.touch()
            with mock.patch.object(
                detect, "get_modified_paths", return_value=set(changed_files)
            ) as diff, mock.patch.dict(
                os.environ, {"GITHUB_OUTPUT": os.fspath(output)}
            ):
                self.assertEqual(detect.main(["--base-ref", "HEAD^"]), 0)
            diff.assert_called_once_with("HEAD^", no_renames=True)
            return output.read_text()

    def test_emits_false_for_an_untouched_pr(self):
        self.assertEqual(self._run_main(["projects/hip/x.cc"]), "rccl=false\n")

    def test_emits_true_for_a_touched_pr(self):
        changed = ["projects/hip/x.cc", "projects/rccl/src/init.cc"]
        self.assertEqual(self._run_main(changed), "rccl=true\n")

    def test_move_out_of_rccl_still_matches(self):
        # get_modified_paths(no_renames=True) reports both sides of the move.
        moved = ["projects/rccl/src/x.cc", "projects/other/x.cc"]
        self.assertEqual(self._run_main(moved), "rccl=true\n")


if __name__ == "__main__":
    unittest.main()
