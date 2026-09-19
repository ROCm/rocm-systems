# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_detect_coco_changes.py.

A false from this script lets rccl-coco-pr.yml's gate pass a PR without running
anything on the clusters, so the paths that must resolve true are the ones
worth pinning.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_detect_coco_changes as detect


class CocoPathsTest(unittest.TestCase):
    def test_rccl_and_rccl_tests_both_match(self):
        # Sibling directories: a pattern for one does not cover the other.
        for path in [
            "projects/rccl/src/init.cc",
            "projects/rccl/tools/ci/run-device-api-ci.sh",
            "projects/rccl-tests/src/all_reduce.cu",
            "projects/rccl-tests/CMakeLists.txt",
        ]:
            with self.subTest(path=path):
                self.assertEqual(detect.coco_paths([path]), [path])

    def test_ci_files_governing_the_gate_match(self):
        # A change to any of these can alter the gate's own verdict, so it has
        # to be tested by the PR making it.
        for path in [
            ".github/workflows/rccl-coco-pr.yml",
            ".github/workflows/rccl-coco-scheduled.yml",
            ".github/workflows/rccl-coco-run.yml",
            ".github/scripts/rccl_coco_matrix.py",
            ".github/scripts/rccl_resolve_coco_run.py",
            ".github/actions/resolve-coco-run/action.yml",
            ".github/actions/checkout-coco-harness/action.yml",
            ".github/scripts/rccl_detect_coco_changes.py",
            ".github/scripts/ci_utils.py",
        ]:
            with self.subTest(path=path):
                self.assertEqual(detect.coco_paths([path]), [path])

    def test_unrelated_paths_do_not_match(self):
        for path in [
            "projects/hip/src/hip.cc",
            "projects/rocshmem/README.md",
            "projects/rdc/CMakeLists.txt",
            "docs/index.md",
            "README.md",
            ".github/workflows/therock-ci.yml",
            # Adjacent names a loose pattern would sweep in.
            "projects/rccl-other/src/x.cc",
            "projects/rcclfoo/x.cc",
        ]:
            with self.subTest(path=path):
                self.assertEqual(detect.coco_paths([path]), [])

    def test_doc_only_paths_are_excluded(self):
        # A README edit under projects/rccl must not book both clusters.
        for path in [
            "projects/rccl/README.md",
            "projects/rccl/docs/api.rst",
            "projects/rccl-tests/README.md",
            "projects/rccl/tools/ci/NOTES.md",
        ]:
            with self.subTest(path=path):
                self.assertEqual(detect.coco_paths([path]), [])

    def test_code_alongside_docs_still_matches(self):
        self.assertEqual(
            detect.coco_paths(["projects/rccl/README.md", "projects/rccl/src/init.cc"]),
            ["projects/rccl/src/init.cc"],
        )

    def test_returns_only_the_matching_subset(self):
        self.assertEqual(
            detect.coco_paths(
                ["docs/index.md", "projects/rccl/src/init.cc", "projects/hip/x.cc"]
            ),
            ["projects/rccl/src/init.cc"],
        )


class TouchesCocoTest(unittest.TestCase):
    def test_matched(self):
        self.assertTrue(detect.touches_coco(["projects/rccl/src/init.cc"]))

    def test_unmatched(self):
        self.assertFalse(detect.touches_coco(["docs/index.md", "projects/hip/x.cc"]))


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
            diff.assert_called_once_with("HEAD^")
            return output.read_text()

    def test_emits_false_for_an_untouched_pr(self):
        self.assertEqual(self._run_main(["docs/index.md"]), "rccl=false\n")

    def test_emits_true_for_a_touched_pr(self):
        self.assertEqual(self._run_main(["projects/rccl/src/init.cc"]), "rccl=true\n")


if __name__ == "__main__":
    unittest.main()
