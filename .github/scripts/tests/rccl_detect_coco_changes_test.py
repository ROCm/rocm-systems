# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_detect_coco_changes.py."""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_detect_coco_changes as detect

HOST_TEST_WORKFLOW = (
    Path(__file__).resolve().parents[2] / "workflows" / "rccl-host-unit-tests.yml"
)


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
            ".github/actions/rccl-resolve-coco-run/action.yml",
            ".github/actions/rccl-checkout-coco-harness/action.yml",
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
            ".github/workflows/therock-multi-arch-ci.yml",
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

    def test_host_only_test_paths_are_excluded_on_develop(self):
        for path in [
            "projects/rccl/test/host/CMakeLists.txt",
            "projects/rccl/test/host/run_host_tests.sh",
            "projects/rccl/test/host/fakes/os_fakes.cc",
            "projects/rccl/test/host/transport/rma-socket-test.cc",
        ]:
            with self.subTest(path=path):
                self.assertEqual(detect.coco_paths([path], "develop"), [])

    def test_host_only_test_paths_match_off_develop(self):
        # rccl-host-unit-tests.yml does not run there, so coco is the only check.
        path = "projects/rccl/test/host/net-test.cc"
        for branch in ["", "main", "rccl-develop", "rocprofiler-compute-develop"]:
            with self.subTest(branch=branch):
                self.assertEqual(detect.coco_paths([path], branch), [path])

    def test_host_test_lookalikes_still_match(self):
        for path in [
            "projects/rccl/test",
            "projects/rccl/test/host",
            "projects/rccl/test/CMakeLists.txt",
            "projects/rccl/test/common/TestBed.cpp",
            "projects/rccl/test/hostile.cc",
            "projects/rccl/test/hostfoo/x.cc",
            "projects/rccl/src/test/host/x.cc",
            "projects/rccl-tests/test/host/x.cc",
        ]:
            with self.subTest(path=path):
                self.assertEqual(detect.coco_paths([path], "develop"), [path])

    def test_coco_paths_alongside_host_tests_still_match(self):
        host = "projects/rccl/test/host/net-test.cc"
        for path in [
            "projects/rccl/src/init.cc",
            "projects/rccl/test/CMakeLists.txt",
            "projects/rccl-tests/src/all_reduce.cu",
            ".github/workflows/rccl-coco-pr.yml",
        ]:
            with self.subTest(path=path):
                self.assertEqual(detect.coco_paths([host, path], "develop"), [path])

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


class HostTestsOnlyTest(unittest.TestCase):
    HOST = "projects/rccl/test/host/net-test.cc"

    def test_host_tests_and_docs_on_develop(self):
        for changed in [
            [self.HOST],
            [self.HOST, "projects/rccl/test/host/fakes/os_fakes.cc"],
            [self.HOST, "projects/rccl/README.md", "docs/index.md"],
        ]:
            with self.subTest(changed=changed):
                self.assertTrue(detect.host_tests_only(changed, "develop"))

    def test_other_branches(self):
        for branch in ["", "main", "rccl-develop", "rocprofiler-compute-develop"]:
            with self.subTest(branch=branch):
                self.assertFalse(detect.host_tests_only([self.HOST], branch))

    def test_any_other_change(self):
        # Anything outside test/host may change librccl or the perf gate itself.
        for other in [
            "projects/rccl/src/init.cc",
            "projects/rccl/test/CMakeLists.txt",
            "projects/rccl/test/hostile.cc",
            "projects/rccl-tests/src/all_reduce.cu",
            "projects/hip/src/hip.cc",
            ".github/workflows/rccl_perf_regression.yml",
        ]:
            with self.subTest(other=other):
                self.assertFalse(detect.host_tests_only([self.HOST, other], "develop"))

    def test_no_code_changes(self):
        for changed in [[], ["projects/rccl/README.md"]]:
            with self.subTest(changed=changed):
                self.assertFalse(detect.host_tests_only(changed, "develop"))


class TouchesCocoTest(unittest.TestCase):
    def test_matched(self):
        self.assertTrue(detect.touches_coco(["projects/rccl/src/init.cc"]))

    def test_unmatched(self):
        self.assertFalse(detect.touches_coco(["docs/index.md", "projects/hip/x.cc"]))


class MainTest(unittest.TestCase):
    """The GITHUB_OUTPUT string is what the workflow actually consumes."""

    def _run_main(self, changed_files, *extra_args):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github_output"
            output.touch()
            with mock.patch.object(
                detect, "get_modified_paths", return_value=set(changed_files)
            ) as diff, mock.patch.dict(
                os.environ, {"GITHUB_OUTPUT": os.fspath(output)}
            ):
                self.assertEqual(detect.main(["--base-ref", "HEAD^", *extra_args]), 0)
            diff.assert_called_once_with("HEAD^", no_renames=True)
            return output.read_text()

    def test_emits_false_for_an_untouched_pr(self):
        self.assertEqual(
            self._run_main(["docs/index.md"]),
            "rccl=false\nhost_tests_only=false\n",
        )

    def test_emits_true_for_a_touched_pr(self):
        self.assertEqual(
            self._run_main(["projects/rccl/src/init.cc"]),
            "rccl=true\nhost_tests_only=false\n",
        )

    def test_emits_host_only_for_a_host_test_only_pr_to_develop(self):
        changed = [
            "projects/rccl/test/host/net-test.cc",
            "projects/rccl/test/host/fakes/os_fakes.cc",
        ]
        self.assertEqual(
            self._run_main(changed, "--base-branch", "develop"),
            "rccl=false\nhost_tests_only=true\n",
        )

    def test_emits_rccl_for_a_host_test_only_pr_elsewhere(self):
        changed = ["projects/rccl/test/host/net-test.cc"]
        for args in [(), ("--base-branch", "main")]:
            with self.subTest(args=args):
                self.assertEqual(
                    self._run_main(changed, *args),
                    "rccl=true\nhost_tests_only=false\n",
                )

    def test_emits_rccl_for_host_tests_plus_rccl_code(self):
        changed = ["projects/rccl/test/host/net-test.cc", "projects/rccl/src/net.cc"]
        self.assertEqual(
            self._run_main(changed, "--base-branch", "develop"),
            "rccl=true\nhost_tests_only=false\n",
        )


class GetModifiedPathsTest(unittest.TestCase):
    """A move into test/host must still report the coco path it left."""

    def test_no_renames_reports_both_sides_of_a_move(self):
        with tempfile.TemporaryDirectory() as tmp:

            # Isolate from the caller's git config, hooks and repo.
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE")
            }
            env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

            def git(*args):
                subprocess.run(
                    ["git", "-C", tmp, *args],
                    check=True,
                    capture_output=True,
                    env=env,
                )

            src = "projects/rccl/src/x.cc"
            dst = "projects/rccl/test/host/x.cc"
            git("init", "-q")
            git("config", "user.email", "t@example.com")
            git("config", "user.name", "t")
            (Path(tmp) / src).parent.mkdir(parents=True)
            (Path(tmp) / src).write_text("int x;\n" * 20)
            git("add", ".")
            git("commit", "-qm", "base")
            (Path(tmp) / dst).parent.mkdir(parents=True)
            git("mv", src, dst)
            git("commit", "-qm", "move")

            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.dict(os.environ, env, clear=True):
                    both = detect.get_modified_paths("HEAD^", no_renames=True)
            finally:
                os.chdir(cwd)

        self.assertEqual(both, {src, dst})
        self.assertTrue(detect.touches_coco(both, "develop"))


class HostTestWorkflowTest(unittest.TestCase):
    """Skipping coco is only safe where rccl-host-unit-tests.yml runs instead."""

    def test_branches_match_host_test_workflow(self):
        text = HOST_TEST_WORKFLOW.read_text(encoding="utf-8")
        block = re.search(r"^\s*branches:\s*\n((?:\s*-\s*\S+\n)+)", text, re.M)
        self.assertIsNotNone(block, f"no branches list in {HOST_TEST_WORKFLOW}")
        branches = set(re.findall(r"-\s*'?([^'\s]+)'?", block.group(1)))
        self.assertEqual(branches, set(detect.HOST_TEST_BRANCHES))

    def test_host_test_workflow_covers_all_of_rccl(self):
        text = HOST_TEST_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("- 'projects/rccl/**'", text)
        self.assertNotIn("paths-ignore", text)


if __name__ == "__main__":
    unittest.main()
