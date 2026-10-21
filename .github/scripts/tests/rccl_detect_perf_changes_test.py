# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_detect_perf_changes.py."""

import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_detect_coco_changes as coco
import rccl_detect_perf_changes as perf

HOST = "projects/rccl/test/host/net-test.cc"
SRC = "projects/rccl/src/net.cc"


class RunGateTest(unittest.TestCase):
    def test_skips_when_all_code_is_covered(self):
        for changed in [
            [HOST],
            [HOST, "projects/rccl/test/host/fakes/os_fakes.cc"],
            [HOST, "projects/rccl/README.md", "docs/index.md"],
        ]:
            with self.subTest(changed=changed):
                self.assertFalse(perf.should_run_gate(changed, "develop"))

    def test_runs_off_rule_branches(self):
        for branch in ["", "main"]:
            with self.subTest(branch=branch):
                self.assertTrue(perf.should_run_gate([HOST], branch))

    def test_runs_when_any_code_is_uncovered(self):
        for other in [
            SRC,
            "projects/rccl/test/CMakeLists.txt",
            "projects/rccl/test/hostile.cc",
            "projects/rccl-tests/src/all_reduce.cu",
            ".github/workflows/rccl_perf_regression.yml",
            ".github/scripts/rccl_change_sets.py",
        ]:
            with self.subTest(other=other):
                self.assertTrue(perf.should_run_gate([HOST, other], "develop"))

    def test_runs_without_code_changes(self):
        # Unchanged from before COVERAGE_RULES: no code means nothing is covered.
        for changed in [[], ["projects/rccl/README.md"]]:
            with self.subTest(changed=changed):
                self.assertTrue(perf.should_run_gate(changed, "develop"))


class BuildRcclTest(unittest.TestCase):
    """Same rule as the `grep '^projects/rccl/'` it replaced."""

    def test_builds_for_any_rccl_path(self):
        for changed in [[SRC], [HOST], ["projects/rccl/README.md"]]:
            with self.subTest(changed=changed):
                self.assertTrue(perf.should_build_rccl(changed))

    def test_no_build_without_rccl_paths(self):
        for changed in [
            [],
            [".github/workflows/rccl_perf_regression.yml"],
            ["projects/rccl-tests/src/all_reduce.cu"],
        ]:
            with self.subTest(changed=changed):
                self.assertFalse(perf.should_build_rccl(changed))


class CocoConsistencyTest(unittest.TestCase):
    """If the perf gate skips as covered, coco must skip too."""

    POOL = [
        HOST,
        "projects/rccl/test/host/fakes/os_fakes.cc",
        SRC,
        "projects/rccl/test/hostile.cc",
        "projects/rccl/README.md",
        "projects/rccl-tests/src/all_reduce.cu",
        "projects/hip/src/hip.cc",
        ".github/scripts/rccl_change_sets.py",
    ]

    def test_skip_implies_no_coco(self):
        for branch in ["develop", "main"]:
            for n in range(1, 3):
                for changed in itertools.combinations(self.POOL, n):
                    if perf.should_run_gate(changed, branch):
                        continue
                    with self.subTest(branch=branch, changed=changed):
                        self.assertFalse(coco.coco_paths(changed, branch))


class MainTest(unittest.TestCase):
    """The GITHUB_OUTPUT string is what the workflow actually consumes."""

    def _run_main(self, changed_files, *extra_args):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github_output"
            output.touch()
            with mock.patch.object(
                perf, "get_modified_paths", return_value=set(changed_files)
            ) as diff, mock.patch.dict(
                os.environ, {"GITHUB_OUTPUT": os.fspath(output)}
            ):
                self.assertEqual(perf.main(["--base-ref", "HEAD^", *extra_args]), 0)
            diff.assert_called_once_with("HEAD^", no_renames=True)
            return output.read_text()

    def test_host_test_only_pr_to_develop(self):
        self.assertEqual(
            self._run_main([HOST], "--base-branch", "develop"),
            "run_gate=false\nbuild_rccl=1\n",
        )

    def test_host_test_only_pr_elsewhere(self):
        for args in [(), ("--base-branch", "main")]:
            with self.subTest(args=args):
                self.assertEqual(
                    self._run_main([HOST], *args), "run_gate=true\nbuild_rccl=1\n"
                )

    def test_rccl_code_change(self):
        self.assertEqual(
            self._run_main([HOST, SRC], "--base-branch", "develop"),
            "run_gate=true\nbuild_rccl=1\n",
        )

    def test_workflow_only_change(self):
        self.assertEqual(
            self._run_main(
                [".github/workflows/rccl_perf_regression.yml"],
                "--base-branch",
                "develop",
            ),
            "run_gate=true\nbuild_rccl=0\n",
        )


class NoRenamesTest(unittest.TestCase):
    """Both gates diff with no_renames so a move into a covered path counts."""

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
                    both = perf.get_modified_paths("HEAD^", no_renames=True)
            finally:
                os.chdir(cwd)

        self.assertEqual(both, {src, dst})
        self.assertFalse(perf.should_run_gate([dst], "develop"))
        self.assertTrue(perf.should_run_gate(both, "develop"))


if __name__ == "__main__":
    unittest.main()
