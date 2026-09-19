# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for check_rccl_cluster_runner_allowlist.py.

The guard's value is entirely in what its string matching catches, and a guard
that has stopped matching is indistinguishable from one that found nothing. The
reference matrix below therefore enumerates the YAML shapes that reach a runner
label: block-sequence items, matrix values and input defaults as well as
`runs-on:`.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import check_rccl_cluster_runner_allowlist as guard

RUNNER = "ruby-linux-slurm-scale-runner"


class LineAssignsRunnerTest(unittest.TestCase):
    def test_shapes_that_reach_a_runner(self):
        for line in [
            "runs-on: ruby-linux-slurm-scale-runner",
            "    runs_on: ruby-linux-slurm-scale-runner",
            # A block sequence and a matrix entry assign a runner just as much
            # as `runs-on:` does, and carry no key of their own.
            "    - ruby-linux-slurm-scale-runner",
            "      runner: [ruby-linux-slurm-scale-runner]",
            "        - {os: ruby-linux-slurm-scale-runner}",
            "  default: ruby-linux-slurm-scale-runner",
            "runs-on: [self-hosted, ruby-linux-slurm-scale-runner]",
        ]:
            with self.subTest(line=line):
                self.assertTrue(guard.line_assigns_runner(line, RUNNER))

    def test_case_insensitive(self):
        # Runner labels are case-insensitive, and one allowlisted label is
        # mixed case.
        mixed = "tw-gfx950-ainic-ROCm-scale-runner"
        self.assertTrue(
            guard.line_assigns_runner(
                "runs-on: tw-gfx950-ainic-rocm-scale-runner", mixed
            )
        )

    def test_ignores_comments_and_blanks(self):
        for line in [
            "# runs-on: ruby-linux-slurm-scale-runner",
            "   # - ruby-linux-slurm-scale-runner",
            "",
            "   ",
        ]:
            with self.subTest(line=line):
                self.assertFalse(guard.line_assigns_runner(line, RUNNER))

    def test_allowlisted_label_does_not_vouch_for_a_longer_one(self):
        # A new runner must not inherit an allowlisted one's permission by
        # sharing its prefix or suffix.
        for line in [
            "runs-on: ruby-linux-slurm-scale-runner-v2",
            "runs-on: ruby-linux-slurm-scale-runner2",
            "runs-on: x-ruby-linux-slurm-scale-runner",
            "runs-on: staging-ruby-linux-slurm-scale-runner-v2",
        ]:
            with self.subTest(line=line):
                self.assertFalse(guard.line_assigns_runner(line, RUNNER))

    def test_unrelated_runner(self):
        self.assertFalse(guard.line_assigns_runner("runs-on: ubuntu-24.04", RUNNER))


class LoadAllowlistTest(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cluster-runners.allowlist"
            path.write_text(text, encoding="utf-8")
            return guard.load_allowlist(path)

    def test_splits_runners_from_workflows(self):
        runners, workflows = self._load(
            "# a comment\n"
            "\n"
            "ruby-linux-slurm-scale-runner\n"
            "oci-linux-slurm-scale-runner  # trailing comment\n"
            ".github/workflows/rccl-coco-pr.yml\n"
            ".github/workflows/rccl-coco-scheduled.yml\n"
            ".github/workflows/rccl-coco-run.yml\n"
        )
        self.assertEqual(
            runners,
            {"ruby-linux-slurm-scale-runner", "oci-linux-slurm-scale-runner"},
        )
        self.assertEqual(
            workflows,
            {
                ".github/workflows/rccl-coco-pr.yml",
                ".github/workflows/rccl-coco-scheduled.yml",
                ".github/workflows/rccl-coco-run.yml",
            },
        )

    def test_empty_allowlist(self):
        runners, workflows = self._load("# nothing but comments\n\n")
        self.assertEqual(runners, set())
        self.assertEqual(workflows, set())

    def test_rejects_a_github_path_outside_workflows(self):
        # Only workflow paths are scanned, so such a line could never match.
        with self.assertRaises(ValueError):
            self._load(".github/actions/resolve-coco-run/action.yml\n")


class GitDiffNewRunnerUsageTest(unittest.TestCase):
    # other.yml carries the runner on a removed line and on a context line,
    # neither of which adds a new reference, so only new.yml is reported.
    DIFF = """diff --git a/.github/workflows/new.yml b/.github/workflows/new.yml
--- a/.github/workflows/new.yml
+++ b/.github/workflows/new.yml
@@ -0,0 +1 @@
+    runs-on: ruby-linux-slurm-scale-runner
diff --git a/.github/workflows/other.yml b/.github/workflows/other.yml
--- a/.github/workflows/other.yml
+++ b/.github/workflows/other.yml
@@ -1 +1 @@
-    runs-on: ruby-linux-slurm-scale-runner
+    runs-on: ubuntu-24.04
     runs-on: ruby-linux-slurm-scale-runner
"""

    def _run(self, stdout: str):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)
        with patch.object(guard.subprocess, "run", return_value=completed):
            return guard.git_diff_new_runner_usage("base", "head", {RUNNER})

    def test_attributes_added_lines_to_their_file(self):
        found = self._run(self.DIFF)
        self.assertEqual(
            found[RUNNER],
            [(".github/workflows/new.yml", "runs-on: ruby-linux-slurm-scale-runner")],
        )

    def test_no_additions(self):
        self.assertEqual(self._run("")[RUNNER], [])


class ArgumentPairingTest(unittest.TestCase):
    def test_base_without_head_is_an_error(self):
        # Silently skipping the new-reference check would report a pass.
        for argv in [
            ["--base", "abc"],
            ["--head", "def"],
        ]:
            with self.subTest(argv=argv):
                # Redirected so that a regression here scans the tree and
                # writes its verdict to a scratch file rather than to the real
                # step output.
                with tempfile.TemporaryDirectory() as tmp:
                    output = os.path.join(tmp, "github_output")
                    with patch.dict(os.environ, {"GITHUB_OUTPUT": output}):
                        with self.assertRaises(SystemExit) as caught:
                            guard.main(argv)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
