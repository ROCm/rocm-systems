# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_check_cluster_runner_allowlist.py.

Covers the YAML shapes that can carry a runner label, and the exit status
main() reports for each verdict.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_check_cluster_runner_allowlist as guard

RUNNER = "ruby-linux-slurm-scale-runner"
# A workflow whose only interesting property is that it names RUNNER.
WORKFLOW = f"jobs:\n  build:\n    runs-on: {RUNNER}\n"


class LineAssignsRunnerTest(unittest.TestCase):
    def test_shapes_that_reach_a_runner(self):
        for line in [
            "runs-on: ruby-linux-slurm-scale-runner",
            "    runs_on: ruby-linux-slurm-scale-runner",
            "    - ruby-linux-slurm-scale-runner",
            "      runner: [ruby-linux-slurm-scale-runner]",
            "        - {os: ruby-linux-slurm-scale-runner}",
            "  default: ruby-linux-slurm-scale-runner",
            "runs-on: [self-hosted, ruby-linux-slurm-scale-runner]",
        ]:
            with self.subTest(line=line):
                self.assertTrue(guard.line_assigns_runner(line, RUNNER))

    def test_case_insensitive(self):
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
            path = Path(tmp) / "rccl-cluster-runners.allowlist"
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
            self._load(".github/actions/rccl-resolve-coco-run/action.yml\n")


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


class GuardRepoMixin:
    """Runs main() against a throwaway repo.

    Allowlist entries and the paths the guard prints are both repo-relative, so
    these build a repo and run from its root. Patching the module's path
    constants to absolute temp paths would make every scanned path compare
    unequal to its allowlist entry.
    """

    def _repo(self, root, *, allowlist, workflows, matrix=None):
        (root / guard.ALLOWLIST_PATH.parent).mkdir(parents=True, exist_ok=True)
        (root / guard.ALLOWLIST_PATH).write_text(allowlist, encoding="utf-8")
        (root / guard.WORKFLOWS_DIR).mkdir(parents=True, exist_ok=True)
        for name, body in workflows.items():
            (root / guard.WORKFLOWS_DIR / name).write_text(body, encoding="utf-8")
        if matrix is not None:
            path, body = matrix
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(body, encoding="utf-8")

    def _run(self, root, argv=()):
        """Returns (exit code, stderr, GITHUB_OUTPUT contents)."""
        output = root / "github_output"
        stderr = io.StringIO()
        with contextlib.chdir(root), patch.dict(
            os.environ, {"GITHUB_OUTPUT": os.fspath(output)}
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(stderr):
                    code = guard.main(list(argv))
        written = output.read_text(encoding="utf-8") if output.is_file() else ""
        return code, stderr.getvalue(), written


class MainVerdictTest(GuardRepoMixin, unittest.TestCase):
    """The exit status main() reports, which is the guard's reason to exist."""

    def test_runner_in_unallowlisted_workflow_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(
                root,
                allowlist=f"{RUNNER}\n.github/workflows/allowed.yml\n",
                workflows={"allowed.yml": WORKFLOW, "rogue.yml": WORKFLOW},
            )
            code, stderr, _ = self._run(root)
        self.assertEqual(code, 1)
        self.assertIn(".github/workflows/rogue.yml", stderr)
        self.assertNotIn(".github/workflows/allowed.yml", stderr)

    def test_runner_only_in_allowlisted_workflows_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(
                root,
                allowlist=f"{RUNNER}\n.github/workflows/allowed.yml\n",
                workflows={
                    "allowed.yml": WORKFLOW,
                    "other.yml": "runs-on: ubuntu-24.04\n",
                },
            )
            code, stderr, _ = self._run(root)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")

    def test_matrix_source_passes_when_every_consumer_is_allowlisted(self):
        for source, consumers in guard.MATRIX_SOURCES.items():
            with self.subTest(source=source.as_posix()):
                allowlist = f"{RUNNER}\n" + "".join(f"{c}\n" for c in sorted(consumers))
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self._repo(
                        root,
                        allowlist=allowlist,
                        workflows={},
                        matrix=(source, f'RUNNER = "{RUNNER}"\n'),
                    )
                    code, stderr, _ = self._run(root)
                self.assertEqual(code, 0, stderr)

    def test_matrix_source_fails_when_a_consumer_is_not_allowlisted(self):
        for source, consumers in guard.MATRIX_SOURCES.items():
            with self.subTest(source=source.as_posix()):
                # Hold back one consumer. The spare keeps the workflow set
                # non-empty, so a failure here cannot come from that check.
                kept = sorted(consumers)[1:]
                allowlist = (
                    f"{RUNNER}\n.github/workflows/allowed.yml\n"
                    + "".join(f"{c}\n" for c in kept)
                )
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self._repo(
                        root,
                        allowlist=allowlist,
                        workflows={},
                        matrix=(source, f'RUNNER = "{RUNNER}"\n'),
                    )
                    code, stderr, _ = self._run(root)
                self.assertEqual(code, 1)
                self.assertIn(source.as_posix(), stderr)

    def test_missing_allowlist_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stderr, _ = self._run(Path(tmp))
        self.assertEqual(code, 1)
        self.assertIn(guard.ALLOWLIST_PATH.as_posix(), stderr)

    def test_allowlist_without_runners_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(
                root,
                allowlist=".github/workflows/allowed.yml\n",
                workflows={"allowed.yml": WORKFLOW},
            )
            code, stderr, _ = self._run(root)
        self.assertEqual(code, 1)
        self.assertIn("No runners", stderr)

    def test_allowlist_without_workflows_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root, allowlist=f"{RUNNER}\n", workflows={})
            code, stderr, _ = self._run(root)
        self.assertEqual(code, 1)
        self.assertIn("No workflows", stderr)


class MainNewReferenceTest(GuardRepoMixin, unittest.TestCase):
    """The PR diff half of main(): the review flag and the added-line verdict.

    The diff names a workflow that is absent from the tree on purpose, so the
    full-tree scan stays silent and only the new-reference branch can fail.
    """

    @staticmethod
    def _diff(path):
        return (
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            "@@ -0,0 +1 @@\n"
            f"+    runs-on: {RUNNER}\n"
        )

    def _run_with_diff(self, root, stdout):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)
        with patch.object(guard.subprocess, "run", return_value=completed):
            return self._run(root, ["--base", "base", "--head", "head"])

    def _clean_repo(self, root):
        self._repo(
            root,
            allowlist=f"{RUNNER}\n.github/workflows/allowed.yml\n",
            workflows={"allowed.yml": WORKFLOW},
        )

    def test_new_reference_in_unallowlisted_workflow_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._clean_repo(root)
            code, stderr, written = self._run_with_diff(
                root, self._diff(".github/workflows/rogue.yml")
            )
        self.assertEqual(code, 1)
        self.assertIn(".github/workflows/rogue.yml", stderr)
        self.assertIn("new_runner_lines=true", written)

    def test_new_reference_in_allowlisted_workflow_only_asks_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._clean_repo(root)
            code, stderr, written = self._run_with_diff(
                root, self._diff(".github/workflows/allowed.yml")
            )
        self.assertEqual(code, 0, stderr)
        self.assertIn("new_runner_lines=true", written)

    def test_no_new_references_leaves_the_review_flag_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._clean_repo(root)
            code, stderr, written = self._run_with_diff(root, "")
        self.assertEqual(code, 0, stderr)
        self.assertIn("new_runner_lines=false", written)


class ArgumentPairingTest(unittest.TestCase):
    def test_base_without_head_is_an_error(self):
        for argv in [
            ["--base", "abc"],
            ["--head", "def"],
        ]:
            with self.subTest(argv=argv):
                # Keep a regression here off the real step output.
                with tempfile.TemporaryDirectory() as tmp:
                    output = os.path.join(tmp, "github_output")
                    with patch.dict(os.environ, {"GITHUB_OUTPUT": output}):
                        with self.assertRaises(SystemExit) as caught:
                            guard.main(argv)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
