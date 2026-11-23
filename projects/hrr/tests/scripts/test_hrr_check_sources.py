#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_check_sources.py."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import hrr_check_sources as chk  # noqa: E402


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Tree(unittest.TestCase):
    """A fake checkout whose runner records the commands it is given."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root = base / "src"
        self.out = base / "out"
        self.out.mkdir()
        write(self.root / chk.API_ARGS_HEADER, "// current header\n")
        self.calls = []
        self.rc = {}

    def tearDown(self):
        self._tmp.cleanup()

    def runner(self, cmd):
        self.calls.append(list(cmd))
        return self.rc.get(Path(cmd[1]).name, 0)

    def quiet(self, fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
        return result, buf.getvalue()


class ApiCompatTest(Tree):
    def test_compares_against_base(self):
        ok, _ = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: ("base header\n", None),
        )
        self.assertTrue(ok)
        cmd = self.calls[0]
        self.assertEqual(Path(cmd[1]).name, "check_hrr_api_compat.py")
        current = Path(cmd[cmd.index("--current-header") + 1])
        self.assertEqual(current, self.root / chk.API_ARGS_HEADER)
        base = Path(cmd[cmd.index("--base-header") + 1])
        self.assertEqual(base.read_text(), "base header\n")

    def test_incompatible_ids_fail(self):
        self.rc["check_hrr_api_compat.py"] = 1
        ok, output = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: ("base\n", None),
        )
        self.assertFalse(ok)
        self.assertIn("not compatible with abc", output)

    def test_missing_base_header_passes(self):
        ok, output = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: (None, None),
        )
        self.assertTrue(ok)
        self.assertEqual(self.calls, [])
        self.assertIn("does not exist at abc", output)

    def test_git_failure_fails_instead_of_passing(self):
        # A base blob the blob:none clone cannot fetch, or a base commit it
        # does not have: not "the file is absent", so no silent pass.
        failure = (None, "cannot read projects/hrr/include/hrr/hrr_api_args.h "
                         "at abc: fatal: could not read from remote repository")
        ok, output = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: failure,
        )
        self.assertFalse(ok)
        self.assertEqual(self.calls, [])
        self.assertIn("::error::cannot read", output)
        self.assertIn("do not pass it silently", output)


class BaseHeaderTest(unittest.TestCase):
    """base_header's telling "the path is absent" from "git broke"."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        for argv in (["init", "-q"], ["config", "user.email", "t@example.com"],
                     ["config", "user.name", "t"]):
            subprocess.run(["git", "-C", str(self.root), *argv], check=True,
                           capture_output=True, text=True)

    def commit(self, relpath: str, text: str) -> str:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", relpath], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-qm", relpath], check=True)
        return subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def test_content_at_base(self):
        sha = self.commit(chk.API_ARGS_HEADER, "// base\n")
        text, failure = chk.base_header(self.root, sha)
        self.assertEqual(text, "// base\n")
        self.assertIsNone(failure)

    def test_path_genuinely_absent_passes_as_none(self):
        # Committed, but never this path: the only "nothing to compare" case.
        sha = self.commit("README.md", "x\n")
        text, failure = chk.base_header(self.root, sha)
        self.assertIsNone(text)
        self.assertIsNone(failure)

    def test_base_commit_the_clone_does_not_have_fails(self):
        text, failure = chk.base_header(self.root, "0" * 40)
        self.assertIsNone(text)
        self.assertIn("cannot read", failure)
        self.assertIn("at " + "0" * 40, failure)


class WorkloadsTest(Tree):
    def test_runs_the_workload_check(self):
        ok, _ = self.quiet(chk.check_workloads, self.root, self.runner)
        self.assertTrue(ok)
        self.assertEqual(Path(self.calls[0][1]).name, "check_matrix.py")
        self.assertIn("--check-workloads", self.calls[0])

    def test_failure_is_reported(self):
        self.rc["check_matrix.py"] = 2
        ok, output = self.quiet(chk.check_workloads, self.root, self.runner)
        self.assertFalse(ok)
        self.assertIn("--check-workloads failed (exit 2)", output)


class ResolveBaseTest(unittest.TestCase):
    def test_explicit_sha(self):
        self.assertEqual(chk.resolve_base(Path("."), "1234abcd"), "1234abcd")


if __name__ == "__main__":
    unittest.main()
