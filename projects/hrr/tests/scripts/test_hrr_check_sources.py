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
    """A fake checkout whose generator writes whatever self.generated holds."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root = base / "src"
        self.out = base / "out"
        self.out.mkdir()
        for tracked in chk.GENERATED_FILES:
            write(self.root / tracked, f"// {tracked}\n")
        write(self.root / chk.MATRIX_EXPECTATIONS, "expect\n")
        self.generated = {t: f"// {t}\n" for t in chk.GENERATED_FILES}
        self.emitted = "expect\n"
        self.calls = []
        self.rc = {}

    def tearDown(self):
        self._tmp.cleanup()

    def runner(self, cmd):
        self.calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "gen_hrr_api_args.py":
            for tracked, name in chk.GENERATED_FILES.items():
                write(self.out / name, self.generated[tracked])
        if "--emit-cxx" in cmd:
            write(Path(cmd[cmd.index("--emit-cxx") + 1]), self.emitted)
        return self.rc.get(script, 0)

    def quiet(self, fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
        return result, buf.getvalue()


class GeneratedTest(Tree):
    def test_up_to_date(self):
        ok, _ = self.quiet(chk.check_generated, self.root, self.out, self.runner)
        self.assertTrue(ok)
        self.assertIn("--check-hrr-coverage", self.calls[0])

    def test_stale_file_is_reported(self):
        stale = "projects/hrr/playback/hip_playback_generated.cpp"
        self.generated[stale] = "// changed\n"
        ok, output = self.quiet(chk.check_generated, self.root, self.out, self.runner)
        self.assertFalse(ok)
        self.assertIn("::error::checked-in HRR generated sources are stale", output)
        self.assertIn(stale, output)
        self.assertIn("+// changed", output)

    def test_line_endings_are_ignored(self):
        tracked = chk.API_ARGS_HEADER
        self.generated[tracked] = f"// {tracked}\r\n"
        ok, _ = self.quiet(chk.check_generated, self.root, self.out, self.runner)
        self.assertTrue(ok)

    def test_generator_failure(self):
        self.rc["gen_hrr_api_args.py"] = 2
        ok, output = self.quiet(chk.check_generated, self.root, self.out, self.runner)
        self.assertFalse(ok)
        self.assertIn("failed (exit 2)", output)


class ApiCompatTest(Tree):
    def regenerate(self):
        self.quiet(chk.check_generated, self.root, self.out, self.runner)
        self.calls.clear()

    def test_compares_against_base(self):
        self.regenerate()
        ok, _ = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: ("base header\n", None),
        )
        self.assertTrue(ok)
        cmd = self.calls[0]
        self.assertEqual(Path(cmd[1]).name, "check_hrr_api_compat.py")
        base = Path(cmd[cmd.index("--base-header") + 1])
        self.assertEqual(base.read_text(), "base header\n")

    def test_incompatible_ids_fail(self):
        self.regenerate()
        self.rc["check_hrr_api_compat.py"] = 1
        ok, output = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: ("base\n", None),
        )
        self.assertFalse(ok)
        self.assertIn("not compatible with abc", output)

    def test_missing_base_header_passes(self):
        self.regenerate()
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
        self.regenerate()
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

    def test_needs_regenerated_header(self):
        ok, output = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: ("x", None),
        )
        self.assertFalse(ok)
        self.assertIn("run the generated check first", output)


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


class ApiMatrixTest(Tree):
    def test_up_to_date(self):
        ok, _ = self.quiet(chk.check_api_matrix, self.root, self.out, self.runner)
        self.assertTrue(ok)
        self.assertEqual(len(self.calls), 3)

    def test_stale_expectations(self):
        self.emitted = "different\n"
        ok, output = self.quiet(chk.check_api_matrix, self.root, self.out, self.runner)
        self.assertFalse(ok)
        self.assertIn("hrr_api_matrix_expectations.h is stale", output)

    def test_step_failure_stops(self):
        self.rc["derive_manifest.py"] = 3
        ok, output = self.quiet(chk.check_api_matrix, self.root, self.out, self.runner)
        self.assertFalse(ok)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("derive_manifest.py failed (exit 3)", output)


class ResolveBaseTest(unittest.TestCase):
    def test_explicit_sha(self):
        self.assertEqual(chk.resolve_base(Path("."), "1234abcd"), "1234abcd")


if __name__ == "__main__":
    unittest.main()
