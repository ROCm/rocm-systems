# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hrr_check_sources.py."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import hrr_check_sources as chk


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
            self.runner, lambda root, sha: "base header\n",
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
            self.runner, lambda root, sha: "base\n",
        )
        self.assertFalse(ok)
        self.assertIn("not compatible with abc", output)

    def test_missing_base_header_passes(self):
        self.regenerate()
        ok, output = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: None,
        )
        self.assertTrue(ok)
        self.assertEqual(self.calls, [])
        self.assertIn("does not exist at abc", output)

    def test_needs_regenerated_header(self):
        ok, output = self.quiet(
            chk.check_api_compat, self.root, self.out, "abc",
            self.runner, lambda root, sha: "x",
        )
        self.assertFalse(ok)
        self.assertIn("run the generated check first", output)


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
