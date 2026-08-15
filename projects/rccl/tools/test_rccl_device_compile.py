#!/usr/bin/env python3
"""Unit tests for tools/rccl-device-compile failure / timing exit behavior.

Guarantees a failed device compile can never look like a successful one:
run_timed() and run() both sys.exit on any subprocess exception, and the
per-kernel timing CSV still records which stage failed (its `finally` block
runs while SystemExit propagates).
"""

import argparse
import csv
import importlib.util
import io
import os
import subprocess
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "rccl-device-compile")


def _load_device_compile():
    # The driver has no .py suffix, so spec_from_file_location alone is not enough.
    loader = SourceFileLoader("rccl_device_compile", SCRIPT)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


dc = _load_device_compile()


class RunTimedExitTest(unittest.TestCase):
    """run_timed must sys.exit on every exception path."""

    def test_sys_exits_with_subprocess_returncode(self):
        err = subprocess.CalledProcessError(17, ["fake-clang", "-c", "x.cpp"])
        with mock.patch.object(dc.subprocess, "check_call", side_effect=err):
            with self.assertRaises(SystemExit) as cm:
                dc.run_timed(["fake-clang", "-c", "x.cpp"], "compile")
        self.assertEqual(cm.exception.code, 17)

    def test_sys_exits_1_on_file_not_found(self):
        with mock.patch.object(
            dc.subprocess, "check_call", side_effect=FileNotFoundError("nope")
        ):
            with self.assertRaises(SystemExit) as cm:
                dc.run_timed(["no-such-bin"], "compile")
        self.assertEqual(cm.exception.code, 1)

    def test_returns_elapsed_seconds_on_success(self):
        with mock.patch.object(dc.subprocess, "check_call", return_value=0):
            elapsed = dc.run_timed(["true"], "compile")
        self.assertIsInstance(elapsed, float)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_reports_failing_command_on_stderr(self):
        err = subprocess.CalledProcessError(5, ["fake-clang", "-c", "x.cpp"])
        with mock.patch.object(dc.subprocess, "check_call", side_effect=err):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit):
                    dc.run_timed(["fake-clang", "-c", "x.cpp"], "compile x.cpp")
        message = stderr.getvalue()
        self.assertIn("compile x.cpp", message)
        self.assertIn("exit code 5", message)
        self.assertIn("fake-clang -c x.cpp", message)


class RunExitTest(unittest.TestCase):
    """run() is a thin wrapper and must exit the same way."""

    def test_sys_exits_with_subprocess_returncode(self):
        err = subprocess.CalledProcessError(9, ["fake-clang"])
        with mock.patch.object(dc.subprocess, "check_call", side_effect=err):
            with self.assertRaises(SystemExit) as cm:
                dc.run(["fake-clang"], "compile")
        self.assertEqual(cm.exception.code, 9)

    def test_sys_exits_1_on_file_not_found(self):
        with mock.patch.object(
            dc.subprocess, "check_call", side_effect=FileNotFoundError("nope")
        ):
            with self.assertRaises(SystemExit) as cm:
                dc.run(["no-such-bin"], "compile")
        self.assertEqual(cm.exception.code, 1)

    def test_no_exit_on_success(self):
        with mock.patch.object(dc.subprocess, "check_call", return_value=0):
            self.assertIsNone(dc.run(["true"], "compile"))


class DoCompileFailureExitTest(unittest.TestCase):
    """A failed compile must exit non-zero and still land in the timing CSV."""

    def _run_do_compile(self, tmp, check_call_side_effect):
        timing_log = os.path.join(tmp, "times.csv")
        source = os.path.join(tmp, "kernel.cpp")
        with open(source, "w") as f:
            f.write("// stub\n")

        args = argparse.Namespace(
            clang="/usr/bin/true",
            arch="gfx942",
            source=source,
            output=os.path.join(tmp, "kernel.o"),
            keep_temps=False,
            timing_log=timing_log,
        )

        with mock.patch.object(dc, "discover_tools", return_value=("fake-clang", None)):
            with mock.patch.object(
                dc.subprocess, "check_call", side_effect=check_call_side_effect
            ):
                with self.assertRaises(SystemExit) as cm:
                    dc.do_compile(args, [])
        return cm.exception.code, timing_log

    def _timing_rows(self, timing_log):
        self.assertTrue(os.path.isfile(timing_log))
        with open(timing_log, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][0], "mode")
        self.assertEqual(len(rows), 2)  # header + one data row
        return rows[1]

    def test_compile_failure_exits_and_logs_stage_with_code(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dc_fail_") as tmp:
            err = subprocess.CalledProcessError(42, ["fake-clang"])
            code, timing_log = self._run_do_compile(tmp, err)

            self.assertEqual(code, 42)
            row = self._timing_rows(timing_log)
            self.assertEqual(row[0], "compile")
            self.assertEqual(row[1], "gfx942")
            self.assertEqual(row[2], "kernel.cpp")
            self.assertEqual(row[7], "compile_failed:42")

    def test_missing_compiler_exits_1_and_logs_stage(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dc_missing_") as tmp:
            code, timing_log = self._run_do_compile(tmp, FileNotFoundError("nope"))

            self.assertEqual(code, 1)
            row = self._timing_rows(timing_log)
            self.assertEqual(row[7], "compile_failed:1")


if __name__ == "__main__":
    unittest.main()
