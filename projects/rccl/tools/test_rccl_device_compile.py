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
import multiprocessing
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


def _fake_tool(cmd):
    """Stand in for amdclang++ / ld.lld: create whatever `-o` names and succeed."""
    if "-o" in cmd:
        out = cmd[cmd.index("-o") + 1]
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out, "w") as f:
            f.write("\t.text\n")
    return 0


def _concurrent_append_worker(args):
    """Top-level (picklable) worker for TimingLogConcurrencyTest."""
    path, index = args
    dc.append_timing_log(path, [
        "compile", "gfx1250", "kernel%d.cpp" % index,
        "1.000", "0.100", "0.200", "1.300", "ok",
    ])


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


class TimingLogPathResolutionTest(unittest.TestCase):
    """--timing-log and RCCL_DEVICE_COMPILE_TIMING_LOG select the CSV path.

    CMake passes the flag (DeviceLinker.cmake); the env var is the manual
    escape hatch documented in the module docstring. If resolution regresses,
    timing silently produces no file rather than failing.
    """

    def _resolve(self, flag, env):
        args = argparse.Namespace(timing_log=flag)
        patched = dict(os.environ)
        patched.pop("RCCL_DEVICE_COMPILE_TIMING_LOG", None)
        if env is not None:
            patched["RCCL_DEVICE_COMPILE_TIMING_LOG"] = env
        with mock.patch.dict(dc.os.environ, patched, clear=True):
            return dc.resolve_timing_log(args)

    def test_flag_is_used_when_present(self):
        self.assertEqual("/tmp/flag.csv", self._resolve("/tmp/flag.csv", None))

    def test_env_var_is_used_when_flag_absent(self):
        self.assertEqual("/tmp/env.csv", self._resolve(None, "/tmp/env.csv"))

    def test_flag_wins_over_env_var(self):
        self.assertEqual("/tmp/flag.csv", self._resolve("/tmp/flag.csv", "/tmp/env.csv"))

    def test_none_when_neither_is_set(self):
        self.assertIsNone(self._resolve(None, None))

    def test_empty_values_are_treated_as_unset(self):
        # An empty -DDEVICE_KERNEL_COMPILE_TIMING_LOG would otherwise resolve to
        # "" and make append_timing_log try to open the cwd.
        self.assertIsNone(self._resolve("", ""))

    def test_missing_attribute_is_tolerated(self):
        # do_link builds its Namespace from a shared parser, but callers that
        # construct args directly (older CMake rules) may omit timing_log.
        with mock.patch.dict(dc.os.environ, {}, clear=True):
            self.assertIsNone(dc.resolve_timing_log(argparse.Namespace()))


class TimingLogSchemaContractTest(unittest.TestCase):
    """install.sh reads this CSV positionally with awk -F, ($1 mode, $3 kernel,
    $4 compile_s, $7 total_s). Reordering _TIMING_LOG_HEADER without updating
    install.sh would silently sort and report the wrong numbers, so pin the
    column positions the shell side depends on."""

    def test_column_positions_match_install_sh_awk_fields(self):
        header = list(dc._TIMING_LOG_HEADER)
        self.assertEqual("mode", header[0], "install.sh: $1 == \"compile\"")
        self.assertEqual("kernel", header[2], "install.sh: $3 is the kernel name")
        self.assertEqual("compile_s", header[3], "install.sh: $4 is compile time")
        self.assertEqual("total_s", header[6], "install.sh: $7 is the sort key")
        self.assertEqual(8, len(header))

    def test_logged_row_has_one_field_per_header_column(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dc_schema_") as tmp:
            path = os.path.join(tmp, "times.csv")
            dc.log_timing(path, "compile", "gfx1250", "k.cpp",
                          1.5, 0.25, 0.125, 2.0, "ok")
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(list(dc._TIMING_LOG_HEADER), rows[0])
        self.assertEqual(len(rows[0]), len(rows[1]))

    def test_durations_are_formatted_to_three_decimals(self):
        # awk does `$7 + 0 > threshold`, so the field must be a plain number --
        # not exponent notation, which awk would parse as 1 (for "1e-05").
        with tempfile.TemporaryDirectory(prefix="rccl_dc_fmt_") as tmp:
            path = os.path.join(tmp, "times.csv")
            dc.log_timing(path, "compile", "gfx1250", "k.cpp",
                          0.0000123, 61.5, 0.5, 62.0, "ok")
            with open(path, newline="") as f:
                row = list(csv.reader(f))[1]
        self.assertEqual("0.000", row[3])
        self.assertEqual("61.500", row[4])
        self.assertEqual("62.000", row[6])
        for field in row[3:7]:
            self.assertNotIn("e", field.lower())


class TimingLogFileFormatTest(unittest.TestCase):
    def test_header_written_once_and_rows_appended(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dc_hdr_") as tmp:
            path = os.path.join(tmp, "times.csv")
            for i in range(3):
                dc.append_timing_log(path, ["compile", "gfx1250", "k%d.cpp" % i,
                                            "1.000", "0.000", "0.000", "1.000", "ok"])
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(4, len(rows))
        self.assertEqual(list(dc._TIMING_LOG_HEADER), rows[0])
        self.assertEqual(["k0.cpp", "k1.cpp", "k2.cpp"], [r[2] for r in rows[1:]])

    def test_parent_directory_is_created(self):
        # CMake points the log at ${PROJECT_BINARY_DIR}, which exists, but the
        # env-var escape hatch can name a fresh path.
        with tempfile.TemporaryDirectory(prefix="rccl_dc_mkdir_") as tmp:
            path = os.path.join(tmp, "nested", "deeper", "times.csv")
            dc.append_timing_log(path, ["compile", "gfx1250", "k.cpp",
                                        "1.000", "0.000", "0.000", "1.000", "ok"])
            self.assertTrue(os.path.isfile(path))

    def test_log_timing_is_a_noop_without_a_path(self):
        # The overwhelmingly common case: timing disabled. Must not create a
        # stray file in the cwd or raise.
        with tempfile.TemporaryDirectory(prefix="rccl_dc_noop_") as tmp:
            before = os.listdir(tmp)
            for path in (None, ""):
                dc.log_timing(path, "compile", "gfx1250", "k.cpp",
                              1.0, 0.0, 0.0, 1.0, "ok")
            self.assertEqual(before, os.listdir(tmp))

    def test_data_is_on_disk_before_the_lock_is_released(self):
        # append_timing_log takes an exclusive flock to make the "is the file
        # empty?" test and the write atomic against other build jobs. That only
        # holds if the bytes reach the kernel before the lock drops: a csv
        # writer's output sitting in Python's buffer is invisible to the next
        # process to grab the lock, which then writes a second header.
        seen = {}
        real_flock = dc.fcntl.flock

        def spy(fd, op):
            if op == dc.fcntl.LOCK_UN:
                seen["size_at_unlock"] = os.fstat(fd).st_size
            return real_flock(fd, op)

        with tempfile.TemporaryDirectory(prefix="rccl_dc_flush_") as tmp:
            path = os.path.join(tmp, "times.csv")
            with mock.patch.object(dc.fcntl, "flock", spy):
                dc.append_timing_log(path, ["compile", "gfx1250", "k.cpp",
                                            "1.000", "0.000", "0.000", "1.000", "ok"])
            final_size = os.path.getsize(path)

        self.assertIn("size_at_unlock", seen)
        self.assertEqual(
            final_size, seen["size_at_unlock"],
            "rows were still buffered when the lock was released, so a "
            "concurrent writer would see a zero-length file and emit a "
            "duplicate header",
        )


class TimingLogConcurrencyTest(unittest.TestCase):
    """A timing build is a parallel build: one process per kernel, -j$(nproc).
    Every row must survive and exactly one header may exist."""

    WORKERS = 32

    def test_concurrent_writers_produce_one_header_and_keep_every_row(self):
        ctx = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory(prefix="rccl_dc_par_") as tmp:
            path = os.path.join(tmp, "times.csv")
            with ctx.Pool(self.WORKERS) as pool:
                pool.map(_concurrent_append_worker,
                         [(path, i) for i in range(self.WORKERS)])
            with open(path, newline="") as f:
                rows = [r for r in csv.reader(f) if r]

        headers = [r for r in rows if r[0] == "mode"]
        data = [r for r in rows if r[0] == "compile"]
        self.assertEqual(1, len(headers),
                         "expected exactly one CSV header, got %d" % len(headers))
        self.assertEqual(self.WORKERS, len(data))
        self.assertEqual(set(range(self.WORKERS)),
                         {int(r[2][len("kernel"):-len(".cpp")]) for r in data})
        for row in data:
            self.assertEqual(len(dc._TIMING_LOG_HEADER), len(row),
                             "interleaved/truncated row: %r" % (row,))


class DoCompileTimingTest(unittest.TestCase):
    """do_compile's per-stage timing row: the success path and each failing
    stage's status annotation."""

    def _args(self, tmp, timing_log=True):
        source = os.path.join(tmp, "kernel.cpp")
        with open(source, "w") as f:
            f.write("// stub\n")
        return argparse.Namespace(
            clang="/usr/bin/true",
            arch="gfx1250",
            source=source,
            output=os.path.join(tmp, "kernel.o"),
            keep_temps=False,
            timing_log=os.path.join(tmp, "times.csv") if timing_log else None,
        )

    def _row(self, path):
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(2, len(rows), rows)
        return rows[1]

    def test_successful_compile_logs_ok_and_every_stage(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dc_ok_") as tmp:
            args = self._args(tmp)
            with mock.patch.object(dc, "discover_tools",
                                   return_value=("fake-clang", None)), \
                 mock.patch.object(dc, "extract_device_function",
                                   return_value=(["\t.text\n"], {"vgpr_count": 8})), \
                 mock.patch.object(dc.subprocess, "check_call",
                                   side_effect=_fake_tool):
                dc.do_compile(args, [])
            row = self._row(args.timing_log)

        self.assertEqual("compile", row[0])
        self.assertEqual("gfx1250", row[1])
        self.assertEqual("kernel.cpp", row[2])
        self.assertEqual("ok", row[7])
        # total_s must cover the whole invocation, so it cannot be less than
        # any single stage.
        stages = [float(x) for x in row[3:6]]
        self.assertGreaterEqual(float(row[6]), max(stages))

    def test_extraction_failure_logs_extract_stage(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dc_extract_") as tmp:
            args = self._args(tmp)
            with mock.patch.object(dc, "discover_tools",
                                   return_value=("fake-clang", None)), \
                 mock.patch.object(dc, "extract_device_function",
                                   side_effect=RuntimeError("no kernel found")), \
                 mock.patch.object(dc.subprocess, "check_call",
                                   side_effect=_fake_tool), \
                 mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as cm:
                    dc.do_compile(args, [])
            self.assertEqual(1, cm.exception.code)
            row = self._row(args.timing_log)
        self.assertEqual("extract_failed:1", row[7])
        # The compile stage completed, so its time is still reported.
        self.assertGreaterEqual(float(row[4]), 0.0)

    def test_assemble_failure_logs_assemble_stage(self):
        calls = {"n": 0}

        def fail_second(cmd):
            calls["n"] += 1
            if calls["n"] == 1:
                return _fake_tool(cmd)
            raise subprocess.CalledProcessError(3, cmd)

        with tempfile.TemporaryDirectory(prefix="rccl_dc_asm_") as tmp:
            args = self._args(tmp)
            with mock.patch.object(dc, "discover_tools",
                                   return_value=("fake-clang", None)), \
                 mock.patch.object(dc, "extract_device_function",
                                   return_value=(["\t.text\n"], {"vgpr_count": 8})), \
                 mock.patch.object(dc.subprocess, "check_call",
                                   side_effect=fail_second), \
                 mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as cm:
                    dc.do_compile(args, [])
            self.assertEqual(3, cm.exception.code)
            row = self._row(args.timing_log)
        self.assertEqual("assemble_failed:3", row[7])

    def test_no_log_written_when_timing_disabled(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dc_off_") as tmp:
            args = self._args(tmp, timing_log=False)
            with mock.patch.dict(dc.os.environ, {}, clear=True), \
                 mock.patch.object(dc, "discover_tools",
                                   return_value=("fake-clang", None)), \
                 mock.patch.object(dc, "extract_device_function",
                                   return_value=(["\t.text\n"], {"vgpr_count": 8})), \
                 mock.patch.object(dc.subprocess, "check_call",
                                   side_effect=_fake_tool):
                dc.do_compile(args, [])
            self.assertEqual([], [n for n in os.listdir(tmp) if n.endswith(".csv")])


class DoLinkTimingTest(unittest.TestCase):
    """do_link logs with mode="link" and its own stage names."""

    def _args(self, tmp):
        obj = os.path.join(tmp, "kernel.o")
        with open(obj, "w") as f:
            f.write("")
        with open(os.path.join(tmp, "kernel.resources.json"), "w") as f:
            f.write("{}")
        dispatcher = os.path.join(tmp, "dispatcher.cpp")
        with open(dispatcher, "w") as f:
            f.write("// stub\n")
        return argparse.Namespace(
            clang="/usr/bin/true",
            arch="gfx1250",
            objects=[obj],
            output=os.path.join(tmp, "device.elf"),
            dispatcher=dispatcher,
            keep_temps=False,
            rocshmem_bitcode=None,
            timing_log=os.path.join(tmp, "times.csv"),
        )

    def _patches(self):
        return (
            mock.patch.object(dc, "discover_tools",
                              return_value=("fake-clang", "fake-lld")),
            mock.patch.object(dc, "aggregate_resources", return_value={
                "vgpr_count": 8, "agpr_count": 0, "sgpr_count": 16,
                "private_segment_fixed_size": 0}),
            mock.patch.object(dc, "patch_dispatcher", side_effect=lambda lines, res: lines),
        )

    def _row(self, path):
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(2, len(rows), rows)
        return rows[1]

    def test_successful_link_logs_link_mode(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dl_ok_") as tmp:
            args = self._args(tmp)
            a, b, c = self._patches()
            with a, b, c, mock.patch.object(dc.subprocess, "check_call",
                                            side_effect=_fake_tool):
                dc.do_link(args, [])
            row = self._row(args.timing_log)
        self.assertEqual("link", row[0])
        self.assertEqual("gfx1250", row[1])
        self.assertEqual("device.elf", row[2])
        self.assertEqual("ok", row[7])

    def test_dispatcher_compile_failure_is_named_in_the_row(self):
        with tempfile.TemporaryDirectory(prefix="rccl_dl_fail_") as tmp:
            args = self._args(tmp)
            a, b, c = self._patches()
            err = subprocess.CalledProcessError(7, ["fake-clang"])
            with a, b, c, \
                 mock.patch.object(dc.subprocess, "check_call", side_effect=err), \
                 mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as cm:
                    dc.do_link(args, [])
            self.assertEqual(7, cm.exception.code)
            row = self._row(args.timing_log)
        self.assertEqual("dispatcher_compile_failed:7", row[7])


class ParseCompilerFlagsTimingTest(unittest.TestCase):
    """--timing-log arrives mixed into a compiler command line (CMake adds it as
    a COMPILE_OPTION), so parse_compiler_flags must claim it for argparse
    instead of forwarding it to amdclang++, which would reject it."""

    def test_joined_form_is_claimed_not_forwarded(self):
        ours, forwarded, sources = dc.parse_compiler_flags(
            ["--compile", "--timing-log=/tmp/t.csv", "-DFOO", "x.cpp"])
        self.assertIn("--timing-log=/tmp/t.csv", ours)
        self.assertNotIn("--timing-log=/tmp/t.csv", forwarded)
        self.assertIn("-DFOO", forwarded)
        self.assertEqual(["x.cpp"], sources)

    def test_separated_form_is_claimed_with_its_value(self):
        ours, forwarded, sources = dc.parse_compiler_flags(
            ["--compile", "--timing-log", "/tmp/t.csv", "-DFOO", "x.cpp"])
        self.assertEqual(["--timing-log", "/tmp/t.csv"],
                         ours[ours.index("--timing-log"):ours.index("--timing-log") + 2])
        self.assertNotIn("/tmp/t.csv", forwarded)
        # The value must not be mistaken for a source file either.
        self.assertEqual(["x.cpp"], sources)

    def test_parser_accepts_the_claimed_form(self):
        # End-to-end through the same parser main() builds, so a flag name typo
        # in either place is caught.
        ours, _, _ = dc.parse_compiler_flags(
            ["--compile", "--timing-log=/tmp/t.csv", "-o", "k.o", "k.cpp"])
        parser = argparse.ArgumentParser()
        parser.add_argument("--compile", action="store_true")
        parser.add_argument("--timing-log", default=None, dest="timing_log")
        parser.add_argument("-o", "--output")
        parser.add_argument("source", nargs="?")
        args = parser.parse_args(ours)
        self.assertEqual("/tmp/t.csv", args.timing_log)


if __name__ == "__main__":
    unittest.main()
