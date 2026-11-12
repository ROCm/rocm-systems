import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace
from unittest import mock

from lib.test_executor import TestExecutor


def _args(**overrides):
    """Build an args namespace with the fields the coverage paths read, so each
    test overrides only what it cares about instead of repeating the block."""
    defaults = dict(
        build_dir=None,
        coverage_report=True,
        output=None,
        report_suffix="",
        skip_tests=False,
        verbose=False,
        host_coverage_lcov=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@contextmanager
def _without_rccl_path_env():
    """Clear RCCL_LIB_PATH / RCCL_BUILD_DIR so setup_directories tests do not
    depend on the developer environment."""
    with mock.patch.dict(os.environ):
        os.environ.pop("RCCL_LIB_PATH", None)
        os.environ.pop("RCCL_BUILD_DIR", None)
        yield


def _directories_executor(workdir, install_flags=None, **arg_overrides):
    executor = TestExecutor.__new__(TestExecutor)
    executor.paths = {"workdir": workdir}
    executor.build_config = {"install_flags": list(install_flags or [])}
    executor.args = _args(**arg_overrides)
    return executor


def _write_full_coverage_cache(build_dir, enabled):
    Path(build_dir, "CMakeCache.txt").write_text(
        f"ENABLE_FULL_COVERAGE:STRING={'ON' if enabled else 'OFF'}\n"
    )


class CoverageReportingTest(unittest.TestCase):
    def test_coverage_report_selects_debug_build_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_rccl_path_env():
            executor = _directories_executor(temp_dir)

            executor.setup_directories()

            self.assertEqual(
                executor.build_dir,
                str(Path(temp_dir) / "build" / "debug"),
            )

    def test_non_coverage_run_selects_release_build_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_rccl_path_env():
            executor = _directories_executor(temp_dir, coverage_report=False)

            executor.setup_directories()

            self.assertEqual(
                executor.build_dir,
                str(Path(temp_dir) / "build" / "release"),
            )

    def test_workspace_isolates_profiles_and_removes_stale_data(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_rccl_path_env():
            workspace = Path(temp_dir) / "coverage-output"
            rawfiles = workspace / "logs" / "rawfiles"
            rawfiles.mkdir(parents=True)
            stale_profile = rawfiles / "stale.profraw"
            stale_profile.write_bytes(b"stale")

            executor = _directories_executor(
                temp_dir, ["--debug"], output=str(workspace)
            )

            executor.setup_directories()

            self.assertEqual(executor.workspace_dir, str(workspace))
            self.assertFalse(stale_profile.exists())

            # The exported LLVM_PROFILE_FILE must be an absolute path under the
            # workspace rawfiles dir; a relative pattern would drop profraws in
            # the cwd and silently yield 0% coverage.
            pattern = executor._coverage_profile_pattern()
            self.assertTrue(os.path.isabs(pattern))
            self.assertEqual(Path(pattern).parent, rawfiles)
            self.assertEqual(
                pattern,
                str(rawfiles / "rccl_tests_%h_%p_%m.profraw"),
            )

    def test_a_stale_host_tracefile_is_removed_with_the_stale_profiles(self):
        # The dispatcher writes host-coverage.lcov beside the profiles. A run that
        # does not reach the host tests leaves it untouched, so one left from the
        # previous run would be unioned in and reported as this run's coverage.
        with tempfile.TemporaryDirectory() as temp_dir, _without_rccl_path_env():
            workspace = Path(temp_dir) / "coverage-output"
            rawfiles = workspace / "logs" / "rawfiles"
            rawfiles.mkdir(parents=True)
            stale_tracefile = rawfiles / "host-coverage.lcov"
            stale_tracefile.write_text("TN:stale\nend_of_record\n")

            executor = _directories_executor(
                temp_dir, ["--debug"], output=str(workspace)
            )

            executor.setup_directories()

            self.assertFalse(stale_tracefile.exists())

    def test_report_only_preserves_existing_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_rccl_path_env():
            workspace = Path(temp_dir) / "coverage-output"
            rawfiles = workspace / "logs" / "rawfiles"
            rawfiles.mkdir(parents=True)
            existing_profile = rawfiles / "existing.profraw"
            existing_profile.write_bytes(b"profile")

            executor = _directories_executor(
                temp_dir, ["--debug"], output=str(workspace), skip_tests=True
            )

            executor.setup_directories()

            self.assertTrue(existing_profile.exists())

    def test_llvm_tool_prefers_resolved_rocm_root_over_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rocm_root = Path(temp_dir) / "rocm"
            llvm_bin = rocm_root / "lib" / "llvm" / "bin"
            llvm_bin.mkdir(parents=True)
            for tool in ("llvm-profdata", "llvm-cov"):
                (llvm_bin / tool).touch()

            executor = TestExecutor.__new__(TestExecutor)
            executor.paths = {"rocm_path": str(rocm_root)}
            executor.args = _args()

            with mock.patch("lib.test_executor.shutil.which",
                            return_value="/usr/bin/llvm-cov"):
                resolved = executor._resolve_llvm_tool("llvm-cov")

            self.assertEqual(resolved, str(llvm_bin / "llvm-cov"))

    def test_missing_profiles_fail_report_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = TestExecutor.__new__(TestExecutor)
            executor.args = _args()
            executor.rawfiles_dir = temp_dir
            executor.rccl_tests_build_config = {}

            self.assertFalse(executor.generate_coverage_report())

    def test_coverage_report_disabled_returns_true(self):
        executor = TestExecutor.__new__(TestExecutor)
        executor.args = _args(coverage_report=False)

        self.assertTrue(executor.generate_coverage_report())

    def test_apply_coverage_profile_env_exports_absolute_pattern(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rawfiles = Path(temp_dir) / "rawfiles"
            rawfiles.mkdir()
            executor = TestExecutor.__new__(TestExecutor)
            executor.args = _args()
            executor.rawfiles_dir = str(rawfiles)
            env = {}

            pattern = executor._apply_coverage_profile_env(env)

            self.assertEqual(pattern, str(rawfiles / "rccl_tests_%h_%p_%m.profraw"))
            self.assertEqual(env["LLVM_PROFILE_FILE"], pattern)

    def test_apply_coverage_profile_env_skips_when_coverage_off(self):
        executor = TestExecutor.__new__(TestExecutor)
        executor.args = _args(coverage_report=False)
        env = {}

        self.assertIsNone(executor._apply_coverage_profile_env(env))
        self.assertNotIn("LLVM_PROFILE_FILE", env)

    def _coverage_executor(self, root, create_lib=True, full_coverage=True):
        """Wire an executor with the directory layout generate_coverage_report
        expects, under <root>."""
        rawfiles = root / "logs" / "rawfiles"
        report_dir = root / "report"
        build_dir = root / "build"
        rawfiles.mkdir(parents=True)
        report_dir.mkdir()
        build_dir.mkdir()
        (rawfiles / "profile.profraw").write_bytes(b"profile")
        if create_lib:
            (build_dir / "librccl.so").write_bytes(b"object")
        _write_full_coverage_cache(build_dir, full_coverage)

        executor = TestExecutor.__new__(TestExecutor)
        executor.args = _args()
        executor.rawfiles_dir = str(rawfiles)
        executor.log_dir = str(root / "logs")
        executor.report_dir = str(report_dir)
        executor.build_dir = str(build_dir)
        executor.paths = {"rocm_path": str(root / "rocm"), "workdir": str(root / "src")}
        executor.rccl_tests_build_config = {}
        return executor, build_dir

    def _run_report(self, executor, command_results):
        with mock.patch.object(executor, "_rocm_root",
                               return_value=str(Path(executor.build_dir).parent / "rocm")), \
             mock.patch.object(executor, "_resolve_llvm_tool",
                               side_effect=lambda name: name), \
             mock.patch("lib.test_executor.subprocess.run",
                        side_effect=command_results):
            return executor.generate_coverage_report()

    def test_llvm_cov_step_failure_fails_report_generation(self):
        # merge, html show, text report, summary report -- each llvm-cov
        # failure must fail the run (merge is index 0 and is pinned here too).
        for fail_at in (0, 1, 2, 3):
            with self.subTest(fail_at=fail_at):
                with tempfile.TemporaryDirectory() as temp_dir:
                    executor, _ = self._coverage_executor(Path(temp_dir))
                    results = [mock.Mock(returncode=0)] * fail_at
                    results.append(
                        CalledProcessError(1, ["llvm-cov"], stderr="bad mapping")
                    )
                    self.assertFalse(self._run_report(executor, results))

    def test_profraw_from_the_host_test_front_door_is_found(self):
        # rccl-UnitTestsHostAll gives each binary it forks a directory of its
        # own under rawfiles/, so the host-only tests' profiles are never at the
        # top level. A flat scan would find only the perf tests' profraw and
        # report no host-test coverage at all, which reads as a source
        # regression rather than as a harness that stopped looking.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)
            nested = Path(executor.rawfiles_dir) / "rccl-UnitTestsMicroInit"
            nested.mkdir()
            (nested / "init.profraw").write_bytes(b"profile")

            self.assertTrue(self._run_report(executor, [mock.Mock(returncode=0)] * 4))

            merged = (root / "logs" / "rawprofiles.list").read_text().split()
            self.assertIn(str(nested / "init.profraw"), merged)

    def test_no_object_files_fails_report_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor, _ = self._coverage_executor(
                Path(temp_dir), create_lib=False, full_coverage=False
            )
            self.assertFalse(self._run_report(executor, [mock.Mock(returncode=0)]))

    def test_device_elf_included_as_coverage_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, build_dir = self._coverage_executor(root, full_coverage=True)
            device_elf = build_dir / "device-gfx942.elf"
            device_elf.write_bytes(b"device")

            recorded = []

            def _record(cmd, *args, **kwargs):
                recorded.append(list(cmd))
                return mock.Mock(returncode=0)

            result = self._run_report(executor, _record)

            self.assertTrue(result)

            # The llvm-cov 'show' argv must attribute the per-arch device ELF via
            # '--object <path>'; the glob here and the DeviceLinker symlink are
            # coupled only by this bare filename convention.
            show_cmd = next(c for c in recorded if "show" in c)
            self.assertIn(str(device_elf), show_cmd)
            self.assertEqual(
                show_cmd[show_cmd.index(str(device_elf)) - 1], "--object"
            )

    def _recording_runner(self, lcov_version="lcov: LCOV version 2.0"):
        """subprocess.run stand-in that succeeds and records the argv of each call."""
        recorded = []

        def _record(cmd, *args, **kwargs):
            recorded.append(list(cmd))
            return mock.Mock(returncode=0, stdout=lcov_version, stderr="")

        return recorded, _record

    def test_host_coverage_tracefile_is_unioned_into_the_report(self):
        # The host binaries compile the same source under different #defines, so their
        # profiles cannot be read against these objects at all. Their coverage arrives
        # as the lcov tracefile the front door wrote, and lcov is where the two can be
        # unioned -- llvm-cov has no lcov input.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)
            host_lcov = root / "host-coverage.lcov"
            host_lcov.write_text("TN:\nend_of_record\n")
            executor.args = _args(host_coverage_lcov=str(host_lcov))

            recorded, runner = self._recording_runner()
            with mock.patch("lib.test_executor.shutil.which",
                            side_effect=lambda tool: f"/usr/bin/{tool}"):
                self.assertTrue(self._run_report(executor, runner))

            export = next(c for c in recorded if "export" in c)
            self.assertIn("--format=lcov", export)
            union = next(c for c in recorded if "--add-tracefile" in c)
            self.assertIn(str(host_lcov), union)
            self.assertIn(str(Path(executor.report_dir) / "runner_coverage.lcov"), union)
            self.assertIn(str(Path(executor.report_dir) / "combined_coverage.lcov"), union)

    def test_report_still_succeeds_without_lcov_installed(self):
        # A GPU run on a machine with no lcov must still get its llvm-cov reports: the
        # union is an addition to them, never a precondition for them.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)
            host_lcov = root / "host-coverage.lcov"
            host_lcov.write_text("TN:\nend_of_record\n")
            executor.args = _args(host_coverage_lcov=str(host_lcov))

            recorded, runner = self._recording_runner()
            with mock.patch("lib.test_executor.shutil.which", return_value=None):
                self.assertTrue(self._run_report(executor, runner))

            self.assertFalse([c for c in recorded if "--add-tracefile" in c])
            self.assertTrue(os.path.isfile(
                os.path.join(executor.report_dir, "coverage_summary.txt")))

    def test_a_failing_union_does_not_fail_the_report(self):
        # Same reason: a malformed or rejected tracefile costs the combined view and
        # nothing else, so a host-side problem cannot turn a green GPU run red.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)
            host_lcov = root / "host-coverage.lcov"
            host_lcov.write_text("TN:\nend_of_record\n")
            executor.args = _args(host_coverage_lcov=str(host_lcov))

            def _fail_the_union(cmd, *args, **kwargs):
                if "--add-tracefile" in cmd:
                    raise CalledProcessError(1, list(cmd), stderr="unsupported")
                return mock.Mock(returncode=0, stdout="lcov: LCOV version 2.0", stderr="")

            with mock.patch("lib.test_executor.shutil.which",
                            side_effect=lambda tool: f"/usr/bin/{tool}"):
                self.assertTrue(self._run_report(executor, _fail_the_union))

    def test_no_lcov_work_when_there_is_no_host_tracefile(self):
        # The library-only run (GPU unit tests, no host tests) is untouched: nothing
        # is exported, nothing is unioned, and the argv is the llvm-cov one it was.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)

            recorded, runner = self._recording_runner()
            with mock.patch("lib.test_executor.shutil.which",
                            side_effect=lambda tool: f"/usr/bin/{tool}"):
                self.assertTrue(self._run_report(executor, runner))

            self.assertFalse([c for c in recorded if "export" in c or "--add-tracefile" in c])

    def test_host_tracefile_is_found_where_run_host_tests_writes_it(self):
        # Found by the path it is written to, under the configured workdir -- not by
        # searching, which would pick up a tracefile from some unrelated tree.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)
            written = Path(executor.paths["workdir"], "test", "host", "build", "coverage")
            written.mkdir(parents=True)
            (written / "host-coverage.lcov").write_text("TN:\nend_of_record\n")

            self.assertEqual(executor._host_coverage_tracefile(),
                             str(written / "host-coverage.lcov"))

    def test_host_tracefile_is_taken_from_where_this_run_wrote_it(self):
        # When this runner drives the host binary itself, LLVM_PROFILE_FILE points at
        # rawfiles/ and the dispatcher writes the tracefile beside the profiles. The
        # build tree's copy is some earlier run_host_tests.sh run's, so preferring it
        # would report stale coverage as this run's.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)
            stale = Path(executor.paths["workdir"], "test", "host", "build", "coverage")
            stale.mkdir(parents=True)
            (stale / "host-coverage.lcov").write_text("TN:stale\nend_of_record\n")
            fresh = Path(executor.rawfiles_dir, "host-coverage.lcov")
            fresh.write_text("TN:fresh\nend_of_record\n")

            self.assertEqual(executor._host_coverage_tracefile(), str(fresh))

    def test_a_missing_explicit_tracefile_is_reported_not_guessed_around(self):
        # --host-coverage-lcov names a file the caller believes in; falling back to a
        # discovered one would report coverage they did not ask for, under a number
        # they would read as theirs.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, _ = self._coverage_executor(root)
            written = Path(executor.paths["workdir"], "test", "host", "build", "coverage")
            written.mkdir(parents=True)
            (written / "host-coverage.lcov").write_text("TN:\nend_of_record\n")
            executor.args = _args(host_coverage_lcov=str(root / "absent.lcov"))

            self.assertIsNone(executor._host_coverage_tracefile())

    def test_host_only_fallback_skips_uninstrumented_device_elf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, build_dir = self._coverage_executor(root, full_coverage=False)
            device_elf = build_dir / "device-gfx942.elf"
            device_elf.write_bytes(b"device")

            recorded = []

            def _record(cmd, *args, **kwargs):
                recorded.append(list(cmd))
                return mock.Mock(returncode=0)

            result = self._run_report(executor, _record)

            self.assertTrue(result)
            show_cmd = next(c for c in recorded if "show" in c)
            self.assertNotIn(str(device_elf), show_cmd)

    def test_run_test_exports_absolute_llvm_profile_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "all_reduce_perf"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
            rawfiles = root / "logs" / "rawfiles"
            rawfiles.mkdir(parents=True)

            executor = TestExecutor.__new__(TestExecutor)
            executor.args = _args()
            executor.paths = {
                "workdir": str(root),
                "rocm_path": str(root / "rocm"),
                "mpi_path": "",
            }
            executor.build_dir = str(root)
            executor.rawfiles_dir = str(rawfiles)
            executor.log_dir = str(root / "logs")
            executor.global_env = {}
            executor.emit_enabled = False
            executor._gpus_per_node = 8
            executor._gpus_per_node_detected = True
            executor._rocm_root_cache = str(root / "rocm")

            captured = {}

            class FakeProc:
                def wait(self, timeout=None):
                    return 0

            def fake_popen(cmd, **kwargs):
                captured["env"] = kwargs.get("env", {})
                return FakeProc()

            with mock.patch("lib.test_executor.subprocess.Popen",
                            side_effect=fake_popen):
                result = executor.run_test(
                    {
                        "name": "all_reduce",
                        "binary": str(binary),
                        "is_gtest": False,
                        "num_ranks": 1,
                        "num_nodes": 1,
                        "num_gpus": 1,
                        "timeout": 0,
                    },
                    {},
                )

            self.assertEqual(result["result"], "PASSED")
            expected = str(rawfiles / "rccl_tests_%h_%p_%m.profraw")
            self.assertEqual(captured["env"]["LLVM_PROFILE_FILE"], expected)

    def test_pytest_run_exports_absolute_llvm_profile_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            test_dir = root / "pytest_suite"
            test_dir.mkdir()
            rawfiles = root / "logs" / "rawfiles"
            rawfiles.mkdir(parents=True)
            junit = root / "logs" / "pytest_pytest_cov.xml"
            junit.write_text(
                '<?xml version="1.0"?><testsuite tests="1" failures="0" '
                'errors="0" skipped="0"/>'
            )

            executor = TestExecutor.__new__(TestExecutor)
            executor.args = _args()
            executor.paths = {
                "workdir": str(root),
                "rocm_path": str(root / "rocm"),
                "mpi_path": "",
            }
            executor.build_dir = str(root)
            executor.rawfiles_dir = str(rawfiles)
            executor.log_dir = str(root / "logs")
            executor._rocm_root_cache = str(root / "rocm")

            captured = {}

            def fake_run(cmd, **kwargs):
                captured["env"] = kwargs.get("env", {})
                return mock.Mock(returncode=0)

            with mock.patch("lib.test_executor.subprocess.run",
                            side_effect=fake_run), \
                 mock.patch(
                     "lib.test_executor.infer_pytest_result_from_junit",
                     return_value="PASSED",
                 ):
                result = executor._run_pytest_test(
                    {
                        "name": "pytest_cov",
                        "test_dir": str(test_dir),
                        "timeout": 0,
                    },
                    {},
                )

            self.assertEqual(result["result"], "PASSED")
            expected = str(rawfiles / "rccl_tests_%h_%p_%m.profraw")
            self.assertEqual(captured["env"]["LLVM_PROFILE_FILE"], expected)


if __name__ == "__main__":
    unittest.main()
