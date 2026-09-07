import os
import tempfile
import unittest
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
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class CoverageReportingTest(unittest.TestCase):
    def test_workspace_isolates_profiles_and_removes_stale_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "coverage-output"
            rawfiles = workspace / "logs" / "rawfiles"
            rawfiles.mkdir(parents=True)
            stale_profile = rawfiles / "stale.profraw"
            stale_profile.write_bytes(b"stale")

            executor = TestExecutor.__new__(TestExecutor)
            executor.paths = {"workdir": temp_dir}
            executor.build_config = {"install_flags": ["--debug"]}
            executor.args = _args(output=str(workspace))

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

    def test_report_only_preserves_existing_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "coverage-output"
            rawfiles = workspace / "logs" / "rawfiles"
            rawfiles.mkdir(parents=True)
            existing_profile = rawfiles / "existing.profraw"
            existing_profile.write_bytes(b"profile")

            executor = TestExecutor.__new__(TestExecutor)
            executor.paths = {"workdir": temp_dir}
            executor.build_config = {"install_flags": ["--debug"]}
            executor.args = _args(output=str(workspace), skip_tests=True)

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

    def _coverage_executor(self, root):
        """Wire an executor with the directory layout generate_coverage_report
        expects, under <root>."""
        rawfiles = root / "logs" / "rawfiles"
        report_dir = root / "report"
        build_dir = root / "build"
        rawfiles.mkdir(parents=True)
        report_dir.mkdir()
        build_dir.mkdir()
        (rawfiles / "profile.profraw").write_bytes(b"profile")
        (build_dir / "librccl.so").write_bytes(b"object")

        executor = TestExecutor.__new__(TestExecutor)
        executor.args = _args()
        executor.rawfiles_dir = str(rawfiles)
        executor.log_dir = str(root / "logs")
        executor.report_dir = str(report_dir)
        executor.build_dir = str(build_dir)
        executor.paths = {"rocm_path": str(root / "rocm")}
        executor.rccl_tests_build_config = {}
        return executor, build_dir

    def test_llvm_cov_failure_fails_report_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor, _ = self._coverage_executor(Path(temp_dir))

            command_results = [
                mock.Mock(returncode=0),
                CalledProcessError(1, ["llvm-cov", "show"], stderr="bad mapping"),
            ]
            with mock.patch.object(executor, "_rocm_root",
                                   return_value=str(Path(temp_dir) / "rocm")), \
                 mock.patch.object(executor, "_resolve_llvm_tool",
                                   side_effect=lambda name: name), \
                 mock.patch("lib.test_executor.subprocess.run",
                            side_effect=command_results):
                result = executor.generate_coverage_report()

            self.assertFalse(result)

    def test_device_elf_included_as_coverage_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executor, build_dir = self._coverage_executor(root)
            device_elf = build_dir / "device-gfx942.elf"
            device_elf.write_bytes(b"device")

            recorded = []

            def _record(cmd, *args, **kwargs):
                recorded.append(list(cmd))
                return mock.Mock(returncode=0)

            with mock.patch.object(executor, "_rocm_root",
                                   return_value=str(root / "rocm")), \
                 mock.patch.object(executor, "_resolve_llvm_tool",
                                   side_effect=lambda name: name), \
                 mock.patch("lib.test_executor.subprocess.run",
                            side_effect=_record):
                result = executor.generate_coverage_report()

            self.assertTrue(result)

            # The llvm-cov 'show' argv must attribute the per-arch device ELF via
            # '--object <path>'; the glob here and the DeviceLinker symlink are
            # coupled only by this bare filename convention.
            show_cmd = next(c for c in recorded if "show" in c)
            self.assertIn(str(device_elf), show_cmd)
            self.assertEqual(
                show_cmd[show_cmd.index(str(device_elf)) - 1], "--object"
            )


if __name__ == "__main__":
    unittest.main()
