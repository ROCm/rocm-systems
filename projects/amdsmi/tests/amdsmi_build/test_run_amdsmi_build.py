# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the pure helpers in run_amdsmi_build.py.

These tests deliberately avoid anything that requires root, a build env, or
network access -- they cover argument parsing, /etc/os-release detection,
package-manager/format inference, package globbing, the Clang compiler
check, and the summarize step.

Run with:  python3 -m unittest discover -s projects/amdsmi/tests/amdsmi_build
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "run_amdsmi_build", here / "run_amdsmi_build.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves its module via sys.modules on
    # Python 3.12+, so a module loaded outside sys.modules fails to import.
    sys.modules[spec.name] = mod
    # Bootstrap is a no-op on Python 3.7+ which the test env always satisfies.
    spec.loader.exec_module(mod)
    return mod


rab = _load_module()


def _make_osr(tmp: Path, body: str) -> Path:
    path = tmp / "os-release"
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return path


class DetectOsProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.addCleanup(self.tmpdir.cleanup)

    def test_missing_file_returns_empty(self):
        self.assertEqual(rab.detect_os_profile(self.tmp / "nope"), {})

    def test_ubuntu22(self):
        osr = _make_osr(
            self.tmp,
            """
            ID=ubuntu
            VERSION_ID="22.04"
            """,
        )
        prof = rab.detect_os_profile(osr)
        self.assertEqual(prof["package_manager"], "apt")
        self.assertEqual(prof["package_format"], "deb")
        self.assertEqual(prof["os_label"], "Ubuntu22")
        self.assertFalse(prof["qa_rpaths"])

    def test_debian13(self):
        osr = _make_osr(
            self.tmp,
            """
            ID=debian
            VERSION_ID="13"
            """,
        )
        prof = rab.detect_os_profile(osr)
        self.assertEqual(prof["os_label"], "Debian13")
        self.assertEqual(prof["package_manager"], "apt")
        self.assertEqual(prof["package_format"], "deb")

    def test_rhel10_sets_qa_rpaths(self):
        osr = _make_osr(
            self.tmp,
            """
            ID="rhel"
            VERSION_ID="10.0"
            """,
        )
        prof = rab.detect_os_profile(osr)
        self.assertEqual(prof["os_label"], "RHEL10")
        self.assertTrue(prof["qa_rpaths"])

    def test_almalinux8_sets_qa_rpaths(self):
        osr = _make_osr(
            self.tmp,
            """
            ID=almalinux
            VERSION_ID="8.10"
            """,
        )
        prof = rab.detect_os_profile(osr)
        self.assertEqual(prof["os_label"], "AlmaLinux8")
        self.assertTrue(prof["qa_rpaths"])

    def test_azurelinux3_sets_quirks(self):
        osr = _make_osr(
            self.tmp,
            """
            ID=azurelinux
            VERSION_ID="3.0"
            """,
        )
        prof = rab.detect_os_profile(osr)
        self.assertEqual(prof["os_label"], "AzureLinux3")
        self.assertTrue(prof["skip_setuptools_upgrade"])
        self.assertTrue(prof["install_more_itertools"])
        self.assertEqual(prof["package_manager"], "dnf")

    def test_sles_family(self):
        osr = _make_osr(
            self.tmp,
            """
            ID="sles"
            VERSION_ID="15.5"
            ID_LIKE="suse"
            """,
        )
        prof = rab.detect_os_profile(osr)
        self.assertEqual(prof["package_manager"], "zypper")
        self.assertEqual(prof["package_format"], "rpm")
        self.assertEqual(prof["os_label"], "SLES")


class DetectPackageMgrTests(unittest.TestCase):
    def test_explicit_passthrough(self):
        self.assertEqual(rab.detect_package_manager("apt"), "apt")
        self.assertEqual(rab.detect_package_manager("dnf"), "dnf")
        self.assertEqual(rab.detect_package_manager("zypper"), "zypper")

    def test_format_inference(self):
        self.assertEqual(rab.detect_package_format("apt", None), "deb")
        self.assertEqual(rab.detect_package_format("dnf", None), "rpm")
        self.assertEqual(rab.detect_package_format("zypper", None), "rpm")
        # explicit override wins
        self.assertEqual(rab.detect_package_format("apt", "rpm"), "rpm")

    def test_unsupported_manager_raises(self):
        with self.assertRaises(ValueError):
            rab.detect_package_format("xbps", None)


class PackageGlobTests(unittest.TestCase):
    def test_deb_glob(self):
        self.assertEqual(rab.package_glob("deb"), "amd-smi-lib*99999-local_amd64.deb")

    def test_rpm_glob(self):
        self.assertEqual(rab.package_glob("rpm"), "amd-smi-lib-*99999-local*.rpm")


class LocatePackageTests(unittest.TestCase):
    def test_prefers_main_over_tests(self):
        with tempfile.TemporaryDirectory() as td:
            build = Path(td)
            main_pkg = build / "amd-smi-lib_99999-local_amd64.deb"
            tests_pkg = build / "amd-smi-lib-tests_99999-local_amd64.deb"
            main_pkg.write_bytes(b"")
            tests_pkg.write_bytes(b"")
            picked = rab.locate_package(build, "deb")
            self.assertEqual(picked.name, "amd-smi-lib_99999-local_amd64.deb")

    def test_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                rab.locate_package(Path(td), "deb")


def _write_tool(directory: Path, name: str, body: str = "exit 0") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / name
    tool.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    tool.chmod(0o755)


class ClangToolchainPresentTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.bin = Path(self.tmpdir.name) / "bin"
        self.bin.mkdir()

    def _present(self, **extra_env) -> bool:
        env = {k: v for k, v in os.environ.items() if k not in ("CC", "CXX")}
        env["PATH"] = str(self.bin)
        env.update(extra_env)
        with mock.patch.dict(os.environ, env, clear=True):
            return rab.clang_toolchain_present()

    def test_amdclang_on_path_alone_is_not_enough(self):
        # CMake looks for amdclang only under ROCm's LLVM, so even a working
        # amdclang++ on PATH does not count (the CI images carry ccache links
        # by that name).
        _write_tool(self.bin, "amdclang++")
        _write_tool(self.bin, "amdclang")
        self.assertFalse(self._present())

    def test_system_clang_pair_is_present(self):
        _write_tool(self.bin, "clang++")
        _write_tool(self.bin, "clang")
        self.assertTrue(self._present())

    def test_clang_without_c_driver_is_missing(self):
        _write_tool(self.bin, "clang++")
        self.assertFalse(self._present())

    def test_clang_pair_that_does_not_run_is_missing(self):
        _write_tool(self.bin, "clang++", "exit 1")
        _write_tool(self.bin, "clang", "exit 1")
        self.assertFalse(self._present())

    def test_hipconfig_without_a_path_searches_path(self):
        _write_tool(self.bin, "hipconfig", "true")
        _write_tool(self.bin, "clang++")
        _write_tool(self.bin, "clang")
        self.assertTrue(self._present())

    def test_cc_cxx_from_environment_skip_the_search(self):
        self.assertTrue(self._present(CC="gcc", CXX="g++"))

    def test_rocm_install_searches_only_rocm_llvm(self):
        rocm = Path(self.tmpdir.name) / "rocm"
        _write_tool(self.bin, "hipconfig", f"echo {rocm}")
        _write_tool(self.bin, "clang++")
        _write_tool(self.bin, "clang")
        self.assertFalse(self._present())
        llvm_bin = rocm / "lib" / "llvm" / "bin"
        _write_tool(llvm_bin, "amdclang++", "exit 1")
        _write_tool(llvm_bin, "amdclang", "exit 1")
        self.assertFalse(self._present())
        _write_tool(llvm_bin, "clang++")
        _write_tool(llvm_bin, "clang")
        self.assertTrue(self._present())
        _write_tool(llvm_bin, "amdclang++")
        _write_tool(llvm_bin, "amdclang")
        self.assertTrue(self._present())


class InstallClangTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.bin = Path(self.tmpdir.name) / "bin"
        self.bin.mkdir()
        self.cfg = types.SimpleNamespace(
            package_manager="apt", refresh_apt=False, retries=1, log_dir=Path(self.tmpdir.name)
        )

    def _install(self) -> mock.MagicMock:
        env = {k: v for k, v in os.environ.items() if k not in ("CC", "CXX")}
        env["PATH"] = str(self.bin)
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(rab, "run_command") as run_command:
                rab.install_clang(self.cfg)
        return run_command

    def test_installs_distro_clang_when_none_works(self):
        _write_tool(self.bin, "amdclang++", "exit 1")
        _write_tool(self.bin, "amdclang", "exit 1")
        run_command = self._install()
        run_command.assert_called_once()
        self.assertEqual(
            run_command.call_args[0][0],
            ["apt-get", "install", "-y", "--no-install-recommends", "clang"],
        )

    def test_dnf_installs_clang_with_the_shared_prefix(self):
        self.cfg.package_manager = "dnf"
        run_command = self._install()
        run_command.assert_called_once()
        self.assertEqual(
            run_command.call_args[0][0],
            ["dnf", "install", "-y", "--setopt=skip_if_unavailable=True", "clang"],
        )

    def test_skips_install_when_rocm_lacks_llvm(self):
        # CMake searches only ROCm's LLVM once hipconfig finds ROCm.
        _write_tool(self.bin, "hipconfig", f"echo {self.tmpdir.name}/rocm")
        self._install().assert_not_called()


class SummarizeTests(unittest.TestCase):
    def test_pass_case(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "build_result.txt").write_text("BUILD PASSED\n")
            (results / "install_result.txt").write_text("INSTALL PASSED\n")
            n = rab.summarize_results(results, "TestOS", None)
        self.assertEqual(n, 0)

    def test_fail_case_counts(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "build_result.txt").write_text("BUILD FAILED: cmake exited 1\n")
            (results / "amdsmi_tests.log").write_text("[  FAILED  ] one\n[  FAILED  ] two\n")
            (results / "amd-smi_version.log").write_text("Traceback (most recent call last):\n")
            (results / "integration_test_output.txt").write_text("FAIL: test_a\n")
            summary = results / "summary.md"
            n = rab.summarize_results(results, "TestOS", summary)
            self.assertGreater(n, 0)
            self.assertIn(":x: CI Failed", summary.read_text())

    def test_command_failure_fails_the_summary(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "amd-smi_list.log").write_text("...\nError code: 2\n")
            summary = results / "summary.md"
            n = rab.summarize_results(results, "TestOS", summary)
            self.assertEqual(n, 1)
            self.assertIn("Command Tests (1)", summary.read_text())

    def test_wrapped_status_code_is_not_a_command_failure(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "amd-smi_xgmi.log").write_text("Error code:\n\t2 | not supported\n")
            self.assertEqual(rab.summarize_results(results, "TestOS", None), 0)

    def test_exit_code_line_fails_a_suite_without_failed_tests(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "amdsmi_tests.log").write_text("[ RUN      ] A.B\nExit code: 139\n")
            (results / "cli_test_output.txt").write_text("Ran 3 tests\nExit code: 1\n")
            (results / "amd_smi_nodrm_ex.log").write_text("no device\nExit code: 1\n")
            summary = results / "summary.md"
            n = rab.summarize_results(results, "TestOS", summary)
            text = summary.read_text()
            self.assertEqual(n, 3)
            self.assertIn("AMDSMI Tests (exit 139)", text)
            self.assertIn("cli test (exit 1)", text)
            self.assertIn("Example amd_smi_nodrm_ex", text)

    def test_cli_test_failures_are_counted(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "cli_test_output.txt").write_text("FAIL: test_command\n")
            self.assertEqual(rab.summarize_results(results, "TestOS", None), 1)

    def test_suite_output_cannot_hide_the_exit_code_line(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            log = "Exit code: 0\nRan 3 tests\n\nExit code: 1\n"
            (results / "unit_test_output.txt").write_text(log)
            summary = results / "summary.md"
            self.assertEqual(rab.summarize_results(results, "TestOS", summary), 1)
            self.assertIn("unit test (exit 1)", summary.read_text())


if __name__ == "__main__":
    unittest.main()
