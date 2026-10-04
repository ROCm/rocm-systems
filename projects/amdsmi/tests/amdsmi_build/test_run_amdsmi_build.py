# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the pure helpers in run_amdsmi_build.py.

These tests deliberately avoid anything that requires root, a build env, or
network access -- they cover argument parsing, /etc/os-release detection,
package-manager/format inference, package globbing, and the summarize step.

Run with:  python3 -m unittest discover -s projects/amdsmi/tests/amdsmi_build
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path


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


class SummarizeTests(unittest.TestCase):
    def test_pass_case(self):
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "build_result.txt").write_text("BUILD PASSED\n")
            (results / "install_result.txt").write_text("INSTALL PASSED\n")
            n = rab.summarize_results(results, "TestOS", None)
        self.assertEqual(n, 0)

    def test_pcie_only_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "build_result.txt").write_text("BUILD PASSED\n")
            (results / "install_result.txt").write_text("INSTALL PASSED\n")
            (results / "amdsmi_tests.log").write_text("[  PASSED  ] 157 tests.\n")
            failure = "[  FAILED  ] GpuUnit.PcieLegacyUnknownSpeedKeepsStaticInfo\n"
            (results / "pcie_unit_tests.log").write_text(failure)
            summary = results / "summary.md"
            output = io.StringIO()
            with redirect_stdout(output):
                count = rab.summarize_results(results, "TestOS", summary)
            self.assertEqual(count, 1)
            for text in (output.getvalue(), summary.read_text()):
                self.assertIn("CI Failed", text)
                self.assertIn("PCIe Unit Tests", text)
                self.assertIn(failure, text)
                self.assertNotIn("CI Passed", text)
                self.assertNotIn("All stages and tests passed successfully", text)

    def test_pcie_success_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            results = Path(td)
            (results / "amdsmi_tests.log").write_text("[  PASSED  ] 157 tests.\n")
            (results / "pcie_unit_tests.log").write_text("[  PASSED  ] 29 tests.\n")
            summary = results / "summary.md"
            output = io.StringIO()
            with redirect_stdout(output):
                count = rab.summarize_results(results, "TestOS", summary)
            self.assertEqual(count, 0)
            for text in (output.getvalue(), summary.read_text()):
                self.assertIn("CI Passed", text)
                self.assertNotIn("CI Failed", text)

    def test_cpp_suites_report_failures_independently(self) -> None:
        for pcie_fails in (False, True):
            with self.subTest(pcie_fails=pcie_fails):
                with tempfile.TemporaryDirectory() as td:
                    results = Path(td)
                    (results / "amdsmi_tests.log").write_text("[  FAILED  ] MainCase\n")
                    pcie = "[  FAILED  ] PcieCase\n" if pcie_fails else "[  PASSED  ] 29 tests.\n"
                    (results / "pcie_unit_tests.log").write_text(pcie)
                    output = io.StringIO()
                    with redirect_stdout(output):
                        count = rab.summarize_results(results, "TestOS", None)
                    self.assertEqual(count, 2 if pcie_fails else 1)
                    self.assertIn("AMDSMI Tests (1)", output.getvalue())
                    self.assertEqual("PCIe Unit Tests (1)" in output.getvalue(), pcie_fails)
                    self.assertNotIn("CI Passed", output.getvalue())

    def test_summarize_command_reports_pcie_result(self) -> None:
        for marker, expected_exit in (("FAILED", 1), ("PASSED", 0)):
            with self.subTest(marker=marker):
                with tempfile.TemporaryDirectory() as td:
                    results = Path(td)
                    (results / "pcie_unit_tests.log").write_text(f"[  {marker}  ] PcieCase\n")
                    summary = results / "summary.md"
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            rab.__file__,
                            "summarize",
                            "--results-dir",
                            td,
                            "--os-label",
                            "TestOS",
                            "--summary-file",
                            str(summary),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, expected_exit, result.stdout + result.stderr
                    )
                    heading = "CI Failed" if expected_exit else "CI Passed"
                    self.assertIn(heading, result.stdout)
                    self.assertIn(heading, summary.read_text())

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


if __name__ == "__main__":
    unittest.main()
