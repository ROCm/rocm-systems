"""Unit tests for the pure helpers in run_amdsmi_build.py.

These tests deliberately avoid anything that requires root, a build env, or
network access -- they cover argument parsing, /etc/os-release detection,
package-manager/format inference, package globbing, and the summarize step.

Run with:  python3 -m unittest discover -s projects/amdsmi/tests/amdsmi_build
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "run_amdsmi_build", here / "run_amdsmi_build.py"
    )
    mod = importlib.util.module_from_spec(spec)
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
        self.assertFalse(prof["debian10_sources"])
        self.assertFalse(prof["qa_rpaths"])

    def test_debian10_sets_archive_sources(self):
        osr = _make_osr(
            self.tmp,
            """
            ID=debian
            VERSION_ID="10"
            """,
        )
        prof = rab.detect_os_profile(osr)
        self.assertEqual(prof["os_label"], "Debian10")
        self.assertTrue(prof["debian10_sources"])
        self.assertEqual(prof["package_manager"], "apt")

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
