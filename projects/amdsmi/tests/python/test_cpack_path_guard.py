#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the CPack Python module path guard."""

import importlib.util
import os
import site
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_guard():
    # In the source tree the guard lives at tests/run_amdsmi_cpack_path_test.py;
    # in the installed tests layout REPO_ROOT is share/amd_smi and it lives at
    # tests/run_amdsmi_cpack_path_test.py there too.
    for cand in (
        REPO_ROOT / "tests" / "run_amdsmi_cpack_path_test.py",
        REPO_ROOT / "run_amdsmi_cpack_path_test.py",
    ):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("amdsmi_cpack_path_guard", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise unittest.SkipTest("run_amdsmi_cpack_path_test.py not found")


class SelectModulePathsTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()

    def test_accepts_canonical_debian_and_rpm_roots(self):
        site_dirs = [
            "/usr/lib/python3/dist-packages",
            "/usr/lib64/python3.9/site-packages",
            "/usr/lib/python3.9/site-packages",
        ]
        paths = [site_dir + "/amdsmi/amdsmi_interface.py" for site_dir in site_dirs]
        self.assertEqual(self.guard.select_module_paths(paths, site_dirs), paths)

    def test_rejects_non_import_paths(self):
        site_dirs = ["/usr/lib/python3.10/dist-packages", "/usr/lib/python3/dist-packages"]
        rejected = [
            "/usr/lib/python3.11/dist-packages/amdsmi/amdsmi_interface.py",
            "/usr/lib/python3/not-site-packages/amdsmi/amdsmi_interface.py",
            "/usr/lib/python3/not-dist-packages/amdsmi/amdsmi_interface.py",
            "/opt/acme/site-packages/amdsmi/amdsmi_interface.py",
            "/usr/share/site-packages/amdsmi/amdsmi_interface.py",
            "/usr/lib/python3/dist-packages-old/amdsmi/amdsmi_interface.py",
            "/usr/lib/python3.10/dist-packages/../../amdsmi/amdsmi_interface.py",
        ]
        for path in rejected:
            with self.subTest(path=path):
                self.assertEqual(self.guard.select_module_paths([path], site_dirs), [])


class ParseDpkgListingTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()

    def test_symlink_target_is_ignored_and_spaces_are_preserved(self):
        listing = (
            "lrwxrwxrwx root/root 0 2026-08-04 12:00 "
            "./usr/lib/python3/dist-packages/amdsmi/link.py -> target.py\n"
            "-rw-r--r-- root/root 10 2026-08-04 12:00 "
            "./usr/lib/python3/dist-packages/amdsmi/path with space.py\n"
        )
        self.assertEqual(
            self.guard.parse_dpkg_listing(listing),
            [
                "/usr/lib/python3/dist-packages/amdsmi/link.py",
                "/usr/lib/python3/dist-packages/amdsmi/path with space.py",
            ],
        )


class MainTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.dpkg = self.base / "dpkg"
        self.dpkg.write_text("#!/bin/sh\nprintf '%s\\n' \"$DPKG_LISTING\"\n", encoding="utf-8")
        self.dpkg.chmod(0o755)
        old_path = os.environ.get("PATH")
        os.environ["PATH"] = str(self.base) + os.pathsep + (old_path or "")
        if old_path is None:
            self.addCleanup(os.environ.pop, "PATH", None)
        else:
            self.addCleanup(os.environ.__setitem__, "PATH", old_path)
        self.pkg = self.base / "pkg.deb"
        self.pkg.touch()

    def test_main_rejects_substring_false_positive(self):
        site_dir = "/usr/lib/python3/dist-packages"
        os.environ["DPKG_LISTING"] = (
            "-rw-r--r-- root/root 10 2026-08-04 12:00 "
            "./usr/lib/python3/dist-packages/amdsmi/amdsmi_interface.py"
        )
        self.addCleanup(os.environ.pop, "DPKG_LISTING", None)
        self.assertEqual(self.guard.main(["--site-dir", site_dir, str(self.pkg)]), 0)

        os.environ["DPKG_LISTING"] = (
            "-rw-r--r-- root/root 10 2026-08-04 12:00 "
            "./usr/lib/python3/not-site-packages/amdsmi/amdsmi_interface.py"
        )
        with self.assertRaises(SystemExit) as raised:
            self.guard.main(["--site-dir", site_dir, str(self.pkg)])
        self.assertNotEqual(raised.exception.code, 0)


class SubprocessMainTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        dpkg = self.base / "dpkg"
        dpkg.write_text("#!/bin/sh\nprintf '%s\\n' \"$DPKG_LISTING\"\n", encoding="utf-8")
        dpkg.chmod(0o755)
        self.pkg = self.base / "pkg.deb"
        self.pkg.touch()

    def _run(self, listing, *extra_args):
        env = os.environ.copy()
        env["PATH"] = str(self.base) + os.pathsep + env.get("PATH", "")
        env["DPKG_LISTING"] = listing
        return subprocess.run(
            [sys.executable, str(Path(self.guard.__file__)), str(self.pkg)] + list(extra_args),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def test_legacy_cli_rejects_substring_false_positive(self):
        result = self._run(
            "-rw-r--r-- root/root 10 2026-08-04 12:00 "
            "./usr/lib/python3/not-site-packages/amdsmi/amdsmi_interface.py"
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("expected site directories", output)

    def test_accepts_the_site_directory_of_the_named_interpreter(self):
        getsitepackages = getattr(site, "getsitepackages", None)
        if getsitepackages is None:
            raise unittest.SkipTest("site.getsitepackages is unavailable")
        site_dirs = getsitepackages()
        if not site_dirs:
            raise unittest.SkipTest("site.getsitepackages returned no directories")
        site_dir = next((path for path in site_dirs if Path(path).exists()), site_dirs[0])
        listing = "-rw-r--r-- root/root 10 2026-08-04 12:00 .{}/amdsmi/amdsmi_interface.py".format(
            site_dir
        )
        result = self._run(listing, "--python", sys.executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_cli_rejects_package_without_amdsmi(self):
        result = self._run(
            "-rw-r--r-- root/root 10 2026-08-04 12:00 ./usr/share/doc/example/README"
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("expected site directories", output)


class DefaultPythonTest(unittest.TestCase):
    """The guard must inspect the same interpreter the packaging targeted."""

    def setUp(self):
        self.guard = _load_guard()

    def _with_existing(self, existing):
        real_exists = self.guard.os.path.exists
        self.guard.os.path.exists = lambda path: path in existing or real_exists(path)
        self.addCleanup(setattr, self.guard.os.path, "exists", real_exists)

    def test_prefers_platform_python(self):
        self._with_existing({"/usr/libexec/platform-python", "/usr/bin/python3"})
        self.assertEqual(self.guard.default_python(), "/usr/libexec/platform-python")

    def test_falls_back_to_usr_bin_python3(self):
        real_exists = self.guard.os.path.exists
        self.guard.os.path.exists = lambda path: path == "/usr/bin/python3"
        self.addCleanup(setattr, self.guard.os.path, "exists", real_exists)
        self.assertEqual(self.guard.default_python(), "/usr/bin/python3")

    def test_falls_back_to_the_running_interpreter(self):
        real_exists = self.guard.os.path.exists
        self.guard.os.path.exists = lambda path: False
        self.addCleanup(setattr, self.guard.os.path, "exists", real_exists)
        self.assertEqual(self.guard.default_python(), sys.executable)


if __name__ == "__main__":
    unittest.main()
