#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Unit tests for the system-wheel builder and the scriptlet install helper.

Exercises tools/build_system_wheel.py (version normalization, RECORD integrity,
file selection) and py-interface/amdsmi_install_wheel.py (site-packages
selection, stdlib-extract fallback, version-scoped uninstall) against synthetic
trees, so both are proven without an installed package, pip, or a GPU.
"""

import base64
import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name, *candidates):
    for cand in candidates:
        if cand.is_file():
            spec = importlib.util.spec_from_file_location(module_name, cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise unittest.SkipTest("{} not found".format(module_name))


def _load_builder():
    return _load(
        "amdsmi_build_system_wheel",
        REPO_ROOT / "tools" / "build_system_wheel.py",
        REPO_ROOT / "build_system_wheel.py",
    )


def _load_installer():
    return _load(
        "amdsmi_install_wheel",
        REPO_ROOT / "py-interface" / "amdsmi_install_wheel.py",
        REPO_ROOT / "amdsmi_install_wheel.py",
    )


def _make_package(root: Path) -> Path:
    pkg = root / "amdsmi"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("__version__ = 'x'\n", encoding="utf-8")
    (pkg / "amdsmi_interface.py").write_text("VALUE = 1\n", encoding="utf-8")
    # A stray .so must never be included in the pure-python wheel.
    (pkg / "libamd_smi.so").write_text("binary", encoding="utf-8")
    return pkg


class NormalizeVersionTest(unittest.TestCase):
    def setUp(self):
        self.builder = _load_builder()

    def test_cases(self):
        cases = {
            "27.0.0": "27.0.0",
            "27.0.0+af02525-dirty": "27.0.0+af02525.dirty",
            "27.0.0+a.b.c": "27.0.0+a.b.c",
            "27.0.0+---": "27.0.0",
            "27.0.0+": "27.0.0",
            "27.0.0+.foo.": "27.0.0+foo",
        }
        for raw, expected in cases.items():
            self.assertEqual(self.builder._normalize_version(raw), expected)


class BuildWheelTest(unittest.TestCase):
    def setUp(self):
        self.builder = _load_builder()
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, version="27.0.0"):
        _make_package(self.base / "pkg")
        out = self.base / "out"
        with mock.patch(
            "sys.argv",
            [
                "build_system_wheel.py",
                "--package-dir",
                str(self.base / "pkg"),
                "--version",
                version,
                "--output-dir",
                str(out),
            ],
        ):
            self.builder.main()
        wheels = list(out.glob("amdsmi-*.whl"))
        self.assertEqual(len(wheels), 1)
        return wheels[0]

    def test_wheel_filename_is_normalized(self):
        wheel = self._build("27.0.0+af02525-dirty")
        self.assertEqual(wheel.name, "amdsmi-27.0.0+af02525.dirty-py3-none-any.whl")

    def test_wheel_excludes_so_and_has_valid_record(self):
        wheel = self._build()
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            self.assertIn("amdsmi/__init__.py", names)
            self.assertNotIn("amdsmi/libamd_smi.so", names)
            record = zf.read("amdsmi-27.0.0.dist-info/RECORD").decode("utf-8")
            # Every non-RECORD entry's recorded hash must match its content.
            for line in record.splitlines():
                name, digest, _size = line.split(",")
                if not digest:
                    continue
                data = zf.read(name)
                want = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode(
                    "ascii"
                ).rstrip("=")
                self.assertEqual(digest, want, name)


class TargetSitelibTest(unittest.TestCase):
    def setUp(self):
        self.installer = _load_installer()

    def _select(self, candidates):
        with mock.patch.object(self.installer.site, "getsitepackages", return_value=candidates):
            return str(self.installer._target_sitelib())

    def test_debian_primary(self):
        self.assertEqual(
            self._select(["/usr/lib/python3.11/site-packages", "/usr/lib/python3/dist-packages"]),
            "/usr/lib/python3/dist-packages",
        )

    def test_rhel_versioned_site_packages(self):
        self.assertEqual(
            self._select(
                ["/usr/lib64/python3.9/site-packages", "/usr/lib/python3.9/site-packages"]
            ),
            "/usr/lib/python3.9/site-packages",
        )

    def test_empty_falls_back_to_purelib(self):
        import sysconfig

        self.assertEqual(self._select([]), sysconfig.get_paths()["purelib"])

    def test_versioned_dist_packages(self):
        self.assertEqual(
            self._select(["/usr/lib/python3.11/dist-packages"]), "/usr/lib/python3.11/dist-packages"
        )

    def test_lib64_only_site_packages(self):
        self.assertEqual(
            self._select(["/usr/lib64/python3.9/site-packages"]),
            "/usr/lib64/python3.9/site-packages",
        )

    def test_getsitepackages_attributeerror_falls_back(self):
        with mock.patch.object(self.installer.site, "getsitepackages", side_effect=AttributeError):
            self.assertTrue(str(self.installer._target_sitelib()))


class WheelVersionTest(unittest.TestCase):
    def setUp(self):
        self.installer = _load_installer()

    def test_parse(self):
        self.assertEqual(
            self.installer._wheel_version(Path("amdsmi-27.0.0+a.b-py3-none-any.whl")), "27.0.0+a.b"
        )


class InstallUninstallTest(unittest.TestCase):
    def setUp(self):
        self.builder = _load_builder()
        self.installer = _load_installer()
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.sitelib = self.base / "site-packages"
        self.sitelib.mkdir()
        # Build a real wheel and stage it under a wheels/ dir next to a fake
        # helper location so _find_wheel() discovers it.
        _make_package(self.base / "pkg")
        wheelhouse = self.base / "share" / "wheels"
        with mock.patch(
            "sys.argv",
            [
                "build_system_wheel.py",
                "--package-dir",
                str(self.base / "pkg"),
                "--version",
                "27.0.0",
                "--output-dir",
                str(wheelhouse),
            ],
        ):
            self.builder.main()
        self.wheel = next(wheelhouse.glob("amdsmi-*.whl"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_extract_fallback_installs_and_replaces(self):
        # Pre-existing stale copy must be removed before extraction.
        stale = self.sitelib / "amdsmi"
        stale.mkdir()
        (stale / "stale.py").write_text("old\n", encoding="utf-8")
        self.installer._extract(self.wheel, self.sitelib)
        self.assertTrue((self.sitelib / "amdsmi" / "__init__.py").is_file())
        self.assertFalse((self.sitelib / "amdsmi" / "stale.py").exists())
        self.assertTrue(any(self.sitelib.glob("amdsmi-27.0.0.dist-info")))

    def test_uninstall_removes_only_our_version(self):
        self.installer._extract(self.wheel, self.sitelib)
        with (
            mock.patch.object(self.installer, "_target_sitelib", return_value=self.sitelib),
            mock.patch.object(self.installer, "_find_wheel", return_value=self.wheel),
            mock.patch.object(self.installer, "_pip_available", return_value=False),
        ):
            self.installer.uninstall()
        self.assertFalse((self.sitelib / "amdsmi").exists())
        self.assertFalse(any(self.sitelib.glob("amdsmi-27.0.0.dist-info")))

    def test_uninstall_leaves_foreign_version(self):
        # A user-installed different version must not be removed.
        foreign_pkg = self.sitelib / "amdsmi"
        foreign_pkg.mkdir()
        (foreign_pkg / "__init__.py").write_text("user\n", encoding="utf-8")
        foreign_di = self.sitelib / "amdsmi-99.9.9.dist-info"
        foreign_di.mkdir()
        with (
            mock.patch.object(self.installer, "_target_sitelib", return_value=self.sitelib),
            mock.patch.object(self.installer, "_find_wheel", return_value=self.wheel),
            mock.patch.object(self.installer, "_pip_available", return_value=False),
        ):
            self.installer.uninstall()
        self.assertTrue(foreign_di.is_dir())
        self.assertTrue((self.sitelib / "amdsmi" / "__init__.py").is_file())

    def test_install_uses_pip_target_then_extract_on_failure(self):
        calls = {}

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd

            class R:
                returncode = 1

            return R()

        with (
            mock.patch.object(self.installer, "_target_sitelib", return_value=self.sitelib),
            mock.patch.object(self.installer, "_find_wheel", return_value=self.wheel),
            mock.patch.object(self.installer, "_pip_available", return_value=True),
            mock.patch.object(self.installer.subprocess, "run", side_effect=fake_run),
        ):
            self.installer.install()
        # pip was invoked with --target pointing at the detected sitelib...
        self.assertIn("--target", calls["cmd"])
        self.assertIn(str(self.sitelib), calls["cmd"])
        # ...and the non-zero pip exit fell back to a working stdlib extract.
        self.assertTrue((self.sitelib / "amdsmi" / "__init__.py").is_file())

    def test_extract_drops_path_traversal_entries(self):
        import zipfile as zf

        evil = self.base / "evil.whl"
        with zf.ZipFile(evil, "w") as archive:
            archive.writestr("amdsmi/__init__.py", "ok\n")
            archive.writestr("amdsmi/../../escape.py", "evil\n")
        self.installer._extract(evil, self.sitelib)
        self.assertFalse((self.base / "escape.py").exists())
        self.assertFalse((self.sitelib / "escape.py").exists())
        self.assertTrue((self.sitelib / "amdsmi" / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
