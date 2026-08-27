#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the package upgrade and downgrade guard."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Result:
    def __init__(self, stdout):
        self.stdout = stdout


class _FakeSubprocess:
    PIPE = object()
    STDOUT = object()

    def __init__(self, module_file, owner_output):
        self.module_file = module_file
        self.owner_output = owner_output

    def check_output(self, cmd, **kwargs):
        return self.module_file

    def run(self, cmd, **kwargs):
        return _Result(self.owner_output)


def _load_guard():
    # Probe both the source-tree and installed-tests layouts.
    for cand in (
        REPO_ROOT / "tests" / "run_amdsmi_upgrade_downgrade_test.py",
        REPO_ROOT / "run_amdsmi_upgrade_downgrade_test.py",
    ):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("amdsmi_upgrade_downgrade_guard", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise unittest.SkipTest("run_amdsmi_upgrade_downgrade_test.py not found")


class InstallCommandTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()

    def test_commands(self):
        cases = (
            (
                "apt",
                False,
                ["apt-get", "install", "-y", "--allow-downgrades", "--reinstall", "pkg"],
            ),
            ("apt", True, ["apt-get", "install", "-y", "--allow-downgrades", "--reinstall", "pkg"]),
            ("dnf", False, ["dnf", "install", "-y", "pkg"]),
            ("dnf", True, ["dnf", "downgrade", "-y", "pkg"]),
            (
                "zypper",
                False,
                ["zypper", "--non-interactive", "install", "--allow-downgrade", "pkg"],
            ),
            (
                "zypper",
                True,
                ["zypper", "--non-interactive", "install", "--allow-downgrade", "pkg"],
            ),
        )
        for manager, downgrade, expected in cases:
            with self.subTest(manager=manager, downgrade=downgrade):
                self.assertEqual(
                    self.guard.install_command(manager, "pkg", downgrade=downgrade), expected
                )

    def test_unknown_manager(self):
        with self.assertRaises(ValueError):
            self.guard.install_command("unknown", "pkg")


class DistinctPackagesTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()

    def test_identical_packages_fail(self):
        with self.assertRaises(SystemExit):
            self.guard.assert_distinct_packages("amd-smi-lib-1", "amd-smi-lib-1")

    def test_different_packages_pass(self):
        self.assertIsNone(self.guard.assert_distinct_packages("amd-smi-lib-1", "amd-smi-lib-2"))


class ImportProvenanceTest(unittest.TestCase):
    """The imported module must belong to the package under test."""

    def setUp(self):
        self.guard = _load_guard()

    def test_ownership_is_matched_exactly(self):
        cases = (
            ("amd-smi-lib: /usr/lib/python3/dist-packages/amdsmi/__init__.py", "apt", True),
            # A sibling package's name CONTAINS the main package's, so a
            # substring test would wrongly accept it.
            ("amd-smi-lib-tests: /usr/lib/python3/dist-packages/amdsmi/__init__.py", "apt", False),
            (
                "dpkg-query: no path found matching pattern "
                "/usr/local/lib/python3.10/dist-packages/amdsmi/__init__.py",
                "apt",
                False,
            ),
            ("", "apt", False),
            ("amd-smi-lib", "dnf", True),
            ("amd-smi-lib-tests", "dnf", False),
            ("file /x is not owned by any package", "dnf", False),
        )
        for output, manager, owned in cases:
            with self.subTest(output=output[:48], manager=manager):
                self.assertEqual(
                    self.guard.import_is_package_owned(output, manager, "amd-smi-lib"), owned
                )

    def test_verify_rejects_unowned_wheel_path(self):
        wheel = "/usr/local/lib/python3.10/dist-packages/amdsmi/__init__.py"
        self.guard.subprocess = _FakeSubprocess(
            wheel, "dpkg-query: no path found matching pattern " + wheel
        )
        with self.assertRaises(SystemExit) as raised:
            self.guard._verify("apt", "amd-smi-lib")
        self.assertIn(wheel, str(raised.exception))

    def test_verify_rejects_a_sibling_package(self):
        path = "/usr/lib/python3/dist-packages/amdsmi/__init__.py"
        self.guard.subprocess = _FakeSubprocess(path, "amd-smi-lib-tests: " + path)
        with self.assertRaises(SystemExit) as raised:
            self.guard._verify("apt", "amd-smi-lib")
        self.assertIn("amd-smi-lib-tests", str(raised.exception))

    def test_verify_owned_import_reaches_cli_check(self):
        path = "/usr/lib/python3/dist-packages/amdsmi/__init__.py"
        self.guard.subprocess = _FakeSubprocess(path, "amd-smi-lib: " + path)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old = os.environ.get("ROCM_PATH")
        os.environ["ROCM_PATH"] = tmp.name
        if old is None:
            self.addCleanup(os.environ.pop, "ROCM_PATH", None)
        else:
            self.addCleanup(os.environ.__setitem__, "ROCM_PATH", old)
        with self.assertRaises(SystemExit) as raised:
            self.guard._verify("apt", "amd-smi-lib")
        self.assertIn("amd-smi", str(raised.exception))
        self.assertNotIn("owned by", str(raised.exception))


class TransitionSequenceTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()
        self.commands = []
        original_run = self.guard._run
        original_verify = self.guard._verify
        original_identity = self.guard.read_package_identity
        self.guard._run = self.commands.append
        self.guard._verify = lambda manager, expected: None
        self.guard.read_package_identity = lambda pkg, manager: (
            "old" if "old" in str(pkg) else "new"
        )
        original_name = self.guard.read_package_name
        self.guard.read_package_name = lambda pkg, manager: "amd-smi-lib"
        self.addCleanup(setattr, self.guard, "read_package_name", original_name)
        self.addCleanup(setattr, self.guard, "_run", original_run)
        self.addCleanup(setattr, self.guard, "_verify", original_verify)
        self.addCleanup(setattr, self.guard, "read_package_identity", original_identity)

    def _run_main(self, manager):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.pkg"
            new = Path(tmp) / "new.pkg"
            old.touch()
            new.touch()
            result = self.guard.main(
                ["--old-package", str(old), "--new-package", str(new), "--package-manager", manager]
            )
            self.assertEqual(result, 0)
            return str(old), str(new)

    def test_dnf_sequence(self):
        old, new = self._run_main("dnf")
        self.assertEqual(
            self.commands,
            [
                ["dnf", "install", "-y", old],
                ["dnf", "install", "-y", new],
                ["dnf", "downgrade", "-y", old],
            ],
        )

    def test_apt_sequence(self):
        old, new = self._run_main("apt")
        self.assertEqual(
            self.commands,
            [
                ["apt-get", "install", "-y", "--allow-downgrades", "--reinstall", old],
                ["apt-get", "install", "-y", "--allow-downgrades", "--reinstall", new],
                ["apt-get", "install", "-y", "--allow-downgrades", "--reinstall", old],
            ],
        )

    def test_identical_packages_fail_before_install(self):
        self.guard.read_package_identity = lambda pkg, manager: "same-package"
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.pkg"
            new = Path(tmp) / "new.pkg"
            old.touch()
            new.touch()
            with self.assertRaises(SystemExit):
                self.guard.main(
                    [
                        "--old-package",
                        str(old),
                        "--new-package",
                        str(new),
                        "--package-manager",
                        "dnf",
                    ]
                )
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
