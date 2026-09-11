#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard maintainer-script ownership boundaries between component packages.

amd-smi-lib-tests depends on amd-smi-lib, so removing or upgrading the tests
package must never disturb the main package's state. These tests assert that
invariant against the packaging templates themselves, deliberately without
prescribing HOW it is achieved: the tests component may be given its own
scriptlets, or simply be left without the main package's, and either satisfies
the contract.

Static assertions on the templates, so they need no root, package manager or
container. run_amdsmi_component_removal_test.py proves the same invariant at
runtime against real packages.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# State that belongs to the main package. The tests package must never touch
# any of it, on any transition.
MAIN_PACKAGE_STATE = (
    "/etc/ld.so.conf.d",
    "ldconfig",
    "/var/log/amd_smi_lib",
    "logrotate",
    "egg-info",
    "gpuv-smi",
)

RPM_SCRIPTLETS = (
    "POST_INSTALL_SCRIPT_FILE",
    "PRE_UNINSTALL_SCRIPT_FILE",
    "POST_UNINSTALL_SCRIPT_FILE",
)


def _read_template(relative_path):
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise unittest.SkipTest("{} not found".format(relative_path))
    return path.read_text(encoding="utf-8")


def _executable_lines(script):
    # The contract is about what a script RUNS, not what it documents, so a
    # maintainer can name the paths that must never be touched in a comment.
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


def _case_branches(script):
    """Map each `case "$1"` branch label to its body."""
    branches = {}
    pattern = re.compile(r"^\s*\(?\s*([a-z|* \t]+?)\s*\)\s*$(?P<body>.*?)^\s*;;", re.M | re.S)
    for match in pattern.finditer(script):
        branches[match.group(1).strip()] = match.group("body")
    return branches


def _removal_commands(script):
    # Matches both a plain `rm -rf ...` and a `find ... -exec rm -rf {} +`.
    return [line for line in script.splitlines() if re.search(r"\brm\b", line)]


class TestsPackageOwnershipTest(unittest.TestCase):
    """The tests package must confine itself to files it owns."""

    def setUp(self):
        self.script = _executable_lines(_read_template("DEBIAN/amd-smi-lib-tests/prerm.in"))

    def test_never_touches_main_package_state(self):
        for value in MAIN_PACKAGE_STATE:
            with self.subTest(state=value):
                self.assertNotIn(value, self.script)

    def test_removals_are_confined_to_the_tests_tree(self):
        # Every removal must name the tests or tools tree, either directly or
        # through a variable holding it, so nothing else can be swept up.
        for command in _removal_commands(self.script):
            with self.subTest(command=command.strip()):
                target = command.lower()
                self.assertTrue(
                    "test" in target or "tool" in target, "removes something outside the tests tree"
                )

    def test_nothing_destructive_on_upgrade(self):
        for label, body in _case_branches(self.script).items():
            if "upgrade" in label:
                with self.subTest(branch=label):
                    self.assertEqual(_removal_commands(body), [])


class MainPackageScriptletTest(unittest.TestCase):
    """The main package may only tear down its own state on a real removal."""

    def test_debian_prerm_removes_ldconfig_only_on_remove(self):
        script = _read_template("DEBIAN/prerm.in")
        calls = re.findall(r"^\s*rm_ldconfig\s*$", script, re.MULTILINE)
        self.assertEqual(calls, ["  rm_ldconfig"])
        body = re.search(
            r'if \[ "\$1" = "remove" \]; then(?P<body>.*?)^fi$', script, re.DOTALL | re.MULTILINE
        )
        self.assertIsNotNone(body, "final remove branch not found")
        self.assertIn("rm_ldconfig", body.group("body"))

    def test_rpm_teardown_is_final_erase_only(self):
        postun = _read_template("RPM/postun.in")
        guard = re.search(
            r'if \[ "\$1" -eq 0 \].*?; then(?P<body>.*?)^fi$', postun, re.DOTALL | re.MULTILINE
        )
        self.assertIsNotNone(guard, "postun is not guarded by an erase check")
        self.assertIn("/etc/ld.so.conf.d", guard.group("body"))

        preun = _read_template("RPM/preun.in")
        guard = re.search(
            r'if \[ "\$1" -eq 0 \]; then(?P<body>.*?)^fi$', preun, re.DOTALL | re.MULTILINE
        )
        self.assertIsNotNone(guard, "preun is not guarded by an erase check")
        for helper in ("rm_leftovers", "rm_logFolder", "rm_rocm_tests_dir", "rm_logrotateConfig"):
            with self.subTest(helper=helper):
                self.assertEqual(len(re.findall(r"^\s*{}\s*$".format(helper), preun, re.M)), 1)
                self.assertIn(helper, guard.group("body"))


class ComponentScopingTest(unittest.TestCase):
    """Scriptlets must name the component that owns the state they touch.

    An unprefixed CPACK_RPM_<KIND>_SCRIPT_FILE is copied into every component
    RPM, and on Debian a component with no control-extra of its own inherits
    the generic CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA. Verified with CPack 3.28.3:
    naming the component (CPACK_RPM_<COMPONENT>_<KIND>) confines the scriptlet
    to it, and a component named nowhere ships none.

    The state at stake -- /etc/ld.so.conf.d, /var/log/amd_smi_lib,
    /etc/logrotate.d -- sits at absolute paths every variant shares, so an
    inheriting component damages the main package even from another prefix.
    """

    def setUp(self):
        self.cmake = _executable_lines(_read_template("CMakeLists.txt"))

    def _is_set(self, name):
        return re.search(r"set\s*\(\s*{}[\s\n]".format(name), self.cmake) is not None

    def test_no_unprefixed_rpm_scriptlet_reaches_every_component(self):
        for kind in RPM_SCRIPTLETS:
            with self.subTest(scriptlet=kind):
                self.assertFalse(
                    self._is_set("CPACK_RPM_{}".format(kind)),
                    "CPACK_RPM_{0} is copied into EVERY component RPM, including asan and "
                    "any component added later. Name the owning component group instead: "
                    "CPACK_RPM_RUNTIME_{0}.".format(kind),
                )

    def test_no_generic_debian_control_extra(self):
        self.assertFalse(
            self._is_set("CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA"),
            "a component without its own control-extra inherits the generic one; "
            "set CPACK_DEBIAN_<COMPONENT>_PACKAGE_CONTROL_EXTRA per component instead",
        )

    def test_the_owning_group_still_declares_its_scriptlets(self):
        # Guards the other direction: scoping must not quietly drop the main
        # package's own ldconfig registration and cleanup.
        #
        # The name is the component GROUP, not the component:
        # generic_package_post() puts dev into the "runtime" group, so CPack
        # emits one package per group and looks up CPACK_*_RUNTIME_*. A
        # CPACK_*_DEV_* variable is accepted by CMake and silently ignored by
        # CPack, which ships a main package with no maintainer scripts at all.
        for kind in RPM_SCRIPTLETS:
            with self.subTest(scriptlet=kind):
                self.assertTrue(
                    self._is_set("CPACK_RPM_RUNTIME_{}".format(kind)),
                    "the main package lost its {}".format(kind),
                )
        self.assertTrue(self._is_set("CPACK_DEBIAN_RUNTIME_PACKAGE_CONTROL_EXTRA"))


if __name__ == "__main__":
    unittest.main()
