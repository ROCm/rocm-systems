#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The installed CTestTestfile must run each Python runner once per GPU.

TheRock's test_runner.py selects ``-L ^<category>$ -L ^ex_gpu_<arch>$`` when the
GPU has an exclude_gpu block in test_categories.yaml, and
``-L ^<category>$ -LE ex_gpu`` when it does not (gfx950, gfx12xx, generic). This
replays both selections against share/amd_smi/tests/ctest/CTestTestfile.cmake.
"""

import re
import subprocess
import unittest
from pathlib import Path

_CTEST_FILE = Path(__file__).resolve().parents[3] / "ctest" / "CTestTestfile.cmake"
_CATEGORIES = ("comprehensive", "full")
_PYTHON_RUNNERS = ("unit_tests.py", "integration_test.py", "cli_unit_test.py")


def _load_tests(text):
    commands = dict(re.findall(r"^add_test\((\S+) (.+)\)$", text, re.MULTILINE))
    labels = {
        name: set(value.split(";"))
        for name, value in re.findall(
            r'^set_tests_properties\((\S+) PROPERTIES LABELS "([^"]*)"', text, re.MULTILINE
        )
    }
    return {name: (commands[name], labels.get(name, set())) for name in commands}


class CtestCategoryLabelsTest(unittest.TestCase):
    def setUp(self):
        if not _CTEST_FILE.is_file():
            self.skipTest(f"{_CTEST_FILE} not installed (built outside rocm-systems)")
        self.tests = _load_tests(_CTEST_FILE.read_text())
        self.archs = sorted(
            {
                label
                for _, labels in self.tests.values()
                for label in labels
                if label.startswith("ex_gpu_")
            }
        )

    def _runner_counts(self, selected):
        return {
            runner: sum(
                1 for name in selected if f"python_unittest/{runner}" in self.tests[name][0]
            )
            for runner in _PYTHON_RUNNERS
        }

    def test_each_runner_selected_once_on_listed_arch(self):
        self.assertTrue(self.archs, "no ex_gpu_* labels in the ctest file")
        for category in _CATEGORIES:
            for arch in self.archs:
                selected = [
                    n for n, (_, labels) in self.tests.items() if {category, arch} <= labels
                ]
                with self.subTest(category=category, arch=arch):
                    self.assertEqual(
                        self._runner_counts(selected), dict.fromkeys(_PYTHON_RUNNERS, 1)
                    )

    def test_each_runner_selected_once_on_unlisted_arch(self):
        for category in _CATEGORIES:
            selected = [
                n
                for n, (_, labels) in self.tests.items()
                if category in labels and not any("ex_gpu" in label for label in labels)
            ]
            with self.subTest(category=category):
                self.assertEqual(self._runner_counts(selected), dict.fromkeys(_PYTHON_RUNNERS, 1))

    def test_root_runners_do_not_have_unprivileged_labels(self):
        unprivileged = {"quick", "pre-commit", "smoke", "standard", "pr", "precheckin"}
        for name, (command, labels) in self.tests.items():
            if "python_unittest/" not in command:
                continue
            with self.subTest(runner=name):
                self.assertFalse(labels & unprivileged)
                self.assertIn("requires_root", labels)

    def test_asic_exclusions_match_legacy(self):
        archs = {
            "gfx906": "vega20",
            "gfx908": "arcturus",
            "gfx90a": "aldebaran",
            "gfx94X": "90400",
            "gfx103X": "sienna_cichlid",
            "gfx1100": "strix_point",
            "gfx115X": "strix_point",
        }
        for arch, asic in archs.items():
            with self.subTest(arch=arch):
                legacy = subprocess.check_output(
                    [
                        "bash",
                        "-c",
                        'source "$1"; printf "%s" "${FILTER[$2]}"',
                        "bash",
                        str(_CTEST_FILE.parent.parent / "amdsmitst.exclude"),
                        asic,
                    ],
                    universal_newlines=True,
                )
                expected = set(filter(None, legacy.split(":"))) | {"*DISABLED*"}
                for category in ("quick", "standard", "comprehensive", "full"):
                    name = f"amdsmitst_{category}_{arch}_suite"
                    self.assertIn(name, self.tests)
                    command = self.tests[name][0]
                    excluded = command.split("--gtest_filter=", 1)[1].split("-", 1)[1]
                    self.assertEqual(set(excluded.split(":")), expected)
        self.assertNotIn("ex_gpu_gfx110X", self.archs)
