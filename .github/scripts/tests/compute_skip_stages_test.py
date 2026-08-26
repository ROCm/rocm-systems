# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for compute_skip_stages.py.

These cover the rocm-systems-owned logic (project parsing, run_all_tests
short-circuit, fan-out policy, and the fail-safe branches) without requiring a
real TheRock checkout. Stage narrowing itself is delegated to TheRock's build
topology and is exercised via a stubbed topology module so the test has no
external dependency.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_skip_stages as css


class ParseProjectsTest(unittest.TestCase):
    def test_strips_prefix_and_whitespace(self):
        self.assertEqual(
            css._parse_projects("projects/clr, projects/rdc"),
            ["clr", "rdc"],
        )

    def test_empty(self):
        self.assertEqual(css._parse_projects(""), [])


class ComputeSkipStagesTest(unittest.TestCase):
    def test_run_all_tests_skips_nothing(self):
        # CI-infra change: full build even if a narrow project set is passed.
        self.assertEqual(
            css.compute_skip_stages("projects/rdc", "_therock", run_all_tests=True),
            [],
        )

    def test_empty_changed_projects_skips_nothing(self):
        self.assertEqual(css.compute_skip_stages("", "_therock"), [])

    def test_fanout_project_skips_nothing(self):
        # ABI-sensitive projects fan out to consumers -> full build.
        self.assertEqual(css.compute_skip_stages("projects/hip", "_therock"), [])
        self.assertEqual(css.compute_skip_stages("projects/amdsmi", "_therock"), [])
        # Any fan-out project in the set forces a full build.
        self.assertEqual(
            css.compute_skip_stages("projects/clr,projects/rdc", "_therock"), []
        )

    def test_topology_load_failure_skips_nothing(self):
        # Nonexistent TheRock path -> import fails -> fail safe (skip nothing).
        self.assertEqual(
            css.compute_skip_stages("projects/rdc", "/no/such/therock"), []
        )

    def _install_fake_topology(self, all_stages, required, known=None):
        """Install a fake _therock_utils.build_topology module on sys.path.

        known: set of resolvable project names. Any project not in `known`
        resolves to None (mirrors BuildTopology.resolve_project_to_artifact).
        """
        known = set(known) if known is not None else None
        mod = types.ModuleType("_therock_utils.build_topology")

        class _Topo:
            def get_all_stage_names(self):
                return set(all_stages)

            def get_stages_for_projects(self, projects):
                return set(required)

            def resolve_project_to_artifact(self, project):
                if known is None:
                    return project  # everything resolves
                return project if project in known else None

        mod.get_topology = lambda: _Topo()
        pkg = types.ModuleType("_therock_utils")
        pkg.__path__ = []  # mark as package
        patcher = patch.dict(
            sys.modules,
            {"_therock_utils": pkg, "_therock_utils.build_topology": mod},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_narrows_to_complement_of_required(self):
        self._install_fake_topology(
            all_stages=["compiler-runtime", "dctools-core", "math-libs", "comm-libs"],
            required=["compiler-runtime", "dctools-core"],
        )
        skip = css.compute_skip_stages("projects/rdc", "_therock")
        self.assertEqual(skip, ["comm-libs", "math-libs"])

    def test_mixed_known_and_unknown_skips_nothing(self):
        # rdc is known, but the unknown project is silently dropped by
        # get_stages_for_projects; we must NOT skip in that case.
        self._install_fake_topology(
            all_stages=["compiler-runtime", "dctools-core", "math-libs"],
            required=["compiler-runtime", "dctools-core"],
            known={"rdc"},
        )
        self.assertEqual(
            css.compute_skip_stages("projects/rdc,projects/unknown", "_therock"),
            [],
        )

    def test_all_known_projects_narrow(self):
        self._install_fake_topology(
            all_stages=["compiler-runtime", "dctools-core", "math-libs"],
            required=["compiler-runtime", "dctools-core"],
            known={"rdc"},
        )
        self.assertEqual(
            css.compute_skip_stages("projects/rdc", "_therock"),
            ["math-libs"],
        )

    def test_no_required_stages_skips_nothing(self):
        self._install_fake_topology(
            all_stages=["compiler-runtime", "math-libs"],
            required=[],
            known={"mystery"},
        )
        self.assertEqual(css.compute_skip_stages("projects/mystery", "_therock"), [])


if __name__ == "__main__":
    unittest.main()
