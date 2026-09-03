# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for compute_build_stages.py.

These cover the rocm-systems-owned logic (project parsing, run_all_tests
short-circuit, full-build policy, and the fail-safe branches) without requiring a
real TheRock checkout. Stage narrowing itself is delegated to TheRock's build
topology and is exercised via a stubbed topology module so the test has no
external dependency.

The script emits an *allowlist* of stages to build; an empty list means "build
everything" (the safe default).
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_build_stages as cbs


class ParseProjectsTest(unittest.TestCase):
    def test_strips_prefix_and_whitespace(self):
        self.assertEqual(
            cbs._parse_projects("projects/clr, projects/rdc"),
            ["clr", "rdc"],
        )

    def test_empty(self):
        self.assertEqual(cbs._parse_projects(""), [])


class ComputeBuildStagesTest(unittest.TestCase):
    def test_run_all_tests_builds_everything(self):
        # CI-infra change: empty allowlist == build all.
        self.assertEqual(
            cbs.compute_build_stages("projects/rdc", "_therock", run_all_tests=True),
            [],
        )

    def test_empty_changed_projects_builds_everything(self):
        self.assertEqual(cbs.compute_build_stages("", "_therock"), [])

    def test_full_build_project_builds_everything(self):
        # ABI-sensitive projects fan out to consumers -> build all (full-build).
        self.assertEqual(cbs.compute_build_stages("projects/hip", "_therock"), [])
        self.assertEqual(cbs.compute_build_stages("projects/amdsmi", "_therock"), [])
        self.assertEqual(
            cbs.compute_build_stages("projects/clr,projects/rdc", "_therock"), []
        )

    def test_legacy_profiler_projects_build_everything(self):
        # Legacy profiler bucket used THEROCK_ENABLE_ALL=ON, so preserve
        # that coverage by falling back to a full build.
        profiler_projects = [
            "aqlprofile",
            "rocprofiler",
            "rocprofiler-compute",
            "rocprofiler-register",
            "rocprofiler-sdk",
            "rocprofiler-systems",
            "roctracer",
        ]

        for project in profiler_projects:
            with self.subTest(project=project):
                self.assertEqual(
                    cbs.compute_build_stages(f"projects/{project}", "_therock"),
                    [],
                )

    def test_topology_load_failure_builds_everything(self):
        # Nonexistent TheRock path -> import fails -> fail safe (build all).
        self.assertEqual(
            cbs.compute_build_stages("projects/rdc", "/no/such/therock"), []
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

    def test_allowlist_is_required_stages(self):
        self._install_fake_topology(
            all_stages=["compiler-runtime", "dctools-core", "math-libs", "comm-libs"],
            required=["compiler-runtime", "dctools-core"],
        )
        self.assertEqual(
            cbs.compute_build_stages("projects/rdc", "_therock"),
            ["compiler-runtime", "dctools-core"],
        )

    def test_mixed_known_and_unknown_builds_everything(self):
        # rdc is known, but the unknown project is silently dropped by
        # get_stages_for_projects; we must NOT narrow in that case.
        self._install_fake_topology(
            all_stages=["compiler-runtime", "dctools-core", "math-libs"],
            required=["compiler-runtime", "dctools-core"],
            known={"rdc"},
        )
        self.assertEqual(
            cbs.compute_build_stages("projects/rdc,projects/unknown", "_therock"),
            [],
        )

    def test_all_stages_required_emits_empty(self):
        # When everything is required, emit empty (== build all) not a full list.
        self._install_fake_topology(
            all_stages=["compiler-runtime", "math-libs"],
            required=["compiler-runtime", "math-libs"],
            known={"widelib"},
        )
        self.assertEqual(cbs.compute_build_stages("projects/widelib", "_therock"), [])

    def test_no_required_stages_builds_everything(self):
        self._install_fake_topology(
            all_stages=["compiler-runtime", "math-libs"],
            required=[],
            known={"mystery"},
        )
        self.assertEqual(cbs.compute_build_stages("projects/mystery", "_therock"), [])


if __name__ == "__main__":
    unittest.main()
