# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for get_changed_projects.py surfacing of non-subtree CI paths.

repos-config.json only lists subtree-synced ``projects/*`` repos, so a PR
confined to in-monorepo ``shared/*`` or ``emulation/*`` directories used to
produce an empty ``changed_projects`` -> TheRock built and tested everything.
These tests pin the surfacing of those directories and the routing of the
shared CTest harness to a full-test run.
"""

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import get_changed_projects as g


def _run(paths):
    with patch.object(g, "get_modified_paths", return_value=set(paths)):
        return g.get_changed_projects("HEAD^")


class SurfaceNonSubtreePathsTest(unittest.TestCase):
    def test_shared_component_is_surfaced(self):
        r = _run(["shared/amdgpu-windows-interop/pal/foo.cpp"])
        self.assertEqual(r.changed_projects, "shared/amdgpu-windows-interop")
        self.assertFalse(r.run_all_tests)
        self.assertFalse(r.skip_tests)

    def test_emulation_components_are_surfaced(self):
        r = _run(["emulation/mirage/src/x.cpp", "emulation/rocjitsu/y.cpp"])
        self.assertEqual(
            sorted(r.changed_projects.split(",")),
            ["emulation/mirage", "emulation/rocjitsu"],
        )

    def test_machine_readable_isa_is_surfaced(self):
        r = _run(["shared/machine-readable-isa/isa/gfx942.json"])
        self.assertEqual(r.changed_projects, "shared/machine-readable-isa")

    def test_ctest_harness_triggers_full_run(self):
        # shared/ctest is test-selection infrastructure, not a component.
        r = _run(["shared/ctest/TestCategories.cmake"])
        self.assertTrue(r.run_all_tests)
        self.assertEqual(r.changed_projects, "")

    def test_unmapped_top_level_dir_runs_all(self):
        # experimental/* has no TheRock mapping and is not a surfaced component;
        # a non-skippable change there is unclassified, so run the full suite
        # rather than narrow (erman-gurses' review on #11379).
        r = _run(["experimental/foo/bar.cpp"])
        self.assertEqual(r.changed_projects, "")
        self.assertTrue(r.run_all_tests)
        self.assertFalse(r.skip_tests)

    def test_projects_still_surface(self):
        r = _run(["projects/rocm-core/CMakeLists.txt"])
        self.assertEqual(r.changed_projects, "projects/rocm-core")

    def test_mixed_recognized_and_unclassified_runs_all(self):
        # A recognized project change alongside an unclassified non-skippable
        # path must NOT narrow to just the recognized subtree (erman-gurses'
        # review on #11379) -- the unclassified change's impact is unknown.
        r = _run(
            [
                "projects/rocm-core/src/x.cpp",
                "tools/rocm-build/helper.py",
            ]
        )
        self.assertTrue(r.run_all_tests)
        self.assertEqual(r.changed_projects, "")
        self.assertFalse(r.skip_tests)

    def test_top_level_nonskippable_file_runs_all(self):
        # A root-level build file belongs to no subtree -> unclassified -> run all.
        r = _run(["CMakeLists.txt"])
        self.assertTrue(r.run_all_tests)
        self.assertEqual(r.changed_projects, "")

    def test_recognized_plus_skippable_still_narrows(self):
        # Docs/skippable files alongside a recognized project must not force a
        # full run -- they are classified as skippable, not unclassified.
        r = _run(["projects/rocm-core/src/x.cpp", "README.md"])
        self.assertFalse(r.run_all_tests)
        self.assertEqual(r.changed_projects, "projects/rocm-core")

    def test_every_surfaced_prefix_directory_exists(self):
        # Guard against drift: each declared CI-relevant prefix must be a real
        # directory in the monorepo (and thus mappable in TheRock).
        repo_root = Path(g.SCRIPT_DIR).parent.parent
        for prefix in g.CI_RELEVANT_NON_SUBTREE_PREFIXES:
            self.assertTrue(
                (repo_root / prefix).is_dir(),
                f"{prefix} is declared CI-relevant but missing on disk",
            )


if __name__ == "__main__":
    unittest.main()
