# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for configure_multi_arch_changed_projects.py.

Covers the rocm-systems-owned per-PR selection: docs skip, CI-infra/full-test
triggers, projects/* matching, surfacing of the non-subtree shared/* +
emulation/* directories, and the unclassified-change fallback. The compare API
is stubbed so the tests need no network or checkout.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_multi_arch_changed_projects as c

_CONFIG = {
    "repositories": [
        {
            "name": "rocm-core",
            "url": "ROCm/rocm-core",
            "branch": "develop",
            "category": "projects",
            "auto_subtree_pull": False,
            "auto_subtree_push": True,
            "monorepo_source_of_truth": True,
        }
    ]
}


def _write_config():
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(_CONFIG, f)
    return path


def _configure(paths, event_name="pull_request"):
    cfg = _write_config()
    try:
        with patch.object(
            c,
            "get_modified_paths_api",
            return_value=set(paths) if paths is not None else None,
        ):
            return c.configure(
                event_name=event_name,
                github_repo="ROCm/rocm-systems",
                base_sha="base",
                head_sha="head",
                config_path=cfg,
            )
    finally:
        os.unlink(cfg)


class ConfigureTest(unittest.TestCase):
    def test_project_change_narrows(self):
        r = _configure(["projects/rocm-core/src/x.cpp"])
        self.assertEqual(r.changed_projects, "projects/rocm-core")
        self.assertFalse(r.run_all_tests)
        self.assertFalse(r.skip_tests)

    def test_shared_component_is_surfaced(self):
        r = _configure(["shared/amdgpu-windows-interop/pal/x.cpp"])
        self.assertEqual(r.changed_projects, "shared/amdgpu-windows-interop")
        self.assertFalse(r.run_all_tests)

    def test_emulation_components_are_surfaced(self):
        r = _configure(["emulation/mirage/a.cpp", "emulation/rocjitsu/b.cpp"])
        self.assertEqual(
            sorted(r.changed_projects.split(",")),
            ["emulation/mirage", "emulation/rocjitsu"],
        )

    def test_machine_readable_isa_is_surfaced(self):
        r = _configure(["shared/machine-readable-isa/isa/g.json"])
        self.assertEqual(r.changed_projects, "shared/machine-readable-isa")

    def test_docs_only_skips(self):
        r = _configure(["projects/rocm-core/docs/readme.md", "README.md"])
        self.assertTrue(r.skip_tests)
        self.assertFalse(r.run_all_tests)

    def test_ci_infra_runs_all(self):
        r = _configure([".github/workflows/therock-multi-arch-ci.yml"])
        self.assertTrue(r.run_all_tests)

    def test_ctest_harness_runs_all(self):
        r = _configure(["shared/ctest/TestCategories.cmake"])
        self.assertTrue(r.run_all_tests)
        self.assertEqual(r.changed_projects, "")

    def test_mixed_recognized_and_unclassified_runs_all(self):
        r = _configure(["projects/rocm-core/src/x.cpp", "tools/build/helper.py"])
        self.assertTrue(r.run_all_tests)
        self.assertEqual(r.changed_projects, "")

    def test_top_level_nonskippable_runs_all(self):
        r = _configure(["CMakeLists.txt"])
        self.assertTrue(r.run_all_tests)

    def test_recognized_plus_skippable_still_narrows(self):
        r = _configure(["projects/rocm-core/src/x.cpp", "README.md"])
        self.assertFalse(r.run_all_tests)
        self.assertEqual(r.changed_projects, "projects/rocm-core")

    def test_no_modified_paths_skips(self):
        r = _configure([])
        self.assertTrue(r.skip_tests)

    def test_truncated_response_runs_all(self):
        r = _configure(None)
        self.assertTrue(r.run_all_tests)

    def test_schedule_runs_all(self):
        r = _configure(["projects/rocm-core/x.cpp"], event_name="schedule")
        self.assertTrue(r.run_all_tests)

    def test_declared_prefixes_are_wellformed(self):
        for prefix in c.CI_RELEVANT_NON_SUBTREE_PREFIXES:
            self.assertEqual(len(prefix.split("/")), 2, prefix)


if __name__ == "__main__":
    unittest.main()
