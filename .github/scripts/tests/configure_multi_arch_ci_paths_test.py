# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for configure_multi_arch_ci_paths.py."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_multi_arch_ci_paths as paths


CONFIG = [
    {
        "prefix": "projects/cuid",
        "build_projects": ["rdc"],
        "test_projects": ["projects/cuid"],
        "platforms": ["linux"],
    },
    {
        "prefix": "shared/amdgpu-windows-interop",
        "build_projects": ["hip"],
        "test_projects": ["hip-clr"],
        "platforms": ["windows"],
    },
    {
        "prefix": "shared/ctest",
        "build_all": True,
        "test_all": True,
        "platforms": ["linux"],
    },
    {
        "prefix": "shared/kpack",
        "build_all": True,
        "test_projects": ["rocm-kpack"],
        "platforms": ["linux"],
    },
    {
        "prefix": "projects/rocprof-trace-decoder",
        "build_projects": ["rocprof-trace-decoder"],
        "test_projects": ["rocprof-trace-decoder"],
        "platforms": ["linux"],
    },
]


class ComputeOutputsTest(unittest.TestCase):
    def test_existing_project_output_is_preserved(self):
        self.assertEqual(
            paths.compute_outputs(
                modified_paths=["projects/rdc/README.md"],
                original_changed_projects="projects/rdc",
                config_paths=CONFIG,
            ),
            {
                "build_changed_projects": "projects/rdc",
                "test_changed_projects": "projects/rdc",
                "build_run_all_tests": "false",
                "test_run_all_tests": "false",
                "linux_enabled": "",
                "windows_enabled": "",
            },
        )

    def test_project_gap_maps_build_and_test_selectors(self):
        self.assertEqual(
            paths.compute_outputs(
                modified_paths=["projects/cuid/README.md"],
                original_changed_projects="",
                config_paths=CONFIG,
            ),
            {
                "build_changed_projects": "rdc",
                "test_changed_projects": "projects/cuid",
                "build_run_all_tests": "false",
                "test_run_all_tests": "false",
                "linux_enabled": "true",
                "windows_enabled": "false",
            },
        )

    def test_windows_only_shared_path_skips_linux(self):
        self.assertEqual(
            paths.compute_outputs(
                modified_paths=["shared/amdgpu-windows-interop/include/a.h"],
                original_changed_projects="",
                config_paths=CONFIG,
            ),
            {
                "build_changed_projects": "hip",
                "test_changed_projects": "hip-clr",
                "build_run_all_tests": "false",
                "test_run_all_tests": "false",
                "linux_enabled": "false",
                "windows_enabled": "true",
            },
        )

    def test_rocprof_trace_decoder_uses_canonical_test_selector(self):
        self.assertEqual(
            paths.compute_outputs(
                modified_paths=["projects/rocprof-trace-decoder/README.md"],
                original_changed_projects="",
                config_paths=CONFIG,
            ),
            {
                "build_changed_projects": "rocprof-trace-decoder",
                "test_changed_projects": "rocprof-trace-decoder",
                "build_run_all_tests": "false",
                "test_run_all_tests": "false",
                "linux_enabled": "true",
                "windows_enabled": "false",
            },
        )

    def test_shared_ctest_runs_all_builds_and_tests(self):
        self.assertEqual(
            paths.compute_outputs(
                modified_paths=["shared/ctest/CMakeLists.txt"],
                original_changed_projects="",
                config_paths=CONFIG,
            ),
            {
                "build_changed_projects": "",
                "test_changed_projects": "",
                "build_run_all_tests": "true",
                "test_run_all_tests": "true",
                "linux_enabled": "true",
                "windows_enabled": "false",
            },
        )

    def test_shared_kpack_builds_all_and_tests_kpack(self):
        self.assertEqual(
            paths.compute_outputs(
                modified_paths=["shared/kpack/CMakeLists.txt"],
                original_changed_projects="",
                config_paths=CONFIG,
            ),
            {
                "build_changed_projects": "",
                "test_changed_projects": "rocm-kpack",
                "build_run_all_tests": "true",
                "test_run_all_tests": "false",
                "linux_enabled": "true",
                "windows_enabled": "false",
            },
        )


if __name__ == "__main__":
    unittest.main()
