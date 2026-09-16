import json
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_configure_ci
import therock_matrix


class ConfigureCITest(unittest.TestCase):
    def setUp(self):
        # Keep run() tests independent of a sibling TheRock checkout and the
        # GitHub Actions step output file.
        select_build_runner_patcher = patch(
            "therock_configure_ci.select_build_runner", return_value=""
        )
        self.addCleanup(select_build_runner_patcher.stop)
        select_build_runner_patcher.start()

        set_github_output_patcher = patch("therock_configure_ci.set_github_output")
        self.addCleanup(set_github_output_patcher.stop)
        set_github_output_patcher.start()

    @patch("therock_configure_ci.get_modified_paths")
    def test_pull_request(self, mock_get_modified):
        args = {"is_pull_request": True, "base_ref": "HEAD^"}

        mock_get_modified.return_value = ["projects/rocminfo/src/main.cpp"]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("therock_configure_ci.get_modified_paths")
    def test_pull_request_empty(self, mock_get_modified):
        args = {"is_pull_request": True, "base_ref": "HEAD^"}

        mock_get_modified.return_value = []

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        # An empty diff should skip CI.
        self.assertEqual(len(project_to_run), 0)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch(self, mock_get_modified):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocminfo projects/clr",
            "base_ref": "HEAD^",
        }

        mock_get_modified.return_value = ["projects/rocminfo/src/main.cpp"]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch_bad_input(self, mock_get_modified):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/invalid$$projects/fake",
            "base_ref": "HEAD^",
        }

        mock_get_modified.return_value = ["projects/rocminfo/src/main.cpp"]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch_all(self, mock_get_modified):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "all",
            "base_ref": "HEAD^",
        }

        mock_get_modified.return_value = ["projects/rocminfo/src/main.cpp"]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch_empty(self, mock_get_modified):
        args = {"is_workflow_dispatch": True, "input_projects": "", "base_ref": "HEAD^"}

        mock_get_modified.return_value = ["projects/rocminfo/src/main.cpp"]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    def test_scheduled_run(self):
        args = {"is_nightly": True, "input_projects": "", "base_ref": "HEAD^"}

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)
        self.assertEqual(test_type, "comprehensive")

    @patch("therock_configure_ci.get_modified_paths")
    def test_is_push(self, mock_get_modified):
        args = {"is_push": True, "base_ref": "HEAD^"}

        mock_get_modified.return_value = ["projects/rocminfo/src/main.cpp"]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    def test_is_path_skippable(self):
        # SKIPPABLE_PATH_PATTERNS contains abstract fnmatch patterns. These
        # cases make the intended matches concrete and verify how patterns
        # behave with paths containing nested directories.
        skippable_paths = [
            "README.md",
            "docs/guide.rst",
            "projects/rocminfo/README.md",
            "projects/rocminfo/docs/api.rst",
            ".gitignore",
            "projects/rocminfo/.gitignore",
            "projects/amdsmi/.pre-commit-config.yaml",
            "projects/amdsmi/.github/CODEOWNERS",
            "projects/amdsmi/LICENSE",
            # A few source files that would normally trigger CI but are
            # excluded based on which directory they are in.
            "tools/systems_pr_bot/policy_check.py",
            "experimental/perf-dkms/CMakeLists.txt",
            "projects/rocr-runtime/libhsakmt/src/dxg/dxgmodule.c",
            # GitHub metadata, workflow, and action files unrelated to TheRock
            # CI are skippable.
            ".github/labeler.yml",
            ".github/labels.yml",
            ".github/actions/rocprofiler-sdk-util/action.yml",
            ".github/workflows/labeler.yml",
            ".github/workflows/rdc-ci.yml",
            ".github/workflows/amdsmi-manylinux-build.yml",
            ".github/workflows/rocjitsu-corpus-tests.yml",
        ]
        for path in skippable_paths:
            with self.subTest(path=path):
                self.assertTrue(therock_configure_ci.is_path_skippable(path))

        non_skippable_paths = [
            "projects/rocminfo/src/main.cpp",
            "CMakeLists.txt",
            "projects/rocminfo/test/test.cpp",
            # TheRock CI workflow, action, and script files are non-skippable so
            # changes to them still trigger CI.
            ".github/workflows/therock-ci.yml",
            ".github/actions/therock-something/action.yml",
            ".github/scripts/therock_configure_ci.py",
        ]
        for path in non_skippable_paths:
            with self.subTest(path=path):
                self.assertFalse(therock_configure_ci.is_path_skippable(path))

    def test_check_for_non_skippable_path(self):
        all_skippable = ["README.md", "docs/guide.rst"]
        self.assertFalse(
            therock_configure_ci.check_for_non_skippable_path(all_skippable)
        )

        some_skippable = ["README.md", "projects/rocminfo/src/main.cpp"]
        self.assertTrue(
            therock_configure_ci.check_for_non_skippable_path(some_skippable)
        )

        none_skippable = [
            "CMakeLists.txt",
            "projects/rocminfo/src/main.cpp",
        ]
        self.assertTrue(
            therock_configure_ci.check_for_non_skippable_path(none_skippable)
        )

    @patch("therock_configure_ci.get_modified_paths")
    def test_docs_only_change_returns_empty_list(self, mock_get_modified):
        args = {"is_pull_request": True, "base_ref": "HEAD^"}

        mock_get_modified.return_value = [
            "README.md",
            "docs/guide.rst",
            "projects/rocprim/docs/api.md",
        ]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("therock_configure_ci.get_modified_paths")
    def test_linux_only_subtrees_returns_empty_list(self, mock_get_modified):
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^",
            "platform": "windows",
        }

        mock_get_modified.return_value = [
            "projects/rocprofiler-compute/src/compute.cpp",
            "projects/rocshmem/src/rocshmem.cpp",
            "projects/amdsmi/src/amdsmi.cpp",
            "projects/rocprofiler-register/src/register.cpp",
            "projects/amdsmi/hello/test.cpp",
            "projects/hipother/hello/test.cpp",
        ]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("therock_configure_ci.get_modified_paths")
    def test_linux_only_subtrees_returns_correct_list(self, mock_get_modified):
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^",
            "platform": "windows",
        }

        mock_get_modified.return_value = [
            "projects/rocprofiler-compute/src/compute.cpp",
            "projects/rocshmem/src/rocshmem.cpp",
            "projects/amdsmi/src/amdsmi.cpp",
            "projects/rocprofiler-register/src/register.cpp",
            "projects/amdsmi/hello/test.cpp",
            "projects/hip/src/hip.cpp",  # contains windows CI trigger
            "projects/clr/src/hip.cpp",  # contains windows CI trigger
            ".github/workflows/therock-ci.yml",  # contains windows CI trigger
        ]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch_explicit_project_not_overridden_by_ci_file_changes(
        self, mock_get_modified
    ):
        """Regression test: workflow_dispatch with an explicit project must not be overridden
        by CI-related file changes (e.g. therock_*.py) in modified_paths.
        Previously, detecting .github/scripts/therock_*.py changes caused all subtrees to be
        evaluated, ignoring the user-specified project selection entirely.
        See: https://github.com/ROCm/rocm-systems/pull/3960
        """
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rdc",
            "base_ref": "HEAD^",
            "platform": "linux",
        }

        # Simulate a branch that touched CI scripts (the triggering condition for the bug)
        mock_get_modified.return_value = [
            ".github/scripts/therock_configure_ci.py",
            ".github/scripts/therock_matrix.py",
        ]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)

        self.assertEqual(len(project_to_run), 1)
        cmake_options = project_to_run[0]["cmake_options"]
        # Must use the dc_tools-specific flags, not the all-projects fallback
        self.assertIn("DTHEROCK_ENABLE_DC_TOOLS=ON", cmake_options)
        self.assertNotIn("DTHEROCK_ENABLE_ALL=ON", cmake_options)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch_windows_not_skipped_by_modified_paths(
        self, mock_get_modified
    ):
        """Regression test: workflow_dispatch on Windows must not be skipped when
        modified_paths doesn't contain Windows-trigger patterns. The user explicitly
        selected a project, so the Windows CI skip logic must not override that.
        """
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/clr",
            "base_ref": "HEAD^",
            "platform": "windows",
        }

        # Simulate a branch whose last commit only touched a Linux-only subtree,
        # which doesn't match any trigger_windows_ci_for_subtrees_paths pattern.
        # Note: .github/scripts/therock_*.py would NOT be appropriate here because
        # it matches the ".github/*/therock*" Windows-trigger pattern.
        mock_get_modified.return_value = [
            "projects/rocprofiler-compute/src/compute.cpp",
        ]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)

        # Windows CI must not be skipped for an explicit workflow_dispatch selection
        self.assertEqual(len(project_to_run), 1)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch_shared_windows_subtree_triggers_windows_ci(
        self, mock_get_modified
    ):
        """workflow_dispatch with shared/amdgpu-windows-interop must trigger Windows CI
        even when the git diff contains no Windows-relevant file paths."""
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "shared/amdgpu-windows-interop",
            "base_ref": "HEAD^",
            "platform": "windows",
        }

        # Explicit workflow_dispatch selections do not depend on modified paths.
        mock_get_modified.return_value = []

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("therock_configure_ci.get_modified_paths")
    def test_workflow_dispatch_windows_only_subtree_skips_linux_ci(
        self, mock_get_modified
    ):
        """workflow_dispatch with a windows_only subtree must not trigger Linux CI."""
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "shared/amdgpu-windows-interop",
            "base_ref": "HEAD^",
            "platform": "linux",
        }

        mock_get_modified.return_value = []

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("therock_configure_ci.get_modified_paths")
    def test_rccl_ci_triggered_by_rccl_paths_pull_request(self, mock_get_modified):
        """A RCCL-only pull request should run RCCL CI but not regular CI."""
        args = {"is_pull_request": True, "base_ref": "HEAD^", "platform": "linux"}

        mock_get_modified.return_value = [
            "projects/rccl/rccl.cpp",
            "projects/rccl/test/test.cpp",
        ]

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertEqual(len(projects), 0)
        self.assertEqual(outputs["run_linux_rccl_ci"], "true")

    @patch("therock_configure_ci.get_modified_paths")
    def test_rccl_ci_not_triggered_by_non_rccl_paths_pull_request(
        self, mock_get_modified
    ):
        """A non-RCCL pull request should run regular CI but not RCCL CI."""
        args = {"is_pull_request": True, "base_ref": "HEAD^", "platform": "linux"}

        mock_get_modified.return_value = ["projects/rocprim/rocprim.cpp"]

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertGreaterEqual(len(projects), 1)
        self.assertEqual(outputs["run_linux_rccl_ci"], "false")

    @patch("therock_configure_ci.get_modified_paths")
    def test_rccl_ci_not_triggered_by_docs_only_change_pull_request(
        self, mock_get_modified
    ):
        """PR with only skippable RCCL doc paths should skip both regular and RCCL CI."""
        args = {"is_pull_request": True, "base_ref": "HEAD^", "platform": "linux"}
    
        mock_get_modified.return_value = [
            "projects/rccl/README.md",
            "projects/rccl/docs/install/building-installing.rst",
            "projects/rccl/.readthedocs.yaml",
        ]
    
        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertEqual(len(projects), 0)
        self.assertEqual(outputs["run_linux_rccl_ci"], "false")

    def test_rccl_ci_triggered_nightly(self):
        """A nightly event should run both regular and RCCL CI."""
        args = {"is_nightly": True, "base_ref": "HEAD^", "platform": "linux"}

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertGreaterEqual(len(projects), 1)
        self.assertEqual(outputs["run_linux_rccl_ci"], "true")

    @patch("therock_configure_ci.get_modified_paths")
    def test_rccl_ci_triggered_workflow_dispatch(self, mock_get_modified):
        """Selecting only RCCL manually should run RCCL CI but not regular CI."""
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rccl",
            "base_ref": "HEAD^",
            "platform": "linux",
        }

        mock_get_modified.return_value = []

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertEqual(len(projects), 0)
        self.assertEqual(outputs["run_linux_rccl_ci"], "true")

    @patch("therock_configure_ci.get_modified_paths")
    def test_non_rccl_push_runs_regular_ci_only(self, mock_get_modified):
        """A non-RCCL push should run regular CI but not RCCL CI."""
        args = {"is_push": True, "base_ref": "HEAD^", "platform": "linux"}

        mock_get_modified.return_value = ["projects/rocprim/rocprim.cpp"]

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertGreaterEqual(len(projects), 1)
        self.assertEqual(outputs["run_linux_rccl_ci"], "false")

    # Push event combinations.
    @patch("therock_configure_ci.get_modified_paths")
    def test_push_only_rccl_skips_regular_ci(self, mock_get_modified):
        """Push with only RCCL changes should skip regular CI but run RCCL CI."""
        args = {"is_push": True, "base_ref": "HEAD^", "platform": "linux"}

        mock_get_modified.return_value = [
            "projects/rccl/src/main.cpp",
            "projects/rccl/test/test.cpp",
        ]

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertEqual(len(projects), 0)
        self.assertEqual(outputs["run_linux_rccl_ci"], "true")

    @patch("therock_configure_ci.get_modified_paths")
    def test_push_rccl_and_known_subtree_runs_both(self, mock_get_modified):
        """Push with RCCL + known subtree changes should run both regular and RCCL CI."""
        args = {"is_push": True, "base_ref": "HEAD^", "platform": "linux"}

        mock_get_modified.return_value = [
            "projects/rccl/src/main.cpp",
            "projects/clr/src/clr.cpp",
        ]

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertGreaterEqual(len(projects), 1)
        self.assertEqual(outputs["run_linux_rccl_ci"], "true")

    @patch("therock_configure_ci.get_modified_paths")
    def test_push_rccl_and_unknown_runs_full_ci(self, mock_get_modified):
        """Push with RCCL + unknown path changes should run full regular CI and RCCL CI."""
        args = {"is_push": True, "base_ref": "HEAD^", "platform": "linux"}

        mock_get_modified.return_value = [
            "projects/rccl/src/main.cpp",
            "unknown/path/file.cpp",
        ]

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        # Should run full CI due to unknown path
        self.assertGreaterEqual(len(projects), 1)
        self.assertEqual(outputs["run_linux_rccl_ci"], "true")

    @patch("therock_configure_ci.get_modified_paths")
    def test_push_known_subtree_only_runs_regular_ci(self, mock_get_modified):
        """Push with only known subtree changes should run regular CI, not RCCL CI."""
        args = {"is_push": True, "base_ref": "HEAD^", "platform": "linux"}

        mock_get_modified.return_value = [
            "projects/clr/src/clr.cpp",
            "projects/rocminfo/src/main.cpp",
        ]

        outputs = therock_configure_ci.run(args)
        projects = json.loads(outputs["projects"])
        self.assertGreaterEqual(len(projects), 1)
        self.assertEqual(outputs["run_linux_rccl_ci"], "false")

    @patch("therock_configure_ci.get_modified_paths")
    def test_hipfile_pr_builds_storage_libs_and_rocprofiler_sdk(self, mock_get_modified):
        """PR with hipfile changes should build storage_libs and rocprofiler-sdk."""
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^",
            "platform": "linux",
        }

        mock_get_modified.return_value = [
            "projects/hipfile/src/hipfile.cpp",
            "projects/hipfile/include/hipfile.h",
        ]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)
        cmake_options = project_to_run[0]["cmake_options"]
        self.assertIn("DTHEROCK_ENABLE_STORAGE_LIBS=ON", cmake_options)
        self.assertIn("DTHEROCK_ENABLE_ROCPROFV3=ON", cmake_options)
        self.assertNotIn("DTHEROCK_ENABLE_ALL=ON", cmake_options)

    @patch("therock_configure_ci.get_modified_paths")
    def test_hipfile_pr_skips_windows_ci(self, mock_get_modified):
        """PR with only hipfile changes should not trigger Windows CI (Linux-only component)."""
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^",
            "platform": "windows",
        }

        mock_get_modified.return_value = [
            "projects/hipfile/src/hipfile.cpp",
            "projects/hipfile/include/hipfile.h",
        ]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("therock_configure_ci.get_modified_paths")
    def test_amdsmi_pr_triggers_amdsmi_group(self, mock_get_modified):
        """PR with amdsmi changes should build everything and test amdsmi plus
        its downstream dependents (blas, rocprofiler-systems, rocrtst)."""
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^",
            "platform": "linux",
        }

        mock_get_modified.return_value = ["projects/amdsmi/src/amdsmi.cpp"]

        project_to_run, _ = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)
        self.assertIn("DTHEROCK_ENABLE_ALL=ON", project_to_run[0]["cmake_options"])
        tests = {
            t.strip() for t in project_to_run[0]["projects_to_test"].split(",")
        }
        self.assertEqual(
            tests,
            {
                "amdsmi",
                "rocblas",
                "hipblas",
                "hipblaslt",
                "rocprofiler-systems",
                "rocrtst",
            },
        )
        # rccl is intentionally excluded: the main Linux build forces
        # THEROCK_ENABLE_RCCL=OFF and RCCL runs in its own dedicated pipeline.
        self.assertNotIn("rccl", tests)

    def test_amdsmi_group_uses_valid_test_targets(self):
        """Every test target in the amdsmi group must be a real TheRock test
        target (a key in fetch_test_configurations.test_matrix)."""
        # Keys of TheRock's fetch_test_configurations.test_matrix. Kept in sync
        # manually since TheRock is not checked out during unit tests.
        valid_test_targets = {
            "sanity", "hip-tests", "hipfile", "rocblas", "rocroller",
            "tensilelite", "origami", "hipblas", "amdsmi", "hipblaslt",
            "hipsolver", "rocsolver", "rocprim", "hipcub", "rocgdb-cpu",
            "rocgdb-gpu", "rocgdb-corefile", "rocr-debug-agent", "rocthrust", "hipsparse",
            "rocsparse", "hipsparselt", "rocrand", "hiprand", "rocfft",
            "hipfft", "miopen", "rccl", "rocshmem", "rocprofiler-sdk",
            "hipdnn", "hipdnn_install", "hipdnn-integration-tests",
            "hipdnn-samples", "miopenprovider", "hipblasltprovider",
            "hipkernelprovider", "rocwmma", "rocalution",
            "rocprofiler-compute", "rocprofiler-systems", "libhipcxx_amdclang",
            "libhipcxx_hiprtc", "hipthreads", "hipthreads_examples",
            "rocdecode", "rocjpeg", "aqlprofile", "rocrtst", "hiptensor",
        }
        amdsmi_tests = {
            t.strip()
            for t in therock_matrix.project_map["amdsmi"][
                "projects_to_test"
            ].split(",")
        }
        self.assertTrue(amdsmi_tests <= valid_test_targets, amdsmi_tests)

    def test_amdsmi_in_core_all_nightly_test_targets(self):
        """amdsmi must be an explicit test target in core, all, and nightly."""
        for group in ("core", "all", "nightly"):
            tests = {
                t.strip()
                for t in therock_matrix.project_map[group][
                    "projects_to_test"
                ].split(",")
            }
            self.assertIn("amdsmi", tests, group)


if __name__ == "__main__":
    unittest.main()
