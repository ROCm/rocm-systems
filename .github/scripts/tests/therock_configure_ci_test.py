from pathlib import Path
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_configure_ci


class ConfigureCITest(unittest.TestCase):
    @patch("subprocess.run")
    def test_pull_request(self, mock_run):
        args = {"is_pull_request": True, "base_ref": "HEAD^"}

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("subprocess.run")
    def test_pull_request_empty(self, mock_run):
        args = {"is_pull_request": True, "base_ref": "HEAD^"}

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        # Empty modified_paths should return empty list (no changes = no CI)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_workflow_dispatch(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocminfo projects/clr",
            "base_ref": "HEAD^",
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("subprocess.run")
    def test_workflow_dispatch_bad_input(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/invalid$$projects/fake",
            "base_ref": "HEAD^",
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_workflow_dispatch_all(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "all",
            "base_ref": "HEAD^",
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("subprocess.run")
    def test_workflow_dispatch_empty(self, mock_run):
        args = {"is_workflow_dispatch": True, "input_projects": "", "base_ref": "HEAD^"}

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    def test_scheduled_run(self):
        args = {"is_nightly": True, "input_projects": "", "base_ref": "HEAD^"}

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)

    @patch("subprocess.run")
    def test_is_push(self, mock_run):
        args = {"is_push": True, "base_ref": "HEAD^"}

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    def test_is_path_skippable(self):
        # Test skippable patterns
        self.assertTrue(therock_configure_ci.is_path_skippable("README.md"))
        self.assertTrue(therock_configure_ci.is_path_skippable("docs/guide.rst"))
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/rocminfo/README.md")
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/rocminfo/docs/api.rst")
        )
        self.assertTrue(therock_configure_ci.is_path_skippable(".gitignore"))

        # Test non-skippable patterns
        self.assertFalse(
            therock_configure_ci.is_path_skippable("projects/rocminfo/src/main.cpp")
        )
        self.assertFalse(therock_configure_ci.is_path_skippable("CMakeLists.txt"))
        self.assertFalse(
            therock_configure_ci.is_path_skippable("projects/rocminfo/test/test.cpp")
        )

    def test_check_for_non_skippable_path(self):
        # All skippable paths
        skippable_paths = [
            "README.md",
            "docs/guide.rst",
            "projects/rocminfo/docs/api.md",
        ]
        self.assertFalse(
            therock_configure_ci.check_for_non_skippable_path(skippable_paths)
        )

        # Mixed paths (has non-skippable)
        mixed_paths = ["README.md", "src/main.cpp"]
        self.assertTrue(therock_configure_ci.check_for_non_skippable_path(mixed_paths))

        # None input
        self.assertFalse(therock_configure_ci.check_for_non_skippable_path(None))

    @patch("subprocess.run")
    def test_docs_only_change_returns_empty_list(self, mock_run):
        args = {"is_pull_request": True, "base_ref": "HEAD^"}

        # Mock git diff to return only doc files
        mock_process = MagicMock()
        mock_process.stdout = "README.md\ndocs/guide.rst\nprojects/rocprim/docs/api.md"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
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

        project_to_run = therock_configure_ci.retrieve_projects(args)
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

        project_to_run = therock_configure_ci.retrieve_projects(args)
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

        project_to_run = therock_configure_ci.retrieve_projects(args)

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

        project_to_run = therock_configure_ci.retrieve_projects(args)

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

        # Branch only touched CI scripts — no Windows-path files in the diff
        mock_get_modified.return_value = [
            ".github/scripts/therock_configure_ci.py",
        ]

        project_to_run = therock_configure_ci.retrieve_projects(args)
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

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)


if __name__ == "__main__":
    unittest.main()
