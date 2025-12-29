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
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^"
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("subprocess.run")
    def test_pull_request_empty(self, mock_run):
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^"
        }

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
            "base_ref": "HEAD^"
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
            "base_ref": "HEAD^"
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
            "base_ref": "HEAD^"
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    @patch("subprocess.run")
    def test_workflow_dispatch_empty(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "",
            "base_ref": "HEAD^"
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_is_push(self, mock_run):
        args = {
            "is_push": True,
            "base_ref": "HEAD^"
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocminfo/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    def test_is_path_skippable(self):
        # Test skippable patterns
        self.assertTrue(therock_configure_ci.is_path_skippable("README.md"))
        self.assertTrue(therock_configure_ci.is_path_skippable("docs/guide.rst"))
        self.assertTrue(therock_configure_ci.is_path_skippable("projects/rocminfo/README.md"))
        self.assertTrue(therock_configure_ci.is_path_skippable("projects/rocminfo/docs/api.rst"))
        self.assertTrue(therock_configure_ci.is_path_skippable(".gitignore"))
        
        # Test non-skippable patterns
        self.assertFalse(therock_configure_ci.is_path_skippable("projects/rocminfo/src/main.cpp"))
        self.assertFalse(therock_configure_ci.is_path_skippable("CMakeLists.txt"))
        self.assertFalse(therock_configure_ci.is_path_skippable("projects/rocminfo/test/test.cpp"))

    def test_check_for_non_skippable_path(self):
        # All skippable paths
        skippable_paths = ["README.md", "docs/guide.rst", "projects/rocminfo/docs/api.md"]
        self.assertFalse(therock_configure_ci.check_for_non_skippable_path(skippable_paths))
        
        # Mixed paths (has non-skippable)
        mixed_paths = ["README.md", "src/main.cpp"]
        self.assertTrue(therock_configure_ci.check_for_non_skippable_path(mixed_paths))
        
        # None input
        self.assertFalse(therock_configure_ci.check_for_non_skippable_path(None))

    @patch("subprocess.run")
    def test_docs_only_change_returns_empty_list(self, mock_run):
        args = {
            "is_pull_request": True,
            "base_ref": "HEAD^"
        }
        
        # Mock git diff to return only doc files
        mock_process = MagicMock()
        mock_process.stdout = "README.md\ndocs/guide.rst\nprojects/rocprim/docs/api.md"
        mock_run.return_value = mock_process
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

if __name__ == "__main__":
    unittest.main()
