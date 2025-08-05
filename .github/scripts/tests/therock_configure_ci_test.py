from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_configure_ci

class ConfigureCITest(unittest.TestCase):
    def test_pull_request(self):
        args = {
            "is_pull_request": True,
            "input_subtrees": "projects/rocprim\nprojects/hipcub"
        }
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)
        
    def test_pull_request_empty(self):
        args = {
            "is_pull_request": True,
            "input_subtrees": ""
        }
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)
        
    def test_workflow_dispatch(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim projects/hipcub"
        }
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)
        
    def test_workflow_dispatch_bad_input(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim$$projects/hipcub"
        }
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)
        
    def test_workflow_dispatch_all(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "all"
        }
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)
        
    def test_workflow_dispatch_empty(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": ""
        }
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)
        
    def test_is_push(self):
        args = {
            "is_push": True,
        }
        
        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

if __name__ == "__main__":
    unittest.main()
