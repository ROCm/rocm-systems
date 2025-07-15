"""
This script determines which build flag and tests to run based on SUBTREES

Required environment variables:
  - SUBTREES
"""

import json
import logging
from therock_matrix import subtree_to_project_map, project_map
from typing import Mapping
import os

logging.basicConfig(level=logging.INFO)

def set_github_output(d: Mapping[str, str]):
    """Sets GITHUB_OUTPUT values.
    See https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    logging.info(f"Setting github output:\n{d}")
    step_output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not step_output_file:
        logging.warning("Warning: GITHUB_OUTPUT env var not set, can't set github outputs")
        return
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())
        
        
def retrieve_projects(args):
    # TODO(geomin12): #590 Enable TheRock CI for forked PRs
    if args.get("is_forked_pr"):
        logging.info("Warning: not enabling any projects due to is_forked_pr. Builds/tests for forked PRs are disabled pending: https://github.com/ROCm/rocm-libraries/issues/590")
        return []
    
    if args.get("is_pull_request"):
        subtrees = args.get("input_subtrees").split("\n")
    
    if args.get("is_workflow_dispatch"):
        if args.get("input_projects") == "all":
            subtrees = list(subtree_to_project_map.keys())
        else:
            subtrees = args.get("input_projects").split()
    
    # If a push event to develop happens, we run tests on all subtrees
    if args.get("is_push"):
        subtrees = list(subtree_to_project_map.keys())
    
    projects = set()
    # collect the associated subtree to project
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))
            
    
    # retrieve the subtrees to checkout, cmake options to build, and projects to test 
    project_to_run = []
    for project in projects:
        if project in project_map:
            project_to_run.append(project_map.get(project))
        
    return project_to_run


def run(args):
    project_to_run = retrieve_projects(args)
    set_github_output({"projects": json.dumps(project_to_run)})


if __name__ == "__main__":
    args = {}
    github_event_name = os.getenv("GITHUB_EVENT_NAME")
    args["is_pull_request"] = github_event_name == "pull_request"
    args["is_push"] = github_event_name == "push"
    args["is_workflow_dispatch"] = github_event_name == "workflow_dispatch"
    
    is_forked_pr = os.getenv("IS_FORKED_PR")
    args["is_forked_pr"] = is_forked_pr == "true"
    
    input_subtrees = os.getenv("SUBTREES", "")
    args["input_subtrees"] = input_subtrees
    
    input_projects = os.getenv("PROJECTS", "")
    args["input_projects"] = input_projects
    
    logging.info(f"Retrieved arguments {args}")

    run(args)
