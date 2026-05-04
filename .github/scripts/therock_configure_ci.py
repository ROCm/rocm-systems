"""
This script determines which build flags and tests to run based on the
GitHub event type and configured subtrees/projects.

For push and pull_request events, SUBTREES is used to decide which parts
of the repository changed and which projects to run. Nightly (schedule)
and some workflow_dispatch invocations do not require SUBTREES.
"""

import fnmatch
import json
import logging
import subprocess
from therock_matrix import (
    subtree_to_project_map,
    project_map,
    trigger_windows_ci_for_subtrees_paths,
    windows_only_subtrees,
)
import time
from typing import Mapping, Optional, Iterable
import os

logging.basicConfig(level=logging.INFO)


def set_github_output(d: Mapping[str, str]):
    """Sets GITHUB_OUTPUT values.
    See https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    logging.info(f"Setting github output:\n{d}")
    step_output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not step_output_file:
        logging.warning(
            "Warning: GITHUB_OUTPUT env var not set, can't set github outputs"
        )
        return
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())


def retry(max_attempts, delay_seconds, exceptions):
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(
                        f"Exception {str(e)} thrown when attempting to run , attempt {attempt} of {max_attempts}"
                    )
                    attempt += 1
                    if attempt < max_attempts:
                        backoff = delay_seconds * (2 ** (attempt - 1))
                        time.sleep(backoff)
            return func(*args, **kwargs)

        return newfn

    return decorator


@retry(max_attempts=3, delay_seconds=2, exceptions=(TimeoutError))
def get_modified_paths(base_ref: str) -> Optional[Iterable[str]]:
    """Returns the paths of modified files relative to the base reference."""
    return subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
        timeout=60,
    ).stdout.splitlines()


GITHUB_WORKFLOWS_CI_PATTERNS = [
    "therock*",
]


def is_path_workflow_file_related_to_ci(path: str) -> bool:
    return any(
        fnmatch.fnmatch(path, ".github/workflows/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    ) or any(
        fnmatch.fnmatch(path, ".github/scripts/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    )


def check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(is_path_workflow_file_related_to_ci(p) for p in paths)


def check_trigger_windows_ci_for_subtree_path(path):
    """Returns true if path matches any of matches windows ci subtree patterns"""
    for windows_ci_subtree_patterns in trigger_windows_ci_for_subtrees_paths:
        if fnmatch.fnmatch(path, windows_ci_subtree_patterns):
            return True
    return False


def check_trigger_windows_ci_for_subtree(subtree: str) -> bool:
    """Returns true if the subtree root corresponds to a Windows CI-triggering subtree.

    Used for workflow_dispatch where explicit subtrees are provided rather than
    modified file paths. Patterns like 'projects/clr/*' are matched by stripping
    the trailing glob to get the subtree prefix 'projects/clr'.
    """
    for pattern in trigger_windows_ci_for_subtrees_paths:
        subtree_prefix = pattern.rstrip("/*").rstrip("/")
        if subtree == subtree_prefix or subtree.startswith(subtree_prefix + "/"):
            return True
    return False


# Paths matching any of these patterns are considered to have no influence over
# build or test workflows so any related jobs can be skipped if all paths
# modified by a commit/PR match a pattern in this list.
SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    ".gitignore",
    "*.md",
    "*.rtf",
    "*.rst",
    "*/.markdownlint-ci2.yaml",
    "*/.readthedocs.yaml",
    "*/.spellcheck.local.yaml",
    "*/.wordlist.txt",
    "projects/*/docs/*",
    "projects/*/.gitignore",
    "projects/rocr-runtime/libhsakmt/src/dxg/*",
    "shared/*/docs/*",
    "shared/*/.gitignore",
    "experimental/python/perfxpert/*",
]


def is_path_skippable(path: str) -> bool:
    """Determines if a given relative path to a file matches any skippable patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if at least one path is not in the skippable set."""
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


def check_rccl_changes(modified_paths: Optional[Iterable[str]]) -> bool:
    """Returns true if any files under projects/rccl/ were modified."""
    if modified_paths is None:
        return False
    return any(path.startswith("projects/rccl/") for path in modified_paths)


def retrieve_projects(args):
    # Nightly (schedule): use same test coverage as TheRock submodule bump PRs —
    # single nightly job with THEROCK_ENABLE_ALL=ON and full projects_to_test list.
    # Run all builds and tests including RCCL.
    if args.get("is_nightly"):
        nightly_config = project_map.get("nightly")
        if not nightly_config:
            logging.warning(
                "No 'nightly' entry in project_map, nightly will have no jobs"
            )
            return []
        # Run full coverage on both Linux and Windows (no path-based skip).
        return [
            {
                "cmake_options": nightly_config.get("cmake_options", ""),
                "projects_to_test": nightly_config.get("projects_to_test", ""),
            }
        ]

    # Check if CI should be skipped based on modified paths
    # (only for push and pull_request events, not workflow_dispatch or nightly)
    base_ref = args.get("base_ref")
    modified_paths = get_modified_paths(base_ref)
    print("modified_paths (max 200):", modified_paths[:200])

    # If only skippable paths were modified, skip CI
    if args.get("is_push") or args.get("is_pull_request"):
        if not check_for_non_skippable_path(modified_paths):
            logging.info("Only skippable paths were modified, skipping CI")
            return []

    # Push event → evaluate all subtrees
    if args.get("is_push"):
        subtrees = list(subtree_to_project_map.keys())

    # Manual workflow dispatch: respect explicit project selection, bypass CI file change detection
    elif args.get("is_workflow_dispatch"):
        if args.get("input_projects") == "all":
            subtrees = list(subtree_to_project_map.keys())
        else:
            subtrees = args.get("input_projects", "").split()

    else:
        # Determine which subtrees were modified (only needed for non-push/dispatch paths)
        matched_subtrees = set()
        for path in modified_paths:
            for subtree in subtree_to_project_map:
                if path.startswith(subtree):
                    matched_subtrees.add(subtree)

        # Change in CI workflow triggers full subtree evaluation
        if check_for_workflow_file_related_to_ci(modified_paths):
            logging.info("CI workflow files changed, evaluating all subtrees")
            subtrees = list(subtree_to_project_map.keys())

        # Pull request
        elif args.get("is_pull_request"):
            if args.get("input_subtrees"):
                subtrees = args.get("input_subtrees").split()
            else:
                subtrees = list(matched_subtrees)

        # Default case
        else:
            subtrees = list(matched_subtrees)

        # If files changed but no subtree matched → evaluate all
        if modified_paths and not subtrees and not check_rccl_changes(modified_paths):
            logging.info(
                "Modified files did not match known subtrees, evaluating all projects"
            )
            subtrees = list(subtree_to_project_map.keys())

    # Holds the python-specific cmake options passed to TheRock build.
    common_python_options = []

    # Linux CI skip logic: exclude Windows-only subtrees so they don't
    # produce Linux projects. If nothing remains, Linux CI is skipped.
    if args.get("platform") == "linux":
        subtrees = [s for s in subtrees if s not in windows_only_subtrees]

        # Common Python executable options for all builds.
        # Replaces TheRock's manylinux build behavior.
        # See build_tools/github_actions/manylinux_config.py in TheRock.
        common_python_options = [
            "-DTHEROCK_SHARED_PYTHON_EXECUTABLES=/opt/python-shared/cp310-cp310/bin/python3;/opt/python-shared/cp311-cp311/bin/python3;/opt/python-shared/cp312-cp312/bin/python3;/opt/python-shared/cp313-cp313/bin/python3;/opt/python-shared/cp314-cp314/bin/python3",
            "-DTHEROCK_DIST_PYTHON_EXECUTABLES=/opt/python/cp310-cp310/bin/python;/opt/python/cp311-cp311/bin/python;/opt/python/cp312-cp312/bin/python;/opt/python/cp313-cp313/bin/python",
        ]

    # Windows CI skip logic: skip if neither the modified file paths nor the
    # explicitly selected subtrees require Windows CI.
    if args.get("platform") == "windows":
        if args.get("is_workflow_dispatch"):
            if not any(check_trigger_windows_ci_for_subtree(s) for s in subtrees):
                logging.info("Selected subtrees do not require Windows CI, skipping")
                return []
        elif not any(
            check_trigger_windows_ci_for_subtree_path(path) for path in modified_paths
        ):
            logging.info("Modified paths do not require Windows CI, skipping")
            return []
    # Determine logical projects impacted
    projects = {
        subtree_to_project_map[subtree]
        for subtree in subtrees
        if subtree in subtree_to_project_map
    }

    if not projects:
        return []

    merged_flags = set()
    merged_tests = set()
    enable_all = False

    for project in projects:
        config = project_map.get(project)
        if not config:
            continue
        cmake_options = config.get("cmake_options", [])
        # Handle both array and string formats for backwards compatibility
        if isinstance(cmake_options, str):
            flags = [f.strip() for f in cmake_options.split()]
        else:
            flags = cmake_options
        if "-DTHEROCK_ENABLE_ALL=ON" in flags:
            enable_all = True
        merged_flags.update(flags)
        tests = config.get("projects_to_test", "")
        if tests:
            merged_tests.update(t.strip() for t in tests.split(","))
    if enable_all:
        final_flags_list = ["-DTHEROCK_ENABLE_ALL=ON"]
    else:
        final_flags_list = sorted(merged_flags)
    # Always append -DTHEROCK_ENABLE_CORE=ON as a default at the end
    final_flags_list.append("-DTHEROCK_ENABLE_CORE=ON")
    # Always append the Python options.
    final_flags_list += common_python_options
    # Removing duplicates
    final_flags_list = list(set(final_flags_list))
    final_flags = " ".join(final_flags_list)

    return [
        {
            "cmake_options": final_flags,
            "projects_to_test": ", ".join(sorted(merged_tests)),
        }
    ]


def run(args):
    project_to_run = retrieve_projects(args)
    outputs = {"projects": json.dumps(project_to_run)}

    # Determine if RCCL CI should run (only relevant for Linux platform)
    if args.get("platform") == "linux":
        if args.get("is_nightly"):
            # Nightly runs always run RCCL CI
            outputs["run_linux_rccl_ci"] = "true"
        elif args.get("is_workflow_dispatch"):
            # For workflow_dispatch, check if projects/rccl was explicitly selected
            input_projects = args.get("input_projects", "")
            if "projects/rccl" in input_projects:
                outputs["run_linux_rccl_ci"] = "true"
            else:
                outputs["run_linux_rccl_ci"] = "false"
        else:
            # For push/PR events, check if any files under projects/rccl/ changed
            base_ref = args.get("base_ref")
            modified_paths = get_modified_paths(base_ref)
            if check_rccl_changes(modified_paths):
                outputs["run_linux_rccl_ci"] = "true"
            else:
                outputs["run_linux_rccl_ci"] = "false"

    set_github_output(outputs)
    return outputs


if __name__ == "__main__":
    args = {}
    github_event_name = os.getenv("GITHUB_EVENT_NAME")
    github_workflow = os.getenv("GITHUB_WORKFLOW", "")
    args["is_pull_request"] = github_event_name == "pull_request"
    args["is_push"] = github_event_name == "push"
    args["is_workflow_dispatch"] = github_event_name == "workflow_dispatch"
    # Nightly: either scheduled run or manual dispatch of the nightly workflow
    args["is_nightly"] = github_event_name == "schedule" or (
        github_event_name == "workflow_dispatch"
        and github_workflow == "TheRock CI Nightly"
    )

    input_subtrees = os.getenv("SUBTREES", "")
    args["input_subtrees"] = input_subtrees

    input_projects = os.getenv("PROJECTS", "")
    args["input_projects"] = input_projects

    input_platform = os.getenv("PLATFORM")
    args["platform"] = input_platform

    args["base_ref"] = os.environ.get("BASE_REF", "HEAD^")

    logging.info(f"Retrieved arguments {args}")

    run(args)
