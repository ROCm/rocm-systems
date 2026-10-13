# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute the build-stage allowlist for a rocm-systems PR.

This is the rocm-systems half of per-PR selective builds. It maps the changed
projects to the minimal set of TheRock build stages needed to build them and
emits that set as ``build_stages`` -- an *allowlist* consumed by TheRock's
``setup_multi_arch.yml`` (added in ROCm/TheRock#7600). Every stage outside the
allowlist is skipped by TheRock outright: no build, no artifact copy. An empty
allowlist means "build every stage" (the safe default).

Why an allowlist (not a skip list): callers declare the stages they *want*, so
adding a new stage to BUILD_TOPOLOGY.toml never silently widens a narrow PR's
scope -- the new stage is simply not requested. TheRock computes the skip
complement and validates the names on its side.

Stage selection is delegated to TheRock's build topology (the single source of
truth), imported from the TheRock checkout so this script owns no dependency
math and cannot drift from the actual build graph:

    BuildTopology.resolve_alias_to_artifact(project)  # project -> artifact
    BuildTopology.get_stages_for_artifacts(artifacts)
        -> the minimal set of stages needed to BUILD the changed projects
           (impacted stages plus their upstream build dependencies).

Fail-safe: whenever the change set cannot be confidently narrowed, this prints
an empty ``build_stages`` so TheRock builds everything. That happens for: a
run_all_tests (CI-infra) change, no changed projects, a full-build project (see
FULL_BUILD_PROJECTS), an unknown/unmapped project, or any topology-load error.

Usage:
    python compute_build_stages.py \
        --changed-projects "projects/rdc,projects/rocdecode" \
        --run-all-tests false \
        --therock-path _therock

It also emits ``windows_amdgpu_families=none`` when every changed project is
disabled on Windows in the topology, so TheRock skips the Windows build. Any
other case emits an empty value, which keeps the workflow's default families.

Output (to $GITHUB_OUTPUT):
    build_stages=<value>: a comma-separated allowlist of stages to build
        (empty string = build all)
    windows_amdgpu_families=<value>: "none" to skip Windows, else empty
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Projects whose changes ripple into downstream consumers. For these we build
# everything (empty allowlist) to preserve coverage, mirroring the pre-multi-arch
# CI where clr/hip/rocr-runtime, amdsmi, and profiler mapped to
# -DTHEROCK_ENABLE_ALL=ON in .github/scripts/therock_matrix.py.
FULL_BUILD_PROJECTS = {
    "clr",
    "hip",
    "hipother",
    "hip-tests",
    "rocr-runtime",
    "amdsmi",
    # Legacy profiler bucket used THEROCK_ENABLE_ALL=ON.
    "aqlprofile",
    "rocprofiler",
    "rocprofiler-register",
    "rocprofiler-sdk",
    "roctracer",
}


def _parse_projects(changed_projects: str) -> List[str]:
    """Normalize 'projects/clr,projects/hip' -> ['clr', 'hip']."""
    projects = []
    for raw in changed_projects.split(","):
        name = raw.strip().rstrip("/").split("/")[-1]
        if name:
            projects.append(name)
    return projects


def compute_build_stages(
    changed_projects: str,
    therock_path: str,
    run_all_tests: bool = False,
) -> List[str]:
    """Return the allowlist of build stages required for this change set.

    Returns [] (build everything) whenever narrowing is not safe.
    """
    # CI-infra changes (workflow/scripts/repos-config) force a full run upstream;
    # never narrow the build in that case.
    if run_all_tests:
        logger.info("run_all_tests set -> build all stages")
        return []

    projects = _parse_projects(changed_projects)
    if not projects:
        logger.info("No changed projects -> build all stages")
        return []

    # Fan-out projects ripple into consumers: build everything.
    if FULL_BUILD_PROJECTS.intersection(projects):
        logger.info("Full-build project changed -> build all stages")
        return []

    # Import TheRock's build topology from the checkout. TheRock is the source of
    # truth for the build graph; we only read it.
    therock_build_tools = (Path(therock_path) / "build_tools").resolve()
    therock_build_tools_str = os.fspath(therock_build_tools)

    if therock_build_tools_str not in sys.path:
        sys.path.insert(0, therock_build_tools_str)

    try:
        from _therock_utils.build_topology import get_topology

        topology = get_topology()
        all_stages = set(topology.get_all_stage_names())
    except Exception as e:  # noqa: BLE001 - never let this analysis break CI
        logger.warning(f"Topology load failed ({e}) -> build all stages")
        return []

    # Fail safe on ANY unrecognized project. get_stages_for_artifacts() silently
    # ignores names it cannot resolve, so a mixed input like "rdc,unknown" would
    # otherwise return only rdc's stages and wrongly narrow the build. Require
    # every changed project to resolve to an artifact before narrowing.
    #
    # NOTE: TheRock removed get_stages_for_projects()/resolve_project_to_artifact()
    # after the previously pinned ref. We resolve projects -> artifacts with
    # resolve_alias_to_artifact() (which understands artifact names, subproject
    # aliases from artifact_subprojects.json, split_databases, and BUILD_TOPOLOGY
    # source_paths) and then map artifacts -> stages with get_stages_for_artifacts().
    resolved = {p: topology.resolve_alias_to_artifact(p) for p in projects}
    unknown = [p for p, art in resolved.items() if art is None]
    if unknown:
        logger.info(f"Unrecognized project(s) {sorted(unknown)} -> build all stages")
        return []

    required = set(topology.get_stages_for_artifacts(list(resolved.values())))
    if not required:
        logger.info("No stages mapped for changed projects -> build all stages")
        return []

    # If everything is required, emit empty (build all) rather than a redundant
    # full list -- semantically identical to TheRock and easier to read in logs.
    if required == all_stages:
        logger.info("All stages required -> build all stages")
        return []

    build_stages = sorted(required)
    logger.info(f"changed projects: {sorted(projects)}")
    logger.info(f"build stages (allowlist): {build_stages}")
    return build_stages


def compute_windows_families(
    changed_projects: str,
    therock_path: str,
    run_all_tests: bool = False,
) -> str:
    """Return "none" when every changed project is disabled on Windows, else "".

    An empty value keeps the workflow's default Windows families.
    """
    if run_all_tests:
        return ""

    projects = _parse_projects(changed_projects)
    if not projects:
        return ""

    therock_build_tools_str = os.fspath((Path(therock_path) / "build_tools").resolve())
    if therock_build_tools_str not in sys.path:
        sys.path.insert(0, therock_build_tools_str)

    try:
        from _therock_utils.build_topology import get_topology

        topology = get_topology()
    except Exception as e:  # noqa: BLE001 - never let this analysis break CI
        logger.warning(f"Topology load failed ({e}) -> keep Windows")
        return ""

    for project in projects:
        artifact = topology.resolve_alias_to_artifact(project)
        if artifact is None:
            logger.info(f"Unrecognized project {project} -> keep Windows")
            return ""
        if "windows" not in topology.artifacts[artifact].disable_platforms:
            logger.info(f"{project} builds on Windows -> keep Windows")
            return ""

    logger.info(
        f"All changed projects are Linux-only {sorted(projects)} -> skip Windows"
    )
    return "none"


def set_github_output(build_stages: List[str], windows_families: str) -> None:
    lines = [
        f"build_stages={','.join(build_stages)}",
        f"windows_amdgpu_families={windows_families}",
    ]
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        print("\n".join(lines))
        return
    with open(output_file, "a") as f:
        f.write("".join(f"{line}\n" for line in lines))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the build-stage allowlist")
    parser.add_argument(
        "--changed-projects",
        default="",
        help="Comma-separated changed projects (e.g. 'projects/rdc,projects/hip')",
    )
    parser.add_argument(
        "--therock-path",
        default="_therock",
        help="Path to the TheRock checkout (contains BUILD_TOPOLOGY.toml)",
    )
    parser.add_argument(
        "--run-all-tests",
        default="false",
        help="When 'true', build all stages. Set from the configure job's "
        "run_all_tests output for CI-infra changes.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_all_tests = str(args.run_all_tests).strip().lower() == "true"
    build_stages = compute_build_stages(
        changed_projects=args.changed_projects,
        therock_path=args.therock_path,
        run_all_tests=run_all_tests,
    )
    windows_families = compute_windows_families(
        changed_projects=args.changed_projects,
        therock_path=args.therock_path,
        run_all_tests=run_all_tests,
    )
    set_github_output(build_stages, windows_families)
    return 0


if __name__ == "__main__":
    sys.exit(main())
