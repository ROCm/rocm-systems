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

    BuildTopology.get_stages_for_projects(projects)
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

Output (to $GITHUB_OUTPUT), written as ``build_stages=<value>``:
    a comma-separated allowlist of stages to build (empty string = build all)
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
    "rocprofiler-compute",
    "rocprofiler-register",
    "rocprofiler-sdk",
    "rocprofiler-systems",
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

    # Fail safe on ANY unrecognized project. get_stages_for_projects() silently
    # ignores names it cannot resolve, so a mixed input like "rdc,unknown" would
    # otherwise return only rdc's stages and wrongly narrow the build. Require
    # every changed project to resolve to an artifact before narrowing.
    unknown = [p for p in projects if topology.resolve_project_to_artifact(p) is None]
    if unknown:
        logger.info(f"Unrecognized project(s) {sorted(unknown)} -> build all stages")
        return []

    required = set(topology.get_stages_for_projects(projects))
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


def set_github_output(build_stages: List[str]) -> None:
    value = ",".join(build_stages)
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        print(f"build_stages={value}")
        return
    with open(output_file, "a") as f:
        f.write(f"build_stages={value}\n")


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
    build_stages = compute_build_stages(
        changed_projects=args.changed_projects,
        therock_path=args.therock_path,
        run_all_tests=str(args.run_all_tests).strip().lower() == "true",
    )
    set_github_output(build_stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
