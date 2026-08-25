# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute build stages that can be fully SKIPPED for a rocm-systems PR.

This is the rocm-systems half of per-PR stage skipping. It lives entirely in
rocm-systems and drives TheRock's *existing* ``prebuilt_stages`` input: TheRock
already gates every per-stage build job on
``!contains(inputs.prebuilt_stages, '<stage>')`` and only runs its
``copy_prebuilt_stages`` (reuse) job when ``baseline_run_id != ''``. So passing
this skip list as ``prebuilt_stages`` with an EMPTY ``baseline_run_id`` skips the
stage's build without copying anything -- a pure skip, with no changes to
TheRock.

    "Skip now, reuse later": when a baseline becomes available, the caller can
    switch ``stage_reuse_mode`` to ``reuse-stage`` and/or supply a
    ``baseline_run_id`` and these same stages become reuse (copy) candidates,
    using the identical TheRock plumbing.

Stage selection is delegated to TheRock's build topology (the single source of
truth), imported from the TheRock checkout so this script owns no dependency
math and cannot drift from the actual build graph:

    BuildTopology.get_stages_for_projects(projects)
        -> the minimal set of stages needed to BUILD the changed projects
           (impacted stages plus their upstream build dependencies).

Every stage NOT in that required set is emitted as skippable.

Fail-safe: whenever the change set cannot be confidently narrowed, this prints an
empty ``skip_stages`` so TheRock builds everything (the safe default). That
happens for: no changed projects, a fan-out project (see FANOUT_PROJECTS), an
unknown/unmapped project, or any topology-load error.

Usage:
    python compute_skip_stages.py \
        --changed-projects "projects/rdc,projects/rocdecode" \
        --therock-path _therock

Output (to $GITHUB_OUTPUT):
    skip_stages: comma-separated stage names to skip (may be empty)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Projects whose changes ripple into downstream consumers. For these we keep the
# full build (skip nothing) to preserve coverage, mirroring the pre-multiarch CI
# where clr/hip/rocr-runtime, amdsmi, and profiler mapped to
# -DTHEROCK_ENABLE_ALL=ON in .github/scripts/therock_matrix.py.
FANOUT_PROJECTS = {
    "clr",
    "hip",
    "hipother",
    "hip-tests",
    "rocr-runtime",
    "amdsmi",
}


def _parse_projects(changed_projects: str) -> List[str]:
    """Normalize 'projects/clr,projects/hip' -> ['clr', 'hip']."""
    projects = []
    for raw in changed_projects.split(","):
        name = raw.strip().rstrip("/").split("/")[-1]
        if name:
            projects.append(name)
    return projects


def compute_skip_stages(changed_projects: str, therock_path: str) -> List[str]:
    """Return the list of build stages that can be fully skipped.

    Returns [] (skip nothing -> full build) whenever narrowing is not safe.
    """
    projects = _parse_projects(changed_projects)
    if not projects:
        logger.info("No changed projects -> skip nothing (full build)")
        return []

    if FANOUT_PROJECTS.intersection(projects):
        logger.info("Fan-out project changed -> skip nothing (full build)")
        return []

    # Import TheRock's build topology from the checkout. TheRock is the source of
    # truth for the build graph; we only read it.
    therock_build_tools = Path(therock_path) / "build_tools"
    sys.path.insert(0, os.fspath(therock_build_tools))
    try:
        from _therock_utils.build_topology import get_topology

        topology = get_topology()
        all_stages = set(topology.get_all_stage_names())
        required = set(topology.get_stages_for_projects(projects))
    except Exception as e:  # noqa: BLE001 - never let skip analysis break CI
        logger.warning(f"Topology analysis failed ({e}) -> skip nothing (full build)")
        return []

    # Unknown project (no stage mapping) -> be safe and build everything.
    if not required:
        logger.info("No stages mapped for changed projects -> skip nothing")
        return []

    skip = sorted(all_stages - required)
    logger.info(f"changed projects: {sorted(projects)}")
    logger.info(f"required stages : {sorted(required)}")
    logger.info(f"skip stages     : {skip}")
    return skip


def set_github_output(skip_stages: List[str]) -> None:
    value = ",".join(skip_stages)
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        print(f"skip_stages={value}")
        return
    with open(output_file, "a") as f:
        f.write(f"skip_stages={value}\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute skippable build stages")
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    skip_stages = compute_skip_stages(
        changed_projects=args.changed_projects,
        therock_path=args.therock_path,
    )
    set_github_output(skip_stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
