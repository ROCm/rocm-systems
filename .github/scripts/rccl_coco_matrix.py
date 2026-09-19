#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit GitHub Actions matrix JSON for RCCL coco PR and scheduled workflows."""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_utils import set_github_output

# prefix names the coco jobs (<prefix>-rccl-pr-{build,smoke}).
PR_CLUSTERS = [
    {
        "name": "Ruby",
        "prefix": "ruby64",
        "runner": "ruby-linux-slurm-scale-runner",
    },
    {
        "name": "OCI",
        "prefix": "oci",
        "runner": "oci-linux-slurm-scale-runner",
    },
]

# mode, job, cluster, kind, timeout_minutes
SCHEDULED_CATALOGUE = [
    ("nightly", "ruby64-rccl-perf-nightly", "ruby", "perf", 720),
    ("nightly", "ruby64-rccl-coverage-nightly", "ruby", "coverage", 720),
    ("nightly", "ruby64-rccl-unit-nightly", "ruby", "unit", 720),
    ("nightly", "oci-rccl-perf-nightly", "oci", "perf", 720),
    ("nightly", "oci-rccl-unit-nightly", "oci", "unit", 720),
    ("nightly", "oci-nixl-nightly", "oci", "nixl", 120),
    ("weekly", "ruby64-rccl-perf-weekly", "ruby", "perf", 2880),
    ("weekly", "oci-rccl-perf-weekly", "oci", "perf", 2880),
    ("monthly", "ruby64-rccl-perf-monthly", "ruby", "perf", 2880),
    ("monthly", "oci-rccl-perf-monthly", "oci", "perf", 2880),
    # Crusoe: job configs exist in tRCCL, but there is no crusoe label in
    # cluster-runners.allowlist yet.
    # ("nightly", "crusoe-rccl-perf-nightly", "crusoe", "perf", 720),
    # ("nightly", "crusoe-rccl-unit-nightly", "crusoe", "unit", 720),
    # ("weekly", "crusoe-rccl-perf-weekly", "crusoe", "perf", 2880),
    # ("monthly", "crusoe-rccl-perf-monthly", "crusoe", "perf", 2880),
]

SCHEDULED_RUNNERS = {
    "ruby": "ruby-linux-slurm-scale-runner",
    "oci": "oci-linux-slurm-scale-runner",
    # "crusoe": "<crusoe runner label>",
}


def pr_clusters() -> None:
    set_github_output({"matrix": json.dumps({"include": PR_CLUSTERS})})


def scheduled_runs(mode: str) -> None:
    include = [
        {
            "job": job,
            "runner": SCHEDULED_RUNNERS[cluster],
            "timeout": timeout,
            "group": f"rccl-coco-{entry_mode}-{cluster}-{kind}",
        }
        for entry_mode, job, cluster, kind, timeout in SCHEDULED_CATALOGUE
        if entry_mode == mode
    ]
    set_github_output(
        {
            "matrix": json.dumps({"include": include}),
            "has_runs": "true" if include else "false",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit matrix outputs for RCCL coco workflows."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "pr-clusters",
        help="Emit the PR build/smoke cluster matrix.",
    )

    scheduled = sub.add_parser(
        "scheduled-runs",
        help="Emit the scheduled run matrix for one periodic mode.",
    )
    scheduled.add_argument(
        "--mode",
        required=True,
        choices=("nightly", "weekly", "monthly"),
    )

    args = parser.parse_args()
    if args.command == "pr-clusters":
        pr_clusters()
    else:
        scheduled_runs(args.mode)


if __name__ == "__main__":
    main()
