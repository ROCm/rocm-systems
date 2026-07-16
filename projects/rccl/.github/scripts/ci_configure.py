# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit the RCCL multi-node CI build matrix for GitHub Actions.

Generalizes the coverage-only matrix to any workload (coverage, rccl-tests, AI ...).
Two tables:
  * CLUSTERS  - one row per cluster (SLURM partition/account/reservation, GPUs/node,
                self-hosted runner label, shared-FS root).
  * TARGETS   - one row per (workload, image, arch) target; each references a cluster
                and carries a `workload` field plus workload-specific `params`.

The reusable workflow passes INPUT_WORKLOAD; setup runs this and fans out one job per
enabled target for that workload. On workflow_dispatch, a non-empty rocm_image override
produces a single-entry matrix from the inputs (using INPUT_CLUSTER for SLURM settings).

Enable a cluster: fill in its CLUSTERS row and register a runner with that label.
Add a target:     append one TARGETS row with enabled=True and the right `workload`.
"""

import json
import os
from typing import Mapping

from orchestrator import DEFAULT_ROCM_IMAGE  # single source of truth for the image


# --- One row per cluster ----------------------------------------------------
# reservation: "" means none (plain partition scheduling). shared_fs_root: ""
# lets the runner default to $HOME (must be a shared FS visible on all nodes).
CLUSTERS = {
    "mi300x-rccl": {
        "partition": "rccl",
        "account": "rccl",
        "reservation": "",            # set to "rccl_415" to pin the shared reservation
        "gpus_per_node": "8",
        "runner": "slurm-login",
        "shared_fs_root": "",
    },
    "alola": {
        "partition": "CHANGEME",
        "account": "CHANGEME",
        "reservation": "",
        "gpus_per_node": "8",
        "runner": "alola-slurm-login",
        "shared_fs_root": "",
    },
    "ruby": {
        "partition": "CHANGEME",
        "account": "CHANGEME",
        "reservation": "",
        "gpus_per_node": "8",
        "runner": "ruby-slurm-login",
        "shared_fs_root": "",
    },
    "tw": {
        "partition": "CHANGEME",
        "account": "CHANGEME",
        "reservation": "",
        "gpus_per_node": "8",
        "runner": "tw-slurm-login",
        "shared_fs_root": "",
    },
}

# --- One row per (workload, image, arch) target -----------------------------
# `params` holds workload-specific fields flattened into the matrix entry (so the
# workflow can forward them as env). Coverage/rccl-tests use test_config/test_suite/
# test_name; an AI workload might use model/script/extra_repo, etc.
TARGETS = [
    {
        "name": "coverage-gfx942-ubuntu24-7.13.0rc2",
        "workload": "coverage",
        "enabled": True,
        "cluster": "mi300x-rccl",
        "rocm_image": DEFAULT_ROCM_IMAGE,
        "dockerfile": "Dockerfile.Multinode.Ubuntu",
        "gpu_arch": "gfx942",
        "nic_type": "mellanox",
        "nodes": "2",
        "params": {"test_config": "mi300x_mellanox_ib.json"},
    },
    # Example rccl-tests target (enable once the payload lands in Phase 4).
    {
        "name": "rccl-tests-gfx942-ubuntu24-7.13.0rc2",
        "workload": "rccl-tests",
        "enabled": False,
        "cluster": "mi300x-rccl",
        "rocm_image": DEFAULT_ROCM_IMAGE,
        "dockerfile": "Dockerfile.Multinode.Ubuntu",
        "gpu_arch": "gfx942",
        "nic_type": "mellanox",
        "nodes": "2",
        "params": {"test_config": "mi300x_mellanox_ib.json"},
    },
]

# Fields copied from the referenced cluster into each matrix entry.
_CLUSTER_FIELDS = ("partition", "account", "reservation", "gpus_per_node", "runner", "shared_fs_root")


def gha_set_output(vars: Mapping[str, str]) -> None:
    """Append key=value pairs to $GITHUB_OUTPUT (or print when running locally)."""
    out = os.getenv("GITHUB_OUTPUT")
    print(f"Setting outputs: {vars}")
    if not out:
        return
    with open(out, "a") as f:
        f.writelines(f"{k}={v}\n" for k, v in vars.items())


def _resolve_cluster(cluster_name: str) -> dict:
    """Return the SLURM fields for a cluster (fail loudly on typos)."""
    if cluster_name not in CLUSTERS:
        raise SystemExit(f"Unknown cluster '{cluster_name}'. Known: {sorted(CLUSTERS)}")
    return {k: CLUSTERS[cluster_name][k] for k in _CLUSTER_FIELDS}


def _flatten(target: dict) -> dict:
    """Matrix entry from a target row: drop control keys, hoist `params`."""
    entry = {k: v for k, v in target.items() if k not in ("enabled", "params")}
    entry.update(target.get("params", {}))
    return entry


def _entry_from_dispatch(env: Mapping[str, str], workload: str) -> dict:
    """Build a single matrix entry from workflow_dispatch inputs."""
    cluster = env.get("INPUT_CLUSTER") or "mi300x-rccl"
    entry = {
        "name": f"{workload}-{env.get('INPUT_GPU_ARCH') or 'gfx942'}-{cluster}",
        "workload": workload,
        "cluster": cluster,
        "rocm_image": env["INPUT_ROCM_IMAGE"],
        "dockerfile": env.get("INPUT_DOCKERFILE") or "Dockerfile.Multinode.Ubuntu",
        "gpu_arch": env.get("INPUT_GPU_ARCH") or "gfx942",
        "nic_type": env.get("INPUT_NIC_TYPE") or "mellanox",
        "nodes": env.get("INPUT_NODES") or "2",
        "test_config": env.get("INPUT_TEST_CONFIG") or "mi300x_mellanox_ib.json",
    }
    entry.update(_resolve_cluster(cluster))
    return entry


def _parse_selection(raw: str) -> set:
    """Parse the comma/space-separated selection ('all' or empty = no filter)."""
    tokens = {t.strip() for t in raw.replace(",", " ").split() if t.strip()}
    tokens.discard("all")
    return tokens


def _matches_selection(target: dict, selection: set) -> bool:
    """Selected if the target's name, gpu_arch, or cluster is in the selection set."""
    return bool({target["name"], target["gpu_arch"], target["cluster"]} & selection)


def build_matrix(env: Mapping[str, str]) -> list:
    """Return the list of matrix entries for this run (filtered by workload)."""
    workload = env.get("INPUT_WORKLOAD", "coverage")
    if env.get("INPUT_ROCM_IMAGE"):
        entries = [_entry_from_dispatch(env, workload)]
    else:
        selection = _parse_selection(env.get("INPUT_TARGETS", ""))
        entries = []
        for t in TARGETS:
            if t.get("workload") != workload or not t.get("enabled"):
                continue
            if selection and not _matches_selection(t, selection):
                continue
            entry = _flatten(t)
            entry.update(_resolve_cluster(t["cluster"]))
            entries.append(entry)
    # Attach shared gtest filters to every entry (empty = all).
    for e in entries:
        e.setdefault("test_suite", env.get("INPUT_TEST_SUITE", ""))
        e.setdefault("test_name", env.get("INPUT_TEST_NAME", ""))
        e.setdefault("test_config", "")
    return entries


def main() -> None:
    entries = build_matrix(os.environ)
    gha_set_output(
        {
            "matrix": json.dumps({"include": entries}),
            "has_targets": "true" if entries else "false",
        }
    )


if __name__ == "__main__":
    main()
