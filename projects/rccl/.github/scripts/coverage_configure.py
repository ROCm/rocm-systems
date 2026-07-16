# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit the RCCL code-coverage build matrix for GitHub Actions.

Two tables:
  * CLUSTERS         - one row per cluster (SLURM partition/account/reservation,
                       GPUs/node, self-hosted runner label, shared-FS root).
  * COVERAGE_TARGETS - one row per (image, arch) target; each references a cluster.

Enable a cluster: fill in its CLUSTERS row and register a runner with that label.
Add a target:     append one COVERAGE_TARGETS row with enabled=True.

The workflow's setup job runs this and fans out one coverage job per enabled row.
On workflow_dispatch, a non-empty rocm_image override produces a single-entry
matrix from the inputs (using INPUT_CLUSTER for SLURM settings).
"""

import json
import os
from typing import Mapping


# --- One row per cluster ----------------------------------------------------
# reservation: "" means none (plain partition scheduling). shared_fs_root: ""
# lets run_coverage.sh default to $HOME (must be a shared FS visible on all nodes).
CLUSTERS = {
    # Current cluster where the RCCL team allocations live (atkulkar's nodes).
    "mi300x-rccl": {
        "partition": "rccl",
        "account": "rccl",
        "reservation": "",            # set to "rccl_415" to pin the shared reservation
        "gpus_per_node": "8",
        "runner": "slurm-login",
        "shared_fs_root": "",
    },
    # --- Fill these in per cluster (discover with `sinfo` / `scontrol show reservation`) ---
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

# --- One row per deployable (image, arch) target ----------------------------
COVERAGE_TARGETS = [
    {
        "name": "gfx942-ubuntu24-7.13.0rc2",
        "enabled": True,
        "cluster": "mi300x-rccl",
        "rocm_image": "registry-sc-harbor.amd.com/framework/therock-release:47_gfx94X_7.13.0rc2_ubuntu24.04_py3.12_pytorch_release-2.11_96bfee1",
        "dockerfile": "Dockerfile.Multinode.Ubuntu",
        "gpu_arch": "gfx942",
        "test_config": "mi300x_mellanox_ib.json",
        "nic_type": "mellanox",
        "nodes": "2",
    },
    # Example: same image on other clusters (enable once their CLUSTERS rows are filled).
    {
        "name": "gfx942-ubuntu24-alola",
        "enabled": False,
        "cluster": "alola",
        "rocm_image": "registry-sc-harbor.amd.com/framework/therock-release:47_gfx94X_7.13.0rc2_ubuntu24.04_py3.12_pytorch_release-2.11_96bfee1",
        "dockerfile": "Dockerfile.Multinode.Ubuntu",
        "gpu_arch": "gfx942",
        "test_config": "mi300x_mellanox_ib.json",
        "nic_type": "mellanox",
        "nodes": "2",
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


def _entry_from_dispatch(env: Mapping[str, str]) -> dict:
    """Build a single matrix entry from workflow_dispatch inputs."""
    cluster = env.get("INPUT_CLUSTER") or "mi300x-rccl"
    entry = {
        "name": f"dispatch-{env.get('INPUT_GPU_ARCH') or 'gfx942'}-{cluster}",
        "cluster": cluster,
        "rocm_image": env["INPUT_ROCM_IMAGE"],
        "dockerfile": env.get("INPUT_DOCKERFILE") or "Dockerfile.Multinode.Ubuntu",
        "gpu_arch": env.get("INPUT_GPU_ARCH") or "gfx942",
        "test_config": env.get("INPUT_TEST_CONFIG") or "mi300x_mellanox_ib.json",
        "nic_type": env.get("INPUT_NIC_TYPE") or "mellanox",
        "nodes": env.get("INPUT_NODES") or "2",
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
    """Return the list of matrix entries for this run."""
    if env.get("INPUT_ROCM_IMAGE"):
        entries = [_entry_from_dispatch(env)]
    else:
        selection = _parse_selection(env.get("INPUT_TARGETS", ""))
        entries = []
        for t in COVERAGE_TARGETS:
            if not t.get("enabled"):
                continue
            if selection and not _matches_selection(t, selection):
                continue
            entry = {k: v for k, v in t.items() if k != "enabled"}
            entry.update(_resolve_cluster(t["cluster"]))
            entries.append(entry)
    # Attach shared gtest filters to every entry.
    for e in entries:
        e.setdefault("test_suite", env.get("INPUT_TEST_SUITE", ""))
        e.setdefault("test_name", env.get("INPUT_TEST_NAME", ""))
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
