# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit the RCCL multi-node CI build matrix for GitHub Actions.

Generalizes the coverage-only matrix to any workload (coverage, rccl-tests, AI ...).
The data lives in ci_targets.yml (clusters + targets); this module only loads,
validates, and turns it into a GitHub Actions matrix — so adding a cluster or target
is a YAML edit, no code change:
  * clusters - one row per cluster (SLURM partition/account/reservation, GPUs/node,
               self-hosted runner label, shared-FS root).
  * targets  - one row per (workload, image, arch) target; references a cluster and
               carries a `workload` field plus workload-specific `params`.

The reusable workflow passes INPUT_WORKLOAD; setup runs this and fans out one job per
enabled target for that workload. On workflow_dispatch, a non-empty rocm_image override
produces a single-entry matrix from the inputs (using INPUT_CLUSTER for SLURM settings).
"""

import json
import os
from pathlib import Path
from string import Template
from typing import Mapping

from orchestrator import DEFAULT_ROCM_IMAGE  # single source of truth for the image

# Data file (clusters + targets); edited far more often than this loader.
CONFIG_FILE = Path(__file__).with_name("ci_targets.yml")
# ${VARS} allowed in YAML string values (keeps the image single-sourced).
_SUBSTITUTIONS = {"DEFAULT_ROCM_IMAGE": DEFAULT_ROCM_IMAGE}
# Fields copied from the referenced cluster into each matrix entry.
_CLUSTER_FIELDS = ("partition", "account", "reservation", "gpus_per_node", "runner", "shared_fs_root")
_REQUIRED_TARGET_KEYS = ("name", "workload", "enabled", "cluster", "rocm_image",
                         "dockerfile", "gpu_arch", "nic_type", "nodes")
_DEFAULT_CLUSTER = "alola"


def _scalar(value):
    """Coerce a YAML scalar to the string the matrix expects (bools stay bools).

    Strings get ${VAR} substitution; numbers (e.g. nodes: 2) become "2" so matrix
    values are stringly-typed exactly as the workflow env expects.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return Template(value).safe_substitute(_SUBSTITUTIONS)
    return str(value)


def _load_config():
    """Load + normalize clusters and targets from ci_targets.yml."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML is required to read ci_targets.yml (pip install pyyaml).")
    with open(CONFIG_FILE) as f:
        data = yaml.safe_load(f) or {}
    clusters = {
        name: {k: _scalar(fields.get(k, "")) for k in _CLUSTER_FIELDS}
        for name, fields in (data.get("clusters") or {}).items()
    }
    targets = []
    for t in (data.get("targets") or []):
        entry = {}
        for k, v in t.items():
            if k == "params":
                entry[k] = {pk: _scalar(pv) for pk, pv in (v or {}).items()}
            elif k == "enabled":
                entry[k] = bool(v)
            else:
                entry[k] = _scalar(v)
        targets.append(entry)
    _validate(clusters, targets)
    return clusters, targets


def _validate(clusters: dict, targets: list) -> None:
    """Fail fast on malformed data (missing keys / dangling cluster refs)."""
    for t in targets:
        missing = [k for k in _REQUIRED_TARGET_KEYS if k not in t]
        if missing:
            raise SystemExit(f"Target {t.get('name', '?')!r} missing keys: {missing}")
        # Only enabled targets must resolve to a real cluster (disabled rows may be WIP).
        if t["enabled"] and t["cluster"] not in clusters:
            raise SystemExit(
                f"Target {t['name']!r} references unknown cluster {t['cluster']!r}; "
                f"known: {sorted(clusters)}"
            )


CLUSTERS, TARGETS = _load_config()


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
    cluster = env.get("INPUT_CLUSTER") or _DEFAULT_CLUSTER
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
