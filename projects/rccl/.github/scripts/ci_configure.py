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
import re
from pathlib import Path
from string import Template
from typing import Mapping

from orchestrator import DEFAULT_ROCM_IMAGE  # single source of truth for the image

# Data file (clusters + targets); edited far more often than this loader. The source
# is overridable so the target list can change WITHOUT a code PR:
#   CI_TARGETS_URL  - fetch the YAML over HTTP(S) (raw git on another branch/repo,
#                     gist, S3, an internal endpoint, ...)
#   CI_TARGETS_FILE - read it from an alternate path (e.g. a separately checked-out
#                     config repo, or a shared-FS location)
# If neither is set, the in-repo file next to this script is used.
DEFAULT_CONFIG_FILE = Path(__file__).with_name("ci_targets.yml")


# --- security limits for the overridable config source ----------------------
# The config drives what runs on self-hosted cluster runners, so treat a remote
# source as sensitive: only https (http requires an explicit opt-in), cap the
# download size, and re-check the scheme on every redirect (blocks file://, ftp://,
# gopher://, and redirect-based SSRF). YAML is parsed with safe_load only.
_MAX_CONFIG_BYTES = 1 << 20          # 1 MiB is far more than a target table needs
_FETCH_TIMEOUT_S = 30


def _check_url_scheme(url: str) -> None:
    from urllib.parse import urlparse
    scheme = urlparse(url).scheme.lower()
    allow_http = os.environ.get("CI_TARGETS_ALLOW_INSECURE") == "1"
    allowed = {"https"} | ({"http"} if allow_http else set())
    if scheme not in allowed:
        hint = "" if allow_http else " (or set CI_TARGETS_ALLOW_INSECURE=1 to allow http)"
        raise SystemExit(f"CI_TARGETS_URL scheme {scheme!r} not allowed; use https{hint}: {url!r}")


def _fetch_url(url: str) -> str:
    import urllib.request

    class _StrictRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            _check_url_scheme(newurl)      # re-validate the redirect target scheme
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    _check_url_scheme(url)
    print(f"Loading ci_targets from URL: {url}")
    opener = urllib.request.build_opener(_StrictRedirect)
    with opener.open(url, timeout=_FETCH_TIMEOUT_S) as resp:
        data = resp.read(_MAX_CONFIG_BYTES + 1)
    if len(data) > _MAX_CONFIG_BYTES:
        raise SystemExit(f"CI_TARGETS_URL response exceeds {_MAX_CONFIG_BYTES} bytes; refusing.")
    return data.decode("utf-8")


def _read_config_text() -> str:
    """Return the ci_targets YAML text from the overridden source or the in-repo file."""
    url = os.environ.get("CI_TARGETS_URL", "").strip()
    if url:
        return _fetch_url(url)
    path = os.environ.get("CI_TARGETS_FILE", "").strip() or str(DEFAULT_CONFIG_FILE)
    print(f"Loading ci_targets from file: {path}")
    with open(path) as f:
        text = f.read(_MAX_CONFIG_BYTES + 1)
    if len(text) > _MAX_CONFIG_BYTES:
        raise SystemExit(f"{path} exceeds {_MAX_CONFIG_BYTES} bytes; refusing.")
    return text
# ${VARS} allowed in YAML string values (keeps the image single-sourced).
_SUBSTITUTIONS = {"DEFAULT_ROCM_IMAGE": DEFAULT_ROCM_IMAGE}
# Fields copied from the referenced cluster into each matrix entry.
_CLUSTER_FIELDS = ("partition", "account", "reservation", "gpus_per_node", "runner", "shared_fs_root")
_REQUIRED_TARGET_KEYS = ("name", "workload", "enabled", "cluster", "rocm_image",
                         "dockerfile", "gpu_arch", "nic_type", "nodes")
_DEFAULT_CLUSTER = "alola"

# Allowlisted value shapes. These values flow (via the matrix -> env) into shell
# commands the orchestrator runs over ssh/docker on self-hosted runners, so we reject
# anything outside a conservative charset (no spaces, quotes, or shell metacharacters
# like ; & | $ ` ( ) < > * ? \). Defense-in-depth on top of the orchestrator's own
# shlex.quote(); empty ("") is allowed where the field is optional.
_TARGET_VALUE_RE = {
    "name": r"[A-Za-z0-9._-]+",
    "workload": r"[A-Za-z0-9._-]+",
    "cluster": r"[A-Za-z0-9._-]+",
    "rocm_image": r"[A-Za-z0-9._:/@-]+",
    "dockerfile": r"[A-Za-z0-9._-]+",
    "gpu_arch": r"[A-Za-z0-9]+",
    "nic_type": r"[A-Za-z0-9_-]+",
    "nodes": r"[0-9]+",
}
_CLUSTER_VALUE_RE = {
    "partition": r"[A-Za-z0-9._-]*",
    "account": r"[A-Za-z0-9._-]*",
    "reservation": r"[A-Za-z0-9._-]*",
    "gpus_per_node": r"[0-9]+",
    "runner": r"[A-Za-z0-9._-]+",
    "shared_fs_root": r"[A-Za-z0-9._/-]*",
}
_PARAM_VALUE_RE = r"[A-Za-z0-9._/,:@=-]*"       # test_config filenames, flags, etc.


def _check_value(where: str, key: str, value, pattern: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise SystemExit(
            f"{where}: field {key!r} value {value!r} is not allowed "
            f"(must match /^{pattern}$/) — rejecting potentially unsafe config."
        )


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
    data = yaml.safe_load(_read_config_text()) or {}
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
    """Fail fast on malformed or unsafe data (missing keys, dangling refs, bad values)."""
    for name, fields in clusters.items():
        _check_value("cluster name", "cluster", name, r"[A-Za-z0-9._-]+")
        for k, pat in _CLUSTER_VALUE_RE.items():
            _check_value(f"cluster {name!r}", k, fields.get(k, ""), pat)
    for t in targets:
        missing = [k for k in _REQUIRED_TARGET_KEYS if k not in t]
        if missing:
            raise SystemExit(f"Target {t.get('name', '?')!r} missing keys: {missing}")
        for k, pat in _TARGET_VALUE_RE.items():
            _check_value(f"target {t.get('name', '?')!r}", k, t[k], pat)
        params = t.get("params") or {}
        for pk, pv in params.items():
            _check_value(f"target {t['name']!r} params", pk, pv, _PARAM_VALUE_RE)
        reserved = set(_REQUIRED_TARGET_KEYS) | set(_CLUSTER_FIELDS)
        overlap = reserved & set(params)
        if overlap:
            raise SystemExit(f"Target {t['name']!r} params contains reserved keys: {sorted(overlap)}")
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
        e.setdefault("extra_volumes", "")   # optional host->ctr bind mounts
        e.setdefault("coverage_report", "") # per-target; empty = coverage on
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
