#!/usr/bin/env python3
"""Run MADEngine AI workloads against CI-built RCCL and track performance.

This script handles:
  1. Installing madengine from source into the CI Python environment
  2. Building a Docker overlay image with the CI-built RCCL
  3. Generating a manifest.json per A/B phase
  4. Running both phases back to back inside one SLURM allocation: the stock
     base image (baseline) and the same image with the CI RCCL laid over it
     (candidate)
  5. Scoring the candidate against the baseline measured on the same nodes
  6. Appending both absolute values and their ratio to a JSONL datastore

Usage from GitHub Actions (on ruby-linux-slurm-scale-runner):
  python3 rocm-systems/projects/rccl/ci/scripts/test_madengine.py \
      --artifact-dir ./build \
      --workload llama-3.1-70b-training \
      --cluster ruby \
      --nodes 2 \
      --results-dir /apps/rccl-ci/madengine/perf \
      --work-dir /apps/rccl-ci/madengine/workdir/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rccl_ci_utils import (
    find_rccl_library,
    send_email_report,
    send_teams_webhook,
    set_github_output,
    write_github_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PERF_DATASTORE = "madengine_results.jsonl"

# madengine's perf_entry_super.json uses short metric names that differ
# from our canonical workload-config keys.  Map each config key to the
# set of madengine metric names we accept as a match.
#
# Score the ``_avg`` rows (Megatron's running mean over measured iterations),
# matching parse_live_log_metrics(); the plain rows are one noisy sample.
_METRIC_ALIASES: dict[str, set[str]] = {
    "tokens_per_second_per_gpu": {"tok_per_s_per_gpu_avg"},
}
_TFLOPS_METRICS = {"TFLOPS_per_gpu_avg"}

# Applied to the candidate/baseline ratio of one same-node A/B pair.
REGRESSION_THRESHOLD_TRAINING = 0.02  # 2%
REGRESSION_THRESHOLD_INFERENCE = 0.05  # 5%

BASELINE = "baseline"
CANDIDATE = "candidate"

# Its own directory because it is also the nodes' build context: WORK_DIR has
# live sbatch logs, and tarring a file that is still growing fails the build.
OVERLAY_CTX = "overlay_ctx"
OVERLAY_DOCKERFILE = "Dockerfile.rccl-overlay"

# Must not predate ROCm/madengine#213: the A/B runs madengine inside salloc.
MADENGINE_REPO = "https://github.com/ROCm/madengine.git"
MADENGINE_REF = "00ee3e80c0a308599633caf74e5d91a72164a2fe"  # develop, has #190 and #213
MAD_REPO = "https://github.com/ROCm/MAD.git"
MAD_REF = "7a9a25e742ef60e8deffb6e5fd9b559857d2018d"  # mad-rccl, 2026-10-09
MAD_BRANCH = "mad-rccl"

WORKLOAD_CONFIGS = {
    "llama-3.1-70b-training": {
        "type": "training",
        "model_repo": "primus_pyt_megatron_lm_train_llama-3.1-70b",
        "model_repo_aliases": [
            "primus_pyt_megatron_lm_train_llama-3.1-70b_overlay",
            "primus_pyt_megatron_lm_train_llama-3.1-70b_scaleout",
        ],
        "base_image": "rocm/primus:v26.4",
        "gpu_target": "gfx950",
        "metric_key": "tokens_per_second_per_gpu",
        "multiple_results": "perf_primus-megatron-Llama-3.1-70B.csv",
        "precisions": ["BF16", "FP8"],  # what MAD's run.sh trains on gfx950
        "reference_values": {
            "2N": 1685,
            "4N": 1600,
            "16N": 1685,
            "32N": 1485,
            "44N": 1432,
        },
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        "time_limit": "04:00:00",  # one allocation, two runs, plus image staging
        "docker_mounts": {"/dev/infiniband": "/dev/infiniband"},
        "docker_run_options": "--privileged --group-add render --shm-size 64G "
            "--device=/dev/infiniband --cap-add IPC_LOCK "
            "--ulimit memlock=-1 -v /sys:/sys:ro -v /run/udev:/run/udev:ro",
    },
    "llama-4-scout-training": {
        "type": "training",
        "model_repo": "primus_pyt_megatron_lm_train_llama-4-scout-17b-16e",
        "model_repo_aliases": [
            "primus_pyt_megatron_lm_train_llama-4-scout-17b-16e_overlay",
            "primus_pyt_megatron_lm_train_llama-4-scout-17b-16e_scaleout",
        ],
        "base_image": "rocm/primus:v26.4",
        "gpu_target": "gfx950",
        "metric_key": "tokens_per_second_per_gpu",
        "multiple_results": "perf_primus-megatron-Llama-4-Scout-17B-16E.csv",
        "precisions": ["BF16"],  # what MAD's run.sh trains on gfx950
        "reference_values": {
            "2N": 2734,
            "4N": 2337,
        },
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        "time_limit": "03:00:00",  # one allocation, two runs, plus image staging
        "docker_mounts": {"/dev/infiniband": "/dev/infiniband"},
        "docker_run_options": "--privileged --group-add render --shm-size 64G "
            "--device=/dev/infiniband --cap-add IPC_LOCK "
            "--ulimit memlock=-1 -v /sys:/sys:ro -v /run/udev:/run/udev:ro",
    },
}

CLUSTER_CONFIGS = {
    "ruby": {
        "gpu_target": "gfx950",
        "slurm_partition": "meta64",
        "slurm_qos": "",
        "slurm_no_gres": True,  # Ruby SLURM has no GPU GRES configured
        "mount_host_ib_libs": True,
        "nccl_env": {
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": "bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_IB_TC": "104",
            "NCCL_IB_QPS_PER_CONNECTION": "4",
            "NCCL_SOCKET_IFNAME": "fenic0",
            "NCCL_DEBUG": "WARN",
        },
        "results_base": "/apps/rccl-ci/madengine/perf",
    },
}


def install_madengine(work_dir: Path) -> Path:
    """Clone and install madengine into the current Python environment."""
    madengine_dir = work_dir / "madengine"
    mad_dir = work_dir / "MAD"

    if not madengine_dir.exists():
        log.info("Cloning madengine at %s...", MADENGINE_REF[:12])
        subprocess.run(
            ["git", "clone", "--depth=1", MADENGINE_REPO, str(madengine_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(madengine_dir), "fetch", "--depth=1", "origin", MADENGINE_REF],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(madengine_dir), "checkout", MADENGINE_REF],
            check=True,
        )

    if not mad_dir.exists():
        log.info("Cloning MAD (%s) at %s...", MAD_BRANCH, MAD_REF[:12])
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch", MAD_BRANCH, MAD_REPO, str(mad_dir)],
            check=True,
        )
        # Verify the clone landed on the expected SHA. The --branch clone
        # gives us the branch tip; fetch+checkout is only needed if the
        # pinned ref differs from the tip (e.g. after the branch moves).
        head = subprocess.run(
            ["git", "-C", str(mad_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not head.startswith(MAD_REF[:12]):
            log.info("MAD HEAD %s != pinned %s, fetching...", head[:12], MAD_REF[:12])
            subprocess.run(
                ["git", "-C", str(mad_dir), "fetch", "--depth=1", "origin", MAD_REF],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(mad_dir), "checkout", MAD_REF],
                check=True,
            )
        else:
            log.info("MAD HEAD matches pinned ref: %s", head[:12])

    log.info("Installing madengine...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(madengine_dir)],
        check=True,
    )

    log.info("madengine installed to: %s", madengine_dir)

    scripts_src = mad_dir / "scripts" / "primus_megatron-lm"
    if not scripts_src.is_dir():
        scripts_src = mad_dir / "scripts" / "primus" / "megatron-lm"
    scripts_dst = work_dir / "scripts" / "primus_megatron-lm"
    if scripts_src.is_dir() and not scripts_dst.exists():
        scripts_dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-r", str(scripts_src), str(scripts_dst)], check=True)
        log.info("Copied MAD primus scripts to %s", scripts_dst)
    elif not scripts_src.is_dir():
        log.error("MAD primus scripts not found — expected at %s", scripts_src)

    result = subprocess.run(
        ["madengine", "--version"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        log.info("madengine version: %s", result.stdout.strip())
    else:
        log.warning("madengine --version failed, but install may still be usable")

    return madengine_dir


def get_rccl_commit(rccl_lib: Path | None = None) -> str:
    """Derive a unique identifier for the RCCL build.

    Checks, in order: RCCL_COMMIT_HASH env, GITHUB_RUN_ID env, sha256 of
    the librccl.so binary.  Does NOT fall back to ``git rev-parse HEAD``
    because in CI the checkout is TheRock (not RCCL), which would produce
    a constant tag and cause stale cache hits on persistent runners.
    """
    commit = os.environ.get("RCCL_COMMIT_HASH", "")
    if commit:
        return commit[:12]

    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if run_id:
        return f"run{run_id}"

    if rccl_lib and rccl_lib.exists():
        h = hashlib.sha256(rccl_lib.read_bytes()).hexdigest()
        return h[:12]

    return "unknown"


def _rccl_uses_kpack(rccl_lib: Path) -> bool:
    """Check if librccl.so uses kpack (GPU kernels in separate .kpack files).

    TheRock builds with kpack produce a small .so (~4MB) with a
    .rocm_kpack_ref section and an empty (NOBITS) .hip_fatbin section.
    These libraries crash on base images whose HIP runtime pre-dates
    kpack support.
    """
    try:
        result = subprocess.run(
            ["readelf", "-S", str(rccl_lib)],
            capture_output=True, text=True, timeout=10,
        )
        return ".rocm_kpack_ref" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_rccl_fingerprint(rccl_lib: Path) -> dict:
    """Extract identifying information from a librccl.so artifact.

    Returns a dict with ``md5``, ``version`` (e.g. ``2.30.4-HEAD:e711c9e``),
    and ``size`` that can be compared against runtime output.
    """
    fp: dict = {"md5": "", "version": "", "size": 0}
    if not rccl_lib or not rccl_lib.exists():
        return fp
    resolved = rccl_lib.resolve()
    fp["size"] = resolved.stat().st_size
    fp["md5"] = hashlib.md5(resolved.read_bytes()).hexdigest()
    try:
        result = subprocess.run(
            ["strings", str(resolved)],
            capture_output=True, text=True, timeout=10,
        )
        # The banner is assembled at run time: init.cc formats VERSION_STRING
        # and rcclGitHash together, while git_version.cmake emits the hash as its
        # own `<branch>:<7 hex><+ if dirty>` literal. So the two halves live in
        # the binary separately and have to be recombined here -- without the
        # hash the fingerprint cannot tell this build from the image's bundled
        # copy of the same version.
        semver = ""
        git_ref = ""
        for line in result.stdout.splitlines():
            if not semver:
                m = re.match(r"^(\d+\.\d+\.\d+)$", line)
                if m:
                    semver = m.group(1)
            if not git_ref:
                m = re.match(r"^([\w./-]+:[0-9a-fA-F]{7}\+?)$", line)
                if m:
                    git_ref = m.group(1)
            if semver and git_ref:
                break
        if semver and git_ref:
            fp["version"] = f"{semver}-{git_ref}"
        elif semver:
            fp["version"] = semver
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return fp


def _split_rccl_version(version: str) -> tuple[str, str, bool]:
    """Split into semver, commit and dirty flag.

    `2.30.7`, `2.30.7-HEAD:e711c9e`, `2.30.7-develop:1b64803+`. The trailing `+`
    is kept apart from the hash: a build with uncommitted changes is not the
    same library as a clean one at that commit, and hashes may be abbreviated
    to different lengths.
    """
    m = re.match(r"(\d+\.\d+\.\d+)(?:-\S*?:([0-9a-fA-F]{6,})(\+?))?", version)
    if not m:
        return version, "", False
    return m.group(1), (m.group(2) or "").lower(), m.group(3) == "+"


def runtime_rccl_versions(work_dir: Path) -> list[str]:
    """Distinct ``RCCL version :`` banners across one phase's node logs."""
    log_dir = work_dir / "slurm_output"
    versions: list[str] = []
    for log_file in sorted(log_dir.glob("*node_*.out")) if log_dir.is_dir() else []:
        for v in re.findall(r"RCCL version\s*:\s*(\S+)", log_file.read_text(errors="replace")):
            if v not in versions:
                versions.append(v)
    return versions


def verify_rccl_replacement(
    work_dir: Path,
    expected: dict,
    nodes: int,
    slurm_job_id: str = "",
) -> tuple[bool, str]:
    """Verify that every node used the CI-built RCCL, not its bundled copy.

    Requires a ``*_node_<i>.out`` log from each of the *nodes* nodes; a library
    that reached only some of them fails.

    Returns (ok, message).  Generic across frameworks — any RCCL-based
    application prints the version string during init.
    """
    if not expected.get("version"):
        return True, "No RCCL version to verify"

    log_dir = work_dir / "slurm_output"
    node_logs = sorted(log_dir.glob("*node_*.out")) if log_dir.is_dir() else []
    if not node_logs:
        return False, "No node logs found — cannot verify RCCL"
    logged = {
        int(m.group(1))
        for log_file in node_logs
        if (m := re.search(r"_node_(\d+)\.out$", log_file.name))
    }
    missing = sorted(set(range(nodes)) - logged)
    if missing:
        return False, (
            f"no log from node(s) {', '.join(map(str, missing))} of {nodes} — "
            f"cannot verify RCCL there"
        )

    verified_nodes: list[str] = []
    for log_file in node_logs:
        node_label = log_file.name
        log_text = log_file.read_text(errors="replace")

        rccl_versions = re.findall(r"RCCL version\s*:\s*(.+)", log_text)
        if not rccl_versions:
            return False, (
                f"{node_label}: no 'RCCL version :' found — "
                f"RCCL may not have initialized on this node"
            )

        # The container spells the build out further than the artifact does:
        # `2.30.7-develop:1b64803+` against a bare `2.30.7`. The commit is what
        # tells the CI build from the image's bundled copy, so it decides
        # whenever the artifact carries one; against a bare fingerprint the two
        # are indistinguishable and this degrades to a version check.
        runtime_version = rccl_versions[0].strip()
        exp_semver, exp_commit, exp_dirty = _split_rccl_version(expected["version"])
        run_semver, run_commit, run_dirty = _split_rccl_version(runtime_version)
        same_commit = (
            exp_commit
            and run_commit
            and (run_commit.startswith(exp_commit) or exp_commit.startswith(run_commit))
            and exp_dirty == run_dirty
        )
        if run_semver != exp_semver or (exp_commit and not same_commit):
            return False, (
                f"{node_label}: RCCL version mismatch: "
                f"expected '{expected['version']}' (from artifact), "
                f"got '{runtime_version}' (from container)"
            )
        verified_nodes.append(node_label)

    return True, (
        f"RCCL verified on {len(verified_nodes)} node(s): "
        f"version={expected['version']}, "
        f"artifact_md5={expected.get('md5', 'N/A')}"
    )


def _soname_bridge(dirs: str, sonames: str) -> str:
    """Shell that links each missing soname in *dirs* onto the copy already there."""
    if not sonames:
        return ""
    return (
        f'    for d in {dirs}; do \\\n'
        f'        for so in {sonames}; do \\\n'
        '            [ -e "$d/$so" ] && continue; \\\n'
        '            base=${so%%.so.*}.so; \\\n'
        '            have=$(for f in "$d/$base".*; do \\\n'
        '                [ -e "$f" ] && [ "$f" != "$d/$so" ] && echo "$f"; \\\n'
        '            done | sort -V | tail -1); \\\n'
        '            [ -n "$have" ] && ln -sfn "$(basename "$have")" "$d/$so"; \\\n'
        '        done; \\\n'
        '    done; \\\n'
    )


def _needed_sonames(lib: Path) -> list[str]:
    """DT_NEEDED entries of *lib* that are safe to bridge across ROCm versions.

    Only `libamd_smi`: RCCL touches a narrow part of it, and it is the one that
    actually skews. Bridging a core runtime like `libamdhip64` would trade a
    clear load failure for an unresolved symbol somewhere deeper.
    """
    try:
        result = subprocess.run(
            ["readelf", "-d", str(lib)], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("readelf unusable (%s); not reconciling dependencies", exc)
        return []
    if result.returncode != 0:
        log.warning("readelf failed on %s; not reconciling dependencies", lib)
        return []
    sonames = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", result.stdout)
    return [s for s in sonames if s.startswith("libamd_smi.")]


def stage_rccl_overlay(
    rccl_lib: Path,
    base_image: str,
    gpu_target: str,
    work_dir: Path,
) -> str:
    """Stage the candidate image (base + CI-built RCCL) under ``overlay_ctx``.

    Returns the tag. Nothing is built here; the allocation's nodes build it.
    """
    rccl_commit = get_rccl_commit(rccl_lib)
    tag = f"{base_image}-rccl-{gpu_target}-{rccl_commit}"

    ctx_dir = work_dir / OVERLAY_CTX
    ctx_dir.mkdir(exist_ok=True)
    dockerfile = ctx_dir / OVERLAY_DOCKERFILE
    rccl_lib_dir = rccl_lib.parent
    uses_kpack = _rccl_uses_kpack(rccl_lib)

    staging_dir = ctx_dir / "rccl_libs"
    staging_dir.mkdir(exist_ok=True)
    for so_file in rccl_lib_dir.glob("librccl*"):
        dest = staging_dir / so_file.name
        if not dest.exists():
            subprocess.run(["cp", "-L", str(so_file), str(dest)], check=True)

    # The CI RCCL is built against a newer ROCm than the base image, so it can
    # name a soname the image predates -- `libamd_smi.so.27` against its `.26` --
    # and then fail to load at all. Reconciled in the image by symlink, the way
    # reconcile_soname_versions() does it for the bind-mount path; copying the
    # artifact's own copy instead would drag in the rocm_sysdeps that
    # quarantine_rocm_sysdeps() exists to keep out of the loader path.
    rccl_needed = " ".join(_needed_sonames(rccl_lib))
    if rccl_needed:
        log.info("RCCL needs: %s", rccl_needed)
    bridge_sdk = _soname_bridge('"$SDK_LIB" "$SDK_DEV"', rccl_needed)
    bridge_dep = _soname_bridge('"$DEP_DIR"', rccl_needed)

    if uses_kpack:
        kpack_files = list(rccl_lib_dir.rglob("*.kpack"))
        if not kpack_files:
            kpack_files = list(rccl_lib.parent.parent.rglob("rccl*.kpack"))
        has_kpack_files = len(kpack_files) > 0
        if has_kpack_files:
            kpack_staging = staging_dir / ".kpack"
            kpack_staging.mkdir(exist_ok=True)
            for kp in kpack_files:
                dest = kpack_staging / kp.name
                if not dest.exists():
                    subprocess.run(["cp", "-L", str(kp), str(dest)], check=True)
            log.info("Found %d kpack file(s): %s",
                     len(kpack_files), [f.name for f in kpack_files])
        else:
            log.warning("RCCL .so has kpack references but no .kpack files found in artifacts")
        log.info(
            "CI-built librccl.so uses kpack (%.1f MB .so). "
            "Building overlay with SDK venv layout for %s.",
            rccl_lib.stat().st_size / 1e6,
            base_image,
        )
        dockerfile.write_text(f"""\
FROM {base_image}
COPY rccl_libs/ /tmp/rccl_ci/
RUN set -e; \\
    SDK_LIB="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"; \\
    SDK_DEV="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib"; \\
    SDK_KPACK="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/.kpack"; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_LIB/librccl.so.1"; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_LIB/librccl.so.1.0" 2>/dev/null || true; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_DEV/librccl.so.1"; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_DEV/librccl.so.1.0"; \\
    if [ -d /tmp/rccl_ci/.kpack ] && ls /tmp/rccl_ci/.kpack/*.kpack >/dev/null 2>&1; then \\
        mkdir -p "$SDK_KPACK"; \\
        cp /tmp/rccl_ci/.kpack/*.kpack "$SDK_KPACK/"; \\
    fi; \\
{bridge_sdk}    rm -rf /tmp/rccl_ci
ENV NCCL_DEBUG=WARN
""")
    else:
        log.info(
            "CI-built librccl.so has embedded GPU kernels (%.1f MB)",
            rccl_lib.stat().st_size / 1e6,
        )
        dockerfile.write_text(f"""\
FROM {base_image}
COPY rccl_libs/ /tmp/rccl_ci/
RUN set -e; \\
    RCCL_REAL=$(readlink -f /opt/rocm/lib/librccl.so 2>/dev/null || \\
                find /opt/rocm*/lib -name 'librccl.so.*.*' -not -type l 2>/dev/null | head -1); \\
    cp /tmp/rccl_ci/librccl.so "$RCCL_REAL"; \\
    DEP_DIR=$(dirname "$RCCL_REAL"); \\
{bridge_dep}    rm -rf /tmp/rccl_ci
ENV NCCL_DEBUG=WARN
""")

    log.info("Candidate (CI RCCL), built on the nodes: %s", tag)
    return tag


def slurm_time_seconds(limit: str) -> int:
    """Seconds in a SLURM ``HH:MM:SS`` time limit."""
    hours, minutes, seconds = (int(part) for part in limit.split(":"))
    return (hours * 60 + minutes) * 60 + seconds


def generate_manifest(
    workload_name: str,
    workload_config: dict,
    cluster_config: dict,
    overlay_image: str,
    nodes: int,
    work_dir: Path,
    rccl_lib: Path | None = None,
    run_dir: Path | None = None,
    pull_only: bool = False,
) -> Path:
    """Generate a madengine manifest.json for the workload.

    Structure follows the reference template from the mad-rccl branch:
    deployment config under ``deployment_config``, env vars inside both
    ``context.docker_env_vars`` and ``deployment_config.env_vars``, mounts
    in ``context.docker_mounts``.

    *run_dir*: madengine's working directory for this phase; gets the manifest.
    *pull_only*: a stock registry image, pulled per node rather than built.
    """
    gpus_per_node = workload_config["gpus_per_node"]

    nccl_env = dict(cluster_config.get("nccl_env", {}))
    if nodes == 1:
        nccl_env.pop("NCCL_NET", None)
        ifname = nccl_env.get("NCCL_SOCKET_IFNAME", "")
        if "," in ifname:
            nccl_env["NCCL_SOCKET_IFNAME"] = ifname.split(",")[0]

    socket_ifname = nccl_env.get("NCCL_SOCKET_IFNAME", "")

    # HF_TOKEN is passed via MAD_SECRETS_HFTOKEN in the process environment
    # (set in the workflow). Do NOT write it into the manifest — the manifest
    # is uploaded as a CI artifact and would leak the credential.

    model_repo = workload_config["model_repo"]
    scripts_dir = work_dir / "scripts" / "primus_megatron-lm"
    if not scripts_dir.is_dir():
        scripts_dir = work_dir / "scripts" / "primus" / "megatron-lm"

    image_key = "overlay"
    gpu_indices = ",".join(str(i) for i in range(gpus_per_node))
    render_ds = [128 + i for i in range(gpus_per_node)]

    docker_env_vars = {
        **nccl_env,
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "0",
        "NCCL_TIMEOUT": "900",
        "IBV_SHOW_WARNINGS": "1",
    }
    if socket_ifname:
        docker_env_vars["GLOO_SOCKET_IFNAME"] = socket_ifname

    docker_mounts = dict(workload_config.get("docker_mounts", {}))
    docker_run_opts = workload_config.get("docker_run_options", "")

    # Mount the host's rdma-core stack into the container.
    # The host's libibverbs has a compiled-in provider search path of
    # /usr/lib64/libibverbs/ so the providers must appear there.
    # The library itself replaces the container's copy so the linker
    # picks it up from the standard search path.
    if cluster_config.get("mount_host_ib_libs"):
        docker_run_opts += (
            " -v /usr/lib64/libibverbs.so.1"
            ":/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro"
            " -v /usr/lib64/libibverbs:/usr/lib64/libibverbs:ro"
            " -v /usr/lib64/libibumad.so.3"
            ":/usr/lib/x86_64-linux-gnu/libibumad.so.3:ro"
        )

    # When --skip-overlay-build is used the base image still has its
    # bundled RCCL.  Bind-mount the CI-built librccl.so (and kpack
    # files if present) over the container's copies so we actually
    # test the artifact, not the image default.
    if rccl_lib is not None:
        host_so = str(rccl_lib.resolve())
        sdk_lib = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
        sdk_dev = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib"
        docker_run_opts += (
            f" -v {host_so}:{sdk_lib}/librccl.so.1:ro"
            f" -v {host_so}:{sdk_lib}/librccl.so.1.0:ro"
            f" -v {host_so}:{sdk_dev}/librccl.so.1:ro"
            f" -v {host_so}:{sdk_dev}/librccl.so.1.0:ro"
        )
        kpack_dir = rccl_lib.resolve().parent.parent / ".kpack"
        if not kpack_dir.is_dir():
            kpack_dir = rccl_lib.resolve().parent / ".kpack"
        if kpack_dir.is_dir():
            sdk_kpack = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/.kpack"
            for kp in kpack_dir.glob("rccl*.kpack"):
                docker_run_opts += f" -v {kp}:{sdk_kpack}/{kp.name}:ro"
            log.info("Bind-mounting RCCL kpack from %s", kpack_dir)
        log.info("Bind-mounting CI-built RCCL: %s", host_so)

    slurm_config = {
        "partition": cluster_config.get("slurm_partition", workload_config["slurm_partition"]),
        "qos": cluster_config.get("slurm_qos", ""),
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "time": workload_config["time_limit"],
        "output_dir": "./slurm_output",
        "exclusive": True,
        "enable_node_check": False,
        "network_interface": socket_ifname,
        "skip_gpus_directive": cluster_config.get("slurm_no_gres", False),
    }

    overlay_dockerfile = work_dir / OVERLAY_CTX / OVERLAY_DOCKERFILE

    if pull_only:
        image_entry = {
            "docker_image": overlay_image,
            "local_image": False,
            "registry_image": overlay_image,
            "registry": None,
            "base_docker": overlay_image,
            "build_status": "SKIPPED",
            "build_duration": 0,
            "gpu_vendor": "AMD",
        }
    else:
        image_entry = {
            "docker_image": overlay_image,
            "local_image": True,
            "registry_image": None,
            "registry": None,
            "base_docker": workload_config["base_image"],
            "build_status": "SKIPPED",
            "build_duration": 0,
            "gpu_vendor": "AMD",
            # Lets a compute node build the image it can neither find nor
            # pull; the context is this file's own directory, overlay_ctx.
            # Advertised only when it exists: --skip-overlay-build writes no
            # Dockerfile, and a dead path would turn madengine's fallback
            # into a guaranteed failure.
            **(
                {"dockerfile": str(overlay_dockerfile)}
                if overlay_dockerfile.is_file()
                else {}
            ),
        }

    manifest = {
        "built_images": {image_key: image_entry},
        "built_models": {
            image_key: {
                "name": model_repo,
                "tags": workload_config.get("tags", ["pyt", "pretrain", "training"]),
                "dockerfile": "N/A (overlay image)",
                "scripts": f"scripts/{scripts_dir.name}/run.sh",
                "n_gpus": "-1",
                "owner": "",
                "training_precision": "",
                "multiple_results": workload_config.get("multiple_results", ""),
                "args": f"--model_repo {model_repo}",
                "additional_docker_run_options": docker_run_opts,
                "data": "",
                "cred": "",
                # None falls back to madengine's 2h default, which would stop a
                # phase early; the allocation's own limit is the real cap.
                "timeout": slurm_time_seconds(workload_config["time_limit"]),
            },
        },
        "context": {
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "docker_gpus": gpu_indices,
            "gpu_renderDs": render_ds,
            "docker_env_vars": docker_env_vars,
            "docker_mounts": docker_mounts,
            "docker_build_arg": {},
        },
        "deployment_config": {
            "target": "slurm",
            "slurm": slurm_config,
            "distributed": {
                "launcher": "primus",
                "backend": "nccl",
                "port": 29500,
                "nnodes": nodes,
                "nproc_per_node": gpus_per_node,
            },
            "env_vars": {
                **docker_env_vars,
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "TORCH_NCCL_HIGH_PRIORITY": "1",
                "OMP_NUM_THREADS": "8",
                "MIOPEN_FIND_MODE": "1",
            },
            "debug": False,
            "docker_gpus": gpu_indices,
        },
    }

    manifest_path = (run_dir or work_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to: %s", manifest_path)
    return manifest_path


def prepare_phase_dir(work_dir: Path, phase: str) -> Path:
    """Create one phase's madengine working directory, with its own scripts copy.

    madengine writes its outputs under fixed names relative to its cwd.

    Recreated from scratch: on a reused --work-dir an earlier run's exit_code,
    perf files and node logs would otherwise still be there, and an allocation
    that never ran a phase would be scored on them as if it had. Fails if the
    old directory cannot be removed rather than run on top of it.
    """
    run_dir = work_dir / "ab" / phase
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            raise RuntimeError(
                f"{run_dir} holds an earlier run's results and cannot be removed "
                f"({exc}); use a fresh --work-dir"
            ) from exc
    run_dir.mkdir(parents=True)

    scripts_src = work_dir / "scripts"
    scripts_dst = run_dir / "scripts"
    if scripts_src.is_dir() and not scripts_dst.exists():
        subprocess.run(["cp", "-r", str(scripts_src), str(scripts_dst)], check=True)

    return run_dir


# Run on every node after both phases. Container output is root-owned, so the
# workspace is removed from inside a local image (never pulled); $1 is kept.
CLEANUP_IMAGES = """\
#!/bin/bash
helper="$1"; shift
if [ -n "${SLURM_JOB_ID}" ]; then
    for ws in "${SLURM_TMPDIR:-/tmp}"/madengine_job_"${SLURM_JOB_ID}"_* \\
              /tmp/madengine_job_"${SLURM_JOB_ID}"_*; do
        [ -d "$ws" ] || continue
        rm -rf "$ws" 2>/dev/null || sudo -n rm -rf "$ws" 2>/dev/null
        for image in "$helper" "$@"; do
            [ -e "$ws" ] || break
            [ -n "$image" ] || continue
            args=(run --rm --pull never -u 0 -v "$(dirname "$ws"):/w"
                  --entrypoint rm "$image" -rf "/w/$(basename "$ws")")
            docker "${args[@]}" >/dev/null 2>&1 \\
                || sudo -n docker "${args[@]}" >/dev/null 2>&1
        done
        if [ -e "$ws" ]; then
            echo "  could not remove $(hostname -s) $ws"
        else
            echo "  removed $(hostname -s) $ws"
        fi
    done
fi
for image in "$@"; do
    if docker image rm -f "$image" >/dev/null 2>&1 \\
        || sudo -n docker image rm -f "$image" >/dev/null 2>&1; then
        echo "  removed $(hostname -s) $image"
    fi
done
exit 0
"""


def _raise_on_term(signum, frame):  # noqa: ARG001 - signal handler signature
    """Make SIGTERM unwind the stack the way SIGINT already does."""
    raise KeyboardInterrupt(f"signal {signum}")


def run_ab_in_one_allocation(
    phases: list[tuple[str, Path]],
    work_dir: Path,
    nodes: int,
    workload_name: str,
    workload_config: dict,
    cluster_config: dict,
    timeout_minutes: int,
    built_images: list[str] | None = None,
) -> dict:
    """Run every phase back to back inside one SLURM allocation, on the same nodes.

    *built_images* are removed from every node afterwards (not on a timeout).

    Returns ``{"exit_codes": {phase: int}, "job_id": str, "nodelist": str}``.
    """
    alloc_info = work_dir / "allocation.txt"
    alloc_info.unlink(missing_ok=True)

    script_lines = [
        "#!/bin/bash",
        "# Generated by test_madengine.py. No `set -e`: a failed baseline must",
        "# still let the candidate run.",
        'echo "Allocation: job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"',
        'printf "%s\\n%s\\n" "${SLURM_JOB_ID}" "${SLURM_JOB_NODELIST}" > '
        + shlex.quote(str(alloc_info)),
    ]
    for phase, run_dir in phases:
        script_lines += [
            "",
            f'echo "===== phase: {phase} ====="',
            f"cd {shlex.quote(str(run_dir))} || exit 1",
            "madengine run -m manifest.json -o perf.csv --live-output --verbose",
            "echo $? > " + shlex.quote(str(run_dir / "exit_code")),
        ]
    cleanup = work_dir / "ab" / "cleanup_images.sh"
    cleanup.parent.mkdir(parents=True, exist_ok=True)
    cleanup.write_text(CLEANUP_IMAGES)
    script_lines += [
        "",
        'echo "===== cleanup: this run\'s workspaces and images on every node ====="',
        'srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 '
        f"--overlap bash {shlex.quote(str(cleanup))} "
        + " ".join(
            shlex.quote(i) for i in [workload_config["base_image"], *(built_images or [])]
        )
        + " || true",
    ]

    script_path = work_dir / "ab" / "run_phases.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines) + "\n")

    # Unique per run, because cancelling by name is the only way to reach a
    # request that is still queued: it has no job id in allocation.txt yet,
    # and SIGKILL, which salloc cannot handle, leaves it queued to start later
    # with nothing attached (InactiveLimit is 0 on ruby). Recorded before
    # salloc starts so the workflow's release step can read it too.
    run_tag = (
        f"{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
        if os.environ.get("GITHUB_RUN_ID") else f"local-{os.getpid()}"
    )
    job_name = f"rccl-ab-{workload_name}-{run_tag}"
    (work_dir / "allocation.name").write_text(job_name + "\n")

    cmd = [
        "salloc",
        "--nodes", str(nodes),
        "--ntasks-per-node", "1",
        "--exclusive",
        "--partition",
        cluster_config.get("slurm_partition", workload_config["slurm_partition"]),
        "--time", workload_config["time_limit"],
        "--job-name", job_name,
    ]
    qos = cluster_config.get("slurm_qos", "")
    if qos:
        cmd += ["--qos", qos]
    cmd += ["bash", str(script_path)]

    log.info("Running: %s", " ".join(cmd))
    log.info("Phases: %s", ", ".join(p for p, _ in phases))
    log.info("Timeout: %d minutes (includes time spent queueing)", timeout_minutes)

    # Pre-warm the bytecode cache: madengine runs `madengine --version` itself,
    # and a cold import of its dependencies off NFS is slow.
    try:
        subprocess.run(["madengine", "--version"], capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        log.warning("madengine --version pre-warm timed out (non-fatal)")
    except Exception:
        pass

    env = os.environ.copy()
    docker_builds_dir = work_dir / "docker_builds"
    docker_builds_dir.mkdir(exist_ok=True)
    env["MAD_DOCKER_BUILDS"] = str(docker_builds_dir)

    # Own process group: killing salloc alone leaves run_phases.sh and its
    # madengine/srun children running on the allocation.
    proc = subprocess.Popen(cmd, cwd=work_dir, env=env, start_new_session=True)

    def release_allocation(reason: str) -> None:
        """Give the nodes back: cancel the job, then kill the whole group.

        The group is in a session of its own, so nothing the caller is killed
        by reaches it. Whatever ends this function has to do it explicitly or
        the allocation keeps the nodes until its own --time expires -- four
        exclusive ones for five hours, in the 405B case.
        """
        log.error("Releasing the allocation: %s", reason)
        try:
            # By name, not by the id in allocation.txt: the name also reaches
            # a request that is still queued, which has no id there yet.
            try:
                subprocess.run(["scancel", "--me", "--name", job_name], timeout=60)
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("scancel --name %s failed: %s", job_name, exc)
        finally:
            # A second signal during scancel raises out of this function; the
            # group must still be killed and reaped.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    # Actions cancels a run with SIGINT and escalates to SIGTERM. SIGINT
    # already arrives as KeyboardInterrupt, but SIGTERM ends the interpreter
    # without unwinding, so neither `except` nor `finally` would run: turn it
    # into the same exception. SIGKILL, which follows, cannot be caught at
    # all -- the allocation's own --time is the only backstop left for it.
    previous_term = signal.signal(signal.SIGTERM, _raise_on_term)
    try:
        proc.wait(timeout=timeout_minutes * 60)
    except subprocess.TimeoutExpired:
        release_allocation(f"timed out after {timeout_minutes} minutes")
    except BaseException:
        release_allocation("the driver was interrupted")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)

    exit_codes = {}
    for phase, run_dir in phases:
        marker = run_dir / "exit_code"
        try:
            exit_codes[phase] = int(marker.read_text().strip())
        except (OSError, ValueError):
            # No marker means the phase never ran: salloc never got the nodes,
            # or the timeout above cut it short.
            log.error("Phase %s produced no exit code — it did not run", phase)
            exit_codes[phase] = 124
        else:
            log.info("Phase %s exit code: %d", phase, exit_codes[phase])

    job_id, granted_nodes = "", ""
    try:
        job_id, granted_nodes = alloc_info.read_text().splitlines()[:2]
    except (OSError, ValueError):
        log.warning("Could not read allocation details from %s", alloc_info)
    else:
        log.info("Allocation: job=%s nodes=%s", job_id, granted_nodes)

    return {"exit_codes": exit_codes, "job_id": job_id, "nodelist": granted_nodes}


def parse_perf_results(work_dir: Path) -> list[dict]:
    """Parse madengine performance results.

    Prefers ``perf_entry_super.json`` (31 fixed columns, per-precision rows
    with ``multi_results``).  Falls back to ``perf.csv`` (variable-width,
    long-format).  Returns a list of result dicts — one per row.
    """
    super_json = work_dir / "perf_entry_super.json"
    if super_json.exists():
        try:
            entries = json.loads(super_json.read_text())
            if entries:
                log.info("Parsed %d result(s) from perf_entry_super.json", len(entries))
                for e in entries:
                    log.info("  model=%s perf=%s metric=%s status=%s precision=%s",
                             e.get("model"), e.get("performance"),
                             e.get("metric"), e.get("status"),
                             e.get("training_precision"))
                return entries
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("Could not parse %s: %s", super_json, exc)

    csv_path = work_dir / "perf.csv"
    if csv_path.exists():
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                log.info("perf.csv row: %s", dict(row))
                rows.append(dict(row))
        if rows:
            log.info("Parsed %d row(s) from perf.csv", len(rows))
            return rows

    log.warning("No perf results found in %s", work_dir)
    return []


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+)"
    r".*throughput per GPU \(TFLOP/s/GPU\):\s*[\d.]+/(?P<tflops_avg>[\d.]+)"
    r".*tokens per GPU \(tokens/s/GPU\):\s*[\d.]+/(?P<tps_avg>[\d.]+)"
)
_RUN_HEADER_RE = re.compile(r"Running:\s+(.+)\s+-\s+(\w+)\s+-\s+(\w+)\s*$")


def parse_live_log_metrics(work_dir: Path) -> list[dict]:
    """Parse madengine live logs for training metrics.

    Detects multiple runs within a single log (e.g. BF16 then FP8) by
    watching for "Running: Model - Precision - Mode" header lines.

    Returns a list of run dicts with keys:
      model, precision, mode, iter, total, tflops_avg,
      tokens_per_second_per_gpu, completed, log_file
    """
    logs = sorted(work_dir.glob("*.run.live.log"))
    if not logs:
        return []

    runs = []
    for log_path in logs:
        current = None
        with open(log_path) as f:
            for raw_line in f:
                line = _ANSI_RE.sub("", raw_line)

                hdr = _RUN_HEADER_RE.search(line)
                if hdr:
                    if current:
                        current.setdefault("iter", 0)
                        current.setdefault("total", 0)
                        current["completed"] = (
                            current["iter"] > 0
                            and current["iter"] == current["total"]
                        )
                        runs.append(current)
                    current = {
                        "model": hdr.group(1).strip(),
                        "precision": hdr.group(2),
                        "mode": hdr.group(3),
                        "log_file": str(log_path),
                    }
                    continue

                m = _ITER_RE.search(line)
                if m:
                    if current is None:
                        current = {"log_file": str(log_path)}
                    current.update({
                        "iter": int(m.group("iter")),
                        "total": int(m.group("total")),
                        "tflops_avg": float(m.group("tflops_avg")),
                        "tokens_per_second_per_gpu": float(m.group("tps_avg")),
                    })

        if current:
            current.setdefault("iter", 0)
            current.setdefault("total", 0)
            current["completed"] = (
                current["iter"] > 0
                and current["iter"] == current["total"]
            )
            runs.append(current)

    return runs


def _usable_metric(val: float) -> bool:
    """Whether a throughput can be scored; NaN would pass any threshold."""
    return math.isfinite(val) and val > 0


def collect_phase_results(run_dir: Path, metric_key: str) -> list[dict]:
    """Per-precision metrics for one phase: structured output, else the live log."""
    perf_results = parse_perf_results(run_dir)
    results: list[dict] = []

    if perf_results:
        # perf_entry_super.json rows are long-format: metric name is a
        # value in the ``metric`` column, performance in ``performance``.
        # Filter to the configured metric_key (or its madengine alias)
        # and key by precision.
        accepted_metrics = _METRIC_ALIASES.get(metric_key, {metric_key})
        for row in perf_results:
            if row.get("metric") not in accepted_metrics:
                continue
            perf_val = row.get("performance", "")
            precision = (row.get("training_precision")
                         or row.get("multi_results", {}).get("precision", ""))
            row_status = row.get("status", "")
            if not perf_val:
                continue
            try:
                val = float(perf_val)
            except (ValueError, TypeError):
                continue
            if not _usable_metric(val):
                log.warning("Ignoring structured result: %s %s = %r",
                            precision, metric_key, val)
                continue
            results.append({
                "precision": precision,
                "metric_value": val,
                "status": ("pass" if row_status.upper() in ("", "PASS", "SUCCESS")
                           else "fail"),
                "source": "structured",
            })
            log.info("Structured result: %s %s = %.1f (status=%s)",
                     precision, metric_key, val, row_status)

        # Attach TFLOPS from companion rows, keyed by precision.
        tflops_by_precision: dict[str, float] = {}
        for row in perf_results:
            if row.get("metric") not in _TFLOPS_METRICS:
                continue
            prec = (row.get("training_precision")
                    or row.get("multi_results", {}).get("precision", ""))
            try:
                tflops_by_precision[prec] = float(row["performance"])
            except (KeyError, ValueError, TypeError):
                pass
        for pr in results:
            if "tflops_avg" not in pr:
                pr["tflops_avg"] = tflops_by_precision.get(pr["precision"])

    if not results:
        for run in parse_live_log_metrics(run_dir):
            val = run.get("tokens_per_second_per_gpu")
            if val is None:
                continue
            if not _usable_metric(val):
                log.warning("Ignoring live-log result: %s = %r",
                            run.get("precision"), val)
                continue
            results.append({
                "precision": run.get("precision"),
                "metric_value": val,
                "tflops_avg": run.get("tflops_avg"),
                "status": "pass" if run.get("completed", False) else "fail",
                "source": "live_log",
                "iter": run.get("iter", 0),
                "total": run.get("total", 0),
                "log_file": run.get("log_file"),
            })
            log.info("Live-log result: %s = %.1f (completed=%s)",
                     run.get("precision"), val, run.get("completed"))

    return results


def check_ab_regression(
    baseline: float,
    candidate: float,
    workload_type: str,
) -> tuple[bool, str]:
    """Score the candidate against the baseline measured on the same nodes.

    Returns (is_regression, message).
    """
    if baseline <= 0:
        return False, "Baseline is not a positive number — cannot compare"

    threshold = (
        REGRESSION_THRESHOLD_TRAINING
        if workload_type == "training"
        else REGRESSION_THRESHOLD_INFERENCE
    )
    ratio = candidate / baseline
    msg = (
        f"candidate {candidate:.1f} vs baseline {baseline:.1f} = "
        f"{ratio:.4f} ({ratio - 1:+.1%}), threshold: -{threshold:.0%}"
    )

    # Not `ratio - 1 < -threshold`: the subtraction rounds, so a loss of
    # exactly the threshold failed for some inputs and passed for others.
    if ratio < 1 - threshold:
        return True, f"REGRESSION DETECTED — {msg}"

    return False, f"No regression — {msg}"


def check_ratio_regression(
    ratio: float,
    previous: float,
    workload_type: str,
) -> tuple[bool, str]:
    """Score this run's candidate/stock ratio against the previous run's.

    The stock baseline is the same image every night, so it divides out the
    node pair the scheduler handed each run: the two ratios are comparable
    where the absolute numbers are not. The vs-stock check above cannot do
    this job on its own -- a workload whose image ships a much slower RCCL
    has a ratio far above 1 and passes it whatever the candidate does.

    Returns (is_regression, message).
    """
    threshold = (
        REGRESSION_THRESHOLD_TRAINING
        if workload_type == "training"
        else REGRESSION_THRESHOLD_INFERENCE
    )
    change = ratio / previous
    msg = (
        f"ratio {ratio:.4f} vs {previous:.4f} previously = "
        f"{change:.4f} ({change - 1:+.1%}), threshold: -{threshold:.0%}"
    )
    if change < 1 - threshold:
        return True, f"REGRESSION DETECTED — {msg}"
    return False, f"No regression — {msg}"


def previous_comparison(
    results_dir: Path,
    workload_name: str,
    scale: str,
    precision: str,
) -> dict | None:
    """The latest earlier passing run's candidate row for this workload.

    Called before this run appends its own rows, so the latest match is the
    previous run's. Same workload, scale and precision only: a ratio depends
    on all three.

    Only runs whose own verdict was a pass count. A failed run's ratio would
    otherwise become the reference: a regression that failed one night would
    pass the next, and a run whose candidate never loaded the CI RCCL would
    set it to about 1.0. Rows written before the verdict was recorded have
    none and are skipped for the same reason.
    """
    try:
        with open(results_dir / PERF_DATASTORE) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (
            row.get("phase") == CANDIDATE
            and row.get("workload") == workload_name
            and row.get("scale") == scale
            and row.get("precision") == precision
            and row.get("verdict") == "pass"
            and isinstance(row.get("ratio"), (int, float))
            and _usable_metric(float(row["ratio"]))
        ):
            return row
    return None


def append_result(
    results_dir: Path,
    workload_name: str,
    scale: str,
    metric_value: float | None,
    status: str,
    rccl_commit: str,
    extra: dict | None = None,
    precision: str | None = None,
    tflops: float | None = None,
    tokens_per_sec: float | None = None,
) -> None:
    """Append a result entry to the JSONL datastore."""
    results_dir.mkdir(parents=True, exist_ok=True)
    datastore = results_dir / PERF_DATASTORE

    entry = {
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": rccl_commit,
        "workload": workload_name,
        "scale": scale,
        "precision": precision,
        "tflops_per_gpu": tflops,
        "tokens_per_sec_per_gpu": tokens_per_sec,
        "metric_value": metric_value,
        "status": status,
    }
    if extra:
        entry.update(extra)

    # Matrix legs append in parallel on a shared filesystem.
    with open(datastore, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry) + "\n")
    log.info("Result appended to %s", datastore)

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)


def pair_by_precision(
    baseline_results: list[dict],
    candidate_results: list[dict],
    expected: list[str] | None = None,
) -> list[dict]:
    """Join the two phases into one comparison row per precision.

    A precision missing from a phase, or *expected* but absent from both, keeps
    a row with None so the gate fails instead of skipping it.
    """
    comparisons: dict[str, dict] = {
        precision: {"precision": precision, BASELINE: None, CANDIDATE: None}
        for precision in expected or []
    }
    for phase, rows in ((BASELINE, baseline_results), (CANDIDATE, candidate_results)):
        for row in rows:
            precision = row.get("precision") or ""
            entry = comparisons.setdefault(
                precision,
                {"precision": precision, BASELINE: None, CANDIDATE: None},
            )
            entry[phase] = row

    for entry in comparisons.values():
        base, cand = entry[BASELINE], entry[CANDIDATE]
        entry["ratio"] = None
        if base and cand:
            base_val, cand_val = base.get("metric_value"), cand.get("metric_value")
            if base_val and cand_val:
                entry["ratio"] = cand_val / base_val

    return [comparisons[p] for p in sorted(comparisons)]


def _format_phase(row: dict | None) -> str:
    """One side of a comparison line, or why it is absent."""
    if not row:
        return "missing"
    val = row.get("metric_value")
    if val is None:
        return "no metric"
    tflops = row.get("tflops_avg")
    suffix = f" ({tflops:.1f} TFLOP/s/GPU)" if tflops else ""
    return f"{val:.1f}{suffix} [{row['status']}]"


def generate_summary_report(
    workload_name: str,
    scale: str,
    exit_code: int,
    regression_msg: str,
    rccl_commit: str,
    cluster: str,
    comparisons: list[dict] | None = None,
    nodelist: str = "",
    job_id: str = "",
    phase_rccl: dict[str, str] | None = None,
) -> str:
    """Generate a plain-text summary report."""
    status = "PASSED" if exit_code == 0 else "FAILED"
    phase_rccl = phase_rccl or {}
    lines = [
        "RCCL MADEngine Workload Test Report",
        "=" * 40,
        f"Status:     {status}",
        f"Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"Workload:   {workload_name}",
        f"Scale:      {scale}",
        f"Cluster:    {cluster}",
        f"Baseline:   RCCL {phase_rccl.get(BASELINE) or 'unknown'}",
        f"Candidate:  RCCL {phase_rccl.get(CANDIDATE) or 'unknown'} (CI build {rccl_commit})",
        f"Nodes:      {nodelist or 'unknown'} (job {job_id or 'unknown'})",
        "",
        "Both numbers below were measured on those same nodes, back to back,",
        "inside one allocation. Baseline is the stock image; candidate is the",
        "same image with the CI-built RCCL laid over it. Throughput in",
        "tok/s/GPU.",
        "",
    ]

    if comparisons:
        for c in comparisons:
            ratio = c.get("ratio")
            ratio_s = f"{ratio:.4f} ({ratio - 1:+.1%})" if ratio else "N/A"
            lines.append(f"{c['precision'] or '?':>5}:  ratio {ratio_s}")
            lines.append(f"        baseline  {_format_phase(c[BASELINE])}")
            lines.append(f"        candidate {_format_phase(c[CANDIDATE])}")
            if c.get("previous_ratio") is not None:
                lines.append(
                    f"        previous  ratio {c['previous_ratio']:.4f} "
                    f"(run {c.get('previous_run_id') or '?'})"
                )
    else:
        lines.append("Throughput: N/A (workload did not produce metrics)")

    lines.append("")
    lines.append("Regression:")
    lines.extend(f"  {line}" for line in regression_msg.splitlines())
    lines.append("")

    run_url = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if run_url and repo and run_id:
        lines.append(f"CI run: {run_url}/{repo}/actions/runs/{run_id}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing CI-built RCCL artifacts",
    )
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
        choices=list(WORKLOAD_CONFIGS.keys()),
        help="Workload to run",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        choices=list(CLUSTER_CONFIGS.keys()),
        help="Target cluster",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=2,
        help="Number of nodes to allocate (default: 2)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for JSONL datastore and run artifacts (default: cluster-specific path)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for madengine install, overlay build, etc.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=330,
        help="Timeout for the whole allocation, queueing included, in minutes "
             "(default: 330 — a 4h allocation plus 90min of queueing)",
    )
    parser.add_argument(
        "--notify-email",
        type=str,
        default="",
        help="Send summary report to this email address",
    )
    parser.add_argument(
        "--teams-webhook",
        type=str,
        default="",
        help="Send summary report to this Teams webhook URL",
    )
    parser.add_argument(
        "--skip-overlay-build",
        action="store_true",
        help="Skip Docker overlay build (use pre-built image specified via --overlay-image)",
    )
    parser.add_argument(
        "--overlay-image",
        type=str,
        default="",
        help="Pre-built overlay image to use (requires --skip-overlay-build)",
    )

    args = parser.parse_args()
    # madengine runs inside our allocation only for multi-node workloads; a
    # single-node phase gets its own sbatch, possibly on other nodes.
    if args.nodes < 2:
        parser.error("--nodes must be at least 2 for a same-node A/B")

    workload_config = WORKLOAD_CONFIGS[args.workload]
    cluster_config = CLUSTER_CONFIGS[args.cluster]
    scale = f"{args.nodes}N/{args.nodes * workload_config['gpus_per_node']}GPU"

    results_dir = args.results_dir or Path(cluster_config["results_base"])
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="madengine_ci_"))
    log.info("Work directory: %s", work_dir)
    log.info("Results directory: %s", results_dir)

    # Step 1: Find RCCL library and fingerprint it
    rccl_lib = find_rccl_library(args.artifact_dir)
    log.info("RCCL library: %s", rccl_lib)
    rccl_commit = get_rccl_commit(rccl_lib)
    log.info("RCCL commit/tag: %s", rccl_commit)
    rccl_fingerprint = get_rccl_fingerprint(rccl_lib)
    if rccl_fingerprint["md5"]:
        log.info("RCCL fingerprint: md5=%s version=%s size=%d",
                 rccl_fingerprint["md5"], rccl_fingerprint["version"],
                 rccl_fingerprint["size"])

    # Step 2: Install madengine
    install_madengine(work_dir)

    # Step 3: Stage the candidate image (or use pre-built). Images are built on
    # the allocation's nodes, never here: this host's disk is shared by runners.
    if args.skip_overlay_build:
        if not args.overlay_image:
            log.error("--skip-overlay-build requires --overlay-image")
            sys.exit(1)
        overlay_image = args.overlay_image
    else:
        overlay_image = stage_rccl_overlay(
            rccl_lib,
            workload_config["base_image"],
            cluster_config["gpu_target"],
            work_dir,
        )

    # Step 4: Generate one manifest per A/B phase
    phases: list[tuple[str, Path]] = []
    for phase, image, pull_only in (
        (BASELINE, workload_config["base_image"], True),
        (CANDIDATE, overlay_image, False),
    ):
        run_dir = prepare_phase_dir(work_dir, phase)
        generate_manifest(
            args.workload,
            workload_config,
            cluster_config,
            image,
            args.nodes,
            work_dir,
            # The bind-mount of the CI library is what makes a --skip-overlay-build
            # run the candidate; the baseline must keep the image's own copy.
            rccl_lib=(
                rccl_lib if args.skip_overlay_build and phase == CANDIDATE else None
            ),
            run_dir=run_dir,
            pull_only=pull_only,
        )
        phases.append((phase, run_dir))

    phase_dirs = dict(phases)

    # Step 5: Run both phases on the same nodes, in one allocation
    allocation = run_ab_in_one_allocation(
        phases,
        work_dir,
        args.nodes,
        args.workload,
        workload_config,
        cluster_config,
        args.timeout_minutes,
        # A --skip-overlay-build image is not ours to remove, and the baseline
        # is the stock base.
        built_images=[] if args.skip_overlay_build else [overlay_image],
    )
    exit_code = max(allocation["exit_codes"].values(), default=1)

    # Step 5b: Verify RCCL replacement. Only the candidate is meant to carry the
    # CI library — the baseline runs the image's bundled copy by design.
    rccl_verification_failed = False
    if rccl_fingerprint.get("version"):
        rccl_ok, rccl_msg = verify_rccl_replacement(
            phase_dirs[CANDIDATE], rccl_fingerprint, args.nodes,
        )
        if rccl_ok:
            log.info("RCCL verification: %s", rccl_msg)
        else:
            log.error("RCCL verification FAILED: %s", rccl_msg)
            rccl_verification_failed = True
            exit_code = max(exit_code, 1)

    # Step 6: Parse each phase's results and save its artifacts
    metric_key = workload_config["metric_key"]
    phase_results = {}
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    for phase, run_dir in phases:
        log.info("Collecting %s results from %s", phase, run_dir)
        phase_results[phase] = collect_phase_results(run_dir, metric_key)

        # Matrix legs share GITHUB_RUN_ID, so the workload keeps them apart.
        run_artifacts = results_dir / "runs" / run_id / args.workload / phase
        try:
            run_artifacts.mkdir(parents=True, exist_ok=True)
            for f in ["perf.csv", "perf_entry_super.csv", "perf_entry_super.json"]:
                src = run_dir / f
                if src.exists():
                    shutil.copy2(str(src), str(run_artifacts / f))
        except OSError as exc:
            log.warning("Could not save %s artifacts to %s: %s",
                        phase, run_artifacts, exc)

    all_rows = [r for rows in phase_results.values() for r in rows]

    # Override exit_code if training actually completed successfully.
    # madengine can report failure (exit code 3) when its perf collector
    # can't parse the output format, even though training ran to completion.
    if exit_code != 0 and not rccl_verification_failed and all_rows:
        every_phase_scored = all(phase_results.get(p) for p, _ in phases)
        all_pass = all(r["status"] == "pass" for r in all_rows)
        has_metric = all(r.get("metric_value") is not None for r in all_rows)
        if every_phase_scored and all_pass and has_metric:
            log.info(
                "Overriding madengine exit code %d → 0: all %d run(s) across "
                "both phases passed with metrics",
                exit_code, len(all_rows),
            )
            exit_code = 0

    # Step 7: Score the candidate against the baseline, per precision
    comparisons = pair_by_precision(
        phase_results.get(BASELINE, []), phase_results.get(CANDIDATE, []),
        expected=workload_config.get("precisions", []),
    )
    regression_msgs = []
    for c in comparisons:
        prec = c["precision"] or "?"
        base, cand = c[BASELINE], c[CANDIDATE]
        if c["ratio"] is None:
            # A missing side is a job failure, not evidence about RCCL.
            regression_msgs.append(
                f"[{prec}] no comparison — baseline={_format_phase(base)}, "
                f"candidate={_format_phase(cand)}"
            )
            log.error("No comparison possible for %s", prec)
            exit_code = max(exit_code, 1)
            continue
        reg, msg = check_ab_regression(
            base["metric_value"], cand["metric_value"], workload_config["type"],
        )
        regression_msgs.append(f"[{prec}] vs stock:    {msg}")
        if reg:
            log.warning(msg)
            exit_code = max(exit_code, 1)
        else:
            log.info(msg)

        # Read before Step 8 appends this run's rows, so it is the previous run.
        prev = previous_comparison(results_dir, args.workload, scale, c["precision"])
        if prev is None:
            regression_msgs.append(
                f"[{prec}] vs previous: no earlier comparison recorded — not scored"
            )
            continue
        c["previous_ratio"] = float(prev["ratio"])
        c["previous_run_id"] = prev.get("run_id", "")
        reg, msg = check_ratio_regression(
            c["ratio"], c["previous_ratio"], workload_config["type"],
        )
        regression_msgs.append(
            f"[{prec}] vs previous: {msg} (run {c['previous_run_id'] or '?'})"
        )
        if reg:
            log.warning(msg)
            exit_code = max(exit_code, 1)
        else:
            log.info(msg)

    if not comparisons:
        regression_msgs.append("no metrics from either phase — nothing to compare")
        exit_code = max(exit_code, 1)
    regression_msg = "\n".join(regression_msgs)

    # Step 8: Append to datastore — one record per phase per precision, with
    # absolute values kept for trend analysis and the ratio on the candidate.
    extra = {
        "cluster": args.cluster,
        "overlay_image": overlay_image,
        "nodelist": allocation["nodelist"],
        "slurm_job_id": allocation["job_id"],
        # run_id names this test run; these name the RCCL binary it tested.
        "artifact_run_id": os.environ.get("ARTIFACT_RUN_ID", ""),
        "rccl_md5": rccl_fingerprint["md5"],
        # The run's own gate verdict, final by now: it covers the phase exit
        # codes, RCCL verification and both regression checks. A later run
        # takes only a passing one as its reference.
        "verdict": "pass" if exit_code == 0 else "fail",
    }
    phase_images = {
        BASELINE: workload_config["base_image"],
        CANDIDATE: overlay_image,
    }
    if all_rows:
        for c in comparisons:
            for phase in (BASELINE, CANDIDATE):
                row = c[phase]
                if row is None:
                    continue
                phase_extra = {**extra, "phase": phase, "image": phase_images[phase]}
                if phase == CANDIDATE and c["ratio"] is not None:
                    phase_extra["ratio"] = c["ratio"]
                    if c.get("previous_ratio") is not None:
                        phase_extra["previous_ratio"] = c["previous_ratio"]
                        phase_extra["previous_run_id"] = c["previous_run_id"]
                append_result(
                    results_dir,
                    args.workload,
                    scale,
                    row.get("metric_value"),
                    row["status"],
                    rccl_commit,
                    extra=phase_extra,
                    precision=c["precision"],
                    tflops=row.get("tflops_avg"),
                    tokens_per_sec=row.get("metric_value"),
                )
    else:
        append_result(
            results_dir,
            args.workload,
            scale,
            None,
            "fail",
            rccl_commit,
            extra=extra,
        )

    # Step 9: Generate and distribute report
    status = "pass" if exit_code == 0 else "fail"
    report = generate_summary_report(
        args.workload, scale, exit_code,
        regression_msg, rccl_commit, args.cluster,
        comparisons=comparisons,
        nodelist=allocation["nodelist"],
        job_id=allocation["job_id"],
        phase_rccl={
            phase: ", ".join(runtime_rccl_versions(run_dir))
            for phase, run_dir in phases
        },
    )
    log.info("\n%s", report)
    write_github_summary(report)
    set_github_output("madengine_status", status)
    candidate_values = [
        c[CANDIDATE]["metric_value"]
        for c in comparisons
        if c[CANDIDATE] and c[CANDIDATE].get("metric_value") is not None
    ]
    if candidate_values:
        set_github_output("madengine_metric", f"{candidate_values[-1]:.1f}")
    ratios = [c["ratio"] for c in comparisons if c["ratio"] is not None]
    if ratios:
        set_github_output("madengine_ratio", f"{min(ratios):.4f}")

    summary_path = work_dir / "madengine_summary.txt"
    summary_path.write_text(report)

    report_status = "PASSED" if exit_code == 0 else "FAILED"
    if args.notify_email:
        send_email_report(report, args.notify_email, report_status,
                          subject_prefix=f"RCCL MADEngine {args.workload}")
    if args.teams_webhook:
        send_teams_webhook(report, args.teams_webhook, report_status,
                           subject_prefix=f"RCCL MADEngine {args.workload}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
