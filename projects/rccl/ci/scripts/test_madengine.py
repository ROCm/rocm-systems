#!/usr/bin/env python3
"""Run MADEngine AI workloads against CI-built RCCL and track performance.

This script handles:
  1. Installing madengine from source into the CI venv
  2. Building a Docker overlay image with the CI-built RCCL
  3. Generating a manifest.json for the requested workload
  4. Running the workload via `madengine run`
  5. Parsing perf.csv results and checking for regressions
  6. Appending results to a JSONL datastore for trend analysis

Usage from GitHub Actions (on ruby-linux-slurm-scale-runner):
  python3 rocm-systems/projects/rccl/ci/scripts/test_madengine.py \
      --artifact-dir ./build \
      --workload llama-3.1-70b-training \
      --cluster ruby \
      --nodes 2 \
      --results-dir /apps/rccl-ci/madengine/perf \
      --work-dir /apps/rccl-ci/madengine/workdir/${GITHUB_RUN_ID}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
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
_METRIC_ALIASES: dict[str, set[str]] = {
    "tokens_per_second_per_gpu": {"tok_per_s_per_gpu"},
}
_TFLOPS_METRICS = {"TFLOPS_per_gpu"}

REGRESSION_WINDOW = 5
REGRESSION_THRESHOLD_TRAINING = 0.02  # 2%
REGRESSION_THRESHOLD_INFERENCE = 0.05  # 5%

MADENGINE_REPO = "https://github.com/ROCm/madengine.git"
MADENGINE_REF = "ec4de0b58c49f05d89dd33e38cc3e81e0fb3d992"
MAD_REPO = "https://github.com/ROCm/MAD.git"
MAD_REF = "63867e6a6e42355fd7b040fbcbd5bf043c9982fc"
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
        "reference_values": {
            "2N": 1685,
            "4N": 1600,
            "16N": 1685,
            "32N": 1485,
            "44N": 1432,
        },
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        "time_limit": "03:00:00",
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
        "multiple_results": "perf_primus-megatron-Llama-3.1-70B.csv",
        "reference_values": {
            "2N": 2734,
            "4N": 2337,
        },
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        "time_limit": "02:00:00",
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
        "results_base": "/apps/rccl-ci/perf",
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


def patch_madengine_for_cluster(
    madengine_dir: Path,
    no_gres: bool = False,
) -> None:
    """Patch madengine source for cluster-specific compatibility."""
    src = madengine_dir / "src" / "madengine"

    if no_gres:
        template = src / "deployment" / "templates" / "slurm" / "job.sh.j2"
        if not template.exists():
            log.warning("SLURM template not found at %s", template)
        else:
            content = template.read_text()
            patched = content.replace(
                "#SBATCH --gpus-per-node={{ gpus_per_node }}\n", ""
            )
            if patched != content:
                template.write_text(patched)
                log.info("Patched SLURM template: removed --gpus-per-node directive")
            else:
                log.info("SLURM template already patched (no --gpus-per-node)")

    template = src / "deployment" / "templates" / "slurm" / "job.sh.j2"
    if template.exists():
        content = template.read_text()
        marker = "# Load required modules"
        if marker in content and "$HOME/.local/bin" not in content:
            patched = content.replace(
                marker,
                'export PATH="$HOME/.local/bin:$PATH"\n\n' + marker,
            )
            template.write_text(patched)
            log.info(
                "Patched SLURM template: added $HOME/.local/bin to PATH "
                "(SLURM jobs do not inherit user shell PATH)"
            )
        else:
            log.info("SLURM template PATH patch already present or marker not found")

    slurm_py = src / "deployment" / "slurm.py"
    if slurm_py.exists():
        content = slurm_py.read_text()
        patched = content.replace(
            '["madengine", "--version"],\n'
            "                capture_output=True,\n"
            "                text=True,\n"
            "                timeout=5,",
            '["madengine", "--version"],\n'
            "                capture_output=True,\n"
            "                text=True,\n"
            "                timeout=120,",
        )
        if patched != content:
            slurm_py.write_text(patched)
            log.info("Patched slurm.py: increased CLI validation timeout to 120s")
        else:
            log.info("slurm.py already patched or timeout string not found")

    template = src / "deployment" / "templates" / "slurm" / "job.sh.j2"
    if template and template.exists():
        content = template.read_text()
        # Patch the MULTI-NODE verification block (inside TASK_SCRIPT_EOF
        # heredoc) to install madengine per-node when the head node's venv
        # is incompatible (Python 3.10 vs 3.9). The single-node block
        # runs on the head node where the venv works — leave it alone.
        #
        # Find the multi-node block by searching for the verification
        # string AFTER the TASK_SCRIPT_EOF heredoc marker.
        heredoc_marker = "TASK_SCRIPT_EOF"
        heredoc_idx = content.find(heredoc_marker)
        if heredoc_idx != -1:
            verify_str = 'echo "Verifying madengine availability..."'
            mn_verify_idx = content.find(verify_str, heredoc_idx)
            if mn_verify_idx == -1:
                mn_verify_idx = content.find(verify_str)
            if mn_verify_idx != -1:
                mn_end_str = "# Create local execution manifest"
                mn_end_idx = content.find(mn_end_str, mn_verify_idx)
                if mn_end_idx != -1:
                    replacement = (
                        'echo "Verifying madengine availability..."\n'
                        'MAD_CLI_COMMAND=""\n'
                        'if command -v madengine >/dev/null 2>&1 && '
                        'madengine --help >/dev/null 2>&1; then\n'
                        '    MAD_CLI_COMMAND="madengine"\n'
                        '    echo "  ✓ madengine available: '
                        '$(madengine --version 2>&1 | head -1)"\n'
                        'fi\n'
                        'if [ -z "$MAD_CLI_COMMAND" ]; then\n'
                        '    echo "  ⚠ madengine not functional — '
                        'installing for this node\'s Python ($(python3 --version))"\n'
                        '    SUBMISSION_DIR={{ manifest_file | dirname }}\n'
                        '    MADENGINE_SRC="$SUBMISSION_DIR/madengine"\n'
                        '    if [ -d "$MADENGINE_SRC" ] && [ -f "$MADENGINE_SRC/pyproject.toml" ]; then\n'
                        '        python3 -m venv "$WORKSPACE/node_venv"\n'
                        '        source "$WORKSPACE/node_venv/bin/activate"\n'
                        '        pip install --upgrade pip setuptools wheel 2>&1 | tail -3\n'
                        '        pip install "$MADENGINE_SRC" 2>&1 | tail -20\n'
                        '        if madengine --version >/dev/null 2>&1; then\n'
                        '            MAD_CLI_COMMAND="madengine"\n'
                        '            echo "  ✓ madengine installed: '
                        '$(madengine --version 2>&1 | head -1)"\n'
                        '        else\n'
                        '            echo "  ✗ madengine install failed"\n'
                        '            exit 1\n'
                        '        fi\n'
                        '    else\n'
                        '        echo "  ✗ madengine source not found at $MADENGINE_SRC"\n'
                        '        exit 1\n'
                        '    fi\n'
                        'fi\n'
                        'echo ""\n\n'
                    )
                    content = content[:mn_verify_idx] + replacement + content[mn_end_idx:]
                    template.write_text(content)
                    log.info("Patched SLURM template: added per-node madengine install (multi-node)")
                else:
                    log.warning("Could not find end of multi-node verification block")
            else:
                log.warning("Could not find multi-node verification block in template")
        else:
            log.warning("TASK_SCRIPT_EOF not found — template may not have multi-node support")

    template = src / "deployment" / "templates" / "slurm" / "job.sh.j2"
    if template and template.exists():
        content = template.read_text()
        old_nfs_pattern = r"\bnfs\b"
        new_nfs_pattern = r"\bnfs[0-9]*\b"
        if old_nfs_pattern in content and new_nfs_pattern not in content:
            content = content.replace(old_nfs_pattern, new_nfs_pattern)
            template.write_text(content)
            log.info("Patched SLURM template: NFS detection now matches nfs4")

    run_orch = src / "orchestration" / "run_orchestrator.py"
    if run_orch.exists():
        content = run_orch.read_text()
        patched = content.replace(
            'print(self.console.sh("yum info rocm-libs", canFail=True))',
            'print(self.console.sh("rpm -qi rocm-libs 2>/dev/null '
            '|| echo rocm-libs not installed as RPM", canFail=True))',
        )
        if patched != content:
            run_orch.write_text(patched)
            log.info(
                "Patched run_orchestrator.py: replaced 'yum info' with 'rpm -qi' "
                "to avoid interactive GPG prompt hang"
            )
        else:
            log.info("run_orchestrator.py already patched or yum string not found")


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
        semver = ""
        head_ref = ""
        for line in result.stdout.splitlines():
            if not semver:
                m = re.match(r"^(\d+\.\d+\.\d+)$", line)
                if m:
                    semver = m.group(1)
            if not head_ref:
                m = re.match(r"^(HEAD:[0-9a-fA-F]{6,})$", line)
                if m:
                    head_ref = m.group(1)
            if semver and head_ref:
                break
        if semver and head_ref:
            fp["version"] = f"{semver}-{head_ref}"
        elif semver:
            fp["version"] = semver
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return fp


def verify_rccl_replacement(
    work_dir: Path,
    expected: dict,
    slurm_job_id: str = "",
) -> tuple[bool, str]:
    """Verify that every node used the CI-built RCCL, not its bundled copy.

    Checks *all* ``*node_*.out`` logs for ``RCCL version :`` and compares
    against the expected fingerprint.  A library that reached only one node
    still fails — this is the multi-node failure this job exists to catch.

    Returns (ok, message).  Generic across frameworks — any RCCL-based
    application prints the version string during init.
    """
    if not expected.get("version"):
        return True, "No RCCL version to verify"

    log_dir = work_dir / "slurm_output"
    node_logs = sorted(log_dir.glob("*node_*.out")) if log_dir.is_dir() else []
    if not node_logs:
        return False, "No node logs found — cannot verify RCCL"

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

        runtime_version = rccl_versions[0].strip()
        if runtime_version != expected["version"]:
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


def build_rccl_overlay_image(
    rccl_lib: Path,
    base_image: str,
    gpu_target: str,
    work_dir: Path,
    registry: str = "",
) -> str:
    """Build a Docker overlay image with the CI-built RCCL and push to registry.

    When a registry is provided, the image is tagged and pushed so that
    SLURM compute nodes can pull it automatically.  Returns the final
    image tag (registry-qualified if pushed).
    """
    rccl_commit = get_rccl_commit(rccl_lib)
    tag = f"{base_image}-rccl-{gpu_target}-{rccl_commit}"

    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
    )
    if result.returncode == 0:
        log.info("Overlay image already exists on head node: %s", tag)
    else:
        dockerfile = work_dir / "Dockerfile.rccl-overlay"
        rccl_lib_dir = rccl_lib.parent
        uses_kpack = _rccl_uses_kpack(rccl_lib)

        staging_dir = work_dir / "rccl_libs"
        staging_dir.mkdir(exist_ok=True)
        for so_file in rccl_lib_dir.glob("librccl*"):
            dest = staging_dir / so_file.name
            if not dest.exists():
                subprocess.run(["cp", "-L", str(so_file), str(dest)], check=True)

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
    rm -rf /tmp/rccl_ci
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
    rm -rf /tmp/rccl_ci
ENV NCCL_DEBUG=WARN
""")

        log.info("Building overlay image: %s", tag)
        subprocess.run(
            ["docker", "build", "-t", tag,
             "-f", str(dockerfile), str(work_dir)],
            check=True,
        )
        log.info("Overlay image built: %s", tag)

    if registry:
        safe_base = base_image.replace("/", "-").replace(":", "-")
        push_tag = f"{registry}/rccl-ci:{safe_base}-{rccl_commit}"
        log.info("Tagging overlay for registry: %s -> %s", tag, push_tag)
        subprocess.run(["docker", "tag", tag, push_tag], check=True)
        log.info("Pushing overlay image to registry: %s", push_tag)
        subprocess.run(["docker", "push", push_tag], check=True)
        log.info("Overlay image pushed: %s", push_tag)
        return push_tag

    return tag


def generate_manifest(
    workload_name: str,
    workload_config: dict,
    cluster_config: dict,
    overlay_image: str,
    nodes: int,
    work_dir: Path,
    nodelist: str = "",
    registry: str = "",
    rccl_lib: Path | None = None,
) -> Path:
    """Generate a madengine manifest.json for the workload.

    Structure follows the reference template from the mad-rccl branch:
    deployment config under ``deployment_config``, env vars inside both
    ``context.docker_env_vars`` and ``deployment_config.env_vars``, mounts
    in ``context.docker_mounts``.
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
        **({"nodelist": nodelist} if nodelist else {}),
    }

    manifest = {
        "built_images": {
            image_key: {
                "docker_image": overlay_image,
                "local_image": not bool(registry),
                "registry_image": overlay_image if registry else None,
                "registry": registry or None,
                "base_docker": workload_config["base_image"],
                "build_status": "SKIPPED",
                "build_duration": 0,
                "gpu_vendor": "AMD",
            },
        },
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
                "timeout": None,
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

    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to: %s", manifest_path)
    return manifest_path


def run_madengine(
    manifest_path: Path,
    output_csv: Path,
    work_dir: Path,
    timeout_minutes: int = 120,
) -> int:
    """Run madengine with the given manifest and return the exit code.

    The manifest already contains deployment_config with slurm, distributed,
    and env_vars sections.  madengine merges deployment_config into
    additional_context automatically (run_orchestrator.py:225-234), so we
    do not need to duplicate those here.
    """
    cmd = [
        "madengine", "run",
        "-m", str(manifest_path),
        "-o", str(output_csv),
        "--live-output",
        "--verbose",
    ]

    log.info("Running: %s", " ".join(cmd))
    log.info("Timeout: %d minutes", timeout_minutes)

    # Pre-warm: madengine's SLURM deployment validates CLI availability by
    # running `madengine --version` with a 5s timeout.  Cold import of
    # madengine's heavy dependencies (kubernetes, aiohttp, paramiko) can
    # exceed 5s.  Running it once beforehand populates the bytecode cache.
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

    try:
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            env=env,
            timeout=timeout_minutes * 60,
        )
        log.info("madengine exit code: %d", proc.returncode)
        return proc.returncode
    except subprocess.TimeoutExpired:
        log.error("madengine timed out after %d minutes", timeout_minutes)
        return 124


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


def check_regression(
    results_dir: Path,
    workload_name: str,
    scale: str,
    current_value: float,
    workload_type: str,
    precision: str | None = None,
) -> tuple[bool, str]:
    """Check if current metric is a regression vs rolling average.

    Returns (is_regression, message).
    """
    datastore = results_dir / PERF_DATASTORE
    if not datastore.exists():
        return False, "No historical data yet — skipping regression check"

    threshold = (
        REGRESSION_THRESHOLD_TRAINING
        if workload_type == "training"
        else REGRESSION_THRESHOLD_INFERENCE
    )

    historical = []
    with open(datastore) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if (
                    entry.get("workload") == workload_name
                    and entry.get("scale") == scale
                    and entry.get("precision") == precision
                    and entry.get("status") == "pass"
                    and entry.get("metric_value") is not None
                ):
                    historical.append(entry["metric_value"])
            except json.JSONDecodeError:
                continue

    if len(historical) < 3:
        return False, f"Only {len(historical)} historical data points — need at least 3 for regression check"

    window = historical[-REGRESSION_WINDOW:]
    rolling_avg = sum(window) / len(window)
    if rolling_avg == 0:
        return False, "Rolling average is 0 — skipping regression check"
    pct_change = (current_value - rolling_avg) / rolling_avg

    msg = (
        f"Current: {current_value:.1f}, "
        f"Rolling avg ({len(window)} runs): {rolling_avg:.1f}, "
        f"Change: {pct_change:+.1%}, "
        f"Threshold: -{threshold:.0%}"
    )

    if pct_change < -threshold:
        return True, f"REGRESSION DETECTED — {msg}"

    return False, f"No regression — {msg}"


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

    with open(datastore, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("Result appended to %s", datastore)

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)


def generate_summary_report(
    workload_name: str,
    scale: str,
    exit_code: int,
    metric_value: float | None,
    regression_msg: str,
    rccl_commit: str,
    cluster: str,
    precision_results: list[dict] | None = None,
) -> str:
    """Generate a plain-text summary report."""
    status = "PASSED" if exit_code == 0 else "FAILED"
    lines = [
        "RCCL MADEngine Workload Test Report",
        "=" * 40,
        f"Status:     {status}",
        f"Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"Workload:   {workload_name}",
        f"Scale:      {scale}",
        f"Cluster:    {cluster}",
        f"RCCL:       {rccl_commit}",
        "",
    ]

    if precision_results:
        for r in precision_results:
            prec = r.get("precision", "?")
            val = r.get("metric_value")
            tflops = r.get("tflops_avg")
            val_s = f"{val:.1f}" if val else "N/A"
            tflops_s = f"{tflops:.1f}" if tflops else ""
            suffix = f" ({tflops_s} TFLOP/s/GPU)" if tflops_s else ""
            lines.append(f"{prec:>4}:       {val_s} tok/s/GPU{suffix} [{r['status']}]")
    elif metric_value is not None:
        lines.append(f"Throughput: {metric_value:.1f} tok/s/GPU")
    else:
        lines.append("Throughput: N/A (workload did not produce metrics)")

    lines.append("")
    lines.append(f"Regression: {regression_msg}")
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
        default=190,
        help="Timeout for madengine run in minutes (default: 190)",
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
        "--registry",
        type=str,
        default="",
        help="Container registry to push overlay image to (e.g. ghcr.io/rocm/rocm-systems)",
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
    madengine_dir = install_madengine(work_dir)

    patch_madengine_for_cluster(
        madengine_dir,
        no_gres=cluster_config.get("slurm_no_gres", False),
    )

    # Step 3: Build overlay image (or use pre-built)
    if args.skip_overlay_build:
        if not args.overlay_image:
            log.error("--skip-overlay-build requires --overlay-image")
            sys.exit(1)
        overlay_image = args.overlay_image
    else:
        overlay_image = build_rccl_overlay_image(
            rccl_lib,
            workload_config["base_image"],
            cluster_config["gpu_target"],
            work_dir,
            registry=args.registry,
        )

    # Step 4: Generate manifest
    # When no registry is configured, the overlay image only exists on the
    # node that built it. Pin the SLURM job to that node so madengine can
    # find the image locally.
    nodelist = ""
    if args.nodes == 1 and not args.registry:
        nodelist = os.environ.get("SLURM_NODELIST", "")
        if not nodelist:
            hostname = subprocess.run(
                ["hostname", "-s"], capture_output=True, text=True,
            ).stdout.strip()
            if hostname:
                nodelist = hostname
        if nodelist:
            log.info("No registry — pinning SLURM job to build node: %s", nodelist)

    manifest_path = generate_manifest(
        args.workload,
        workload_config,
        cluster_config,
        overlay_image,
        args.nodes,
        work_dir,
        nodelist=nodelist,
        registry=args.registry,
        rccl_lib=rccl_lib if args.skip_overlay_build else None,
    )

    # Step 5: Run the workload
    output_csv = work_dir / "perf.csv"
    exit_code = run_madengine(
        manifest_path, output_csv, work_dir, args.timeout_minutes,
    )

    # Step 5b: Verify RCCL replacement (runs in both overlay and bind-mount modes)
    rccl_verification_failed = False
    if rccl_fingerprint.get("version"):
        rccl_ok, rccl_msg = verify_rccl_replacement(
            work_dir, rccl_fingerprint,
        )
        if rccl_ok:
            log.info("RCCL verification: %s", rccl_msg)
        else:
            log.error("RCCL verification FAILED: %s", rccl_msg)
            rccl_verification_failed = True
            exit_code = max(exit_code, 1)

    # Step 6: Parse results — structured output first, live log fallback
    perf_results = parse_perf_results(work_dir)
    live_log_runs = parse_live_log_metrics(work_dir)

    # Save run artifacts
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_artifacts = results_dir / "runs" / run_id
    try:
        run_artifacts.mkdir(parents=True, exist_ok=True)
        for f in ["perf.csv", "perf_entry_super.csv", "perf_entry_super.json"]:
            src = work_dir / f
            if src.exists():
                shutil.copy2(str(src), str(run_artifacts / f))
    except OSError as exc:
        log.warning("Could not save run artifacts to %s: %s", run_artifacts, exc)

    # Build per-precision results from structured output (primary) or
    # live-log scraping (fallback).  Each entry carries precision, metric
    # value, and a pass/fail status so downstream regression checks and
    # datastore writes are driven from one list.
    metric_key = workload_config["metric_key"]
    precision_results: list[dict] = []

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
            precision_results.append({
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
        for pr in precision_results:
            if "tflops_avg" not in pr:
                pr["tflops_avg"] = tflops_by_precision.get(pr["precision"])

    if not precision_results and live_log_runs:
        for run in live_log_runs:
            val = run.get("tokens_per_second_per_gpu")
            if val is None:
                continue
            precision_results.append({
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

    metric_value = precision_results[-1]["metric_value"] if precision_results else None

    # Override exit_code if training actually completed successfully.
    # madengine can report failure (exit code 3) when its perf collector
    # can't parse the output format, even though training ran to completion.
    if exit_code != 0 and not rccl_verification_failed and precision_results:
        all_pass = all(r["status"] == "pass" for r in precision_results)
        has_metric = all(r.get("metric_value") is not None for r in precision_results)
        if all_pass and has_metric:
            log.info(
                "Overriding madengine exit code %d → 0: all %d precision run(s) "
                "passed with metrics",
                exit_code, len(precision_results),
            )
            for r in precision_results:
                log.info("  %s: %.1f %s", r.get("precision", "?"),
                         r["metric_value"], metric_key)
            exit_code = 0

    # Step 7: Per-precision regression check
    regression_msg = "N/A"
    is_regression = False
    if precision_results:
        regression_msgs = []
        for pr in precision_results:
            if pr["metric_value"] is not None:
                reg, msg = check_regression(
                    results_dir, args.workload, scale, pr["metric_value"],
                    workload_config["type"], precision=pr.get("precision"),
                )
                regression_msgs.append(f"[{pr.get('precision', '?')}] {msg}")
                if reg:
                    is_regression = True
                    log.warning(msg)
                else:
                    log.info(msg)
        regression_msg = "; ".join(regression_msgs) if regression_msgs else "N/A"
        if is_regression:
            exit_code = max(exit_code, 1)

    # Step 8: Append result to datastore (one record per precision run)
    extra = {"cluster": args.cluster, "overlay_image": overlay_image}
    if precision_results:
        for pr in precision_results:
            append_result(
                results_dir,
                args.workload,
                scale,
                pr["metric_value"],
                pr["status"],
                rccl_commit,
                extra=extra,
                precision=pr.get("precision"),
                tflops=pr.get("tflops_avg"),
                tokens_per_sec=pr["metric_value"],
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
        args.workload, scale, exit_code, metric_value,
        regression_msg, rccl_commit, args.cluster,
        precision_results=precision_results if precision_results else None,
    )
    log.info("\n%s", report)
    write_github_summary(report)
    set_github_output("madengine_status", status)
    if metric_value is not None:
        set_github_output("madengine_metric", f"{metric_value:.1f}")

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
