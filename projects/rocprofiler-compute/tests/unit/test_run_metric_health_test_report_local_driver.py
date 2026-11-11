# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the metric-health local driver."""

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "run_metric_health_test_report_local_driver.sh"
EXAMPLE = REPO_ROOT / "scripts" / ".metric_health_sites.env.example"

GFX942_SITE_BASE = """
GFX942_HOST=host.example
GFX942_REMOTE_ROOT=/remote/root
GFX942_THEROCK=/opt/therock
GFX942_REMOTE_LOG_DIR=/remote/logs
"""
GFX1151_SITE = """
GFX1151_HOST=apu.example
GFX1151_REMOTE_USER=user
GFX1151_REMOTE_ROOT=/remote/root
GFX1151_THEROCK=/opt/therock
GFX1151_REMOTE_LOG_DIR=/remote/logs
"""
SLURM_SITE = """
SLURM_HOST_LOGIN=login.example
SLURM_PARTITION=gpu
SLURM_REMOTE_HOME=/home/user
GFX908_NODE=node1
GFX908_GRES=gpu:1
"""
GFX90A_SITE = SLURM_SITE + "GFX90A_NODE=node90a\nGFX90A_GRES=gpu:1\n"
MEGA_SKIP_LINE = (
    "WORKLOAD_DONE name=mega_kernel mode=single-pass profile=SKIP "
    "analyze=SKIP duration_s=0 iters=500\n"
)

# Site and override variables the driver reads from the environment. A host
# that exports any of them must not change what these tests see.
SITE_VAR_RE = re.compile(r"^(GFX\d+|SLURM)_")
DRIVER_ENV_OVERRIDES = (
    "ARTIFACTS",
    "GRES",
    "HOST",
    "HOST_LOGIN",
    "METRIC_HEALTH_SITES_ENV",
    "NODE",
    "NODE_ROOT",
    "PARTITION",
    "PROXY",
    "REMOTE_LOG",
    "REMOTE_LOG_DIR",
    "REMOTE_NOHUP",
    "REMOTE_PASS",
    "REMOTE_ROOT",
    "REMOTE_USER",
    "ROCR_VISIBLE_DEVICES",
    "SBATCH_OUT",
    "WAIT_TIMEOUT",
    "WALLTIME",
)

# Stubs a remote host whose worker already exited without ALL_DONE.
SSH_WORKER_GONE = """
ssh_cmd() {
  case "$1" in
    *"kill -0"*) echo gone ;;
    *"grep -q"*) return 1 ;;
    *) echo "make: *** [mega_kernel] Error 1" ;;
  esac
}
"""
SSH_WORKER_RUNNING = """
ssh_cmd() {
  case "$1" in
    *"kill -0"*) echo running ;;
    *"grep -q"*) return 1 ;;
  esac
}
"""
SSH_WORKER_DONE = """
ssh_cmd() {
  case "$1" in
    *"kill -0"*) echo gone ;;
    *"grep -q"*) return 0 ;;
    *) echo "ALL_DONE tag=t" ;;
  esac
}
"""
SLURM_JOB_GONE = """
ssh_login() {
  case "$1" in
    *"squeue -h"*) echo SQUEUE_OK ;;
    *"grep -q"*) return 1 ;;
  esac
}
"""
RECORDING_STUB = """#!/usr/bin/env bash
printf '%s SSHPASS=%s\\n' "$(basename "$0") $*" "${SSHPASS:-}" >>"$STUB_RECORD"
if [[ "$(basename "$0")" == sshpass ]]; then
  shift
  exec "$@"
fi
"""


def driver_env(
    site_text: str,
    tmp_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return a clean environment for one driver run with this site file."""
    site = tmp_path / "sites.env"
    site.write_text(site_text, encoding="utf-8")
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in DRIVER_ENV_OVERRIDES and not SITE_VAR_RE.match(name)
    }
    env["METRIC_HEALTH_SITES_ENV"] = str(site)
    env["DRIVER"] = str(DRIVER)
    if extra_env:
        env.update(extra_env)
    return env


def run_sourced_driver(
    args: List[str],
    body: str,
    env: Dict[str, str],
) -> subprocess.CompletedProcess:
    """Source the driver with these arguments, then run a bash body."""
    script = f'set -- {" ".join(args)}; source "$DRIVER"\n{body}'
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def gfx942_rocr_default(
    site_text: str,
    tmp_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> str:
    """Source the driver and return the gfx942 ROCR default."""
    completed = run_sourced_driver(
        ["--arch", "gfx942"],
        'printf "%s\\n" "$ROCR_DEFAULT"',
        driver_env(site_text, tmp_path, extra_env),
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def wait_with_stub(
    arch: str, site_text: str, stub: str, tmp_path: Path, timeout: str = "60"
) -> subprocess.CompletedProcess:
    """Run wait_for_done against stubbed remote commands."""
    return run_sourced_driver(
        ["--arch", arch, "--tag", "t"],
        stub + "POLL_SLEEP=0\nwait_for_done\n",
        driver_env(site_text, tmp_path, {"WAIT_TIMEOUT": timeout}),
    )


def write_run_meta(artifact_dir: Path, resolved: str) -> Path:
    log_dir = artifact_dir / "logs" / "gfx942"
    log_dir.mkdir(parents=True)
    (log_dir / "run_meta.env").write_text(
        f"RUN=develop\nROCPROF_COMPUTE_RESOLVED={resolved}\n", encoding="utf-8"
    )
    for name in ("vcopy", "nbody", "mega_kernel"):
        (log_dir / f"{name}_single-pass.log").write_text("", encoding="utf-8")
    return artifact_dir


def check_profilers(
    tmp_path: Path, baseline_bin: str, candidate_bin: str, body: str
) -> subprocess.CompletedProcess:
    baseline = write_run_meta(tmp_path / "baseline", baseline_bin)
    candidate = write_run_meta(tmp_path / "candidate", candidate_bin)
    return run_sourced_driver(
        ["--arch", "gfx942", "--baseline", str(baseline)],
        "python3() { echo REPORT_RAN; }\n" + body,
        driver_env(GFX942_SITE_BASE, tmp_path, {"ARTIFACTS": str(candidate)}),
    )


def gfx90a_report(
    tmp_path: Path, log_names: List[str], health_log: str
) -> subprocess.CompletedProcess:
    """Run generate_report for gfx90a with these analyze logs and health log."""
    artifacts = tmp_path / "artifacts"
    log_dir = artifacts / "logs" / "gfx90a"
    log_dir.mkdir(parents=True)
    for name in log_names:
        (log_dir / f"{name}_single-pass.log").write_text("", encoding="utf-8")
    (artifacts / "metric_health_t.log").write_text(health_log, encoding="utf-8")
    return run_sourced_driver(
        ["--arch", "gfx90a", "--tag", "t"],
        "python3() { echo REPORT_RAN; }\ngenerate_report\n",
        driver_env(GFX90A_SITE, tmp_path, {"ARTIFACTS": str(artifacts)}),
    )


def pack_workload_dirs(
    root: Path, tag: str, tmp_path: Path
) -> subprocess.CompletedProcess:
    """Run the remote workload-archive script against a local directory."""
    script_path = tmp_path / "archive.sh"
    sourced = run_sourced_driver(
        ["--arch", "gfx908", "--tag", "t"],
        "workload_dir_archive_script "
        + shlex.quote(str(root))
        + " "
        + shlex.quote(tag)
        + " > "
        + shlex.quote(str(script_path)),
        driver_env(SLURM_SITE, tmp_path),
    )
    assert sourced.returncode == 0, sourced.stderr
    return subprocess.run(
        ["bash", str(script_path)],
        check=False,
        capture_output=True,
    )


def test_gfx942_device_id_comes_from_site_config(tmp_path: Path) -> None:
    """gfx942 uses the site-config device id."""
    site_text = GFX942_SITE_BASE + "GFX942_ROCR_VISIBLE_DEVICES=17\n"
    assert gfx942_rocr_default(site_text, tmp_path) == "17"


def test_gfx942_device_id_is_empty_without_site_value(tmp_path: Path) -> None:
    """An unset site device id does not fall back to a baked-in number."""
    assert gfx942_rocr_default(GFX942_SITE_BASE, tmp_path) == ""


def test_exported_gfx942_device_id_wins_over_site_file(tmp_path: Path) -> None:
    """An already-exported site variable wins over the file."""
    site_text = GFX942_SITE_BASE + "GFX942_ROCR_VISIBLE_DEVICES=17\n"
    resolved = gfx942_rocr_default(
        site_text,
        tmp_path,
        {"GFX942_ROCR_VISIBLE_DEVICES": "3"},
    )
    assert resolved == "3"


def test_host_site_exports_do_not_leak_into_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device id exported on the test host does not override the site file."""
    monkeypatch.setenv("GFX942_ROCR_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    site_text = GFX942_SITE_BASE + "GFX942_ROCR_VISIBLE_DEVICES=17\n"
    assert gfx942_rocr_default(site_text, tmp_path) == "17"
    assert gfx942_rocr_default(GFX942_SITE_BASE, tmp_path) == ""


def test_example_names_gfx942_device_without_a_value() -> None:
    """The committed example names the variable and leaves it empty."""
    text = EXAMPLE.read_text(encoding="utf-8")
    device_lines = [
        line
        for line in text.splitlines()
        if line.startswith("GFX942_ROCR_VISIBLE_DEVICES=")
    ]
    assert device_lines == ["GFX942_ROCR_VISIBLE_DEVICES="]
    assert "17" not in text
    driver = DRIVER.read_text(encoding="utf-8")
    assert "ROCR_DEFAULT=17" not in driver


def test_password_log_fetch_passes_password_to_scp(tmp_path: Path) -> None:
    """The ssh-pass log fetch runs scp under sshpass with the password."""
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name in ("sshpass", "scp"):
        stub = stub_bin / name
        stub.write_text(RECORDING_STUB, encoding="utf-8")
        stub.chmod(0o755)
    record = tmp_path / "record.txt"
    env = driver_env(
        GFX1151_SITE,
        tmp_path,
        {
            "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
            "REMOTE_PASS": "secret",
            "STUB_RECORD": str(record),
        },
    )

    completed = run_sourced_driver(
        ["--arch", "gfx1151"], "remote_scp user@apu.example:/log /tmp/log", env
    )

    assert completed.returncode == 0, completed.stderr
    calls = record.read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith("sshpass -e scp ")
    assert calls[0].endswith("SSHPASS=secret")
    assert calls[1].startswith("scp -o StrictHostKeyChecking=no ")
    assert calls[1].endswith("user@apu.example:/log /tmp/log SSHPASS=secret")


def test_wait_fails_when_ssh_worker_exits_without_all_done(tmp_path: Path) -> None:
    completed = wait_with_stub("gfx942", GFX942_SITE_BASE, SSH_WORKER_GONE, tmp_path)

    assert completed.returncode == 1
    assert "stopped without ALL_DONE" in completed.stderr
    assert "Error 1" in completed.stderr


def test_wait_times_out_while_worker_keeps_running(tmp_path: Path) -> None:
    completed = wait_with_stub(
        "gfx942", GFX942_SITE_BASE, SSH_WORKER_RUNNING, tmp_path, timeout="0"
    )

    assert completed.returncode == 1
    assert "no ALL_DONE" in completed.stderr
    assert "after 0s" in completed.stderr


def test_wait_succeeds_when_worker_wrote_all_done(tmp_path: Path) -> None:
    completed = wait_with_stub("gfx942", GFX942_SITE_BASE, SSH_WORKER_DONE, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "ALL_DONE" in completed.stdout


def test_wait_fails_when_slurm_job_left_the_queue(tmp_path: Path) -> None:
    completed = wait_with_stub("gfx908", SLURM_SITE, SLURM_JOB_GONE, tmp_path)

    assert completed.returncode == 1
    assert "stopped without ALL_DONE" in completed.stderr


def test_sbatch_job_name_names_the_tag(tmp_path: Path) -> None:
    sbatch = tmp_path / "job.sbatch"
    completed = run_sourced_driver(
        ["--arch", "gfx908", "--tag", "t1"],
        f'write_sbatch /remote/root "{sbatch}"',
        driver_env(SLURM_SITE, tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "#SBATCH -J metric-health-t1" in sbatch.read_text(encoding="utf-8")


def test_report_refuses_runs_from_the_same_profiler(tmp_path: Path) -> None:
    shared = "/opt/therock/bin/rocprof-compute"

    completed = check_profilers(tmp_path, shared, shared, "generate_report")

    assert completed.returncode == 1
    assert f"both ran {shared}" in completed.stderr
    assert "REPORT_RAN" not in completed.stdout


def test_report_compares_runs_from_different_profilers(tmp_path: Path) -> None:
    completed = check_profilers(
        tmp_path,
        "/develop/install/bin/rocprof-compute",
        "/branch/install/bin/rocprof-compute",
        "generate_report",
    )

    assert completed.returncode == 0, completed.stderr
    assert "baseline rocprof-compute: /develop/install/bin/rocprof-compute" in (
        completed.stdout
    )
    assert "candidate rocprof-compute: /branch/install/bin/rocprof-compute" in (
        completed.stdout
    )
    assert "REPORT_RAN" in completed.stdout


def test_sbatch_keeps_a_preset_visible_device(tmp_path: Path) -> None:
    """The sbatch snippet defaults devices only when they are unset."""
    sbatch = tmp_path / "job.sbatch"
    completed = run_sourced_driver(
        ["--arch", "gfx908", "--tag", "t1"],
        f'write_sbatch /remote/root "{sbatch}"',
        driver_env(SLURM_SITE, tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    text = sbatch.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in text
    assert ': "${ROCR_VISIBLE_DEVICES:=0}"' in text
    assert ': "${HIP_VISIBLE_DEVICES:=0}"' in text
    assert "export ROCR_VISIBLE_DEVICES=0" not in text
    assert "export HIP_VISIBLE_DEVICES=0" not in text
    snippet = "\n".join(line for line in text.splitlines() if "VISIBLE_DEVICES" in line)
    show = 'printf "%s %s\\n" "$ROCR_VISIBLE_DEVICES" "$HIP_VISIBLE_DEVICES"'
    preset = subprocess.run(
        ["bash", "-c", f"{snippet}\n{show}"],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "ROCR_VISIBLE_DEVICES": "2",
            "HIP_VISIBLE_DEVICES": "3",
        },
    )
    assert preset.returncode == 0, preset.stderr
    assert preset.stdout.strip() == "2 3"
    defaulted = subprocess.run(
        ["bash", "-c", f"{snippet}\n{show}"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert defaulted.returncode == 0, defaulted.stderr
    assert defaulted.stdout.strip() == "0 0"


def test_gfx90a_log_poll_does_not_request_gpus(tmp_path: Path) -> None:
    """A gfx90a poll reads the log without the workload's GPU request."""
    completed = run_sourced_driver(
        ["--arch", "gfx90a", "--tag", "t"],
        'ssh_login() { printf "%s\\n" "$1"; }\non_log_host "grep -q ALL_DONE /log"\n',
        driver_env(GFX90A_SITE, tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    command = completed.stdout
    assert "--gres=none" in command
    assert "gpu:1" not in command
    assert "--overlap" not in command


def test_gfx90a_skipped_mega_kernel_does_not_fail_the_report(tmp_path: Path) -> None:
    """profile=SKIP for mega_kernel warns and still generates the report."""
    completed = gfx90a_report(tmp_path, ["vcopy", "nbody"], MEGA_SKIP_LINE)

    assert completed.returncode == 0, completed.stderr
    assert "profile=SKIP" in completed.stderr
    assert "REPORT_RAN" in completed.stdout


def test_gfx90a_missing_vcopy_still_fails_the_report(tmp_path: Path) -> None:
    """A workload that was supposed to run still fails when its log is missing."""
    completed = gfx90a_report(tmp_path, ["nbody"], MEGA_SKIP_LINE)

    assert completed.returncode == 1
    assert "vcopy_single-pass.log" in completed.stderr
    assert "REPORT_RAN" not in completed.stdout


def test_gfx90a_missing_mega_without_skip_fails_the_report(tmp_path: Path) -> None:
    """A missing mega_kernel log without profile=SKIP is still an error."""
    completed = gfx90a_report(tmp_path, ["vcopy", "nbody"], "ALL_DONE\n")

    assert completed.returncode == 1
    assert "mega_kernel_single-pass.log" in completed.stderr
    assert "REPORT_RAN" not in completed.stdout


def test_workload_archive_fetches_dirs_whose_names_contain_the_tag(
    tmp_path: Path,
) -> None:
    """A quoted tag packs that directory and does not expand a glob in the tag."""
    tag = "a*b"
    root = tmp_path / "workloads"
    matched = root / f"vcopy_{tag}_single-pass"
    matched.mkdir(parents=True)
    (matched / "marker.txt").write_text("ok", encoding="utf-8")
    (root / "vcopy_aXb_single-pass").mkdir()
    packed = pack_workload_dirs(root, tag, tmp_path)

    assert packed.returncode == 0, packed.stderr
    archive = tmp_path / "wl.tar.gz"
    archive.write_bytes(packed.stdout)
    dest = tmp_path / "out"
    dest.mkdir()
    extracted = run_sourced_driver(
        ["--arch", "gfx908", "--tag", "t"],
        "extract_tagged_workload_archive "
        + shlex.quote(str(archive))
        + " "
        + shlex.quote(str(dest))
        + " "
        + shlex.quote(tag),
        driver_env(SLURM_SITE, tmp_path),
    )

    assert extracted.returncode == 0, extracted.stderr
    assert (dest / matched.name / "marker.txt").read_text(encoding="utf-8") == "ok"
    assert not (dest / "vcopy_aXb_single-pass").exists()


def test_workload_archive_with_no_match_is_empty(tmp_path: Path) -> None:
    """No matching directory is a successful empty archive."""
    root = tmp_path / "workloads"
    root.mkdir()
    (root / "vcopy_other_single-pass").mkdir()
    packed = pack_workload_dirs(root, "a*b", tmp_path)

    assert packed.returncode == 0, packed.stderr
    assert packed.stdout == b""


def test_workload_archive_reports_a_missing_directory(tmp_path: Path) -> None:
    """A missing workload root is a real tar/cd failure, not an empty match."""
    packed = pack_workload_dirs(tmp_path / "missing", "tag", tmp_path)

    assert packed.returncode != 0


def test_extract_rejects_an_archive_without_the_tag(tmp_path: Path) -> None:
    """A valid archive that does not contain the tag is not installed."""
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "file.txt").write_text("x", encoding="utf-8")
    archive = tmp_path / "plain.tar.gz"
    subprocess.run(
        ["tar", "czf", str(archive), "-C", str(tmp_path), "plain"],
        check=True,
    )
    dest = tmp_path / "out"
    dest.mkdir()
    extracted = run_sourced_driver(
        ["--arch", "gfx908", "--tag", "t"],
        "extract_tagged_workload_archive "
        + shlex.quote(str(archive))
        + " "
        + shlex.quote(str(dest))
        + " "
        + shlex.quote("wanted"),
        driver_env(SLURM_SITE, tmp_path),
    )

    assert extracted.returncode == 1
    assert "no matching directory" in extracted.stderr
    assert not (dest / "plain").exists()
