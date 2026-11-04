# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the metric-health remote worker script."""

import os
import subprocess
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "scripts" / "run_metric_health_test_report.sh"

# Tools the native-tool probe looks for. Each stub succeeds, so the probe
# passes on any test host.
PROBE_TOOLS = ("cmake", "pkg-config")
SUCCEEDING_STUB = "#!/usr/bin/env bash\nexit 0\n"


def worker_env(tmp_path: Path) -> Dict[str, str]:
    """Return an environment whose PATH finds stub probe tools first."""
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir(exist_ok=True)
    for name in PROBE_TOOLS:
        stub = stub_bin / name
        stub.write_text(SUCCEEDING_STUB, encoding="utf-8")
        stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}{os.pathsep}{env['PATH']}"
    env["WORKER"] = str(WORKER)
    return env


def run_sourced_worker(
    tmp_path: Path, extra_args: List[str], body: str
) -> subprocess.CompletedProcess:
    """Source the gfx942 worker with these arguments, then run a bash body."""
    therock = tmp_path / "therock"
    therock.mkdir(exist_ok=True)
    args = [
        "--arch",
        "gfx942",
        "--tag",
        "t",
        "--root",
        str(tmp_path / "root"),
        "--therock",
        str(therock),
        "--log",
        str(tmp_path / "worker.log"),
        *extra_args,
    ]
    script = f'set -- {" ".join(args)}; source "$WORKER"\n{body}'
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=worker_env(tmp_path),
    )


def native_decision(tmp_path: Path, extra_args: List[str]) -> List[str]:
    """Return USE_NATIVE_TOOL and the profile flags decide_native picks."""
    body = (
        "decide_native >/dev/null\n"
        'printf "%s\\n" "$USE_NATIVE_TOOL" "${PROFILE_EXTRA[@]}"'
    )
    completed = run_sourced_worker(tmp_path, extra_args, body)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.split()


def install_shared_profiler(tmp_path: Path) -> Path:
    """Put a rocprof-compute link under the TheRock prefix; return its target."""
    target = tmp_path / "shared" / "rocprof-compute"
    target.parent.mkdir()
    target.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    target.chmod(0o755)
    therock_bin = tmp_path / "therock" / "bin"
    therock_bin.mkdir(parents=True)
    (therock_bin / "rocprof-compute").symlink_to(target)
    return target


def test_probe_mode_attempts_native_tool_when_dependencies_exist(
    tmp_path: Path,
) -> None:
    assert native_decision(tmp_path, []) == ["true", "--no-roof"]


def test_no_native_tool_flag_overrides_probe_mode(tmp_path: Path) -> None:
    """--no-native-tool wins even when the probe finds every dependency."""
    assert native_decision(tmp_path, ["--no-native-tool"]) == [
        "false",
        "--no-native-tool",
        "--no-roof",
    ]


def test_run_meta_records_resolved_profiler(tmp_path: Path) -> None:
    """A link to a shared install records the file it resolves to."""
    target = install_shared_profiler(tmp_path)

    completed = run_sourced_worker(
        tmp_path, [], "resolve_rocprof_compute >/dev/null\nwrite_run_meta"
    )

    assert completed.returncode == 0, completed.stderr
    meta = tmp_path / "root" / "workloads" / "logs_gfx942_t" / "run_meta.env"
    lines = meta.read_text(encoding="utf-8").splitlines()
    assert f"ROCPROF_COMPUTE_RESOLVED={target.resolve()}" in lines
