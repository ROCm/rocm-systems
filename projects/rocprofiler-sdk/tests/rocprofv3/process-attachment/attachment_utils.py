# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Helpers for rocprofv3 live process-attachment integration tests."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import pytest

# Prefixed lines make mixed CTest output easy to read (app vs attach vs harness).
LOG_TEST = "[TEST]"
LOG_APP = "[APP]"
LOG_ATTACH = "[ATTACH]"

OUTPUT_SUBDIR = "attachment-output"
SKIP_MARKER = "skipped"
EXECUTE_FAILED_MARKER = "execute-failed"
# Unique line for CTest SKIP_REGULAR_EXPRESSION (must not appear in pytest tracebacks).
PTRACE_SKIP_MESSAGE = "PROCESS_ATTACHMENT_SKIP: ptrace unavailable"
PC_SAMPLING_SKIP_MESSAGE = "PROCESS_ATTACHMENT_SKIP: PC sampling unavailable"
MPI_SKIP_MESSAGE = "PROCESS_ATTACHMENT_SKIP: MPI unavailable"
MPI_SKIP_MARKER = "mpi-skipped"
MPI_ATTACH_NUMPROCS = 2
MPI_ATTACH_STARTUP_SEC = 3
# Wall-clock runtime for mpi-simple-attach (one mpiexec tree, no bash respawn loop).
MPI_ATTACH_DEFAULT_DURATION_SEC = 90
MPI_ATTACH_HIP_READY_TIMEOUT_SEC = 30
ATTACH_TARGET_MAX_ATTEMPTS = 5
# Maps entries that indicate HIP/HS A is loaded (therock may use libhsa-amd-* not libamdhip64).
MPI_ATTACH_MAP_MARKERS = (
    "libamdhip64",
    "libamdhip",
    "libhsa-amd",
    "rocprofiler-register",
)
OPENMP_ATTACH_STARTUP_SEC = 2
# Default wall-clock runtime for openmp-attach (single stable PID, no bash respawn loop).
OPENMP_ATTACH_DEFAULT_DURATION_SEC = 90
APP_STARTUP_SEC = 2

# transpose: many iterations so PMC attach runs long enough to stress rocpd detach
TRANSPOSE_LONG_PMC_APP_ARGS = ("4", "100000", "10")
TRANSPOSE_LONG_PMC_WAIT_AFTER_ATTACH_SEC = 180
PMC_ROCPD_LONG_ATTACH_MSEC = "60000"
TRANSPOSE_PMC_APP_STARTUP_SEC = 1

# attachment-test with high GPU thread count (burn-style workload proxy)
PMC_MULTITHREAD_APP_ARGS = ("8", "1")
PMC_MULTITHREAD_ATTACH_MSEC = "10000"

# Light PMC + rocpd smoke (CI-friendly; passes on fixed ROCm builds)
PMC_ROCPD_SMOKE_APP_ARGS = ("4", "1")
PMC_ROCPD_SMOKE_ATTACH_MSEC = "5000"

# transpose + PC sampling + selected-regions (enough iterations to outlive attach)
TRANSPOSE_PC_SAMPLING_APP_ARGS = ("4", "10000", "10")
TRANSPOSE_PC_SAMPLING_APP_STARTUP_SEC = 2
TRANSPOSE_PC_SAMPLING_ATTACH_MSEC = "5000"
TRANSPOSE_PC_SAMPLING_INTERVAL = "1000000"
PC_SAMPLING_SKIP_MARKER = "pc-sampling-skipped"


def output_subdir(output_dir: str | Path) -> Path:
    return Path(output_dir) / OUTPUT_SUBDIR


def skip_marker_path(output_dir: str | Path) -> Path:
    return output_subdir(output_dir) / SKIP_MARKER


def execute_failed_marker_path(output_dir: str | Path) -> Path:
    return output_subdir(output_dir) / EXECUTE_FAILED_MARKER


def pc_sampling_skip_marker_path(output_dir: str | Path) -> Path:
    return output_subdir(output_dir) / PC_SAMPLING_SKIP_MARKER


def mpi_skip_marker_path(output_dir: str | Path) -> Path:
    return output_subdir(output_dir) / MPI_SKIP_MARKER


def results_json_path(output_dir: str | Path, output_name: str) -> Path:
    return output_subdir(output_dir) / f"{output_name}_results.json"


def ptrace_permissions_ok() -> bool:
    """Return True when live ptrace attach is expected to work."""
    scope_path = Path("/proc/sys/kernel/yama/ptrace_scope")
    if not scope_path.is_file():
        return True

    try:
        scope = int(scope_path.read_text().strip())
    except (OSError, ValueError):
        return True

    if scope == 0:
        return True

    if os.geteuid() == 0:
        return True

    if _has_cap_sys_ptrace("self"):
        return True

    python3 = shutil.which("python3")
    if python3 and _has_cap_sys_ptrace(python3):
        return True

    return False


def _has_cap_sys_ptrace(target: str) -> bool:
    getpcaps = shutil.which("getpcaps")
    if not getpcaps:
        return False
    try:
        proc = subprocess.run(
            [getpcaps, target],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return "cap_sys_ptrace" in (proc.stdout + proc.stderr)


def rocprofv3_supports_attach_sync_output(rocprofv3: str) -> bool:
    """--attach-sync-output exists in rocprofv3 1.3+ only."""
    try:
        proc = subprocess.run(
            [rocprofv3, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "attach-sync-output" in (proc.stdout + proc.stderr)


def pc_sampling_selected_regions_attach_extra(
    rocprofv3: str,
    *,
    duration_msec: str = TRANSPOSE_PC_SAMPLING_ATTACH_MSEC,
) -> list[str]:
    """Host-trap PC sampling, marker trace, and roctx selected-regions (JSON output)."""
    return [
        "--attach-duration-msec",
        duration_msec,
        "--pc-sampling-beta-enabled",
        "--pc-sampling-unit",
        "time",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        TRANSPOSE_PC_SAMPLING_INTERVAL,
        "--marker-trace",
        "--selected-regions",
        "--output-format",
        "json",
    ]


def _rocm_install_prefix(rocprofv3: str) -> Path:
    """ROCm prefix containing bin/rocprofv3 (e.g. install_18May)."""
    return Path(rocprofv3).resolve().parent.parent


def _list_avail_environment(rocprofv3: str) -> dict[str, str]:
    """Environment for rocprofv3 --list-avail / -L (see using-pc-sampling.rst)."""
    env = os.environ.copy()
    env["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "on"
    prefix = _rocm_install_prefix(rocprofv3)
    for lib in (
        prefix / "lib/rocprofiler-sdk/librocprofv3-list-avail.so",
        prefix / "lib/librocprofiler-sdk-rocprofv3-list-avail.so",
    ):
        if lib.is_file():
            env["ROCPROF_LIST_AVAIL_TOOL_LIBRARY"] = str(lib)
            break
    metrics = prefix / "share/rocprofiler-sdk"
    if metrics.is_dir():
        env["ROCPROFILER_METRICS_PATH"] = str(metrics)
    return env


def query_pc_sampling_support(rocprofv3: str) -> tuple[bool, str]:
    """Query GPU PC sampling support via rocprofv3 --list-avail (-L).

    Matches the workflow documented in using-pc-sampling.rst and implemented in
    rocprofv3.py (loads librocprofv3-list-avail.so, calls rocprofv3-avail info
    --pc-sampling). Requires ROCPROFILER_PC_SAMPLING_BETA_ENABLED.
    """
    env = _list_avail_environment(rocprofv3)
    try:
        proc = subprocess.run(
            [rocprofv3, "--list-avail"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"failed to run rocprofv3 --list-avail: {exc}"

    output = proc.stdout + proc.stderr
    if "PC sampling unavailable" in output:
        return False, "rocprofv3 reports PC sampling unavailable"
    if proc.returncode != 0:
        tail = output.strip().splitlines()[-3:]
        return False, (
            f"rocprofv3 --list-avail exited {proc.returncode}"
            + (f": {' | '.join(tail)}" if tail else "")
        )
    # Attach test uses host-trap; documented -L output lists Method :host_trap under configs.
    if "host_trap" not in output.lower():
        return False, "no host_trap PC sampling configuration in --list-avail output"
    return True, ""


def require_pc_sampling_available(output_dir: str | Path, rocprofv3: str) -> None:
    """Skip before attach when rocprofv3 --list-avail shows no PC sampling support.

    CMake may also DISABLE this test via rocprofiler_sdk_pc_sampling_disabled().
    On supported systems, a later missing JSON/output is a test failure, not a skip.
    """
    supported, reason = query_pc_sampling_support(rocprofv3)
    if supported:
        return
    pc_sampling_skip_marker_path(output_dir).touch()
    print(PC_SAMPLING_SKIP_MESSAGE, flush=True)
    pytest.skip(reason or "PC sampling not available on this system")


def _json_file_contains(json_path: Path, needle: str) -> bool:
    with open(json_path, "rb") as inp:
        return needle.encode() in inp.read()


def assert_pc_sampling_attach_output(output_dir: str | Path, output_name: str) -> None:
    """Fail (not skip) when attach on a supported GPU did not produce PC sampling JSON."""
    json_path = results_json_path(output_dir, output_name)
    if not json_path.is_file() or json_path.stat().st_size == 0:
        raise AssertionError(
            f"PC sampling attach did not write results JSON: {json_path}"
        )
    if not _json_file_contains(json_path, '"pc_sample_host_trap"'):
        raise AssertionError(
            f"PC sampling attach output missing pc_sample_host_trap records: {json_path}"
        )


def pmc_rocpd_attach_extra(
    rocprofv3: str,
    *,
    duration_msec: str = PMC_ROCPD_LONG_ATTACH_MSEC,
) -> list[str]:
    """PMC + kernel-trace + rocpd flags for live attach (--pmc SQ_WAVES, -f rocpd)."""
    extra = [
        "--attach-duration-msec",
        duration_msec,
        "--pmc",
        "SQ_WAVES",
        "--kernel-trace",
        "-f",
        "rocpd",
    ]
    if rocprofv3_supports_attach_sync_output(rocprofv3):
        extra.append("--attach-sync-output")
    return extra


def _raise_if_process_failed(proc: subprocess.Popen, label: str) -> None:
    """Fail the test when a child exited non-zero or was killed by a signal."""
    rc = proc.returncode
    if rc is None:
        return
    if rc < 0:
        raise RuntimeError(f"{label} terminated by signal {-rc}")
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc}")


def _raise_if_attach_log_reports_failure(log: str) -> None:
    """Detect rocattach detach/attach failures reported in rocprofv3 output."""
    if "returned non-zero status" in log:
        raise RuntimeError("rocprofv3 attach/detach returned non-zero status")
    if "Detachment failed" in log or "rocattach_detach_tree failed" in log:
        raise RuntimeError("rocprofv3 detach failed")
    if "rocattach_attach failed" in log or "rocattach call failed" in log:
        raise RuntimeError("rocprofv3 attach failed")


def mpi_attach_env() -> dict[str, str]:
    """OpenMPI run-as-root (same as mpi-ranks / rocpd multiproc tests)."""
    return {
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }


def openmp_attach_env() -> dict[str, str]:
    """Match openmp-tools integration test offload settings."""
    return {
        "OMP_NUM_THREADS": "2",
        "OMP_TARGET_OFFLOAD": "mandatory",
        "ROCR_VISIBLE_DEVICES": "0",
    }


def skip_if_mpi_unavailable(output_dir: str | Path, mpiexec: str | None) -> None:
    if mpiexec and Path(mpiexec).is_file():
        return
    mpi_skip_marker_path(output_dir).touch()
    print(MPI_SKIP_MESSAGE, flush=True)
    pytest.skip("MPI launcher not available for process-attachment MPI test")


def build_mpi_simple_attach_cmd(
    mpiexec: str,
    mpi_numproc_flag: str,
    mpi_simple_attach: str,
    *,
    num_procs: int = MPI_ATTACH_NUMPROCS,
    duration_sec: int = MPI_ATTACH_DEFAULT_DURATION_SEC,
    host_sleep_ms: int = 100,
) -> list[str]:
    """mpiexec + mpi-simple-attach (stable MPI tree, long GPU loop per rank)."""
    return [
        mpiexec,
        mpi_numproc_flag,
        str(num_procs),
        mpi_simple_attach,
        str(duration_sec),
        str(host_sleep_ms),
    ]


def build_openmp_attach_cmd(
    openmp_attach: str,
    *,
    duration_sec: int = OPENMP_ATTACH_DEFAULT_DURATION_SEC,
    host_sleep_ms: int = 100,
) -> list[str]:
    """Launch openmp-attach (one process, stable PID) for live attach tests."""
    return [openmp_attach, str(duration_sec), str(host_sleep_ms)]


def hip_rocpd_attach_extra(
    rocprofv3: str,
    *,
    duration_msec: str = "5000",
) -> list[str]:
    """HIP + kernel trace + rocpd for live attach (attachment-test, simple-transpose, openmp-target)."""
    extra = [
        "--attach-duration-msec",
        duration_msec,
        "--hip-trace",
        "--kernel-trace",
        "-f",
        "rocpd",
    ]
    if rocprofv3_supports_attach_sync_output(rocprofv3):
        extra.append("--attach-sync-output")
    return extra


def _find_rocprofiler_register_library(rocprofv3: str) -> Path | None:
    """Locate librocprofiler-register under the ROCm prefix used by rocprofv3."""
    prefix = _rocm_install_prefix(rocprofv3)
    for directory in (prefix / "lib", prefix / "lib/rocprofiler-sdk"):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.glob("librocprofiler-register.so*")):
            if entry.is_file():
                return entry.resolve()
    return None


def _preload_env(
    *,
    extra: dict[str, str] | None = None,
    rocprofv3: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"
    preload_parts: list[str] = []
    if rocprofv3:
        register = _find_rocprofiler_register_library(rocprofv3)
        if register:
            preload_parts.append(str(register))
    for key in ("ROCPROF_PRELOAD", "LD_PRELOAD"):
        value = os.environ.get(key)
        if value:
            preload_parts.append(value)
    if preload_parts:
        env["LD_PRELOAD"] = ":".join(preload_parts)
    if extra:
        env.update(extra)
    return env


def _log(prefix: str, message: str) -> None:
    """Print a message with a stable source tag (one tag per line)."""
    for line in message.splitlines():
        print(f"{prefix} {line}", flush=True)


def _prefix_stream_to_console(stream, prefix: str) -> None:
    """Forward a subprocess stream to stdout with a fixed prefix on each line."""
    try:
        for line in stream:
            print(f"{prefix} {line}", end="", flush=True)
    finally:
        stream.close()


def _format_process_exit(proc: subprocess.Popen) -> str:
    rc = proc.returncode
    if rc is None:
        return "still running"
    if rc < 0:
        return f"terminated by signal {-rc}"
    return f"exit code {rc}"


def _terminate_process(proc: subprocess.Popen, *, grace_sec: float = 10) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def prepare_attachment_output(output_dir: str | Path) -> Path:
    """Create a clean attachment-output directory."""
    out_path = output_subdir(output_dir)
    shutil.rmtree(out_path, ignore_errors=True)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def skip_if_ptrace_unavailable(output_dir: str | Path) -> None:
    """Skip when YAMA/ptrace permissions block live attach (CTest SKIP regex)."""
    if ptrace_permissions_ok():
        return
    skip_marker_path(output_dir).touch()
    print(PTRACE_SKIP_MESSAGE, flush=True)
    pytest.skip(
        "ptrace_scope is not 0, user is not root, and CAP_SYS_PTRACE is not present"
    )


def launch_test_application(
    app_cmd: Sequence[str],
    env: dict[str, str],
) -> tuple[subprocess.Popen, threading.Thread]:
    """Start the GPU workload and stream its stdout/stderr with [APP] tags."""
    _log(LOG_TEST, f"Launching application: {' '.join(app_cmd)}")
    app_proc = subprocess.Popen(
        app_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert app_proc.stdout is not None
    log_thread = threading.Thread(
        target=_prefix_stream_to_console,
        args=(app_proc.stdout, LOG_APP),
        daemon=True,
        name="process-attachment-app-log",
    )
    log_thread.start()
    _log(LOG_TEST, f"Application pid={app_proc.pid} (stream tagged {LOG_APP})")
    return app_proc, log_thread


def _join_app_log_thread(log_thread: threading.Thread | None) -> None:
    if log_thread is not None and log_thread.is_alive():
        log_thread.join(timeout=10)


def wait_for_app_startup(app_proc: subprocess.Popen, startup_sec: float) -> None:
    """Let the application initialize before rocprofv3 attach."""
    time.sleep(startup_sec)
    if app_proc.poll() is not None:
        raise RuntimeError(
            "Test application failed to start or exited early "
            f"({_format_process_exit(app_proc)})"
        )


def _proc_maps_contains(pid: int, markers: Sequence[str]) -> bool:
    maps_path = Path(f"/proc/{pid}/maps")
    if not maps_path.is_file():
        return False
    try:
        content = maps_path.read_text(errors="replace")
    except OSError:
        return False
    return any(marker in content for marker in markers)


def wait_for_proc_maps_markers(
    pid: int,
    markers: Sequence[str],
    *,
    timeout_sec: float,
    label: str,
) -> None:
    """Block until /proc/<pid>/maps shows at least one marker (HIP ready for attach)."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if _proc_maps_contains(pid, markers):
            _log(LOG_TEST, f"{label} pid={pid} has HIP/register libraries loaded")
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"timed out waiting for {label} pid={pid} to load libraries matching {markers!r}"
    )


def _proc_argv0(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    if not raw:
        return ""
    return raw.split(b"\0", 1)[0].decode(errors="replace")


def _is_mpi_rank_binary(pid: int, binary_path: str) -> bool:
    """True when argv0 is the simple-transpose executable (not a bash -c wrapper)."""
    argv0 = _proc_argv0(pid)
    if not argv0:
        return False
    if argv0 == binary_path:
        return True
    return Path(argv0).name == Path(binary_path).name and binary_path in argv0


def resolve_child_attach_pid(
    parent_proc: subprocess.Popen,
    binary_path: str,
    *,
    map_markers: Sequence[str] = MPI_ATTACH_MAP_MARKERS,
    timeout_sec: float = MPI_ATTACH_HIP_READY_TIMEOUT_SEC,
    label: str = "child",
) -> int:
    """Return a child PID running binary_path (argv0) with GPU libraries loaded."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if parent_proc.poll() is not None:
            raise RuntimeError(
                f"{label} parent exited before {binary_path!r} was found for attach "
                f"({_format_process_exit(parent_proc)})"
            )
        try:
            proc = subprocess.run(
                ["pgrep", "-f", f"^{binary_path}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except OSError as exc:
            raise RuntimeError(f"pgrep failed while locating MPI ranks: {exc}") from exc

        for pid_str in proc.stdout.split():
            if not pid_str.strip().isdigit():
                continue
            pid = int(pid_str)
            if pid == parent_proc.pid:
                continue
            if not _is_mpi_rank_binary(pid, binary_path):
                continue
            maps_text = Path(f"/proc/{pid}/maps").read_text(errors="replace")
            if not any(marker in maps_text for marker in map_markers):
                continue
            if "rocprofiler-register" not in maps_text:
                continue
            _log(
                LOG_TEST,
                f"{label} attach target pid={pid} (parent pid={parent_proc.pid})",
            )
            return pid
        time.sleep(0.05)

    raise RuntimeError(
        f"timed out waiting for {label} running {binary_path!r} with HIP/register loaded"
    )


def resolve_mpi_rank_attach_pid(
    launcher_proc: subprocess.Popen,
    binary_path: str,
    *,
    timeout_sec: float = MPI_ATTACH_HIP_READY_TIMEOUT_SEC,
) -> int:
    return resolve_child_attach_pid(
        launcher_proc,
        binary_path,
        timeout_sec=timeout_sec,
        label="MPI rank",
    )


def mpi_attach_rocprofv3_extra(rocprofv3: str) -> list[str]:
    """HIP rocpd attach to mpiexec process tree (documented MPI attach workflow)."""
    return hip_rocpd_attach_extra(rocprofv3)


def openmp_attach_rocprofv3_extra(rocprofv3: str) -> list[str]:
    """HIP rocpd attach to openmp-attach main process (stable PID)."""
    return hip_rocpd_attach_extra(rocprofv3)


def build_attach_command(
    rocprofv3: str | Path,
    app_pid: int,
    out_path: Path,
    log_level: str,
    output_name: str,
    rocprofv3_extra: Sequence[str],
) -> list[str]:
    return [
        str(rocprofv3),
        "--attach",
        str(app_pid),
        "-d",
        str(out_path),
        "--log-level",
        log_level,
        "-o",
        output_name,
        *rocprofv3_extra,
    ]


def run_rocprofv3_attach(
    app_proc: subprocess.Popen,
    attach_cmd: Sequence[str],
    env: dict[str, str],
) -> str:
    """
    Run rocprofv3 --attach, stream its log, and fail if the app exits during attach.

    Returns the full attach log text.
    """
    _log(LOG_TEST, f"Starting rocprofv3 attach to application pid={app_proc.pid}")
    _log(LOG_TEST, f"Attach command: {' '.join(attach_cmd)}")
    _log(LOG_TEST, f"Attach process output is tagged {LOG_ATTACH}")
    attach_proc = subprocess.Popen(
        attach_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    attach_log_lines: list[str] = []
    try:
        assert attach_proc.stdout is not None
        for line in attach_proc.stdout:
            print(f"{LOG_ATTACH} {line}", end="", flush=True)
            attach_log_lines.append(line)
            if app_proc.poll() is not None:
                _log(
                    LOG_TEST,
                    "Application exited during attach "
                    f"(app {_format_process_exit(app_proc)})",
                )
                _terminate_process(attach_proc, grace_sec=30)
                raise RuntimeError(
                    "Test application exited or aborted during rocprofv3 attach"
                )
        attach_proc.wait()
        attach_log = "".join(attach_log_lines)
        _raise_if_attach_log_reports_failure(attach_log)
        _raise_if_process_failed(attach_proc, "rocprofv3")
        return attach_log
    except Exception:
        _terminate_process(attach_proc, grace_sec=30)
        raise


def wait_for_app_after_attach(
    app_proc: subprocess.Popen,
    wait_sec: float | None,
) -> None:
    """Wait for the application to finish after detach (optional timeout)."""
    if wait_sec is not None:
        try:
            app_proc.wait(timeout=wait_sec)
        except subprocess.TimeoutExpired:
            _terminate_process(app_proc, grace_sec=30)
    else:
        app_proc.wait()
    _raise_if_process_failed(app_proc, "Test application")


def cleanup_attachment_processes(
    app_proc: subprocess.Popen,
    attach_proc: subprocess.Popen | None,
    app_log_thread: threading.Thread | None = None,
) -> None:
    if attach_proc is not None:
        _terminate_process(attach_proc, grace_sec=10)
    _terminate_process(app_proc, grace_sec=10)
    _join_app_log_thread(app_log_thread)


def list_attachment_output(out_path: Path) -> None:
    _log(LOG_TEST, f"Attachment output directory: {out_path}")
    for entry in sorted(out_path.iterdir()):
        _log(LOG_TEST, f"  {entry.name}")


def run_process_attachment(
    *,
    rocprofv3: str,
    output_dir: str | Path,
    log_level: str,
    output_name: str,
    rocprofv3_extra: Sequence[str],
    app_cmd: Sequence[str],
    app_startup_sec: float = APP_STARTUP_SEC,
    app_wait_after_attach_sec: float | None = None,
    app_env: dict[str, str] | None = None,
    attach_target_binary: str | None = None,
) -> Path:
    """
    Launch a GPU test app, attach rocprofv3, wait for the app to finish.

    Returns the directory containing profiler output files.
    """
    output_dir = Path(output_dir)
    out_path = prepare_attachment_output(output_dir)
    _log(
        LOG_TEST,
        "Log tags: [TEST]=harness, [APP]=application under test, "
        "[ATTACH]=rocprofv3/rocattach",
    )
    skip_if_ptrace_unavailable(output_dir)

    rocprofv3_path = Path(rocprofv3)
    if not rocprofv3_path.is_file():
        raise FileNotFoundError(f"rocprofv3 not found: {rocprofv3}")

    env = _preload_env(extra=app_env, rocprofv3=rocprofv3)
    app_proc, app_log_thread = launch_test_application(app_cmd, env)
    try:
        if attach_target_binary:
            wait_for_app_startup(app_proc, app_startup_sec)
            last_error: Exception | None = None
            for attempt in range(ATTACH_TARGET_MAX_ATTEMPTS):
                try:
                    attach_pid = resolve_child_attach_pid(
                        app_proc,
                        attach_target_binary,
                        label="GPU workload",
                    )
                    attach_cmd = build_attach_command(
                        rocprofv3_path,
                        attach_pid,
                        out_path,
                        log_level,
                        output_name,
                        rocprofv3_extra,
                    )
                    run_rocprofv3_attach(app_proc, attach_cmd, env)
                    last_error = None
                    break
                except RuntimeError as exc:
                    last_error = exc
                    _log(
                        LOG_TEST,
                        f"Attach attempt {attempt + 1}/{ATTACH_TARGET_MAX_ATTEMPTS} failed: {exc}",
                    )
            if last_error is not None:
                raise last_error
        else:
            wait_for_app_startup(app_proc, app_startup_sec)
            attach_pid = app_proc.pid
            attach_cmd = build_attach_command(
                rocprofv3_path,
                attach_pid,
                out_path,
                log_level,
                output_name,
                rocprofv3_extra,
            )
            run_rocprofv3_attach(app_proc, attach_cmd, env)
        wait_for_app_after_attach(app_proc, app_wait_after_attach_sec)
        _log(LOG_TEST, f"Application finished ({_format_process_exit(app_proc)})")
    except Exception:
        execute_failed_marker_path(output_dir).touch()
        cleanup_attachment_processes(app_proc, None, app_log_thread)
        raise

    _join_app_log_thread(app_log_thread)
    list_attachment_output(out_path)
    return out_path


def require_options(config, names: Iterable[str]) -> dict[str, str]:
    values = {}
    for name in names:
        value = config.getoption(name)
        if not value:
            pytest.fail(f"pytest option --{name.replace('_', '-')} is required")
        values[name] = value
    return values
