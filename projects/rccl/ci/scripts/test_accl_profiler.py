#!/usr/bin/env python3
"""Run the ACCL profiler against five RCCL collectives on a Slurm cluster.

The script consumes a staged TheRock install containing RCCL, rccl-tests, the
ACCL profiler, and accl_report.py. It submits one allocation, runs the five
collectives sequentially, validates every JSONL file, and writes a manifest and
human-readable decomposition report.

AllToAll is intentionally excluded. Its current fallback is represented as
grouped P2P Send/Recv tasks, while the ACCL plugin currently correlates only
collective event parents. That work belongs in a separate RCCL/profiler ticket.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


COLLECTIVES = {
    "all_reduce_perf": "AllReduce",
    "reduce_scatter_perf": "ReduceScatter",
    "all_gather_perf": "AllGather",
    "reduce_perf": "Reduce",
    "broadcast_perf": "Broadcast",
}

DECOMPOSITION_FIELDS = {
    "enqueue_to_kernel_us",
    "gpu_kernel_avg_us",
    "gpu_kernel_min_us",
    "gpu_kernel_max_us",
    "proxy_gpu_wait_us",
    "proxy_network_us",
    "proxy_peer_wait_us",
    "proxy_flush_us",
    "proxy_gpu_recv_wait_us",
    "n_proxy_ops",
    "n_send_ops",
    "n_recv_ops",
}

RUBY_RCCL_ENV = {
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_HCA": (
        "bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,"
        "bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1"
    ),
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_TC": "104",
    "NCCL_IB_QPS_PER_CONNECTION": "4",
    "NCCL_SOCKET_IFNAME": "fenic0",
    "HSA_NO_SCRATCH_RECLAIM": "1",
}


def _raise_keyboard_interrupt(signum, _frame) -> None:
    """Turn runner termination into the cancellation path for the Slurm job."""
    raise KeyboardInterrupt(f"received signal {signum}")


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    rccl_library: Path
    profiler_plugin: Path
    report_script: Path
    binaries: dict[str, Path]
    kpack_files: tuple[Path, ...]


@dataclass(frozen=True)
class RunConfig:
    nodes: int = 2
    gpus_per_node: int = 8
    partition: str = "meta64"
    constraint: str = ""
    timeout_minutes: int = 90
    min_bytes: str = "1K"
    max_bytes: str = "256M"
    step_factor: int = 2
    iterations: int = 20
    warmup_iterations: int = 5

    @property
    def ranks(self) -> int:
        return self.nodes * self.gpus_per_node


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_artifact(root: Path, relative_path: str, name: str) -> Path:
    preferred = root / relative_path
    if preferred.is_file():
        return preferred.resolve()

    matches = sorted({path.resolve() for path in root.rglob(name) if path.is_file()})
    if not matches:
        raise FileNotFoundError(f"{name} not found under artifact tree {root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous artifact {name}; expected {preferred}, found: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def discover_artifacts(artifact_dir: Path) -> ArtifactPaths:
    root = artifact_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {root}")

    binaries = {
        binary: _find_artifact(root, f"bin/{binary}", binary)
        for binary in COLLECTIVES
    }
    return ArtifactPaths(
        root=root,
        rccl_library=_find_artifact(root, "lib/librccl.so", "librccl.so"),
        profiler_plugin=_find_artifact(
            root, "lib/librccl-profiler-accl.so", "librccl-profiler-accl.so"
        ),
        report_script=_find_artifact(
            root, "share/rccl/accl/accl_report.py", "accl_report.py"
        ),
        binaries=binaries,
        kpack_files=tuple(
            sorted(path.resolve() for path in root.rglob("*.kpack") if path.is_file())
        ),
    )


def _shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def validate_kpack_layout(paths: ArtifactPaths) -> None:
    """Require both RCCL and rccl-tests archives in a split artifact tree.

    Each kpack-stripped ELF contains its own .rocm_kpack_ref that selects the
    matching archive. Do not combine the archives in ROCM_KPACK_PATH: that
    global override can make an architecture-compatible RCCL archive shadow
    the rccl-tests archive (or vice versa), resulting in hipErrorInvalidImage.
    """
    if not paths.kpack_files:
        return

    names = {path.name for path in paths.kpack_files}
    required = {"rccl_lib_gfx950.kpack", "rccl_test_gfx950.kpack"}
    missing = sorted(required - names)
    if missing:
        raise FileNotFoundError(
            "TheRock kpack artifact tree is incomplete; missing: "
            + ", ".join(missing)
        )


def render_slurm_script(
    paths: ArtifactPaths,
    work_dir: Path,
    config: RunConfig,
) -> str:
    validate_kpack_layout(paths)
    raw_dir = work_dir / "raw"
    test_log_dir = work_dir / "rccl-tests"
    status_file = work_dir / "collective-status.tsv"
    preflight_log = work_dir / "preflight.log"
    lib_dir = paths.rccl_library.parent
    sysdeps_dir = paths.root / "lib" / "rocm_sysdeps" / "lib"
    ld_paths = [str(lib_dir)]
    if sysdeps_dir.is_dir():
        ld_paths.append(str(sysdeps_dir.resolve()))

    # The GitHub runner and the Ruby compute nodes use different Linux
    # distributions. In particular, setup-python adds a Python toolcache to
    # PATH and LD_LIBRARY_PATH that requires a newer glibc than the compute
    # nodes provide. Keep the runner environment for this Python driver, but
    # give the Slurm payload a node-local PATH and only the fetched artifact
    # libraries it needs.
    path_entries = [
        str(paths.root / "bin"),
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]

    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        (
            "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV "
            "ROCM_KPACK_PATH ROCM_KPACK_PATH_PREFIX || true"
        ),
        _shell_export("PATH", ":".join(path_entries)),
        _shell_export("LD_LIBRARY_PATH", ":".join(ld_paths)),
        _shell_export("ROCM_PATH", str(paths.root)),
        _shell_export("HIP_PATH", str(paths.root)),
        _shell_export("NCCL_PROFILER_PLUGIN", str(paths.profiler_plugin)),
        _shell_export("ACCL_PROFILER_MIN_SIZE_BYTES", "0"),
        _shell_export("ACCL_RCCL_LIB", str(paths.rccl_library)),
        _shell_export("ACCL_PLUGIN", str(paths.profiler_plugin)),
        _shell_export("ACCL_PREFLIGHT_BINARY", str(paths.binaries["all_reduce_perf"])),
        _shell_export(
            "ACCL_TEST_BINARIES",
            " ".join(str(paths.binaries[name]) for name in COLLECTIVES),
        ),
        _shell_export(
            "ACCL_KPACK_FILES", " ".join(str(path) for path in paths.kpack_files)
        ),
        _shell_export("ACCL_KPACK_COUNT", str(len(paths.kpack_files))),
        _shell_export("ACCL_EXPECT_RCCL_SHA256", sha256_file(paths.rccl_library)),
        _shell_export("ACCL_EXPECT_PLUGIN_SHA256", sha256_file(paths.profiler_plugin)),
    ]
    lines.extend(_shell_export(name, value) for name, value in RUBY_RCCL_ENV.items())
    lines.extend([
        f"mkdir -p {shlex.quote(str(raw_dir))} {shlex.quote(str(test_log_dir))}",
        f": > {shlex.quote(str(status_file))}",
        "preflight_rc=0",
        (
            f"srun --nodes={config.nodes} --ntasks={config.nodes} --ntasks-per-node=1 "
            "--kill-on-bad-exit=1 bash -c '"
        ),
        "set -eu",
        "host=$(hostname -s)",
        "resolved_rccl=$(ldd \"$ACCL_PREFLIGHT_BINARY\" | awk '\"'\"'/librccl[.]so/{print $3; exit}'\"'\"')",
        "test -n \"$resolved_rccl\"",
        "test \"$(readlink -f \"$resolved_rccl\")\" = \"$(readlink -f \"$ACCL_RCCL_LIB\")\"",
        "resolved_hip=$(ldd \"$ACCL_PREFLIGHT_BINARY\" | awk '\"'\"'/libamdhip64[.]so/{print $3; exit}'\"'\"')",
        "test -n \"$resolved_hip\"",
        (
            "case \"$(readlink -f \"$resolved_hip\")\" in "
            "\"$ROCM_PATH\"/*) ;; *) echo \"unexpected HIP runtime: $resolved_hip\" >&2; exit 1 ;; esac"
        ),
        "test \"$(sha256sum \"$ACCL_RCCL_LIB\" | awk '\"'\"'{print $1}'\"'\"')\" = \"$ACCL_EXPECT_RCCL_SHA256\"",
        "test \"$(sha256sum \"$ACCL_PLUGIN\" | awk '\"'\"'{print $1}'\"'\"')\" = \"$ACCL_EXPECT_PLUGIN_SHA256\"",
        "readelf -Ws \"$ACCL_PLUGIN\" | grep -q '\"'\"' ncclProfiler_v5$'\"'\"'",
        "if [ \"$ACCL_KPACK_COUNT\" -gt 0 ]; then",
        "  for kpack in $ACCL_KPACK_FILES; do test -s \"$kpack\"; done",
        "  readelf -SW \"$ACCL_RCCL_LIB\" | grep -q '\"'\"'[.]rocm_kpack_ref'\"'\"'",
        "  for binary in $ACCL_TEST_BINARIES; do",
        "    readelf -SW \"$binary\" | grep -q '\"'\"'[.]rocm_kpack_ref'\"'\"'",
        "  done",
        "else",
        "  if readelf -SW \"$ACCL_RCCL_LIB\" | grep -q '\"'\"'[.]rocm_kpack_ref'\"'\"'; then",
        "    echo \"librccl.so is kpack-stripped but no kpack archives were fetched\" >&2",
        "    exit 1",
        "  fi",
        "  for binary in $ACCL_TEST_BINARIES; do",
        "    if readelf -SW \"$binary\" | grep -q '\"'\"'[.]rocm_kpack_ref'\"'\"'; then",
        "      echo \"$binary is kpack-stripped but no kpack archives were fetched\" >&2",
        "      exit 1",
        "    fi",
        "  done",
        "fi",
        # Prefer the native rocminfo executable. rocm_agent_enumerator is a
        # Python script, so use an explicit node-local interpreter only as a
        # fallback instead of its /usr/bin/env shebang.
        "if [ -x \"$ROCM_PATH/bin/rocminfo\" ]; then",
        (
            "  archs=$(\"$ROCM_PATH/bin/rocminfo\" 2>/dev/null | sed -n "
            "'\"'\"'s/.*Name:[[:space:]]*\\(gfx[0-9a-f]*\\).*/\\1/p'\"'\"' "
            "| sort -u | paste -sd, -)"
        ),
        "elif [ -x /usr/bin/python3 ] && [ -f \"$ROCM_PATH/bin/rocm_agent_enumerator\" ]; then",
        "  archs=$(/usr/bin/python3 \"$ROCM_PATH/bin/rocm_agent_enumerator\" | sort -u | paste -sd, -)",
        "elif [ -x /usr/local/bin/python3 ] && [ -f \"$ROCM_PATH/bin/rocm_agent_enumerator\" ]; then",
        "  archs=$(/usr/local/bin/python3 \"$ROCM_PATH/bin/rocm_agent_enumerator\" | sort -u | paste -sd, -)",
        "elif command -v rocminfo >/dev/null 2>&1; then",
        (
            "  archs=$(rocminfo 2>/dev/null | sed -n "
            "'\"'\"'s/.*Name:[[:space:]]*\\(gfx[0-9a-f]*\\).*/\\1/p'\"'\"' "
            "| sort -u | paste -sd, -)"
        ),
        "else",
        "  archs=",
        "fi",
        "case \",$archs,\" in *,gfx950,*) ;; *) echo \"unexpected GPU architecture(s): $archs\" >&2; exit 1 ;; esac",
        (
            "printf '\"'\"'host=%s arch=%s rccl=%s plugin=%s hip=%s kpacks=%s\\n'\"'\"' "
            "\"$host\" \"$archs\" \"$ACCL_EXPECT_RCCL_SHA256\" "
            "\"$ACCL_EXPECT_PLUGIN_SHA256\" \"$(readlink -f \"$resolved_hip\")\" "
            "\"$ACCL_KPACK_COUNT\""
        ),
        "' "
        f"> {shlex.quote(str(preflight_log))} 2>&1 || preflight_rc=$?",
        "if [ \"$preflight_rc\" -ne 0 ]; then",
        f"  printf '%s\\t%s\\n' preflight \"$preflight_rc\" >> {shlex.quote(str(status_file))}",
    ])
    for binary in COLLECTIVES:
        lines.append(
            f"  printf '%s\\t%s\\n' {shlex.quote(binary)} 125 >> "
            f"{shlex.quote(str(status_file))}"
        )
    lines.extend([
        "  exit \"$preflight_rc\"",
        "fi",
        f"printf '%s\\t%s\\n' preflight 0 >> {shlex.quote(str(status_file))}",
        "overall_rc=0",
    ])

    for index, binary in enumerate(COLLECTIVES):
        output_dir = raw_dir / binary
        log_file = test_log_dir / f"{binary}.log"
        debug_level = "INFO" if index == 0 else "WARN"
        command = [
            "srun",
            f"--nodes={config.nodes}",
            f"--ntasks={config.ranks}",
            f"--ntasks-per-node={config.gpus_per_node}",
            "--kill-on-bad-exit=0",
            str(paths.binaries[binary]),
            "-b", config.min_bytes,
            "-e", config.max_bytes,
            "-f", str(config.step_factor),
            "-g", "1",
            "-n", str(config.iterations),
            "-w", str(config.warmup_iterations),
        ]
        lines.extend([
            f"mkdir -p {shlex.quote(str(output_dir))}",
            _shell_export("ACCL_PROFILER_OUTPUT_DIR", str(output_dir)),
            _shell_export("NCCL_DEBUG", debug_level),
            f"echo 'Running {binary} with ACCL profiler'",
            f"{' '.join(shlex.quote(part) for part in command)} 2>&1 | tee {shlex.quote(str(log_file))}",
            "collective_rc=${PIPESTATUS[0]}",
            (
                f"printf '%s\\t%s\\n' {shlex.quote(binary)} \"$collective_rc\" >> "
                f"{shlex.quote(str(status_file))}"
            ),
            "if [ \"$collective_rc\" -ne 0 ]; then overall_rc=1; fi",
        ])
    lines.extend(["exit \"$overall_rc\"", ""])
    return "\n".join(lines)


def submit_slurm_job(script_path: Path, work_dir: Path, config: RunConfig) -> str:
    command = [
        "sbatch",
        "--parsable",
        "--job-name=rccl-accl-profiler",
        f"--partition={config.partition}",
        f"--nodes={config.nodes}",
        f"--ntasks={config.ranks}",
        f"--ntasks-per-node={config.gpus_per_node}",
        "--exclusive",
        f"--time={config.timeout_minutes}:00",
        f"--output={work_dir / 'slurm' / 'accl-%j.out'}",
        f"--error={work_dir / 'slurm' / 'accl-%j.err'}",
    ]
    if config.constraint:
        command.append(f"--constraint={config.constraint}")
    command.append(str(script_path))
    log.info("Submitting: %s", shlex.join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse Slurm job ID from: {result.stdout!r}")
    (work_dir / "slurm" / "job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    return job_id


def cancel_slurm_job(job_id: str) -> None:
    if not job_id:
        return
    log.warning("Cancelling Slurm job %s", job_id)
    subprocess.run(["scancel", job_id], check=False)


def wait_for_slurm_job(job_id: str, timeout_minutes: int) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_minutes * 60
    previous_state = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
            capture_output=True,
            text=True,
            check=False,
        )
        state = result.stdout.strip().splitlines()
        if not state:
            break
        current_state = state[0]
        if current_state != previous_state:
            log.info("Slurm job %s state: %s", job_id, current_state)
            previous_state = current_state
        time.sleep(10)
    else:
        cancel_slurm_job(job_id)
        raise TimeoutError(f"Slurm job {job_id} exceeded {timeout_minutes} minutes")

    for _ in range(6):
        result = subprocess.run(
            [
                "sacct", "--noheader", "--parsable2", "--allocations",
                "--jobs", job_id, "--format=State,ExitCode",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = [line for line in result.stdout.splitlines() if line.strip()]
        if rows:
            fields = rows[0].split("|")
            return fields[0], fields[1] if len(fields) > 1 else ""
        time.sleep(2)
    return "UNKNOWN", ""


def parse_collective_status(status_file: Path) -> dict[str, int]:
    statuses: dict[str, int] = {}
    if not status_file.is_file():
        return statuses
    for line in status_file.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            continue
        try:
            statuses[fields[0]] = int(fields[1])
        except ValueError:
            continue
    return statuses


def validate_collective_output(
    output_dir: Path,
    expected_collective: str,
    expected_ranks: int,
    warmup_iterations: int,
) -> dict:
    files = sorted(output_dir.glob("*.jsonl")) if output_dir.is_dir() else []
    if not files:
        raise ValueError(f"No profiler JSONL files in {output_dir}")

    rank_sizes: dict[int, set[int]] = {}
    sample_counts: dict[tuple[int, int], int] = {}
    record_count = 0

    for path in files:
        if path.stat().st_size == 0:
            raise ValueError(f"Profiler output is empty: {path}")
        objects = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_number}: {exc}") from exc

        if not objects:
            raise ValueError(f"Profiler output has no JSON objects: {path}")
        summaries = [obj["summary"] for obj in objects if "summary" in obj]
        if len(summaries) != 1 or "summary" not in objects[-1]:
            raise ValueError(f"Expected one final summary in {path}")
        summary = summaries[0]
        if (
            summary.get("complete") is not True
            or summary.get("dropped_collectives", 0) != 0
            or summary.get("leaked_collectives", 0) != 0
        ):
            raise ValueError(f"Incomplete profiler summary in {path}: {summary}")

        file_records = 0
        for obj in objects:
            coll_perf = obj.get("coll_perf")
            if coll_perf is None:
                continue
            header = obj.get("header")
            if not isinstance(header, dict):
                raise ValueError(f"Missing header in collective record from {path}")
            if coll_perf.get("coll") != expected_collective:
                raise ValueError(
                    f"Unexpected collective in {path}: {coll_perf.get('coll')!r}; "
                    f"expected {expected_collective!r}"
                )
            rank = header.get("rank")
            n_ranks = header.get("n_ranks")
            size = coll_perf.get("coll_msg_size_bytes")
            if not isinstance(rank, int) or not isinstance(size, int):
                raise ValueError(f"Invalid rank or message size in {path}")
            if n_ranks != expected_ranks:
                raise ValueError(
                    f"Rank {rank} in {path} reports n_ranks={n_ranks}; "
                    f"expected {expected_ranks}"
                )
            decomposition = coll_perf.get("decomposition")
            if not isinstance(decomposition, dict) or not DECOMPOSITION_FIELDS.issubset(decomposition):
                missing = DECOMPOSITION_FIELDS - set(decomposition or {})
                raise ValueError(f"Missing decomposition fields in {path}: {sorted(missing)}")
            kernel_events = obj.get("event_trace_ts", {}).get("kernel_events")
            if not isinstance(kernel_events, list) or not kernel_events:
                raise ValueError(f"Missing kernel events in {path}")

            rank_sizes.setdefault(rank, set()).add(size)
            key = (rank, size)
            sample_counts[key] = sample_counts.get(key, 0) + 1
            file_records += 1
            record_count += 1
        if file_records == 0:
            raise ValueError(f"No collective records in {path}")

    expected_rank_set = set(range(expected_ranks))
    observed_rank_set = set(rank_sizes)
    if observed_rank_set != expected_rank_set:
        raise ValueError(
            f"Rank coverage mismatch in {output_dir}: observed={sorted(observed_rank_set)}, "
            f"expected={sorted(expected_rank_set)}"
        )

    reference_sizes = rank_sizes[0]
    for rank, sizes in rank_sizes.items():
        if sizes != reference_sizes:
            raise ValueError(
                f"Message-size coverage differs for rank {rank}: "
                f"observed={sorted(sizes)}, expected={sorted(reference_sizes)}"
            )
    too_short = {
        f"rank={rank},size={size}": count
        for (rank, size), count in sample_counts.items()
        if count <= warmup_iterations
    }
    if too_short:
        raise ValueError(
            "No post-warmup sample for one or more rank/size groups: "
            + json.dumps(too_short, sort_keys=True)
        )

    return {
        "files": len(files),
        "records": record_count,
        "ranks": sorted(observed_rank_set),
        "message_sizes": sorted(reference_sizes),
        "minimum_samples_per_rank_size": min(sample_counts.values()),
    }


def generate_reports(paths: ArtifactPaths, work_dir: Path, warmup: int) -> list[str]:
    errors = []
    report_dir = work_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    combined_parts = []
    for binary in COLLECTIVES:
        raw_dir = work_dir / "raw" / binary
        output = report_dir / f"{binary}.txt"
        if not raw_dir.is_dir():
            errors.append(f"{binary}: raw output directory is missing")
            continue
        result = subprocess.run(
            [
                sys.executable,
                str(paths.report_script),
                "single",
                "--input", str(raw_dir),
                "--warmup", str(warmup),
                "--output", str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            errors.append(
                f"{binary}: accl_report.py failed with {result.returncode}: "
                f"{result.stderr.strip()}"
            )
            continue
        combined_parts.append(f"### {binary}\n\n{output.read_text(encoding='utf-8')}")
    (work_dir / "accl_report.txt").write_text(
        "\n\n".join(combined_parts), encoding="utf-8"
    )
    return errors


def build_manifest(
    paths: ArtifactPaths,
    config: RunConfig,
    statuses: dict[str, int],
    validation: dict[str, dict],
    errors: list[str],
    job_id: str,
    slurm_state: str,
    slurm_exit_code: str,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "repository_sha": os.environ.get("GITHUB_SHA", ""),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "artifact_run_id": os.environ.get("ARTIFACT_RUN_ID", ""),
        "slurm": {
            "job_id": job_id,
            "state": slurm_state,
            "exit_code": slurm_exit_code,
            "partition": config.partition,
            "nodes": config.nodes,
            "gpus_per_node": config.gpus_per_node,
            "ranks": config.ranks,
        },
        "artifacts": {
            "root": str(paths.root),
            "librccl": str(paths.rccl_library),
            "librccl_sha256": sha256_file(paths.rccl_library),
            "profiler": str(paths.profiler_plugin),
            "profiler_sha256": sha256_file(paths.profiler_plugin),
            "report_script": str(paths.report_script),
            "kpacks": [
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in paths.kpack_files
            ],
        },
        "sweep": {
            "min_bytes": config.min_bytes,
            "max_bytes": config.max_bytes,
            "step_factor": config.step_factor,
            "iterations": config.iterations,
            "warmup_iterations": config.warmup_iterations,
        },
        "collectives": {
            binary: {
                "expected_name": COLLECTIVES[binary],
                "exit_code": statuses.get(binary),
                "validation": validation.get(binary),
            }
            for binary in COLLECTIVES
        },
        "deferred": {
            "alltoall_perf": (
                "Requires ACCL support for ncclProfileP2p parent correlation; "
                "tracked separately from AICOMRCCL-2172."
            )
        },
        "environment": RUBY_RCCL_ENV,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--partition", default="meta64")
    parser.add_argument("--constraint", default="")
    parser.add_argument("--timeout-minutes", type=int, default=90)
    parser.add_argument("--min-bytes", default="1K")
    parser.add_argument("--max-bytes", default="256M")
    parser.add_argument("--step-factor", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the Slurm script after artifact discovery without submitting it",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)

    if args.nodes < 1 or args.gpus_per_node < 1:
        parser.error("--nodes and --gpus-per-node must be positive")
    config = RunConfig(
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        partition=args.partition,
        constraint=args.constraint,
        timeout_minutes=args.timeout_minutes,
        min_bytes=args.min_bytes,
        max_bytes=args.max_bytes,
        step_factor=args.step_factor,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )

    paths = discover_artifacts(args.artifact_dir)
    work_dir = args.work_dir.resolve()
    (work_dir / "slurm").mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "run-accl-profiler.sbatch"
    script_path.write_text(
        render_slurm_script(paths, work_dir, config), encoding="utf-8"
    )
    script_path.chmod(0o755)
    if args.dry_run:
        log.info("Dry run complete; Slurm script written to %s", script_path)
        return 0

    job_id = ""
    slurm_state = "NOT_SUBMITTED"
    slurm_exit_code = ""
    errors: list[str] = []
    try:
        job_id = submit_slurm_job(script_path, work_dir, config)
        log.info("Submitted Slurm job %s", job_id)
        slurm_state, slurm_exit_code = wait_for_slurm_job(
            job_id, config.timeout_minutes + 5
        )
        log.info(
            "Slurm job %s completed: state=%s exit_code=%s",
            job_id, slurm_state, slurm_exit_code,
        )
    except (KeyboardInterrupt, SystemExit):
        cancel_slurm_job(job_id)
        raise
    except Exception as exc:  # Preserve artifacts and manifest on infrastructure errors.
        cancel_slurm_job(job_id)
        slurm_state = "ERROR"
        errors.append(str(exc))

    statuses = parse_collective_status(work_dir / "collective-status.tsv")
    if statuses.get("preflight") != 0:
        errors.append(f"preflight failed or did not run: {statuses.get('preflight')}")

    validation = {}
    for binary, expected_name in COLLECTIVES.items():
        if statuses.get(binary) != 0:
            errors.append(f"{binary} exit code: {statuses.get(binary)}")
        try:
            validation[binary] = validate_collective_output(
                work_dir / "raw" / binary,
                expected_name,
                config.ranks,
                config.warmup_iterations,
            )
        except (OSError, ValueError) as exc:
            errors.append(f"{binary}: {exc}")

    errors.extend(generate_reports(paths, work_dir, config.warmup_iterations))
    if not slurm_state.startswith("COMPLETED"):
        errors.append(
            f"Slurm allocation did not complete successfully: "
            f"state={slurm_state}, exit_code={slurm_exit_code}"
        )

    manifest = build_manifest(
        paths,
        config,
        statuses,
        validation,
        errors,
        job_id,
        slurm_state,
        slurm_exit_code,
    )
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if errors:
        for error in errors:
            log.error("%s", error)
        return 1
    log.info("All five ACCL-profiler collective runs passed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
