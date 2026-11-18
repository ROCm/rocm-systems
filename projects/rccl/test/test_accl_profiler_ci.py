"""Unit tests for the Ruby ACCL-profiler CI driver."""

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from accl_test_records import make_record as _record

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ci", "scripts"))
import test_accl_profiler as accl_ci  # noqa: E402
from test_accl_profiler import (  # noqa: E402
    ArtifactPaths,
    COLLECTIVES,
    RunConfig,
    discover_artifacts,
    parse_collective_status,
    render_node_preflight_script,
    render_slurm_script,
    submit_slurm_job,
    validate_collective_output,
    wait_for_slurm_job,
)
from plan_rccl_ci import plan  # noqa: E402


def _artifact_paths(tmp_path: Path, kpack_names: tuple[str, ...] = ()) -> ArtifactPaths:
    files = {
        "rccl_library": tmp_path / "librccl.so",
        "profiler_plugin": tmp_path / "librccl-profiler-accl.so",
        "report_script": tmp_path / "accl_report.py",
    }
    for path in files.values():
        path.write_text("fixture", encoding="utf-8")
    binaries = {}
    for binary in COLLECTIVES:
        path = tmp_path / binary
        path.write_text("fixture", encoding="utf-8")
        binaries[binary] = path
    kpack_dir = tmp_path / ".kpack"
    kpack_files = []
    for name in kpack_names:
        kpack_dir.mkdir(exist_ok=True)
        path = kpack_dir / name
        path.write_text("fixture", encoding="utf-8")
        kpack_files.append(path)
    return ArtifactPaths(
        root=tmp_path,
        rccl_library=files["rccl_library"],
        profiler_plugin=files["profiler_plugin"],
        report_script=files["report_script"],
        binaries=binaries,
        kpack_files=tuple(kpack_files),
    )


def _run_config(tmp_path: Path, **kwargs) -> RunConfig:
    mpi_root = tmp_path / "openmpi"
    (mpi_root / "bin").mkdir(parents=True, exist_ok=True)
    (mpi_root / "lib").mkdir()
    launcher = mpi_root / "bin" / "mpirun"
    launcher.write_text("fixture", encoding="utf-8")
    launcher.chmod(0o755)
    return RunConfig(mpi_root=mpi_root, **kwargs)


def _write_rank_file(
    output_dir: Path,
    rank: int,
    coll: str = "AllReduce",
    complete: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    for size in (1024, 2048):
        objects.extend(_record(rank, size, coll) for _ in range(3))
    objects.append(
        {
            "summary": {
                "dropped_collectives": 0 if complete else 1,
                "leaked_collectives": 0,
                # The CI reader does not depend on the pool-size key. Use the
                # reporter schema from #11881; the older reporter also reads it.
                "coll_pool_size": 256,
                "complete": complete,
            }
        }
    )
    path = output_dir / f"rank{rank}.jsonl"
    path.write_text(
        "\n".join(json.dumps(obj) for obj in objects) + "\n", encoding="utf-8"
    )


def test_slurm_script_runs_only_five_supported_collectives(tmp_path):
    paths = _artifact_paths(tmp_path)
    script = render_slurm_script(
        paths, tmp_path / "work", _run_config(tmp_path, nodes=2)
    )
    for binary in COLLECTIVES:
        assert binary in script
    assert "alltoall_perf" not in script
    assert "srun --nodes=2 --ntasks=2 --ntasks-per-node=1" in script
    assert "--ntasks=16" not in script
    expected_mpi = f"{tmp_path}/openmpi/bin/mpirun --prefix {tmp_path}/openmpi -np 16"
    assert expected_mpi in script
    assert '--host "$ACCL_MPI_HOSTS"' in script
    assert "--mca pml ob1 --mca btl '^vader,openib'" in script
    assert "plm_rsh_args" in script
    assert "ssh -p 2224" not in script
    assert "oob_tcp_if_include" not in script
    assert "ulimit -l unlimited" in script
    assert f"export NCCL_PROFILER_PLUGIN={paths.profiler_plugin}" in script


def test_slurm_script_validates_mpi_rank_placement(tmp_path):
    paths = _artifact_paths(tmp_path)
    script = render_slurm_script(
        paths, tmp_path / "work", _run_config(tmp_path, nodes=2)
    )

    assert 'scontrol show hostnames "$SLURM_JOB_NODELIST"' in script
    assert "${host}:${ACCL_GPUS_PER_NODE}" in script
    assert '"${OMPI_COMM_WORLD_SIZE:-}" = "$ACCL_EXPECT_RANKS"' in script
    assert '"${OMPI_COMM_WORLD_LOCAL_SIZE:-}" = "$ACCL_GPUS_PER_NODE"' in script
    assert "-x NCCL_PROFILER_PLUGIN" in script
    assert "-x ACCL_PROFILER_OUTPUT_DIR" in script
    assert "NCCL_IGNORE_CPU_AFFINITY" in script
    assert 'set -e; test "${OMPI_COMM_WORLD_SIZE:-}"' in script
    assert (
        f"export ACCL_PROFILER_OUTPUT_DIR={tmp_path}/work/raw/all_reduce_perf" in script
    )


def test_rendered_slurm_script_has_valid_bash_syntax(tmp_path):
    paths = _artifact_paths(tmp_path)
    script_path = tmp_path / "run.sbatch"
    script_path.write_text(
        render_slurm_script(paths, tmp_path / "work", _run_config(tmp_path)),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", "-n", str(script_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

    result = subprocess.run(
        ["bash", "-n"],
        input=render_node_preflight_script(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_slurm_script_does_not_inherit_runner_python_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PATH",
        "/apps/actions-runner/_work/_tool/Python/3.12.14/x64/bin:/usr/bin",
    )
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/apps/actions-runner/_work/_tool/Python/3.12.14/x64/lib",
    )

    script = render_slurm_script(
        _artifact_paths(tmp_path), tmp_path / "work", _run_config(tmp_path)
    )

    assert "_tool/Python" not in script
    assert (
        f"export PATH={tmp_path}/openmpi/bin:{tmp_path}/bin:"
        "/usr/bin:/bin:/usr/sbin:/sbin" in script
    )
    assert f"export LD_LIBRARY_PATH={tmp_path}" in script
    assert (
        "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV "
        "ROCM_KPACK_PATH ROCM_KPACK_PATH_PREFIX" in script
    )
    for variable in (
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_RUNTIME_URL",
        "ACTIONS_RESULTS_URL",
    ):
        assert variable in script
    assert 'rocminfo_bin="$ROCM_PATH/bin/rocminfo"' in script
    assert '"$python_bin" "$ROCM_PATH/bin/rocm_agent_enumerator"' in script


def test_slurm_script_uses_embedded_kpack_references(tmp_path):
    paths = _artifact_paths(
        tmp_path,
        ("rccl_lib_gfx950.kpack", "rccl_test_gfx950.kpack"),
    )

    script = render_slurm_script(paths, tmp_path / "work", _run_config(tmp_path))

    assert "export ROCM_KPACK_PATH=" not in script
    assert "export ROCM_KPACK_PATH_PREFIX=" not in script
    assert "rccl_lib_gfx950.kpack" in script
    assert "rccl_test_gfx950.kpack" in script
    assert 'has_kpack_ref "$ACCL_RCCL_LIB"' in script
    assert 'has_kpack_ref "$binary"' in script
    assert 'resolved_hip=$(ldd "$ACCL_PREFLIGHT_BINARY"' in script
    assert '"$ROCM_PATH"/*' in script


def test_slurm_script_rejects_incomplete_kpack_layout(tmp_path):
    paths = _artifact_paths(tmp_path, ("rccl_lib_gfx950.kpack",))

    with pytest.raises(FileNotFoundError, match="rccl_test_gfx950.kpack"):
        render_slurm_script(paths, tmp_path / "work", _run_config(tmp_path))


def test_discover_artifacts_prefers_packaged_layout(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "share" / "rccl" / "accl").mkdir(parents=True)
    (tmp_path / "lib" / "librccl.so").write_text("rccl", encoding="utf-8")
    plugin = tmp_path / "lib" / "librccl-profiler-accl.so"
    plugin.write_text("plugin", encoding="utf-8")
    report = tmp_path / "share" / "rccl" / "accl" / "accl_report.py"
    report.write_text("report", encoding="utf-8")
    # A preferred package location wins over a duplicate elsewhere in the tree.
    (tmp_path / "duplicate").mkdir()
    (tmp_path / "duplicate" / plugin.name).write_text("other plugin")
    for binary in COLLECTIVES:
        (tmp_path / "bin" / binary).write_text(binary, encoding="utf-8")

    paths = discover_artifacts(tmp_path)

    assert paths.profiler_plugin == plugin.resolve()
    assert paths.report_script == report.resolve()
    assert set(paths.binaries) == set(COLLECTIVES)


def test_discover_artifacts_fallback_and_kpacks(tmp_path):
    expected = _artifact_paths(
        tmp_path, ("rccl_lib_gfx950.kpack", "rccl_test_gfx950.kpack")
    )
    assert discover_artifacts(tmp_path) == expected


def test_discover_artifacts_rejects_ambiguous_fallback(tmp_path):
    _artifact_paths(tmp_path)
    (tmp_path / "duplicate").mkdir()
    (tmp_path / "duplicate" / "librccl-profiler-accl.so").write_text("other plugin")
    with pytest.raises(
        RuntimeError, match="Ambiguous artifact librccl-profiler-accl.so"
    ):
        discover_artifacts(tmp_path)


@pytest.mark.parametrize("collective", COLLECTIVES.values())
def test_message_sizes_match_requested_sweep(collective):
    sizes = accl_ci.message_sizes(RunConfig(), collective)
    divisor = 16 if collective in {"AllGather", "ReduceScatter"} else 1
    assert sizes == {1024 * 2**index // divisor for index in range(19)}


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_bytes": "0"},
        {"min_bytes": "invalid"},
        {"max_bytes": "512"},
        {"step_factor": 1},
        {"min_bytes": "4"},
    ],
)
def test_message_sizes_reject_invalid_sweeps(overrides):
    with pytest.raises(ValueError):
        accl_ci.message_sizes(RunConfig(**overrides), "AllGather")


def test_validate_collective_output_rejects_uniformly_truncated_sweep(tmp_path):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    with pytest.raises(ValueError, match="Requested message-size coverage mismatch"):
        validate_collective_output(tmp_path, "AllReduce", 2, 2, {1024, 2048, 4096})


@pytest.mark.parametrize(
    "world,local_size,local_rank,passes",
    [
        (16, 8, 0, True),
        (16, 8, 7, True),
        (8, 8, 0, False),
        (16, 4, 0, False),
        (16, 8, 8, False),
    ],
)
def test_mpi_preflight_executes_fail_fast(
    tmp_path, world, local_size, local_rank, passes
):
    script = render_slurm_script(
        _artifact_paths(tmp_path), tmp_path / "work", _run_config(tmp_path)
    )
    line = next(line for line in script.splitlines() if "set -e; test" in line)
    argv = shlex.split(line)
    payload = argv[argv.index("-c") + 1]
    result = subprocess.run(
        ["bash", "-c", payload],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ACCL_EXPECT_RANKS": "16",
            "ACCL_GPUS_PER_NODE": "8",
            "OMPI_COMM_WORLD_SIZE": str(world),
            "OMPI_COMM_WORLD_LOCAL_SIZE": str(local_size),
            "OMPI_COMM_WORLD_LOCAL_RANK": str(local_rank),
            "OMPI_COMM_WORLD_RANK": "0",
        },
    )
    assert (result.returncode == 0) == passes, result.stderr
    assert ("mpi_host=" in result.stdout) == passes


@pytest.mark.parametrize(
    "corrupt", [None, "librccl.so", "librccl-profiler-accl.so", *COLLECTIVES]
)
def test_node_preflight_checks_all_artifact_digests(tmp_path, corrupt):
    paths = _artifact_paths(tmp_path)
    env = {
        **os.environ,
        "ROCM_PATH": str(tmp_path),
        "ACCL_RCCL_LIB": str(paths.rccl_library),
        "ACCL_PLUGIN": str(paths.profiler_plugin),
        "ACCL_PREFLIGHT_BINARY": str(paths.binaries["all_reduce_perf"]),
        "ACCL_EXPECT_RCCL_SHA256": accl_ci.sha256_file(paths.rccl_library),
        "ACCL_EXPECT_PLUGIN_SHA256": accl_ci.sha256_file(paths.profiler_plugin),
        "ACCL_EXPECT_BINARY_SHA256": "\n".join(
            f"{accl_ci.sha256_file(p)}  {p}" for p in paths.binaries.values()
        ),
        "ACCL_TEST_BINARIES": " ".join(map(str, paths.binaries.values())),
        "ACCL_KPACK_COUNT": "0",
    }
    if corrupt:
        (tmp_path / corrupt).write_text("tampered")
    # Emulate node discovery, but exercise the real Bash checks and sha256sum.
    mocks = """
ldd() { printf 'librccl.so => %s\nlibamdhip64.so => %s/libamdhip64.so\n' "$ACCL_RCCL_LIB" "$ROCM_PATH"; }
readelf() { if [ "$1" = -Ws ]; then echo ' ncclProfiler_v5'; fi; }
rocminfo() { echo 'Name: gfx950'; }
"""
    result = subprocess.run(
        ["bash", "-c", mocks + render_node_preflight_script()],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (corrupt is None), result.stdout + result.stderr


@pytest.mark.parametrize(
    "failure",
    [None, "validation", "report", "process", "preflight", "slurm", "infrastructure"],
)
def test_main_manifest_summary_and_exit_agree(tmp_path, monkeypatch, failure):
    paths = _artifact_paths(tmp_path)
    report_source = (
        Path(__file__).resolve().parents[1] / "plugins/profiler/accl/accl_report.py"
    )
    paths.report_script.write_text(report_source.read_text())
    if failure == "report":
        paths.report_script.write_text("import sys; sys.exit(1)\n")
    config = _run_config(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    statuses = {"preflight": 0, **dict.fromkeys(COLLECTIVES, 0)}
    if failure in {"process", "preflight"}:
        statuses["all_reduce_perf" if failure == "process" else "preflight"] = 1
    (work / "collective-status.tsv").write_text(
        "".join(f"{key}\t{value}\n" for key, value in statuses.items())
    )
    for binary, collective in COLLECTIVES.items():
        for rank in range(2):
            output = work / "raw" / binary
            _write_rank_file(output, rank, collective, complete=failure != "validation")
            # Two-rank AG/RS descriptor bytes are half the requested size.
            if collective in {"AllGather", "ReduceScatter"}:
                path = output / f"rank{rank}.jsonl"
                objects = [json.loads(line) for line in path.read_text().splitlines()]
                for obj in objects:
                    if "coll_perf" in obj:
                        obj["coll_perf"]["coll_msg_size_bytes"] //= 2
                path.write_text("\n".join(map(json.dumps, objects)) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_accl_profiler.py",
            "--artifact-dir",
            str(paths.root),
            "--work-dir",
            str(work),
            "--mpi-root",
            str(config.mpi_root),
            "--nodes",
            "1",
            "--gpus-per-node",
            "2",
            "--min-bytes",
            "1K",
            "--max-bytes",
            "2K",
        ],
    )
    monkeypatch.setattr(accl_ci.signal, "signal", lambda *_: None)

    def submit(*_):
        if failure == "infrastructure":
            raise RuntimeError("scheduler unavailable")
        return "12345"

    monkeypatch.setattr(accl_ci, "submit_slurm_job", submit)
    monkeypatch.setattr(accl_ci, "cancel_slurm_job", lambda *_: None)
    monkeypatch.setattr(
        accl_ci,
        "wait_for_slurm_job",
        lambda *_: ("FAILED" if failure == "slurm" else "COMPLETED", "0:0"),
    )
    summaries = []
    monkeypatch.setattr(accl_ci, "write_github_summary", summaries.append)
    assert accl_ci.main() == (0 if failure is None else 1)
    manifest = json.loads((work / "manifest.json").read_text())
    assert bool(manifest["errors"]) == (failure is not None)
    assert f"Overall: {'PASS' if failure is None else 'FAIL'}" in summaries[0]
    if failure in {"validation", "report", "process"}:
        assert "all_reduce_perf | FAIL" in summaries[0]
    if failure is None:
        assert set(path.stem for path in (work / "reports").glob("*.txt")) == set(
            COLLECTIVES
        )
        assert (work / "accl_report.txt").stat().st_size > 0


def test_validate_collective_output_accepts_complete_rank_coverage(tmp_path):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    result = validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)
    assert result["files"] == 2
    assert result["ranks"] == [0, 1]
    assert result["message_sizes"] == [1024, 2048]
    assert result["minimum_samples_per_rank_size"] == 3


def test_validate_collective_output_requires_nested_kernel_events(tmp_path):
    _write_rank_file(tmp_path, 0)
    path = tmp_path / "rank0.jsonl"
    objects = [json.loads(line) for line in path.read_text().splitlines()]
    event_trace = objects[0]["coll_perf"].pop("event_trace_ts")
    objects[0]["event_trace_ts"] = event_trace
    path.write_text(
        "\n".join(json.dumps(obj) for obj in objects) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Missing kernel events"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


@pytest.mark.parametrize("event_trace", [None, [], "invalid"])
def test_validate_collective_output_rejects_invalid_event_trace(tmp_path, event_trace):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    path = tmp_path / "rank0.jsonl"
    objects = [json.loads(line) for line in path.read_text().splitlines()]
    objects[0]["coll_perf"]["event_trace_ts"] = event_trace
    path.write_text(
        "\n".join(json.dumps(obj) for obj in objects) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Missing kernel events"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_rejects_incomplete_summary(tmp_path):
    _write_rank_file(tmp_path, 0, complete=False)
    _write_rank_file(tmp_path, 1)
    with pytest.raises(ValueError, match="Incomplete profiler summary"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_rejects_missing_rank(tmp_path):
    _write_rank_file(tmp_path, 0)
    with pytest.raises(ValueError, match="Rank coverage mismatch"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_rejects_wrong_collective(tmp_path):
    _write_rank_file(tmp_path, 0, coll="Broadcast")
    _write_rank_file(tmp_path, 1)
    with pytest.raises(ValueError, match="Unexpected collective"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_rejects_wrong_rank_count(tmp_path):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    path = tmp_path / "rank1.jsonl"
    objects = [json.loads(line) for line in path.read_text().splitlines()]
    objects[0]["header"]["n_ranks"] = 3
    path.write_text("\n".join(json.dumps(obj) for obj in objects) + "\n")
    with pytest.raises(ValueError, match="reports n_ranks=3"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_rejects_missing_decomposition_field(tmp_path):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    path = tmp_path / "rank0.jsonl"
    objects = [json.loads(line) for line in path.read_text().splitlines()]
    objects[0]["coll_perf"]["decomposition"].pop("proxy_network_us")
    path.write_text("\n".join(json.dumps(obj) for obj in objects) + "\n")
    with pytest.raises(ValueError, match="Missing decomposition fields"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_rejects_size_coverage_divergence(tmp_path):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    path = tmp_path / "rank1.jsonl"
    objects = [json.loads(line) for line in path.read_text().splitlines()]
    objects = [
        obj
        for obj in objects
        if obj.get("coll_perf", {}).get("coll_msg_size_bytes") != 2048
    ]
    path.write_text("\n".join(json.dumps(obj) for obj in objects) + "\n")
    with pytest.raises(ValueError, match="Message-size coverage differs"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_requires_post_warmup_sample(tmp_path):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    with pytest.raises(ValueError, match="No post-warmup sample"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=3)


def test_plan_uses_all_scope_for_saturday_schedule():
    runs_ci, scope, _ = plan(
        "schedule", "smoke", datetime(2026, 10, 3, tzinfo=timezone.utc)
    )
    assert runs_ci is True
    assert scope == "all"


def test_parse_collective_status_ignores_malformed_rows(tmp_path):
    path = tmp_path / "collective-status.tsv"
    path.write_text(
        "preflight\t0\nall_reduce_perf\t0\nbad\nreduce_perf\tnot-an-int\n",
        encoding="utf-8",
    )
    assert parse_collective_status(path) == {
        "preflight": 0,
        "all_reduce_perf": 0,
    }


def test_submit_slurm_job_persists_raw_output_before_parsing(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    (work_dir / "slurm").mkdir(parents=True)
    config = _run_config(tmp_path)

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="12345;ruby\n", stderr="scheduler notice\n"
        )

    monkeypatch.setattr(accl_ci.subprocess, "run", fake_run)
    job_id = submit_slurm_job(tmp_path / "run.sbatch", work_dir, config)

    assert job_id == "12345"
    assert (work_dir / "slurm" / "sbatch.stdout").read_text() == "12345;ruby\n"
    assert (work_dir / "slurm" / "sbatch.stderr").read_text() == "scheduler notice\n"
    assert (work_dir / "slurm" / "job_id.txt").read_text() == "12345\n"


def test_submit_slurm_job_preserves_unparseable_output(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    (work_dir / "slurm").mkdir(parents=True)
    config = _run_config(tmp_path)

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="unexpected response\n", stderr="scheduler notice\n"
        )

    monkeypatch.setattr(accl_ci.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Could not parse Slurm job ID"):
        submit_slurm_job(tmp_path / "run.sbatch", work_dir, config)

    assert (work_dir / "slurm" / "sbatch.stdout").read_text() == "unexpected response\n"
    assert not (work_dir / "slurm" / "job_id.txt").exists()


def test_wait_for_slurm_job_retries_and_waits_for_terminal_state(monkeypatch):
    responses = iter(
        [
            subprocess.CompletedProcess([], 1, stdout="", stderr="controller busy"),
            subprocess.CompletedProcess([], 0, stdout="RUNNING\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="RUNNING|0:0\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="COMPLETING|0:0\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="COMPLETED|0:0\n", stderr=""),
        ]
    )

    def fake_run(_command, **_kwargs):
        return next(responses)

    monkeypatch.setattr(accl_ci.subprocess, "run", fake_run)
    monkeypatch.setattr(accl_ci.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(accl_ci.time, "monotonic", lambda: 0.0)

    assert wait_for_slurm_job("12345", 90) == ("COMPLETED", "0:0")


def test_wait_for_slurm_job_rejects_persistent_squeue_errors(monkeypatch):
    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="controller unavailable"
        )

    monkeypatch.setattr(accl_ci.subprocess, "run", fake_run)
    monkeypatch.setattr(accl_ci.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="squeue failed 30 consecutive times"):
        wait_for_slurm_job("12345", 90)


def test_wait_for_slurm_job_starts_timeout_at_running(monkeypatch):
    squeue_states = iter(["PENDING\n", "RUNNING\n", "RUNNING\n"])
    monotonic_values = iter([1000.0, 1061.0])
    cancelled = []

    def fake_run(command, **_kwargs):
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command, 0, stdout=next(squeue_states), stderr=""
            )
        if command[0] == "scancel":
            cancelled.append(command[1])
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(accl_ci.subprocess, "run", fake_run)
    monkeypatch.setattr(accl_ci.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(accl_ci.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(TimeoutError, match="exceeded 1 running minutes"):
        wait_for_slurm_job("12345", 1)
    assert cancelled == ["12345"]
