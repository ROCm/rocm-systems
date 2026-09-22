"""Unit tests for the Ruby ACCL-profiler CI driver."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ci", "scripts"))
from test_accl_profiler import (  # noqa: E402
    ArtifactPaths,
    COLLECTIVES,
    RunConfig,
    parse_collective_status,
    render_slurm_script,
    validate_collective_output,
)


def _artifact_paths(
    tmp_path: Path, kpack_names: tuple[str, ...] = ()
) -> ArtifactPaths:
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


def _record(rank: int, size: int, coll: str = "AllReduce") -> dict:
    return {
        "header": {"rank": rank, "n_ranks": 2},
        "coll_perf": {
            "coll": coll,
            "coll_sn": size + rank,
            "coll_msg_size_bytes": size,
            "decomposition": {
                "enqueue_to_kernel_us": 1,
                "gpu_kernel_avg_us": 2,
                "gpu_kernel_min_us": 2,
                "gpu_kernel_max_us": 2,
                "proxy_gpu_wait_us": 3,
                "proxy_network_us": 4,
                "proxy_peer_wait_us": 5,
                "proxy_flush_us": 6,
                "proxy_gpu_recv_wait_us": 7,
                "n_proxy_ops": 8,
                "n_send_ops": 4,
                "n_recv_ops": 4,
            },
        },
        "event_trace_ts": {
            "kernel_events": [{"channel_id": 0, "duration_us": 2}]
        },
    }


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
    objects.append({
        "summary": {
            "dropped_collectives": 0 if complete else 1,
            "leaked_collectives": 0,
            "pool_size": 256,
            "complete": complete,
        }
    })
    path = output_dir / f"rank{rank}.jsonl"
    path.write_text(
        "\n".join(json.dumps(obj) for obj in objects) + "\n", encoding="utf-8"
    )


def test_slurm_script_runs_only_five_supported_collectives(tmp_path):
    paths = _artifact_paths(tmp_path)
    script = render_slurm_script(paths, tmp_path / "work", RunConfig(nodes=2))
    for binary in COLLECTIVES:
        assert binary in script
    assert "alltoall_perf" not in script
    assert "--nodes=2 --ntasks=16 --ntasks-per-node=8" in script
    assert "NCCL_PROFILER_PLUGIN" in script


def test_rendered_slurm_script_has_valid_bash_syntax(tmp_path):
    paths = _artifact_paths(tmp_path)
    script_path = tmp_path / "run.sbatch"
    script_path.write_text(
        render_slurm_script(paths, tmp_path / "work", RunConfig()),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", "-n", str(script_path)], capture_output=True, text=True, check=False
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
        _artifact_paths(tmp_path), tmp_path / "work", RunConfig()
    )

    assert "_tool/Python" not in script
    assert f"export PATH={tmp_path}/bin:/usr/bin:/bin:/usr/sbin:/sbin" in script
    assert f"export LD_LIBRARY_PATH={tmp_path}" in script
    assert (
        "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV "
        "ROCM_KPACK_PATH ROCM_KPACK_PATH_PREFIX" in script
    )
    assert '"$ROCM_PATH/bin/rocminfo"' in script
    assert '/usr/bin/python3 "$ROCM_PATH/bin/rocm_agent_enumerator"' in script


def test_slurm_script_uses_embedded_kpack_references(tmp_path):
    paths = _artifact_paths(
        tmp_path,
        ("rccl_lib_gfx950.kpack", "rccl_test_gfx950.kpack"),
    )

    script = render_slurm_script(paths, tmp_path / "work", RunConfig())

    assert "export ROCM_KPACK_PATH=" not in script
    assert "export ROCM_KPACK_PATH_PREFIX=" not in script
    assert "rccl_lib_gfx950.kpack" in script
    assert "rccl_test_gfx950.kpack" in script
    assert 'readelf -SW "$ACCL_RCCL_LIB"' in script
    assert 'readelf -SW "$binary"' in script
    assert 'resolved_hip=$(ldd "$ACCL_PREFLIGHT_BINARY"' in script
    assert '"$ROCM_PATH"/*' in script


def test_slurm_script_rejects_incomplete_kpack_layout(tmp_path):
    paths = _artifact_paths(tmp_path, ("rccl_lib_gfx950.kpack",))

    with pytest.raises(FileNotFoundError, match="rccl_test_gfx950.kpack"):
        render_slurm_script(paths, tmp_path / "work", RunConfig())


def test_validate_collective_output_accepts_complete_rank_coverage(tmp_path):
    for rank in range(2):
        _write_rank_file(tmp_path, rank)
    result = validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)
    assert result["files"] == 2
    assert result["ranks"] == [0, 1]
    assert result["message_sizes"] == [1024, 2048]
    assert result["minimum_samples_per_rank_size"] == 3


def test_validate_collective_output_rejects_incomplete_summary(tmp_path):
    _write_rank_file(tmp_path, 0, complete=False)
    _write_rank_file(tmp_path, 1)
    with pytest.raises(ValueError, match="Incomplete profiler summary"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


def test_validate_collective_output_rejects_missing_rank(tmp_path):
    _write_rank_file(tmp_path, 0)
    with pytest.raises(ValueError, match="Rank coverage mismatch"):
        validate_collective_output(tmp_path, "AllReduce", 2, warmup_iterations=2)


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
