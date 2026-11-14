# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import glob
import os
import struct
import sys
import time

import pytest
from bench import runner
from bench.runner import RunConfig, parse_markers, run_command

PY = sys.executable


def cfg(**kw):
    return RunConfig(**{"runs": 2, "warmup": 1, "pause_ms": 0, **kw})


def run(command, **kw):
    return list(run_command("A", command, cfg(**kw)))


def test_true_and_false():
    ok = run("true")
    bad = run("false")
    assert len(ok) == 2 and all(r.ok for r in ok)
    assert all(r.exit == 1 and not r.ok for r in bad)
    assert ok[0].wall_ns > 0 and ok[0].max_rss_kb > 0


def test_timeout_recorded_as_error():
    (r,) = run("sleep 5", runs=1, warmup=0, timeout=0.2)
    assert "timeout" in r.error
    assert r.wall_ns < 3e9


def test_spawn_failure_is_error():
    (r,) = run("/nonexistent/binary", runs=1, warmup=0)
    assert "spawn failed" in r.error


def test_checksum_from_last_line():
    code = "print('checksum: aaa'); print('noise'); print('checksum: beef')"
    (r,) = run(f'{PY} -c "{code}"', runs=1, warmup=0)
    assert r.checksum == "beef"


def test_markers_via_bench_fd():
    code = (
        "import os,struct,time;fd=int(os.environ['BENCH_FD']);"
        "w=lambda i:os.write(fd,struct.pack('<IIQQ',i,0,time.monotonic_ns(),time.process_time_ns()));"
        "w(1);time.sleep(0.05);w(2)"
    )
    (r,) = run(f'{PY} -c "{code}"', runs=1, warmup=0)
    assert r.ok
    assert r.t1_t2_ns >= 50e6
    assert r.t0_t1_ns > 0 and r.t2_t3_ns > 0
    assert r.t0_t1_ns + r.t1_t2_ns + r.t2_t3_ns == pytest.approx(r.wall_ns, abs=1e6)


def test_no_markers_gives_none():
    (r,) = run("true", runs=1, warmup=0)
    assert r.t0_t1_ns is r.t1_t2_ns is r.t2_t3_ns is None


def test_parse_markers_first_begin_last_end_and_partial_tail():
    data = (
        struct.pack("<IIQQ", 1, 0, 5, 50)
        + struct.pack("<IIQQ", 2, 0, 9, 90)
        + struct.pack("<IIQQ", 2, 0, 12, 120)
        + b"xx"
    )
    assert parse_markers(data) == {1: [(5, 50)], 2: [(9, 90), (12, 120)]}


def test_clean_removes_paths_between_runs(tmp_path):
    target = tmp_path / "trace.out"
    code = f"open({str(target)!r},'w').write('12345')"
    results = run(f'{PY} -c "{code}"', runs=2, warmup=1, clean=[str(target)])
    assert not target.exists()
    assert [r.output_bytes for r in results] == [5, 5]


def test_mem_sampling_only_when_enabled():
    off = run("sleep 0.1", runs=1, warmup=0)[0]
    on = run("sleep 0.1", runs=1, warmup=0, mem_sample_ms=10)[0]
    assert off.sampled_rss_kb is None
    assert on.sampled_rss_kb > 0


def test_affinity_prefixes_taskset():
    from bench.runner import build_argv

    assert build_argv("a b", "2-3")[:3] == ["taskset", "-c", "2-3"]
    assert build_argv("a b", None) == ["a", "b"]


def test_gc_restored():
    import gc

    assert gc.isenabled()
    run("true", runs=1, warmup=0)
    assert gc.isenabled()


def test_parse_output_reads_only_valid_metrics_and_last_checksum(tmp_path):
    from bench.runner import parse_output

    f = tmp_path / "out.txt"
    f.write_text(
        "checksum: old\nhello\nmetric gpu_kernel_us 12.5\nmetric bad name 3\n"
        "metric gpu_gap_us 1e-1\nmetric x notnumber\nchecksum: ab\n"
    )
    assert parse_output(str(f)) == ("ab", {"gpu_kernel_us": 12.5, "gpu_gap_us": 0.1})


def test_run_collects_app_reported_metrics():
    cfg = RunConfig(runs=2, warmup=0, pause_ms=0)
    cmd = "python3 -c \"print('metric gpu_interval_ms 3.25'); print('checksum: ok')\""
    results = list(run_command("A", cmd, cfg))
    assert [r.metrics for r in results] == [{"gpu_interval_ms": 3.25}] * 2
    assert all(r.checksum == "ok" for r in results)


def test_env_added_only_to_its_command(monkeypatch):
    monkeypatch.setenv("BENCH_TEST_INHERITED", "1")
    check = 'sh -c \'test "$BENCH_TEST_X" = y && test "$BENCH_TEST_INHERITED" = 1\''
    assert all(r.ok for r in run_command("A", check, cfg(), {"BENCH_TEST_X": "y"}))
    assert not any(r.ok for r in run_command("A", check, cfg()))


def test_timeout_kills_children_of_the_command(tmp_path):
    marker = tmp_path / "survived"
    (r,) = run(
        f"sh -c 'sh -c \"sleep 1; touch {marker}\" & sleep 5'",
        runs=1,
        warmup=0,
        timeout=0.2,
    )
    time.sleep(1.5)
    assert "timeout" in r.error
    assert not marker.exists()


def test_interrupt_kills_child_and_removes_temp_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)
    pids = []

    def interrupted_wait4(pid, options):
        pids.append(pid)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.os, "wait4", interrupted_wait4)
    with pytest.raises(KeyboardInterrupt):
        runner.run_once("x", 0, ["sleep", "30"], RunConfig())
    with pytest.raises(ProcessLookupError):  # killed and reaped, not orphaned
        os.kill(pids[0], 0)
    assert not glob.glob(str(tmp_path / "bench_out_*"))
