# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from conftest import NGPUS
from collectives import COLLECTIVES, COLLECTIVES_WITH_ROOT
from test_runner import run_rccl_perf, run_rccl_mpi


_MODES = [
    pytest.param("standalone", id="standalone"),
    pytest.param("mpi",        id="MPI", marks=pytest.mark.mpi),
]


def _run(mode, executable, gpu_count, args):
    """Dispatch to standalone or MPI runner with correct thread/rank layout."""
    if mode == "mpi":
        return run_rccl_mpi(executable, gpu_count, ["-t", "1", "-g", "1"] + args)
    return run_rccl_perf(executable, ["-t", str(gpu_count), "-g", "1"] + args)


# ---------------------------------------------------------------------------
# Group 2: Size sweep modes
# ---------------------------------------------------------------------------

def test_stepfactor(gpu_count):
    """Multiplicative size stepping (-f): 1K → 1G in powers of 2."""
    run_rccl_perf("all_reduce_perf", [
        "-t", str(gpu_count), "-g", "1",
        "-b", "1K", "-e", "1G", "-f", "2", "-d", "float", "-o", "sum"])


def test_stepbytes(gpu_count):
    """Additive size stepping (-i): 1M → 64M in 1M increments."""
    run_rccl_perf("all_reduce_perf", [
        "-t", str(gpu_count), "-g", "1",
        "-b", "1M", "-e", "64M", "-i", "1M", "-d", "float", "-o", "sum"])


# ---------------------------------------------------------------------------
# Group 3: Correctness checking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("check_iters", ["1", "2"], ids=lambda c: f"check{c}")
def test_correctness_check(check_iters, mode, gpu_count):
    """Correctness verification logic (-c)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-c", check_iters, "-d", "float", "-o", "sum"])


# ---------------------------------------------------------------------------
# Group 4: Execution modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
def test_blocking(mode, gpu_count):
    """Blocking collective mode (-z 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-z", "1", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_null_stream(mode, gpu_count):
    """NULL stream path (-y 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-y", "1", "-d", "float", "-o", "sum"])


# test_parallel_init is disabled:
# preiliminary investigation suggests that concurrent ncclInitKernelsForDevice()
# calls race on cudaFuncSetAttribute() with no lock held, corrupting kernel
# shared-memory configuration and causing GPU page faults during subsequent collective.
# Re-enable once fixed in RCCL.
#
# def test_parallel_init(gpu_count):
#     """Threaded NCCL/RCCL init (-p 1)."""
#     run_rccl_perf("all_reduce_perf", [
#         "-t", str(gpu_count), "-g", "1",
#         "-p", "1", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("placement", ["0", "1"], ids=["in_place", "out_of_place"])
def test_placement(placement, mode, gpu_count):
    """In-place vs out-of-place buffer paths (-O)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-O", placement, "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_hip_graph(mode, gpu_count):
    """HIP graph capture and replay (-G 2)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-G", "2", "-d", "float", "-o", "sum"])


# ---------------------------------------------------------------------------
# Group 5: Multi-GPU threading model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
def test_thread_per_gpu(mode, gpu_count):
    """-t <NGPUS> -g 1 standalone, one rank per GPU in MPI."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-d", "float", "-o", "sum"])


# Standalone: one thread owning N GPUs, N from 2 up to the full machine.
# On a 1-GPU host this range is empty; surface that as a single skipped item
# (param value 1 is a placeholder — the skip marker prevents it from running)
# instead of an empty parametrize that would silently emit a pytest warning.
if NGPUS >= 2:
    _MULTI_GPU_STANDALONE = [pytest.param(n, id=f"g{n}") for n in range(2, NGPUS + 1)]
else:
    _MULTI_GPU_STANDALONE = [pytest.param(
        1, id="g1", marks=pytest.mark.skip(reason="requires NGPUS >= 2"))]

# MPI: all (np, g) pairs where np * g = NGPUS, exercising the full
# rank-vs-GPU-per-rank tradeoff space
_MULTI_GPU_MPI_PAIRS = [(k, NGPUS // k)
                        for k in range(1, NGPUS + 1) if NGPUS % k == 0]


# Collectives currently known to fail validation under multi-GPU sweeps on
# MI355X (ROCm 7.0 / RCCL 2.28.3). See run on smci355-ccs-aus-m03-17:
#   hypercube_perf -t 1 -g {4,8} -d float at the default 32 MiB:
#     '# Out of bounds values : 2 FAILED' -> exit code 7.
# Remove this set once the upstream collective is fixed.
_KNOWN_BROKEN_MULTI_GPU = {"hypercube"}


@pytest.mark.parametrize("collective", COLLECTIVES, ids=lambda c: c.name)
@pytest.mark.parametrize("n_gpus", _MULTI_GPU_STANDALONE)
def test_multi_gpu_per_thread(n_gpus, collective):
    """-t 1 -g N across all collectives, N from 2 to NGPUS."""
    if collective.name in _KNOWN_BROKEN_MULTI_GPU:
        pytest.skip(
            f"known RCCL bug (untracked): {collective.executable} fails "
            "validation under -t 1 -g N on MI355X (RCCL 2.28.3)"
        )
    args = ["-t", "1", "-g", str(n_gpus), "-d", "float"]
    if collective.has_ops:
        args += ["-o", "sum"]
    if collective.has_root:
        args += ["-r", "0"]
    run_rccl_perf(collective.executable, args)


@pytest.mark.mpi
@pytest.mark.parametrize("collective", COLLECTIVES, ids=lambda c: c.name)
@pytest.mark.parametrize("np_count,g_count", _MULTI_GPU_MPI_PAIRS,
                         ids=[f"np{np}_g{g}" for np, g in _MULTI_GPU_MPI_PAIRS])
def test_multi_gpu_per_thread_mpi(np_count, g_count, collective):
    """-np K -g M across all collectives for all K*M=NGPUS combinations."""
    if collective.name in _KNOWN_BROKEN_MULTI_GPU:
        pytest.skip(
            f"known RCCL bug (untracked): {collective.executable} fails "
            "validation under -np K -g M on MI355X (RCCL 2.28.3)"
        )
    args = ["-t", "1", "-g", str(g_count), "-d", "float"]
    if collective.has_ops:
        args += ["-o", "sum"]
    if collective.has_root:
        args += ["-r", "0"]
    run_rccl_mpi(collective.executable, np_count, args)


# ---------------------------------------------------------------------------
# Group 6: Rooted collectives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("collective", COLLECTIVES_WITH_ROOT, ids=lambda c: c.name)
def test_root_explicit(collective, mode, gpu_count):
    """Non-default root rank (-r <last>) across all rooted collectives."""
    if gpu_count < 2:
        pytest.skip("non-default root requires >= 2 GPUs (gpu_count=1 collapses to -r 0)")
    args = ["-r", str(gpu_count - 1), "-d", "float"]
    if collective.has_ops:
        args += ["-o", "sum"]
    _run(mode, collective.executable, gpu_count, args)


# ---------------------------------------------------------------------------
# Group 7: Iteration control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
def test_custom_iters(mode, gpu_count):
    """Non-default iteration count (-n 5)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-n", "5", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_aggregated_iters(mode, gpu_count):
    """Aggregated/batched iterations (-m 2)."""
    pytest.skip(
        "known RCCL bug (untracked): all_reduce_perf -m 2 at default 32 MiB "
        "raises 'an illegal memory access was encountered' on MI355X "
        "(RCCL 2.28.3); see common.cu.cpp:641 / all_reduce.cu.cpp:603"
    )
    _run(mode, "all_reduce_perf", gpu_count,
         ["-m", "2", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_multi_cycle(mode, gpu_count):
    """Multi-cycle outer loop (-N 2)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-N", "2", "-d", "float", "-o", "sum"])


# ---------------------------------------------------------------------------
# Group 8: Output & reporting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
def test_cpu_time_report(mode, gpu_count):
    """CPU time reporting (-C 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-C", "1", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_timestamps(mode, gpu_count):
    """Timestamp output (-S 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-S", "1", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("avg_mode", ["0", "2", "3"],
                         ids=["avg_rank0", "avg_min", "avg_max"])
def test_average_mode(avg_mode, mode, gpu_count):
    """Average reporting modes (-a)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-a", avg_mode, "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_json_output(mode, gpu_count, tmp_path):
    """JSON file output (-J)."""
    pytest.skip(
        "known rccl-tests bug (untracked): all_reduce_perf -J <file> "
        "segfaults at process exit (signal 11) on MI355X (rccl-tests 2.17.9); "
        "no RCCL frames in backtrace - the crash is inside the JSON writer"
    )
    outfile = str(tmp_path / "rccl_test.json")
    _run(mode, "all_reduce_perf", gpu_count,
         ["-J", outfile, "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_memory_report(mode, gpu_count):
    """Memory usage reporting (-M 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-M", "1", "-d", "float", "-o", "sum"])


# ---------------------------------------------------------------------------
# Group 9: RCCL-specific features
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("mem_type", ["fine", "host", "managed"],
                         ids=lambda m: f"mem_{m}")
def test_memory_type(mem_type, mode, gpu_count):
    """Non-default memory types (-Y)."""
    env = {"HSA_FORCE_FINE_GRAIN_PCIE": "1"} if mem_type == "fine" else None
    if mode == "mpi":
        run_rccl_mpi("all_reduce_perf", gpu_count,
                     ["-t", "1", "-g", "1", "-Y", mem_type, "-d", "float", "-o", "sum"],
                     env_overrides=env)
    else:
        run_rccl_perf("all_reduce_perf",
                      ["-t", str(gpu_count), "-g", "1", "-Y", mem_type,
                       "-d", "float", "-o", "sum"],
                      env_overrides=env)


@pytest.mark.parametrize("mode", _MODES)
def test_rotating_tensor(mode, gpu_count):
    """Rotating tensor pattern (-E 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-E", "1", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_rccl_reporter_csv(mode, gpu_count, tmp_path):
    """RCCL Reporter CSV output (-Z csv -X file)."""
    outfile = str(tmp_path / "rccl_report.csv")
    _run(mode, "all_reduce_perf", gpu_count,
         ["-Z", "csv", "-X", outfile, "-d", "float", "-o", "sum"])


# ---------------------------------------------------------------------------
# Group 10: Buffer registration & algo reporting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
def test_local_register(mode, gpu_count):
    """Local buffer registration (-R 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-R", "1", "-d", "float", "-o", "sum"])


@pytest.mark.parametrize("mode", _MODES)
def test_algo_proto_channels(mode, gpu_count):
    """Algo/proto/channel reporting (-A 1)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-A", "1", "-d", "float", "-o", "sum"])


# ---------------------------------------------------------------------------
# Group 11: Environment variables
# ---------------------------------------------------------------------------

def test_env_device_override():
    """NCCL_TESTS_DEVICE override (single GPU, standalone only)."""
    run_rccl_perf("all_reduce_perf", [
        "-t", "1", "-g", "1",
        "-d", "float", "-o", "sum"],
        env_overrides={"NCCL_TESTS_DEVICE": "0"})


@pytest.mark.parametrize("mode", _MODES)
def test_env_min_bw(mode, gpu_count):
    """NCCL_TESTS_MIN_BW pass/fail threshold."""
    if mode == "mpi":
        run_rccl_mpi("all_reduce_perf", gpu_count,
                     ["-t", "1", "-g", "1", "-d", "float", "-o", "sum"],
                     env_overrides={"NCCL_TESTS_MIN_BW": "0.001"})
    else:
        run_rccl_perf("all_reduce_perf",
                      ["-t", str(gpu_count), "-g", "1", "-d", "float", "-o", "sum"],
                      env_overrides={"NCCL_TESTS_MIN_BW": "0.001"})


# ---------------------------------------------------------------------------
# Group 12: Internal timeout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
def test_internal_timeout(mode, gpu_count):
    """RCCL-Tests internal timeout mechanism (-T)."""
    _run(mode, "all_reduce_perf", gpu_count,
         ["-T", "30", "-d", "float", "-o", "sum"])
