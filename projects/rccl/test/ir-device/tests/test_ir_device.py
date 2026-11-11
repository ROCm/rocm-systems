# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Functional tests for librccl_device.bc, driven through the GoogleTest
binaries built from bindings/ir/test/IR_test.cpp (single process) and
bindings/ir/test/IR_gin_mpi_test.cpp (multi-rank, under mpirun).

Each pytest case runs a binary restricted to a single --gtest_filter so
failures map cleanly to the underlying device API. Outcomes are mapped by
`summarize_gtest` (conftest.py) so the pytest result mirrors what GoogleTest actually did:

  * any failure / non-zero exit            -> pytest FAILED
  * at least one real pass                 -> pytest PASSED
  * clean run where every case SKIPPED     -> pytest SKIPPED (with the reason)

That last rule matters for the GIN/composite cases: when the host lacks the
hardware/runtime to actually exercise a barrier (e.g. <2 GPUs, or no cuMem /
symmetric-memory support), GoogleTest SKIPs and exits 0 — without this mapping
those would misleadingly show as PASSED.
"""

import os
import re
import subprocess

import pytest


@pytest.mark.ir_device
@pytest.mark.peer_pointer
def test_get_peer_pointer_team(run_gtest, summarize_gtest):
    """[A] ncclGetPeerPointerTeam — 20 pointer-arithmetic cases."""
    proc, log = run_gtest("IRDeviceTest.A_GetPeerPointerTeam",
                          "ir_peer_pointer.log")
    summarize_gtest(proc, log, "ncclGetPeerPointerTeam")


@pytest.mark.ir_device
@pytest.mark.coop
def test_coop_init_and_accessors(run_gtest, summarize_gtest):
    """[B1-B5] ncclCoopAnyInit* + ncclCoopSize/ThreadRank/NumThreads."""
    proc, log = run_gtest(
        "IRDeviceTest.B1_*:IRDeviceTest.B2_*:IRDeviceTest.B3*:"
        "IRDeviceTest.B4*:IRDeviceTest.B5_*",
        "ir_coop_init.log",
    )
    summarize_gtest(proc, log, "ncclCoopAny init/accessors")


@pytest.mark.ir_device
@pytest.mark.coop
def test_coop_sync(run_gtest, summarize_gtest):
    """[B6] ncclCoopSync — all five coop types complete."""
    proc, log = run_gtest("IRDeviceTest.B6_CoopSync", "ir_coop_sync.log")
    summarize_gtest(proc, log, "ncclCoopSync")


@pytest.mark.ir_device
@pytest.mark.lsa_barrier
def test_lsa_barrier_session(run_gtest, summarize_gtest):
    """[B7] ncclLsaBarrierSession structural check (+ runtime cases skipped)."""
    proc, log = run_gtest("IRDeviceTest.B7*", "ir_lsa_barrier.log")
    summarize_gtest(proc, log, "ncclLsaBarrierSession")


@pytest.mark.ir_device
@pytest.mark.gin_barrier
@pytest.mark.composite_barrier
def test_bucket_c_symbols_linkable(run_gtest, summarize_gtest):
    """[C0] All four bucket-C thunks resolve/export from librccl_device.bc.

    Takes the device address of ncclGinBarrierSession{Init,Sync} and
    ncclBarrierSession{Init,Sync}; a missing/renamed/internalized export
    fails the device link or yields a null address. No live resources needed.
    """
    proc, log = run_gtest("IRDeviceTest.C0_BucketCSymbolsLinkable",
                          "ir_bucket_c_symbols.log")
    summarize_gtest(proc, log, "bucket C symbol link coverage")


@pytest.mark.ir_device
@pytest.mark.gin_barrier
def test_gin_barrier_session(run_gtest, summarize_gtest):
    """[C1] ncclGinBarrierSession structural check (+ runtime case skipped)."""
    proc, log = run_gtest("IRDeviceTest.C1*", "ir_gin_barrier.log")
    summarize_gtest(proc, log, "ncclGinBarrierSession")


@pytest.mark.ir_device
@pytest.mark.composite_barrier
def test_composite_barrier_session(run_gtest, summarize_gtest):
    """[C2] composite ncclBarrierSession structural check (+ runtime skipped)."""
    proc, log = run_gtest("IRDeviceTest.C2*", "ir_composite_barrier.log")
    summarize_gtest(proc, log, "ncclBarrierSession")


@pytest.mark.ir_device
@pytest.mark.reduce_copy
def test_reduce_copy_typed_exports(ir_bitcode, paths):
    """[D0] librccl_device.bc exports exactly the typed ReduceCopy thunks.

    The op and type matrix is read from where the thunks are defined (src/util.h
    type lists, expanded for HIP by the preprocessor, and the
    NCCL_IR_DEFINE_API_* lines of nccl_device_wrapper__impl.h), so a thunk the
    Makefile/CMake public-API lists miss is reported as missing, and an
    exported thunk outside that matrix as unexpected.
    """
    ir_dir = os.path.join(paths.RCCL_DIR, "bindings", "ir")
    types = subprocess.run(
        [os.path.join(paths.LLVM_BIN, "clang++"), "-E", "-P", "-x", "c++",
         "-D__HIP_PLATFORM_AMD__", "-I", ir_dir, "-"],
        input='#include "src/util.h"\n#define NCCL_IR_TYPE_NAME(name, type) name\n'
              "ALL NCCL_IR_TYPES(NCCL_IR_TYPE_NAME)\n"
              "MULTIMEM NCCL_IR_MULTIMEM_TYPES(NCCL_IR_TYPE_NAME)\n",
        stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    types = dict(line.split(None, 1) for line in types.splitlines() if line.strip())
    types = {k: v.split() for k, v in types.items()}
    with open(os.path.join(ir_dir, "nccl_device_wrapper__impl.h")) as fh:
        ops = re.findall(r"^NCCL_IR_DEFINE_API_(ALL|MULTIMEM)_TYPES\((\w+)\)", fh.read(), re.M)
    assert ops and types.get("ALL") and types.get("MULTIMEM"), (
        f"could not read the ReduceCopy op/type matrix: ops={ops} types={types}")
    expected = {f"{op}_{t}" for kind, op in ops for t in types[kind]}

    ir = subprocess.run([os.path.join(paths.LLVM_BIN, "llvm-dis"), ir_bitcode, "-o", "-"],
                        stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    exported = set(re.findall(r"^define (?!internal |private )[^@\n]*@(\w+)\(", ir, re.M))
    thunk = re.compile(r"(?:%s)_[A-Z][A-Z0-9]*$" % "|".join(op for _, op in ops))
    missing = sorted(expected - exported)
    unexpected = sorted(s for s in exported - expected if thunk.match(s))
    assert not missing and not unexpected, (
        f"typed ReduceCopy thunks in {ir_bitcode}: {len(missing)} of {len(expected)} "
        f"not exported ({', '.join(missing[:6])}); {len(unexpected)} exported but not "
        f"defined in util.h ({', '.join(unexpected[:6])})")


@pytest.mark.ir_device
@pytest.mark.reduce_copy
def test_reduce_copy_typed_local(run_gtest, summarize_gtest, paths, monkeypatch):
    """[D1] ncclLocalReduceSumCopy_{I8,F32,BF16}: 3 sources -> 2 destinations."""
    # D1 raises the stack limit (up to 64 KiB). HSA_NO_SCRATCH_RECLAIM=1 makes
    # ROCr map that for every wave slot of the GPU, more than MI300X's scratch
    # aperture holds; with reclaim it maps what this dispatch needs.
    monkeypatch.setenv("HSA_NO_SCRATCH_RECLAIM", "0")
    # Too small a stack can fault, return wrong sums or hang, so every failure
    # names the stack limit the probe in IR_reduce_copy_test.cpp chose.
    log = os.path.join(paths.LOGDIR, "ir_reduce_copy.log")
    try:
        proc, log = run_gtest("IRReduceCopyTest.*", "ir_reduce_copy.log")
        timed_out = None
    except subprocess.TimeoutExpired as e:
        proc, timed_out = None, e
    try:
        with open(log) as fh:
            out = fh.read()
    except OSError:
        out = ""
    stack = re.search(r"\[IRReduceCopyTest\] (.*)", out)
    stack = stack.group(1) if stack else "stack probe did not report"
    if timed_out:
        raise AssertionError(f"typed ReduceCopy thunks hung ({timed_out}) [{stack}]")
    try:
        summarize_gtest(proc, log, "typed ReduceCopy thunks")
    except AssertionError as e:
        raise AssertionError(f"{e} [{stack}]") from None
    # an empty filter match would also exit 0
    for t in ("I8", "F32", "BF16"):
        assert f"[       OK ] IRReduceCopyTest.D1_LocalReduceSumCopy_{t}" in out, (
            f"D1 {t} did not run, see {log} [{stack}]")


@pytest.mark.ir_device
@pytest.mark.gin_barrier
@pytest.mark.mpi
def test_gin_barrier_session_functional_mpi(run_gin_mpi_gtest, summarize_gtest):
    """[C1 functional] ncclGinBarrierSession{Init,Sync} over a live 2-rank GIN
    rail barrier, under `mpirun -np 2`.

    Reports SKIPPED (not PASSED) when the host can't actually run it — e.g.
    fewer than 2 GPUs, or no cuMem/symmetric-memory support.
    """
    proc, log = run_gin_mpi_gtest(
        "IRGinBarrierMPITest.C1_GinBarrierSession_TwoRanks",
        "ir_gin_barrier_mpi.log",
    )
    summarize_gtest(proc, log, "GIN barrier functional (2-rank)")


@pytest.mark.ir_device
@pytest.mark.composite_barrier
@pytest.mark.mpi
def test_composite_barrier_session_functional_mpi(run_gin_mpi_gtest, summarize_gtest):
    """[C2 functional] composite ncclBarrierSession{Init,Sync} (inner LSA +
    outer GIN) over a live 2-rank comm. Same skip semantics as the GIN case.
    """
    proc, log = run_gin_mpi_gtest(
        "IRGinBarrierMPITest.C2_CompositeBarrierSession_TwoRanks",
        "ir_composite_barrier_mpi.log",
    )
    summarize_gtest(proc, log, "composite barrier functional (2-rank)")
