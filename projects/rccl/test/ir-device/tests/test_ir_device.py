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
`_summarize_gtest` so the pytest result mirrors what GoogleTest actually did:

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


def _summarize_gtest(proc, log, what):
    """Translate a GoogleTest binary run into a pytest outcome.

    Reads the captured log so we can tell a genuine pass from a run in which
    every selected case was SKIPPED (both exit 0).
    """
    try:
        with open(log) as fh:
            text = fh.read()
    except OSError:
        text = ""

    if proc.returncode != 0 or "[  FAILED  ]" in text:
        raise AssertionError(f"{what} failed, see {log}")

    def _count(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    passed = _count(r"\[  PASSED  \] (\d+) test")
    skipped = _count(r"\[  SKIPPED \] (\d+) test")

    # Nothing passed but something skipped -> surface a real pytest SKIP with
    # the GoogleTest skip reason (the line printed right after ": Skipped").
    if passed == 0 and skipped > 0:
        m = re.search(r": Skipped\s*\n([^\[]+)", text)
        reason = " ".join(m.group(1).split()) if m else "all cases skipped"
        pytest.skip(f"{what}: {reason} (see {log})")


@pytest.mark.ir_device
@pytest.mark.peer_pointer
def test_get_peer_pointer_team(run_gtest):
    """[A] ncclGetPeerPointerTeam — 20 pointer-arithmetic cases."""
    proc, log = run_gtest("IRDeviceTest.A_GetPeerPointerTeam",
                          "ir_peer_pointer.log")
    _summarize_gtest(proc, log, "ncclGetPeerPointerTeam")


@pytest.mark.ir_device
@pytest.mark.coop
def test_coop_init_and_accessors(run_gtest):
    """[B1-B5] ncclCoopAnyInit* + ncclCoopSize/ThreadRank/NumThreads."""
    proc, log = run_gtest(
        "IRDeviceTest.B1_*:IRDeviceTest.B2_*:IRDeviceTest.B3*:"
        "IRDeviceTest.B4*:IRDeviceTest.B5_*",
        "ir_coop_init.log",
    )
    _summarize_gtest(proc, log, "ncclCoopAny init/accessors")


@pytest.mark.ir_device
@pytest.mark.coop
def test_coop_sync(run_gtest):
    """[B6] ncclCoopSync — all five coop types complete."""
    proc, log = run_gtest("IRDeviceTest.B6_CoopSync", "ir_coop_sync.log")
    _summarize_gtest(proc, log, "ncclCoopSync")


@pytest.mark.ir_device
@pytest.mark.lsa_barrier
def test_lsa_barrier_session(run_gtest):
    """[B7] ncclLsaBarrierSession structural check (+ runtime cases skipped)."""
    proc, log = run_gtest("IRDeviceTest.B7*", "ir_lsa_barrier.log")
    _summarize_gtest(proc, log, "ncclLsaBarrierSession")


@pytest.mark.ir_device
@pytest.mark.gin_barrier
@pytest.mark.composite_barrier
def test_bucket_c_symbols_linkable(run_gtest):
    """[C0] All four bucket-C thunks resolve/export from librccl_device.bc.

    Takes the device address of ncclGinBarrierSession{Init,Sync} and
    ncclBarrierSession{Init,Sync}; a missing/renamed/internalized export
    fails the device link or yields a null address. No live resources needed.
    """
    proc, log = run_gtest("IRDeviceTest.C0_BucketCSymbolsLinkable",
                          "ir_bucket_c_symbols.log")
    _summarize_gtest(proc, log, "bucket C symbol link coverage")


@pytest.mark.ir_device
@pytest.mark.gin_barrier
def test_gin_barrier_session(run_gtest):
    """[C1] ncclGinBarrierSession structural check (+ runtime case skipped)."""
    proc, log = run_gtest("IRDeviceTest.C1*", "ir_gin_barrier.log")
    _summarize_gtest(proc, log, "ncclGinBarrierSession")


@pytest.mark.ir_device
@pytest.mark.composite_barrier
def test_composite_barrier_session(run_gtest):
    """[C2] composite ncclBarrierSession structural check (+ runtime skipped)."""
    proc, log = run_gtest("IRDeviceTest.C2*", "ir_composite_barrier.log")
    _summarize_gtest(proc, log, "ncclBarrierSession")


@pytest.mark.ir_device
@pytest.mark.gin_barrier
@pytest.mark.mpi
def test_gin_barrier_session_functional_mpi(run_gin_mpi_gtest):
    """[C1 functional] ncclGinBarrierSession{Init,Sync} over a live 2-rank GIN
    rail barrier, under `mpirun -np 2`.

    Reports SKIPPED (not PASSED) when the host can't actually run it — e.g.
    fewer than 2 GPUs, or no cuMem/symmetric-memory support.
    """
    proc, log = run_gin_mpi_gtest(
        "IRGinBarrierMPITest.C1_GinBarrierSession_TwoRanks",
        "ir_gin_barrier_mpi.log",
    )
    _summarize_gtest(proc, log, "GIN barrier functional (2-rank)")


@pytest.mark.ir_device
@pytest.mark.composite_barrier
@pytest.mark.mpi
def test_composite_barrier_session_functional_mpi(run_gin_mpi_gtest):
    """[C2 functional] composite ncclBarrierSession{Init,Sync} (inner LSA +
    outer GIN) over a live 2-rank comm. Same skip semantics as the GIN case.
    """
    proc, log = run_gin_mpi_gtest(
        "IRGinBarrierMPITest.C2_CompositeBarrierSession_TwoRanks",
        "ir_composite_barrier_mpi.log",
    )
    _summarize_gtest(proc, log, "composite barrier functional (2-rank)")


# ---------------------------------------------------------------------------
# [D] ReduceCopy typed C-ABI thunks (upstream NCCL bbb1375fa382)
# ---------------------------------------------------------------------------

# Independent oracle for the exported ReduceCopy surface. Deliberately NOT
# derived from bindings/ir/util.h, CMakeLists.txt or the Makefile, so a type
# or op silently dropped from any of them fails here.
_RC_ALL_TYPES = ["I8", "U8", "I32", "U32", "I64", "U64",
                 "F16", "F32", "F64", "BF16"]
_RC_MULTIMEM_TYPES = ["I32", "U32", "I64", "U64", "F16", "F32", "F64", "BF16"]
_RC_ALL_OPS = ["ncclLsaReduceSum", "ncclLsaCopy", "ncclLsaReduceSumCopy",
               "ncclLocalReduceSumCopy"]
_RC_MULTIMEM_OPS = ["ncclMultimemReduceSum", "ncclMultimemCopy",
                    "ncclMultimemReduceSumCopy"]
# RCCL's device ReduceSum/Copy has no fp8 support: the HIP bitcode must not
# advertise FP8 entry points that upstream (CUDA) exposes.
_RC_FP8_TYPES = ["F8E4M3", "F8E5M2"]

_RC_EXPECTED = (
    {f"{op}_{t}" for op in _RC_ALL_OPS for t in _RC_ALL_TYPES}
    | {f"{op}_{t}" for op in _RC_MULTIMEM_OPS for t in _RC_MULTIMEM_TYPES}
)
_RC_SYMBOL_RE = re.compile(
    r"^nccl(Lsa|Local|Multimem)(ReduceSum|Copy|ReduceSumCopy)_[A-Z0-9]+$")


def _llvm_nm(paths):
    for cand in (os.path.join(paths.ROCM_PATH, "lib", "llvm", "bin", "llvm-nm"),
                 os.path.join(paths.ROCM_PATH, "llvm", "bin", "llvm-nm")):
        if os.path.isfile(cand):
            return cand
    pytest.skip(f"llvm-nm not found under {paths.ROCM_PATH}")


@pytest.mark.ir_device
@pytest.mark.reduce_copy
def test_reduce_copy_symbol_inventory(ir_test_binary, paths):
    """[D-sym] librccl_device.bc exports exactly the 64 ReduceCopy thunks.

    Checks, at the symbol level of the shipped artifact:
      * every expected thunk is a defined, externally visible (T) function —
        catches a name missing from the opt internalize allow-list;
      * the names are unmangled (C linkage) — catches a lost extern "C";
      * no unexpected ReduceCopy-shaped symbols, in particular no FP8 ones.
    """
    nm = _llvm_nm(paths)
    proc = subprocess.run([nm, "--defined-only", paths.BITCODE],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    assert proc.returncode == 0, f"llvm-nm failed:\n{proc.stdout}"

    exported, internal = set(), set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        kind, name = parts[-2], parts[-1]
        if not _RC_SYMBOL_RE.match(name):
            continue
        (exported if kind == "T" else internal).add(name)

    missing = sorted(_RC_EXPECTED - exported)
    unexpected = sorted(exported - _RC_EXPECTED)
    assert not missing, (
        f"{len(missing)} ReduceCopy thunks not exported (internalized or "
        f"dropped): {missing}; non-exported definitions seen: {sorted(internal)}")
    assert not unexpected, f"unexpected ReduceCopy exports: {unexpected}"
    fp8 = sorted(s for s in exported | internal
                 if any(s.endswith("_" + t) for t in _RC_FP8_TYPES))
    assert not fp8, f"FP8 ReduceCopy thunks must not exist on HIP: {fp8}"


@pytest.mark.ir_device
@pytest.mark.reduce_copy
def test_reduce_copy_symbols_linkable(run_gtest):
    """[D0] Every ReduceCopy thunk resolves through the device-side LTO link."""
    proc, log = run_gtest("IRDeviceTest.D0_ReduceCopySymbolsLinkable",
                          "ir_reduce_copy_symbols.log")
    _summarize_gtest(proc, log, "ReduceCopy symbol link coverage")


@pytest.mark.ir_device
@pytest.mark.reduce_copy
def test_local_reduce_sum_copy_typed(run_gtest):
    """[D1] ncclLocalReduceSumCopy_<T> functional across all 10 types."""
    proc, log = run_gtest("IRReduceCopyLocalTest/*",
                          "ir_local_reduce_sum_copy.log")
    _summarize_gtest(proc, log, "ncclLocalReduceSumCopy_<T>")


@pytest.mark.ir_device
@pytest.mark.reduce_copy
@pytest.mark.mpi
@pytest.mark.parametrize("case", ["E1_LsaReduceSum", "E2_LsaCopy",
                                  "E3_LsaReduceSumCopy"])
def test_lsa_reduce_copy_typed_mpi(run_reducecopy_mpi_gtest, case):
    """[E1-E3] ncclLsa{ReduceSum,Copy,ReduceSumCopy}_<T> over a live LSA team,
    all 10 types, under `mpirun -np IR_RC_NRANKS` (default 2).

    SKIPPED (not PASSED) when the host cannot run it, e.g. no cuMem or fewer
    GPUs than ranks.
    """
    proc, log = run_reducecopy_mpi_gtest(f"IRReduceCopyMPITest/*.{case}",
                                         f"ir_rc_mpi_{case}.log")
    _summarize_gtest(proc, log, f"{case} (multi-rank)")
