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


# Every device-API helper that does a system-scope 128-bit access. librccl_device.bc covers only
# the GIN proxy descriptor store: the LL all-to-all and hip_compat helpers are not in its public API.
_B128_PROBE_SRC = r"""
#include "nccl.h"
#include "nccl_device.h"
extern "C" __global__ void st_volatile(void* p, uint32_t a) { nccl_st_volatile_v4_u32(p, a, a + 1, a + 2, a + 3); }
extern "C" __global__ void ld_volatile(const void* p, uint32_t* o) {
  uint32_t a, b, c, d;
  nccl_ld_volatile_v4_u32(p, a, b, c, d);
  o[0] = a ^ b ^ c ^ d;
}
extern "C" __global__ void lla2a_store(uint32_t* p, uint32_t a) { amdLLA2aStoreLine(p, a, a + 1, a + 2, a + 3); }
extern "C" __global__ void lla2a_load(const uint32_t* p, uint32_t* o) {
  uint32_t a, b, c, d;
  amdLLA2aLoadLine(p, a, b, c, d);
  o[0] = a ^ b ^ c ^ d;
}
"""

_LEGACY_B128 = r"@llvm\.amdgcn\.global\.(?:store|load)\.b128\b"


@pytest.mark.ir_device
def test_b128_accesses_use_upstream_builtins(ir_bitcode, paths, tmp_path):
    """The device API's 128-bit system-scope accesses use __builtin_amdgcn_av_{store,load}_b128
    when the compiler has them. The AMD-only llvm.amdgcn.global.{store,load}.b128 they replace
    abort upstream-LLVM consumers such as Triton, while AMD's LLVM lowers both to the same
    instructions, so only the IR can tell them apart. Checks librccl_device.bc (the GIN proxy
    descriptor store) and a probe of the LL all-to-all and hip_compat helpers, which the bitcode
    does not contain.
    """
    has_av = subprocess.run(
        [paths.AMDCLANGPP, "-x", "hip", "--cuda-device-only", f"--offload-arch={paths.ARCH}", "-E", "-"],
        input="#if __has_builtin(__builtin_amdgcn_av_store_b128) && __has_builtin(__builtin_amdgcn_av_load_b128)\n"
        "RCCL_AV_B128_AVAILABLE\n#endif\n",
        capture_output=True, text=True, check=True,
    )
    if "RCCL_AV_B128_AVAILABLE" not in has_av.stdout:
        pytest.skip(f"compiler lacks __builtin_amdgcn_av_{{store,load}}_b128 for {paths.ARCH}")

    bitcode_ir = subprocess.run([paths.LLVM_DIS, ir_bitcode, "-o", "-"], capture_output=True, text=True, check=True).stdout
    legacy = sorted(set(re.findall(_LEGACY_B128, bitcode_ir)))
    assert not legacy, f"{ir_bitcode} still calls the AMD-only {legacy}; use RCCL_STORE_B128_SYS / RCCL_LOAD_B128_SYS"
    # Not vacuous: the GIN barrier's proxy descriptor writes are 128-bit stores.
    assert re.search(r"@llvm\.amdgcn\.av\.store\.b128", bitcode_ir), f"no 128-bit system-scope store in {ir_bitcode}"

    src = tmp_path / "b128_probe.hip"
    src.write_text(_B128_PROBE_SRC)
    probe = subprocess.run(
        [paths.AMDCLANGPP, "-x", "hip", "--cuda-device-only", f"--offload-arch={paths.ARCH}", "-O3",
         "-D__HIP_PLATFORM_AMD__=1", "-DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1",
         f"-I{paths.HIPIFY_INC}", f"-I{os.path.join(paths.HIPIFY_INC, 'nccl_device')}", f"-I{paths.GENERATED_INC}",
         "-emit-llvm", "-S", "-o", "-", str(src)],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0, f"compiling the b128 probe failed:\n{probe.stderr[-2000:]}"
    legacy = sorted(set(re.findall(_LEGACY_B128, probe.stdout)))
    assert not legacy, f"the LL all-to-all / hip_compat helpers still call the AMD-only {legacy}"
    for op in ("store", "load"):
        assert re.search(rf"@llvm\.amdgcn\.av\.{op}\.b128", probe.stdout), f"no av.{op}.b128 in the helper probe"
