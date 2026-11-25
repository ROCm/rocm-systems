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
    NCCL_IR_DEFINE_API_* lines of nccl_device_wrapper__impl.h). The bitcode
    checked here comes from the bindings/ir Makefile, so its public-API list
    is covered by what the bitcode exports; the -DEMIT_LLVM_IR=ON bitcode is
    covered by comparing the thunk list bindings/ir/CMakeLists.txt builds
    with the same matrix.
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
    problems = []

    def compare(where, have, only_thunks):
        missing = sorted(expected - have)
        unexpected = sorted(s for s in have - expected if not only_thunks or thunk.match(s))
        if missing:
            problems.append(f"{where}: {len(missing)} of {len(expected)} missing "
                            f"({', '.join(missing[:6])})")
        if unexpected:
            problems.append(f"{where}: {len(unexpected)} not in the src/util.h / nccl_device_wrapper__impl.h matrix "
                            f"({', '.join(unexpected[:6])})")

    compare(f"exported from {ir_bitcode}", exported, only_thunks=True)

    # CMake builds its own public-API list (foreach(_op ...) x foreach(_ty
    # ${<types var>})); expand it the same way from the set() type lists.
    with open(os.path.join(ir_dir, "CMakeLists.txt")) as fh:
        cmake = fh.read()
    cmake_types = {name: vals.split() for name, vals in
                   re.findall(r"^\s*set\((RCCL_IR_\w+_TYPES)\s+([^)]*)\)", cmake, re.M)}
    loops = re.findall(r"foreach\(_op\s+([^)]*)\)\s*foreach\(_ty\s+\$\{(\w+)\}\)", cmake)
    assert loops and all(var in cmake_types for _, var in loops), (
        f"could not read the ReduceCopy thunk list from bindings/ir/CMakeLists.txt: "
        f"loops={loops} types={cmake_types}")
    compare("bindings/ir/CMakeLists.txt public API",
            {f"{op}_{t}" for op_list, var in loops
             for op in op_list.split() for t in cmake_types[var]},
            only_thunks=False)

    assert not problems, "typed ReduceCopy thunks: " + "; ".join(problems)


@pytest.mark.ir_device
@pytest.mark.reduce_copy
def test_reduce_copy_typed_local(run_gtest, summarize_gtest, paths, monkeypatch):
    """[D1] ncclLocalReduceSumCopy_{I8,F32,BF16}: 3 sources -> 2 destinations."""
    # D1 raises the stack limit (to 32 KiB). HSA_NO_SCRATCH_RECLAIM=1 makes
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
        with open(log, errors="replace") as fh:
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


# Every device-API helper that does a system-scope 128-bit access. librccl_device.bc covers only
# the GIN proxy descriptor store: the LL all-to-all, flow LL/LL128 and hip_compat helpers are not in
# its public API. One kernel per helper, so the check can name the helper that regressed.
_B128_PROBE_SRC = r"""
#include "nccl.h"
#include "nccl_device.h"
// nccl_device.h pulls the flow helpers in only under __CUDACC_EXTENDED_LAMBDA__; they stand alone.
#include "nccl_device/impl/flow__ll_funcs.h"
#include "nccl_device/impl/flow__ll128_funcs.h"
extern "C" __global__ void compat_store(void* p, uint32_t a) { nccl_st_volatile_v4_u32(p, a, a + 1, a + 2, a + 3); }
extern "C" __global__ void compat_load(const void* p, uint32_t* o) {
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
extern "C" __global__ void flow_ll_store(ncclLLFifoLine* p, uint64_t d, uint32_t f) { ncclFlowLLStore(p, d, f); }
extern "C" __global__ void flow_ll_load(ncclLLFifoLine* p, uint32_t f, uint32_t* abortFlag, uint64_t* o) {
  o[0] = ncclFlowLLLoad(p, f, abortFlag);
}
extern "C" __global__ void flow_ll128_store(uint64_t* p, uint64_t lo, uint64_t hi) { ncclFlowLL128Store128(p, lo, hi); }
extern "C" __global__ void flow_ll128_load(const uint64_t* p, uint64_t* o) {
  uint64_t lo, hi;
  ncclFlowLL128Load128(p, lo, hi);
  o[0] = lo ^ hi;
}
"""

_PROBE_KERNELS = ("compat_store", "compat_load", "lla2a_store", "lla2a_load",
                  "flow_ll_store", "flow_ll_load", "flow_ll128_store", "flow_ll128_load")
# Kernels whose helper takes its 128-bit system-scope path on each arch: hip_compat only on
# gfx942/gfx950, and the flow LL FIFO only on gfx1250 (RCCL_LL_FIFO_SYS_SCOPE).
_PROBE_B128_KERNELS = {
    "gfx942": set(_PROBE_KERNELS) - {"flow_ll_store", "flow_ll_load"},
    "gfx950": set(_PROBE_KERNELS) - {"flow_ll_store", "flow_ll_load"},
    "gfx1250": set(_PROBE_KERNELS) - {"compat_store", "compat_load"},
}

_LEGACY_B128 = r"@llvm\.amdgcn\.global\.(?:store|load)\.b128\b"


def _has_av_b128(paths, arch):
    """True if the compiler offers __builtin_amdgcn_av_{store,load}_b128 for `arch`; False if not, or if
    it cannot target `arch` at all."""
    proc = subprocess.run(
        [paths.AMDCLANGPP, "-x", "hip", "--cuda-device-only", f"--offload-arch={arch}", "-E", "-"],
        input="#if __has_builtin(__builtin_amdgcn_av_store_b128) && __has_builtin(__builtin_amdgcn_av_load_b128)\n"
        "RCCL_AV_B128_AVAILABLE\n#endif\n",
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and "RCCL_AV_B128_AVAILABLE" in proc.stdout


def _kernel_bodies(ir):
    """Map each `define` in textual IR to its body."""
    return {m.group(1): m.group(2) for m in re.finditer(r"^define [^\n]*@(\w+)\([^\n]*\{\n(.*?)^\}", ir, re.S | re.M)}


@pytest.mark.ir_device
def test_b128_accesses_use_upstream_builtins(ir_bitcode, paths, tmp_path):
    """The device API's 128-bit system-scope accesses use __builtin_amdgcn_av_{store,load}_b128
    when the compiler has them. The AMD-only llvm.amdgcn.global.{store,load}.b128 they replace
    abort upstream-LLVM consumers such as Triton, while AMD's LLVM lowers both to the same
    instructions, so only the IR can tell them apart. Checks librccl_device.bc (the GIN proxy
    descriptor store) and a probe of the LL all-to-all, flow LL/LL128 and hip_compat helpers, which
    the bitcode does not contain. The probe is also compiled for gfx1250, where the flow LL FIFO
    takes its 128-bit path, when the compiler can target it.
    """
    if not _has_av_b128(paths, paths.ARCH):
        pytest.skip(f"compiler lacks __builtin_amdgcn_av_{{store,load}}_b128 for {paths.ARCH}")

    bitcode_ir = subprocess.run([os.path.join(paths.LLVM_BIN, "llvm-dis"), ir_bitcode, "-o", "-"], capture_output=True, text=True, check=True).stdout
    legacy = sorted(set(re.findall(_LEGACY_B128, bitcode_ir)))
    assert not legacy, f"{ir_bitcode} still calls the AMD-only {legacy}; use RCCL_STORE_B128_SYS / RCCL_LOAD_B128_SYS"
    # Not vacuous: the GIN barrier's proxy descriptor writes are 128-bit stores.
    assert re.search(r"@llvm\.amdgcn\.av\.store\.b128", bitcode_ir), f"no 128-bit system-scope store in {ir_bitcode}"

    src = tmp_path / "b128_probe.hip"
    src.write_text(_B128_PROBE_SRC)
    archs = [paths.ARCH] + (["gfx1250"] if paths.ARCH != "gfx1250" and _has_av_b128(paths, "gfx1250") else [])
    for arch in archs:
        probe = subprocess.run(
            [paths.AMDCLANGPP, "-x", "hip", "--cuda-device-only", f"--offload-arch={arch}", "-O3",
             "-D__HIP_PLATFORM_AMD__=1", "-DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1",
             f"-I{paths.HIPIFY_INC}", f"-I{os.path.join(paths.HIPIFY_INC, 'nccl_device')}", f"-I{paths.GENERATED_INC}",
             "-emit-llvm", "-S", "-o", "-", str(src)],
            capture_output=True, text=True,
        )
        assert probe.returncode == 0, f"compiling the b128 probe for {arch} failed:\n{probe.stderr[-2000:]}"
        bodies = _kernel_bodies(probe.stdout)
        assert set(_PROBE_KERNELS) <= set(bodies), f"{arch}: probe kernels missing from IR: {set(_PROBE_KERNELS) - set(bodies)}"
        legacy = {k for k in _PROBE_KERNELS if re.search(_LEGACY_B128, bodies[k])}
        assert not legacy, f"{arch}: these helpers still call the AMD-only global b128 builtins: {sorted(legacy)}"
        # Not vacuous: every helper with a 128-bit path on this arch must reach the av intrinsic.
        missing = {k for k in _PROBE_B128_KERNELS.get(arch, ()) if not re.search(r"@llvm\.amdgcn\.av\.(?:store|load)\.b128", bodies[k])}
        assert not missing, f"{arch}: no av b128 access in {sorted(missing)}"
