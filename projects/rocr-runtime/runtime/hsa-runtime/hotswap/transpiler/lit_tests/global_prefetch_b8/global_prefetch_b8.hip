// ARCH: gfx1250
//
// Lit fixture for FLAT `global_prefetch_b8` —
// the gfx1250-only VMEM prefetch hint that brings global memory into
// the GL2 cache. Exercises both the no-offset path and the byte
// pointer-offset (GEP) path, the two shapes the AMDGPU backend lowers
// to `global_prefetch_b8 v[..], off [offset:N]`.
//
// === Why this kernel must NEVER be dispatched ===
//
// The clang builtin `__builtin_amdgcn_global_prefetch` lowers to the
// FLAT VFLAT 0x05D real (`global_prefetch_b8`), which only executes
// on gfx1250 hardware (`FeatureVmemPrefInsts`, gated by
// `vmem-pref-insts`). The fixture exists exclusively to compile a
// `.co` file that the raiser can disassemble and lift; running the
// kernel on any non-gfx1250 device would surface as a hardware
// exception.
//
// === Companion lit tests ===
//
// `global_prefetch_b8.ll` (this directory) — pins the cross-target
// (gfx1250 → gfx942) loud-refusal contract. The handler in
// `handle_flat.cpp` refuses via `RaiseFailure::unsupportedShape`
// because gfx942 has no VMEM-prefetch unit and the matching LLVM
// intrinsic `int_amdgcn_global_prefetch`
// (IntrinsicsAMDGPU.td:3211) is gated by `HasVmemPrefInsts`. The
// closest sibling `amdgcn.s.prefetch.data` requires a UNIFORM SGPR
// pointer, which we cannot prove for the divergent VGPR address
// here without divergence analysis — silently dropping the hint
// would mask both the cross-target capability gap and any
// software-pipelining regression downstream (the gfx1250 producers
// that use this — Triton block-pipelined matmul kernels — rely on
// the prefetch overlapping with prior MFMA chains).
//
// `global_prefetch_b8_same_target.ll` (this directory) — pins the
// same-target (gfx1250 → gfx1250) intrinsic-emit shape:
// `inttoptr i64 → ptr addrspace(1)` for the global address (or
// `inreg` SGPR-base + VGPR voff sum for the SADDR variant), an
// optional non-inbounds `i8`-typed GEP for the FLAT
// `flat_offset` immediate (named `prefetch_addr` when emitted),
// and the canonical
// `call void @llvm.amdgcn.global.prefetch(ptr addrspace(1), i32)`
// with the `cpol` immediate threaded through. The call is NOT
// wrapped in `ctx.emitUnderExec` because the intrinsic carries
// the EXEC mask implicitly (`IntrInaccessibleMemOrArgMemOnly`
// — semantically a hint, with no observable side effect on
// inactive lanes).
//
// === Builtin signature (BuiltinsAMDGPU.def via #150466) ===
//
//   void __builtin_amdgcn_global_prefetch(
//       const __attribute__((address_space(1))) void *gaddr,
//       unsigned int                                 cpol);
//
// The second argument encodes the cache-scope / hint bits
// (TH_LOAD_NT, SCOPE_SE, etc.) — we exercise `cpol = 8`
// (SCOPE_SE) here, matching the dominant Triton corpus shape
// the kerneldex sweep observed.

#include <hip/hip_runtime.h>

extern "C" __global__ void
global_prefetch_b8_kernel(int *gptr) {
  int tid = (int)threadIdx.x;
  // Plain (no flat_offset) form — lowers to
  //   global_prefetch_b8 v[..], off
  __builtin_amdgcn_global_prefetch(
      __builtin_addressof(gptr[tid]),
      /*cpol=*/8);
  // Byte-offset (flat_offset != 0) form — the backend folds the
  //   getelementptr i8, ptr addrspace(1) %p, i64 256
  // tail onto the FLAT instruction's `offset:` immediate, so the
  // emitted real is
  //   global_prefetch_b8 v[..], off offset:256
  __builtin_amdgcn_global_prefetch(
      (const __attribute__((address_space(1))) void *)((const char __attribute__((address_space(1))) *)
          __builtin_addressof(gptr[tid]) + 256),
      /*cpol=*/8);
}
