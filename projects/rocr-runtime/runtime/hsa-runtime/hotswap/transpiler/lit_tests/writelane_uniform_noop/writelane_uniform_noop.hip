// ARCH: gfx1250
//
// Fixture for the uniform-operand `v_writelane_b32` symmetry
// contract (lit_tests/writelane_uniform_noop). Pins the half of
// `rewrite_cross_lane_divergent.cpp` that REWRITES a writelane site
// even when both the scalar `val` operand AND the `old` operand are
// provably wave-uniform — the dominant Tensile / rocBLAS writelane
// shape.
//
// Naming: the directory keeps the historical `_noop` suffix because
// renaming lit fixtures churns the CMake glob and references in
// docs, but the checked contract has inverted: the pass is no longer
// a no-op on uniform operands. See the regenerated `.ll` companion
// for the full rationale, or hotswap/docs/wave-size-translation.md
// §5.6.3 "symmetry contract" for the cross-reference into the design
// doc.
//
// Why the contract changed. The original rule ("preserve uniform
// writelanes") left hardware lane N + W_s unpopulated, which paired
// unsoundly with any `ds_bpermute`-rewritten readlane on the same
// VGPR (readlane reads BOTH N and N+W_s, sees undef on the upper
// half, feeds it into i64 address math, faults at dispatch). The
// symmetry contract rewrites every writelane site unconditionally.
// Uniform operands still produce identical values at lanes N and
// N+W_s, so the rewrite is correctness-preserving — the only cost
// is a trivial codegen tax (one extra `select`).
//
// Shape. `val` = `blockIdx.x` (uniform — `llvm.amdgcn.workgroup.id.x`
// is workgroup-invariant, therefore wave-uniform). `old` = literal
// integer 0 forced via `+v` tied constraint so hipcc emits
// `v_mov_b32 v<n>, 0` before the writelane; under cross-widening the
// raiser sees a `ConstantInt 0` stored into the output VGPR before
// the writelane lifts. Both operands are uniform; the rewrite fires
// anyway.

#include <hip/hip_runtime.h>

extern "C" __global__ void writelane_uniform_noop_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // `val` = uniform workgroup_id. `old` = uniform 0 via the `+v`
  // tied constraint's pre-asm materialisation. Both operands are
  // uniform in the oracle's view -> `rewriteCrossLaneDivergent`
  // increments `uniformPreserved` and leaves the intrinsic in place.
  int wg_id = blockIdx.x;
  int lane_slot = 0;
  asm volatile("v_writelane_b32 %[out], %[val], 0\n\t"
               : [out] "+v"(lane_slot)
               : [val] "s"(wg_id));

  out[tid] = lane_slot;
}
