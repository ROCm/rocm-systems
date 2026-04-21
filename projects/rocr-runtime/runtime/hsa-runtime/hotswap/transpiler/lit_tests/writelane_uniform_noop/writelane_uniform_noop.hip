// ARCH: gfx1250
//
// Fixture for the uniform-operand `v_writelane_b32` no-op (lit_tests/
// writelane_uniform_noop). Pins the half of
// `rewrite_cross_lane_divergent.cpp` that LEAVES a writelane site
// untouched when both the scalar `val` operand AND the `old` operand
// are provably wave-uniform — the dominant Tensile / rocBLAS
// writelane shape.
//
// This fixture is the correctness counterpart to
// `writelane_divergent_rewrite`: without a uniform no-op guard the
// rewrite pass would unnecessarily convert every well-behaved
// writelane in the corpus into a per-lane `select` + `mbcnt` helper,
// regressing codegen quality on every kernel that does not actually
// leak cross-widen-divergent state.
//
// Shape. `val` = `blockIdx.x` (uniform — `llvm.amdgcn.workgroup.id.x`
// is workgroup-invariant, therefore wave-uniform). `old` = literal
// integer 0 forced via `+v` tied constraint so hipcc emits
// `v_mov_b32 v<n>, 0` before the writelane; under cross-widening the
// raiser sees a `ConstantInt 0` stored into the output VGPR before
// the writelane lifts, keeping `old` uniform in the oracle's view.

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
