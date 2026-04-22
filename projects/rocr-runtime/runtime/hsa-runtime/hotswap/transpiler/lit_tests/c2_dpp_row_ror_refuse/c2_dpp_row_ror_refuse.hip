// ARCH: gfx1250
//
// Negative fixture for the DPP cross-widen rewrite.  Pins the
// refusal path for `dpp_ctrl` values OUTSIDE the supported family
// (quad_perm / row_shl / row_shr).  Exercises `row_ror:1` (ctrl =
// 0x121), which the `isDppCtrlRewritable` predicate in
// `rewrite_cross_lane_divergent.cpp` explicitly refuses.
//
// Per the project's "fail loud" rule, this path must surface a
// principled refusal with a readable diagnostic — NOT a silent
// pass-through that would leave the `row_ror` DPP's wave-size-
// dependent semantics intact across the source/target boundary.
// The companion positive fixture is `c2_dpp_quad_perm.ll`.
//
// When `row_ror` lowering lands (add the case to `buildDppLaneMap`
// and the branch to `isDppCtrlRewritable`), this fixture should be
// flipped to a POSITIVE test of the rewrite shape — same protocol
// as the c2_dpp_quad_perm fixture's MAINTENANCE block.

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_dpp_row_ror_refuse_kernel(int *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[tid];
  // row_ror:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
  // ctrl = DPP::ROW_ROR0 + 1 = 0x121
  asm volatile(
      "v_mov_b32_dpp %[v], %[v] row_ror:1 row_mask:0xf "
      "bank_mask:0xf bound_ctrl:1\n\t"
      : [v] "+v"(v));
  a[tid] = v;
}
