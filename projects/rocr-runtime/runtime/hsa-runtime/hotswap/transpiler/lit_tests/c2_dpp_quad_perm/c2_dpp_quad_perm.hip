// ARCH: gfx1250
//
// Fixture for the Class-2 DPP-modifier cross-lane sub-category
// (lit_tests/c2_dpp_quad_perm).
//
// SPE_DESIGN.md §3 Class 2 / gpt-oss-derisking.md §7.3 flag DPP
// modifiers as the single largest outstanding correctness hazard in
// the GPT-OSS corpus today — 5 of 15 kernels use DPP. The raiser
// canonicalises DPP modifiers away before any handler sees them
// (opcode_map.cpp:buildDppToBaseMap), producing same-lane IR for an
// instruction whose intent is cross-lane. Until CROSS_LANE_SURVEY.md
// item P5 (`update.dpp` intrinsic lift) lands, every kernel that
// uses any DPP modifier is silently miscompiled on the cross-wave
// path.
//
// This fixture pins `v_mov_b32_dpp quad_perm:[1,0,3,2]` — a
// pairwise-swap shuffle inside each 4-lane quad. It is one of the
// simplest DPP patterns and matches the shape emitted by the
// bitmatrix_metadata_compute_stage{1,2} kernels in GPT-OSS.
//
// MAINTENANCE. Once P5 lands, this test should be flipped (see the
// c2_permlane_swap.ll MAINTENANCE block for the same protocol).

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_dpp_quad_perm_kernel(int *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[tid];
  // quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
  asm volatile(
      "v_mov_b32_dpp %[v], %[v] quad_perm:[1,0,3,2] row_mask:0xf "
      "bank_mask:0xf bound_ctrl:1\n\t"
      : [v] "+v"(v));
  a[tid] = v;
}
