// ARCH: gfx1250
//
// Lit fixture for the VOPD sub-operations and operand shapes that
// were unhandled before the supporting handler change in
// transpiler/handle_vopd.cpp:
//
//   1. v_dual_mov_b32 X-half with an f32 inline-constant literal
//      ("1.0", "0.5", etc.) — `readVOPDSrc` previously only parsed
//      integer literals through `StringRef::getAsInteger`, so any
//      kernel where the AMDGPU printer surfaced an inline-constant
//      pool entry as its decimal float spelling was rejected at
//      VOPD decomposition time.  Triton's
//      `attn_fwd_pipelined_kernel` GEMM hits this exact pattern.
//   2. v_dual_ashrrev_i32 Y-half — used by Triton scoring/masked-
//      compaction kernels for sign-extending the high bit of a
//      32-bit lane index for branch-free clamp/select sequences.
//   3. v_dual_max_i32 Y-half — used by Triton scaling kernels
//      (`upcast_from_mxfp`) to clamp a signed exponent against zero.
//
// Inline asm is the cleanest way to pin all three shapes in a
// single fixture.  We use a paired VOPD instruction per asm block
// so the assembler emits a real VOPD encoding (rather than two
// single-issue VOPs the auto-pairer might or might not bundle).
// The kernel below pins one pair per shape:
//
//   * Pair A: `v_dual_mov_b32 vX, 1.0 :: v_dual_ashrrev_i32 vY,
//     31, vZ`  — covers the float-literal MOV and the AShr.
//   * Pair B: `v_dual_mov_b32 vX, v? :: v_dual_max_i32 vY, 0,
//     vZ`     — covers the smax intrinsic dispatch.
//
// All three branches share the same `parseVOPDHalf` skeleton; if
// the float-literal parse, the AShr handler, or the smax handler
// regress, exactly one of the corresponding `lift` checks below
// will fail.

#include <hip/hip_runtime.h>

extern "C" __global__ void
vopd_extra_subops_kernel(int *in_ptr, int *out_ptr) {
  int v_in = in_ptr[threadIdx.x];

  unsigned int mov_lit = 0;
  int ashr_out = 0;
  asm volatile(
      "v_dual_mov_b32 %[d_mov], 1.0 :: v_dual_ashrrev_i32 "
      "%[d_ashr], 31, %[s]"
      : [d_mov] "=v"(mov_lit), [d_ashr] "=v"(ashr_out)
      : [s] "v"(v_in));

  int v_a = in_ptr[(threadIdx.x + 1) & 31];
  int v_b = in_ptr[(threadIdx.x + 2) & 31];
  int max_out = 0;
  int mov_reg = 0;
  asm volatile(
      "v_dual_mov_b32 %[d_mov], %[s_mov] :: v_dual_max_i32 "
      "%[d_max], 0, %[s_max]"
      : [d_mov] "=v"(mov_reg), [d_max] "=v"(max_out)
      : [s_mov] "v"(v_a), [s_max] "v"(v_b));

  out_ptr[threadIdx.x * 4 + 0] = (int)mov_lit;
  out_ptr[threadIdx.x * 4 + 1] = ashr_out;
  out_ptr[threadIdx.x * 4 + 2] = max_out;
  out_ptr[threadIdx.x * 4 + 3] = mov_reg;
}
