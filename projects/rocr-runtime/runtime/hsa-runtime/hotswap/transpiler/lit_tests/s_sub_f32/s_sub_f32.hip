// ARCH: gfx1250
//
// Lift fixture for SOP2 s_sub_f32 (gfx11+ scalar FP subtract). See
// the S_SUB_F32 SemOp in transpiler/semop.hpp and the matching
// handler block in transpiler/handle_sop2.cpp; this fixture pins
// the IR shape both reference (handler ↔ test back-reference is
// non-negotiable per the transpiler test policy).
//
// Operands are funneled through __builtin_amdgcn_readfirstlane on
// their i32 bit-patterns to satisfy the inline asm "s" constraints
// (kernarg-broadcast values land in VGPRs by default on gfx1250).
// Same idiom as the s_mul_hi_i32 / s_cmov_b32 fixtures next door.
// Operand and result are typed as `int` and reinterpreted on both
// sides to keep the asm operand-class consistent — the SOP2
// encoding takes plain SGPRs irrespective of the f32-ness.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_sub_f32_kernel(float *out, float a, float b) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int sa_bits =
      __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, a));
  int sb_bits =
      __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, b));
  int result_bits;
  asm volatile("s_sub_f32 %[r], %[a], %[b]\n\t"
               : [r] "=s"(result_bits)
               : [a] "s"(sa_bits), [b] "s"(sb_bits));
  out[tid] = __builtin_bit_cast(float, result_bits);
}
