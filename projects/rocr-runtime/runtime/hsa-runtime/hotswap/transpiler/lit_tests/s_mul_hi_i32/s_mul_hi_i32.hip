// ARCH: gfx1250
//
// Lift fixture for SOP2 s_mul_hi_i32 (scalar signed mul-high). See
// the S_MUL_HI_I32 SemOp in transpiler/semop.hpp and the matching
// handler block in transpiler/handle_sop2.cpp; this fixture pins
// the IR shape both reference (handler ↔ test back-reference is
// non-negotiable per the transpiler test policy).
//
// Operands are funneled through __builtin_amdgcn_readfirstlane to
// satisfy the inline asm "s" constraints (the kernarg-broadcast
// values land in VGPRs by default on gfx1250 and the assembler
// rejects mixed s/v operands on s_mul_hi_i32). Same idiom as the
// s_cmov_b32 fixture next door.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_mul_hi_i32_kernel(int *out, int a, int b) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int sa = __builtin_amdgcn_readfirstlane(a);
  int sb = __builtin_amdgcn_readfirstlane(b);
  int result;
  asm volatile("s_mul_hi_i32 %[r], %[a], %[b]\n\t"
               : [r] "=s"(result)
               : [a] "s"(sa), [b] "s"(sb));
  out[tid] = result;
}
