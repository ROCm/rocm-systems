// ARCH: gfx1250
//
// Lift fixture for VOP3 v_minmax_num_f32 (gfx12 IEEE-2019
// renamed v_minmax_f32). Pins:
//   dst = minnum(maxnum(s0, s1), s2)
// — clamps the result to at most s2 but no less than the larger
// of (s0, s1). The .NUM suffix carries IEEE-754 2008 minNum/maxNum
// semantics (NaN-pruning), distinct from the IEEE-754 2019
// V_MINIMUMMAXIMUM_F32 (NaN-propagating, opcode 0x26c).
//
// See the V_MINMAX_NUM_F32 SemOp in transpiler/semop.hpp and the
// matching handler block in transpiler/handle_valu.cpp; this
// fixture pins the IR shape both reference (handler ↔ test
// back-reference is non-negotiable per the transpiler test
// policy).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_minmax_num_f32_kernel(float *out,
                                                    const float *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a = in[3 * tid + 0];
  float b = in[3 * tid + 1];
  float c = in[3 * tid + 2];
  float r;
  asm volatile("v_minmax_num_f32 %[r], %[a], %[b], %[c]\n\t"
               : [r] "=v"(r)
               : [a] "v"(a), [b] "v"(b), [c] "v"(c));
  out[tid] = r;
}
