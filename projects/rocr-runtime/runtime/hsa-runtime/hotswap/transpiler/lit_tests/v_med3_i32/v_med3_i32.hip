// ARCH: gfx1250
//
// Lift fixture for VOP3 v_med3_i32. Pins the signed median-of-three
// shape:
//   med3_i32(a, b, c) = smax(smin(a, b), smin(smax(a, b), c))
// — the canonical LLVM IR form the AMDGPU backend pattern-matches
// back to V_MED3_I32 via AMDGPUInstructions.td.
//
// See the V_MED3_I32 SemOp in transpiler/semop.hpp and the matching
// handler block in transpiler/handle_valu.cpp; this fixture pins the
// IR shape both reference (handler <-> test back-reference is
// non-negotiable per the transpiler test policy).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_med3_i32_kernel(int *out, const int *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int a = in[3 * tid + 0];
  int b = in[3 * tid + 1];
  int c = in[3 * tid + 2];
  int r;
  asm volatile("v_med3_i32 %[r], %[a], %[b], %[c]\n\t"
               : [r] "=v"(r)
               : [a] "v"(a), [b] "v"(b), [c] "v"(c));
  out[tid] = r;
}
