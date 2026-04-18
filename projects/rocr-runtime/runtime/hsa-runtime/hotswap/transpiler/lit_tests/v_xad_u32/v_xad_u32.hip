// ARCH: gfx1250
//
// Lift fixture for VOP3 v_xad_u32 (gfx9+ ternary xor-then-add:
// `dst = (src0 ^ src1) + src2`). See the V_XAD_U32 SemOp in
// transpiler/semop.hpp and the matching handler block in
// transpiler/handle_valu.cpp; this fixture pins the IR shape
// both reference (handler ↔ test back-reference is
// non-negotiable per the transpiler test policy).
//
// VOP3 vector op — operands live in VGPRs natively; pointer-load
// the inputs to keep the compiler from constant-folding past the
// asm barrier.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_xad_u32_kernel(unsigned int *out,
                                            const unsigned int *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int a = in[3 * tid + 0];
  unsigned int b = in[3 * tid + 1];
  unsigned int c = in[3 * tid + 2];
  unsigned int r;
  asm volatile("v_xad_u32 %[r], %[a], %[b], %[c]\n\t"
               : [r] "=v"(r)
               : [a] "v"(a), [b] "v"(b), [c] "v"(c));
  out[tid] = r;
}
