// ARCH: gfx1250
//
// Lift fixture for VOP3 v_alignbit_b32 (gfx6+ funnel-shift
// right):  dst = ((src0 << 32) | src1) >> (src2 & 0x1F)) [31:0].
// See the V_ALIGNBIT_B32 SemOp in transpiler/semop.hpp and the
// matching handler block in transpiler/handle_valu.cpp; this
// fixture pins the IR shape both reference (handler ↔ test
// back-reference is non-negotiable per the transpiler test
// policy).
//
// VOP3 vector op — operands live in VGPRs natively; pointer-load
// the inputs to keep the compiler from constant-folding past
// the asm barrier. We use inline asm rather than the natural
// `__builtin_amdgcn_alignbit` (which exists pre-LLVM-19 only)
// or a generic funnel-shift expression (the AMDGPU backend
// would constant-fold a HIP-level fshr around the asm if we
// folded the operands inline).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_alignbit_b32_kernel(unsigned int *out,
                                                 const unsigned int *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int hi = in[3 * tid + 0];
  unsigned int lo = in[3 * tid + 1];
  unsigned int sh = in[3 * tid + 2];
  unsigned int r;
  asm volatile("v_alignbit_b32 %[r], %[h], %[l], %[s]\n\t"
               : [r] "=v"(r)
               : [h] "v"(hi), [l] "v"(lo), [s] "v"(sh));
  out[tid] = r;
}
