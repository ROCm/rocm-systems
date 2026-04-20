// ARCH: gfx1250
//
// Lift fixture for VOP3 v_bfi_b32 (bit-field insert).  Hardware
// semantic (AMDGPUbfiPattern in VOP3Instructions.td):
//   dst = (src0 & src1) | (~src0 & src2)
// i.e. src0 is the per-bit mask, src1 provides bits where the mask
// is 1, src2 provides bits where the mask is 0.  See the
// V_BFI_B32 SemOp in transpiler/semop.hpp and the matching handler
// block in transpiler/handle_valu.cpp; this fixture pins the IR
// shape both reference (handler <-> test back-reference is
// non-negotiable per the transpiler test policy).
//
// Why inline asm: hipcc aggressively decomposes the ternary mask-
// and-merge pattern to `and` / `andn2` / `or` separately and rarely
// emits the fused `v_bfi_b32` opcode from C source alone.  The
// kerneldex corpus producer that actually exercises the opcode is
// libdevice asin (see kernels/triton/corpus_asin_fp32.py in the
// compare_correctness tool), which lowers copysign-style sign-stitch
// idioms to v_bfi_b32.  Inline asm pins the gfx1250 binary to the
// real opcode so the lift is exercised.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_bfi_b32_kernel(unsigned int *out,
                                            const unsigned int *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int mask = in[3 * tid + 0];
  unsigned int one_src = in[3 * tid + 1];
  unsigned int zero_src = in[3 * tid + 2];
  unsigned int r;
  asm volatile("v_bfi_b32 %0, %1, %2, %3"
               : "=v"(r)
               : "v"(mask), "v"(one_src), "v"(zero_src));
  out[tid] = r;
}
