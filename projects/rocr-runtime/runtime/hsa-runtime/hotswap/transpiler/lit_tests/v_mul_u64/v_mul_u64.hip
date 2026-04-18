// ARCH: gfx1250
//
// Lift fixture for VOP2 v_mul_u64 (gfx1250+ vector 64-bit unsigned
// multiply, low 64 bits). See the V_MUL_U64 SemOp in
// transpiler/semop.hpp and the matching handler block in
// transpiler/handle_valu.cpp; this fixture pins the IR shape both
// reference (handler ↔ test back-reference is non-negotiable per the
// transpiler test policy).
//
// V_MUL_U64 is gfx1250-only, defined in
// AMDGPU/VOP2Instructions.td:942 as
//   defm V_MUL_U64 : VOP2Inst <"v_mul_u64", VOP_I64_I64_I64,
//                              DivergentBinFrag<mul>>;
// Its SDPattern is the divergent form of plain `mul`, so the
// handler emits a single i64 `mul` against the two 64-bit VGPR
// pairs. We let the inputs be ordinary VGPR-resident u64 values
// here (no readfirstlane); v_mul_u64 takes VSrc_b64 operands and
// the dest is a VReg_64 (AlignTarget) — see the matcher entry in
// AMDGPUGenAsmMatcher.inc for V_MUL_U64_e64_gfx1250.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_mul_u64_kernel(unsigned long long *out,
                                            const unsigned long long *a_ptr,
                                            const unsigned long long *b_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long a = a_ptr[tid];
  unsigned long long b = b_ptr[tid];
  unsigned long long result;
  asm volatile("v_mul_u64 %[r], %[a], %[b]\n\t"
               : [r] "=v"(result)
               : [a] "v"(a), [b] "v"(b));
  out[tid] = result;
}
