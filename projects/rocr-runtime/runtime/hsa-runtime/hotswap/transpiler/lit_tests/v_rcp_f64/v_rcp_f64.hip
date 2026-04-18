// ARCH: gfx1250
//
// Lift fixture for the VOP1 transcendental v_rcp_f64 (TRANS-class
// 64-bit reciprocal approximation). See the V_RCP_F64 SemOp in
// transpiler/semop.hpp and the matching handler block in
// transpiler/handle_valu.cpp; this fixture pins the IR shape both
// reference (handler ↔ test back-reference is non-negotiable per the
// transpiler test policy).
//
// Why inline asm instead of a C `1.0 / s` divide:
//   * The AMDGPU backend lowers a default-precision `fdiv 1.0, x` on
//     gfx1250 to a software divide sequence (mul + Newton-Raphson
//     refinement on top of v_rcp_f64), which would not produce a
//     bare v_rcp_f64 in the HSACO. Forcing it via `__builtin_amdgcn_rcp`
//     also routes through type-promoted overloads.
//   * The simplest way to pin "hardware emits a single v_rcp_f64" is
//     inline asm, matching the style of the other decoder/lift
//     fixtures (e.g. decoder_madmk_v_fmamk_f32.hip).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_rcp_f64_kernel(double *out, double *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double s = in[tid];
  double r;
  asm volatile("v_rcp_f64 %[r], %[s]\n\t"
               : [r] "=v"(r)
               : [s] "v"(s));
  out[tid] = r;
}
