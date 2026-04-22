// ARCH: gfx1250
//
// Lift fixture for VOP3 v_mad_nc_u64_u32 (gfx1250+ no-carry unsigned
// 64-bit multiply-add).  Sibling of `v_mad_nc_i64_i32`; see the
// rationale block in that fixture's .hip for the full "handler ↔
// decoder ↔ test" back-reference policy, which applies identically
// here.
//
// v_mad_nc_u64_u32 is gfx1250-only, defined in
// AMDGPU/VOP3Instructions.td at encoding 0x2fa.  Same operand
// layout as the signed sibling (`VOP_I32_I32_I64_DPP` profile —
// `VOPProfile<[i64, i32, i32, i64]>`), but the 32→64 widening is
// zero-extending rather than sign-extending; the u64 accumulator
// operand is still an unsigned-interpreted 64-bit VGPR pair.
// Semantically `D.u64 = zext(S0.u32) * zext(S1.u32) + S2.u64` with
// no carry output.  The backend's
// `SelectMad64_32(Signed=false)` matcher (AMDGPUISelDAGToDAG.cpp:1220)
// re-emits `v_mad_nc_u64_u32` on gfx1250 and the older
// `v_mad_co_u64_u32` (carry-to-SGPR, discarded) on gfx942 from the
// same canonical `add(mul(zext s0, zext s1), s2_i64)` IR we emit.
// This fixture provides a sibling "prove the decoder + handler
// work" probe; no corpus kernel surfaces the unsigned variant
// today, but the opcode exists in gfx1250's VOP3 table and a future
// MXFP-family kernel using unsigned tensor indexing could hit it
// (same shape as the signed version in `downcast_to_mxfp_*`).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_mad_nc_u64_u32_kernel(unsigned long long *out,
                                                   const unsigned int *a_ptr,
                                                   const unsigned int *b_ptr,
                                                   const unsigned long long *c_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int a = a_ptr[tid];
  unsigned int b = b_ptr[tid];
  unsigned long long c = c_ptr[tid];
  unsigned long long result;
  asm volatile("v_mad_nc_u64_u32 %[r], %[a], %[b], %[c]\n\t"
               : [r] "=v"(result)
               : [a] "v"(a), [b] "v"(b), [c] "v"(c));
  out[tid] = result;
}
