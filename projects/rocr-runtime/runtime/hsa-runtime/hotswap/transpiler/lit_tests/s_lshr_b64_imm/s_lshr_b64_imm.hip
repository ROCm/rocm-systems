// ARCH: gfx1250
//
// Lift fixture for SOP2 s_lshr_b64 with an immediate shift count.
// See the S_LSHR_B64 SemOp in transpiler/semop.hpp; the matching
// handler block in transpiler/handle_sop2.cpp under
// `if (sop == SemOp::S_LSHR_B64) { ... }`; and the SOP2 mapping in
// transpiler/opcode_map.cpp. Bidirectional handler ↔ test
// back-reference is non-negotiable per the transpiler test policy.
//
// `s_lshr_b64 sdst, src0, IMM` is the dominant shape across the
// kerneldex corpus — every observed `s_lshr_b64` in the gfx1250
// scope_discovery + tensilelite indexes uses an immediate shift
// (specifically `>> 16`, used to extract the high half of a packed
// 64-bit pointer/integer for slow-path scalar address arithmetic).
// Pinning this shape covers the load-bearing case; the register-
// shift-count path is exercised end-to-end by the same handler
// branch, so a regression there would surface in the per-kernel
// raise sweep within minutes.
//
// The 64-bit src0 is broadcast through two readfirstlane calls
// (low/high halves recombined) so the kernarg-resolution layer in
// handle_smem.cpp doesn't try to fold the inline asm operand into
// a half-overlapping kernarg load. Same idiom as the s_sub_nc_u64
// / s_mul_hi_i32 fixtures next door.
//
// The shift amount is a *literal* `16` in the inline asm, so the
// AMDGPU disassembler emits `s_lshr_b64 sdst, src0, 16` (a SOP2
// with src1 as an inline constant), which is exactly the corpus
// shape. A %[s] "i" constraint with `16` as the C value would
// also work but the literal form makes the intent unambiguous in
// the .hip source.

#include <hip/hip_runtime.h>

static __device__ inline unsigned long long
broadcast_u64(unsigned long long v) {
  unsigned int lo = __builtin_amdgcn_readfirstlane((unsigned int)v);
  unsigned int hi = __builtin_amdgcn_readfirstlane((unsigned int)(v >> 32));
  return ((unsigned long long)hi << 32) | lo;
}

extern "C" __global__ void
s_lshr_b64_imm_kernel(unsigned long long *out, unsigned long long a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long sa = broadcast_u64(a);
  unsigned long long result;
  asm volatile("s_lshr_b64 %[r], %[a], 16\n\t"
               : [r] "=s"(result)
               : [a] "s"(sa));
  out[tid] = result;
}
