// ARCH: gfx1250
//
// Lift fixture for the SOPC bit-test family (SOPInstructions.td:1411-
// 1414; gfx6+ on every AMDGPU generation):
//
//   s_bitcmp0_b32 / s_bitcmp1_b32 / s_bitcmp0_b64 / s_bitcmp1_b64
//
// See SemOp::S_BITCMP{0,1}_{B32,B64} in transpiler/semop.hpp and the
// shared handler block in transpiler/handle_sopc.cpp; the SOPC
// mapping lives in transpiler/opcode_map.cpp.  Closes the kerneldex
// corpus blocker on scope_discovery `_attn_fwd` (a single
// `s_bitcmp0_b32 s68, 1` that the pre-fix handler dropped on the
// floor).
//
// All four variants share exactly one code path in handle_sopc.cpp
// (only the source-operand width and the final icmp predicate
// differ), so a single fixture that exercises every permutation is
// the regression guard we want — a rewrite that split the handler
// back into four independent branches has to keep every one
// passing.
//
// The shift amount (src1) is deliberately put through
// `__builtin_amdgcn_readfirstlane` to force the assembler into the
// SGPR operand class on every generation; without the RFL the
// kernarg value lands in a VGPR and `s_bitcmp0_b32` / friends would
// be rejected by the assembler.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_bitcmp_kernel(unsigned int *out,
                                           unsigned int a32,
                                           unsigned int b32,
                                           unsigned long long a64,
                                           unsigned long long b64,
                                           unsigned int shamt) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Force every source into an SGPR (readfirstlane) so the SOPC
  // assembler accepts the operand classes.
  unsigned int sa32 = __builtin_amdgcn_readfirstlane(a32);
  unsigned int sb32 = __builtin_amdgcn_readfirstlane(b32);
  unsigned int sh = __builtin_amdgcn_readfirstlane(shamt);
  unsigned long long sa64;
  unsigned long long sb64;
  {
    unsigned int lo_a = __builtin_amdgcn_readfirstlane((unsigned int)a64);
    unsigned int hi_a = __builtin_amdgcn_readfirstlane(
        (unsigned int)(a64 >> 32));
    sa64 = ((unsigned long long)hi_a << 32) | lo_a;
    unsigned int lo_b = __builtin_amdgcn_readfirstlane((unsigned int)b64);
    unsigned int hi_b = __builtin_amdgcn_readfirstlane(
        (unsigned int)(b64 >> 32));
    sb64 = ((unsigned long long)hi_b << 32) | lo_b;
  }

  // Four SOPC bit-tests, each followed by a s_cselect_b32 so SCC
  // becomes an observable VGPR write — mirrors the s_cmp_eq_u64
  // fixture's SCC materialisation pattern.  Without the cselect the
  // SCC store would be dead-code-eliminated and the FileCheck pins
  // would disappear.
  unsigned int r0, r1, r2, r3;

  asm volatile("s_bitcmp0_b32 %[s], %[sh]\n\t"
               "s_cselect_b32 %[r], 1, 0\n\t"
               : [r] "=s"(r0)
               : [s] "s"(sa32), [sh] "s"(sh));

  asm volatile("s_bitcmp1_b32 %[s], %[sh]\n\t"
               "s_cselect_b32 %[r], 1, 0\n\t"
               : [r] "=s"(r1)
               : [s] "s"(sb32), [sh] "s"(sh));

  asm volatile("s_bitcmp0_b64 %[s], %[sh]\n\t"
               "s_cselect_b32 %[r], 1, 0\n\t"
               : [r] "=s"(r2)
               : [s] "s"(sa64), [sh] "s"(sh));

  asm volatile("s_bitcmp1_b64 %[s], %[sh]\n\t"
               "s_cselect_b32 %[r], 1, 0\n\t"
               : [r] "=s"(r3)
               : [s] "s"(sb64), [sh] "s"(sh));

  out[4 * tid + 0] = r0;
  out[4 * tid + 1] = r1;
  out[4 * tid + 2] = r2;
  out[4 * tid + 3] = r3;
}
