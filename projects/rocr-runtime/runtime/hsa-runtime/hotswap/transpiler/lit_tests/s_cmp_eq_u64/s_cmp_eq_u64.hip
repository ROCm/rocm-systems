// ARCH: gfx1250
//
// Lift fixture for SOPC s_cmp_eq_u64 (gfx8+ scalar 64-bit unsigned
// equality compare). See SemOp::S_CMP_EQ_U64 in transpiler/semop.hpp;
// the matching handler block in transpiler/handle_sopc.cpp under
// `if (sop == SemOp::S_CMP_EQ_U64) { ... }`; and the SOPC mapping in
// transpiler/opcode_map.cpp. Bidirectional handler ↔ test
// back-reference is non-negotiable per the transpiler test policy.
//
// `s_cmp_eq_u64` is the 64-bit sibling of `s_cmp_eq_u32` and the
// only 64-bit unsigned scalar compare in SOPInstructions.td besides
// `s_cmp_lg_u64` (the SOPC_CMP_64 record family is closed at these
// two; there are no signed/ordered 64-bit SOPC compares on any
// AMDGPU generation, so the family is fully covered by the same
// commit). The lift writes to SCC only — there is no destination
// register, mirroring every other SOPC compare in the project's
// existing handler.
//
// Operands are funneled through __builtin_amdgcn_readfirstlane on
// each 32-bit half and recombined to a u64 — that's the same
// idiom as the s_sub_nc_u64 fixture next door (the kernarg-broadcast
// values land in VGPRs by default on gfx1250 and the assembler
// rejects mixed s/v operands on s_cmp_eq_u64).

#include <hip/hip_runtime.h>

static __device__ inline unsigned long long
broadcast_u64(unsigned long long v) {
  unsigned int lo = __builtin_amdgcn_readfirstlane((unsigned int)v);
  unsigned int hi = __builtin_amdgcn_readfirstlane((unsigned int)(v >> 32));
  return ((unsigned long long)hi << 32) | lo;
}

extern "C" __global__ void
s_cmp_eq_u64_kernel(unsigned int *out, unsigned long long a,
                    unsigned long long b) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long sa = broadcast_u64(a);
  unsigned long long sb = broadcast_u64(b);
  // Set SCC = (sa == sb), then materialise via s_cselect_b32 so SCC
  // becomes an observable VGPR write the lift will preserve through
  // its SCC-writeback path. Without the cselect the SCC store is
  // dead-code-eliminated and the FileCheck pin disappears.
  unsigned int scc_val;
  asm volatile("s_cmp_eq_u64 %[a], %[b]\n\t"
               "s_cselect_b32 %[r], 1, 0\n\t"
               : [r] "=s"(scc_val)
               : [a] "s"(sa), [b] "s"(sb));
  out[tid] = scc_val;
}
