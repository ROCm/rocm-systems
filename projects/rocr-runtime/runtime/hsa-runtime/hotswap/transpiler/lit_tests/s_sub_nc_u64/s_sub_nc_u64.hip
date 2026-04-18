// ARCH: gfx1250
//
// Lift fixture for SOP2 s_sub_nc_u64 (gfx12+ scalar 64-bit subtract,
// no-carry). See the S_SUB_NC_U64 SemOp in transpiler/semop.hpp and
// the matching handler block in transpiler/handle_sop2.cpp; this
// fixture pins the IR shape both reference (handler ↔ test
// back-reference is non-negotiable per the transpiler test policy).
//
// Note on naming: LLVM's MC opcode is `S_SUB_U64` but the gfx12
// AsmString is `s_sub_nc_u64` (SOPInstructions.td 2311). The
// disassembler we feed raise_cli emits the asm form, so we use that
// here. opcode_map.cpp folds both into SemOp::S_SUB_NC_U64.
//
// Operands are funneled through __builtin_amdgcn_readfirstlane on
// each 32-bit half and then recombined to a u64 — that's the only
// way to get a 64-bit SGPR pair out of a kernarg-broadcast value
// without tripping the kernarg-resolution layer's "load fully
// covers param" check (a u64 readfirstlane shrinks the kernarg
// load to 32 bits and rebuilds via s_bfe_i64, which the resolver
// rejects). Same constraint idiom as the s_mul_hi_i32 fixture.

#include <hip/hip_runtime.h>

static __device__ inline unsigned long long
broadcast_u64(unsigned long long v) {
  unsigned int lo = __builtin_amdgcn_readfirstlane((unsigned int)v);
  unsigned int hi = __builtin_amdgcn_readfirstlane((unsigned int)(v >> 32));
  return ((unsigned long long)hi << 32) | lo;
}

extern "C" __global__ void s_sub_nc_u64_kernel(unsigned long long *out,
                                               unsigned long long a,
                                               unsigned long long b) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long sa = broadcast_u64(a);
  unsigned long long sb = broadcast_u64(b);
  unsigned long long result;
  asm volatile("s_sub_nc_u64 %[r], %[a], %[b]\n\t"
               : [r] "=s"(result)
               : [a] "s"(sa), [b] "s"(sb));
  out[tid] = result;
}
