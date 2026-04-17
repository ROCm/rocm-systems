// Scalar bit-clear (read-modify-write) via s_bitset0_b32.
//
// Target handler: S_BITSET0_B32 (handle_sop1.cpp).  The instruction
// clears bit `src[4:0]` in `sdst`, with `sdst_in` tied to `sdst` so the
// pre-existing value survives the RMW.  We drive scalars by loading
// per-block uniform values and forcing them into SGPRs via
// `readfirstlane`; lane 0 of each block performs the RMW via inline
// asm with a `+s` constraint on the tied destination.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_bitset0_b32(const uint32_t *initial,
                                         const uint32_t *pos,
                                         uint32_t *out, int n) {
  int bi = blockIdx.x;
  if (bi >= n) return;
  if (threadIdx.x != 0) return;
  uint32_t v  = __builtin_amdgcn_readfirstlane(initial[bi]);
  uint32_t bp = __builtin_amdgcn_readfirstlane(pos[bi]);
  asm volatile("s_bitset0_b32 %0, %1" : "+s"(v) : "s"(bp));
  out[bi] = v;
}
