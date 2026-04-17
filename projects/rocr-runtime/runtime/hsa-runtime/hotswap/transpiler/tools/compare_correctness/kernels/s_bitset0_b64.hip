// 64-bit scalar bit-clear (read-modify-write) via s_bitset0_b64.
//
// Target handler: S_BITSET0_B64 (handle_sop1.cpp).  The instruction
// clears bit `src[5:0]` in a 64-bit sdst, with `sdst_in` tied to `sdst`
// so the pre-existing value survives the RMW.  Bit index is an
// SReg_32; the destination is an SReg_64 pair.  Per-block uniform
// values are forced into SGPRs via `readfirstlane` and the RMW runs on
// lane 0 of each block through an inline-asm `+s` constraint on the
// tied destination.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_bitset0_b64(const uint64_t *initial,
                                         const uint32_t *pos,
                                         uint64_t *out, int n) {
  int bi = blockIdx.x;
  if (bi >= n) return;
  if (threadIdx.x != 0) return;
  uint32_t lo = __builtin_amdgcn_readfirstlane(uint32_t(initial[bi]));
  uint32_t hi = __builtin_amdgcn_readfirstlane(uint32_t(initial[bi] >> 32));
  uint64_t v  = (uint64_t(hi) << 32) | uint64_t(lo);
  uint32_t bp = __builtin_amdgcn_readfirstlane(pos[bi]);
  asm volatile("s_bitset0_b64 %0, %1" : "+s"(v) : "s"(bp));
  out[bi] = v;
}
