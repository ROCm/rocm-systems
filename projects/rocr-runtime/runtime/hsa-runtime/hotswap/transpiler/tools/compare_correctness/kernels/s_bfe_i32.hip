// Signed scalar bit-field extract via s_bfe_i32.
//
// Target handler: S_BFE_I32 (handle_sop2.cpp).  `s_bfe_i32 sdst, src,
// ctrl` extracts `ctrl[22:16]` bits at offset `ctrl[4:0]` from `src`
// and sign-extends from bit (width-1).  Scalar-side equivalent of
// v_bfe_i32 — one per block, not per lane.
//
// We drive scalars by loading a per-block uniform value and passing it
// through `__builtin_amdgcn_readfirstlane` to guarantee it lives in an
// SGPR at the inline-asm site.  Lane 0 of each block performs the
// s_bfe_i32 and writes one output dword per block.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_bfe_i32(const uint32_t *src,
                                     const uint32_t *ctrl,
                                     uint32_t *out, int n) {
  int bi = blockIdx.x;
  if (bi >= n) return;
  if (threadIdx.x != 0) return;
  uint32_t si = __builtin_amdgcn_readfirstlane(src[bi]);
  uint32_t ci = __builtin_amdgcn_readfirstlane(ctrl[bi]);
  uint32_t r;
  asm volatile("s_bfe_i32 %0, %1, %2" : "=s"(r) : "s"(si), "s"(ci));
  out[bi] = r;
}
