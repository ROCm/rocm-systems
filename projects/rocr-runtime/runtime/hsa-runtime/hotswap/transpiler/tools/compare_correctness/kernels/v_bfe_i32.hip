// Signed bit-field extract via v_bfe_i32.
//
// Target handler: V_BFE_I32 (handle_valu.cpp).  `v_bfe_i32 dst, src,
// off, w` extracts `w` bits at offset `off` from `src` and sign-extends
// from bit (w-1).  Only the low 5 bits of `off` and `w` matter.  hipcc
// lowers `(int32_t)(src << (32-off-w)) >> (32-w)` patterns to a
// shift-pair on gfx1250 rather than emitting this fused opcode, so we
// force it with inline asm.
//
// Output shape: 1:1.  Inputs are src[N], off[N] (u32), w[N] (u32);
// output is the sign-extracted field as int32_t reinterpreted as
// uint32_t.  The CPU reference follows the hardware definition:
// zero-width extracts return 0, otherwise the low `w` bits of
// `src >> off` are sign-extended from bit (w-1).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_bfe_i32(const uint32_t *src,
                                     const uint32_t *off,
                                     const uint32_t *w,
                                     uint32_t *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t si = src[i], oi = off[i], wi = w[i];
  uint32_t r;
  asm volatile("v_bfe_i32 %0, %1, %2, %3"
               : "=v"(r)
               : "v"(si), "v"(oi), "v"(wi));
  out[i] = r;
}
