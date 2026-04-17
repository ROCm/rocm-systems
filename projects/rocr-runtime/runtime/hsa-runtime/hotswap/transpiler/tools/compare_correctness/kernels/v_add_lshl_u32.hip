// (a + b) << (c & 31) per lane, via v_add_lshl_u32.
//
// Target handler: V_ADD_LSHL_U32 (handle_valu.cpp).  hipcc does not
// naturally emit this fused opcode on gfx1250 (the arithmetic pattern
// lowers to a v_add_u32 + v_lshlrev_b32 pair), so we force it with
// inline asm.  Both gfx942 and gfx1250 assemblers accept v_add_lshl_u32,
// so the same source builds for both compile paths.
//
// Output shape: 1:1.  Inputs are three u32 arrays giving the two
// addends and the shift count per lane; output is ((a+b) << (c & 31)).
// The only-low-5-bits-of-shift behaviour and 32-bit wrap-around of the
// addition are both reflected in the CPU reference.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_add_lshl_u32(const uint32_t *a,
                                          const uint32_t *b,
                                          const uint32_t *c,
                                          uint32_t *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t ai = a[i], bi = b[i], ci = c[i];
  uint32_t r;
  asm volatile("v_add_lshl_u32 %0, %1, %2, %3"
               : "=v"(r)
               : "v"(ai), "v"(bi), "v"(ci));
  out[i] = r;
}
