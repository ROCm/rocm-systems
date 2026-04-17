// Bit-field mask generation via v_bfm_b32.
//
// Target handler: V_BFM_B32 (handle_valu.cpp).  `v_bfm_b32 dst, w, off`
// computes `((1 << (w & 31)) - 1) << (off & 31)` — a contiguous run
// of `w` 1-bits starting at bit `off`, with modulo-32 masking of both
// counts.  hipcc does not naturally emit this opcode on gfx1250 (the
// pattern `(1 << w) - 1` lowers to lshlrev+not), so we force the
// instruction with inline asm.  Both gfx942 and gfx1250 assemblers
// accept it, so the same source works for both compile paths.
//
// Output shape: 1:1.  Inputs are two u32 arrays giving the width and
// offset per lane; output is the mask.  The only-five-bits-matter
// behaviour of the instruction is reflected in the CPU reference.

#include <hip/hip_runtime.h>

extern "C" __global__ void bfm_b32(const uint32_t *w, const uint32_t *off,
                                   uint32_t *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t wi  = w[i];
  uint32_t oi  = off[i];
  uint32_t out;
  asm volatile("v_bfm_b32 %0, %1, %2" : "=v"(out) : "v"(wi), "v"(oi));
  c[i] = out;
}
