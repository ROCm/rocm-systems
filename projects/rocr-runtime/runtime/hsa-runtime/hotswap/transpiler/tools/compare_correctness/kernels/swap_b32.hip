// Register-pair swap via v_swap_b32.
//
// Target handler: V_SWAP_B32 (handle_valu.cpp).  `v_swap_b32 dst, src`
// atomically exchanges the two operands' 32-bit values — a single
// instruction that writes BOTH its operands, which is unusual for
// VALU and is specifically what the handler must get right.
//
// hipcc rarely emits it spontaneously (the RA prefers copies), so we
// force it with an inline asm constraining both operands as input/
// output (`+v`).  Both gfx942 and gfx1250 assemblers accept the
// opcode, so the kernel compiles for both engines.
//
// Output shape: 1:1.  The kernel consumes adjacent input pairs
// (`in[2i], in[2i+1]`), swaps them with v_swap_b32, and writes them
// back — so `out[2i] = in[2i+1]` and `out[2i+1] = in[2i]`.  N must be
// even; the recipe's validate hook guards that.

#include <hip/hip_runtime.h>

extern "C" __global__ void swap_b32(const uint32_t *in, uint32_t *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int base = 2 * i;
  if (base + 1 >= n) return;
  uint32_t x = in[base];
  uint32_t y = in[base + 1];
  asm volatile("v_swap_b32 %0, %1" : "+v"(x), "+v"(y));
  out[base]     = x;
  out[base + 1] = y;
}
