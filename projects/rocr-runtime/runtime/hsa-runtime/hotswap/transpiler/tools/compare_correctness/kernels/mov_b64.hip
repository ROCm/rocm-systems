// 64-bit VGPR-to-VGPR move via v_mov_b64.
//
// Target handler: V_MOV_B64 (handle_valu.cpp).  `v_mov_b64` was added
// in gfx11 and is accepted by the gfx1250 assembler; gfx942 predates
// it and rejects the opcode outright.  We force emission on gfx1250
// with inline asm, and fall back to a natural 64-bit copy on gfx942
// so the native-mode compile still succeeds.  Both compile paths
// produce the same value, so the CPU reference (pure copy) is
// engine-agnostic.
//
// Output shape: 1:1.  Input and output are u64 arrays of length N.

#include <hip/hip_runtime.h>

extern "C" __global__ void mov_b64(const uint64_t *a, uint64_t *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t x = a[i];
  uint64_t y;
#if defined(__gfx1250__)
  asm volatile("v_mov_b64 %0, %1" : "=v"(y) : "v"(x));
#else
  y = x;
#endif
  c[i] = y;
}
