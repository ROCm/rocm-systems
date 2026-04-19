// ARCH: gfx1250
//
// Lift fixture for the gfx8+ HasD16LoadStore D16_HI partial-store
// family — `ds_store_b16_d16_hi` (and its companion
// `ds_store_b8_d16_hi`, which shares the exact same handler block;
// both are pinned via the same value-name breadcrumb and the same
// lshr-16 + trunc shape). See SemOp::DS_WRITE_B16_D16_HI in
// transpiler/semop.hpp; the matching handler block in
// transpiler/handle_ds.cpp under
// `if (sop == SemOp::DS_WRITE_B16_D16_HI || ...)`; and the DS
// mapping in transpiler/opcode_map.cpp. Bidirectional handler ↔
// test back-reference is non-negotiable per the transpiler test
// policy.
//
// `ds_store_b16_d16_hi` writes the UPPER 16 bits of the source
// VGPR (bits [31:16]) to LDS. The dominant corpus pattern (18
// tensilelite gemm-tile-spill kernels) is packing the upper
// half of a packed-bf16 register pair into an LDS staging slot
// during the K-strip tile redistribution between the global
// load and the WMMA accumulator setup. A regression that wrote
// bits [15:0] instead of [31:16] would silently corrupt every
// other tile.
//
// We use a __shared__ array as the LDS slot and write through
// inline asm so the compiler is forced to emit
// `ds_store_b16_d16_hi` (not the lower-half ds_store_b16). The
// "v" constraint on the data input keeps the whole i32 in a
// VGPR so the encoded source register holds bits [31:16] of the
// kernel-arg-broadcast input.

#include <hip/hip_runtime.h>

extern "C" __global__ void
ds_store_b16_d16_hi_kernel(unsigned int packed) {
  __shared__ unsigned short slot[2];

  int tid = threadIdx.x;
  unsigned int slot_addr;
  unsigned int *slot_ptr = (unsigned int *)&slot[0];
  asm volatile("v_mov_b32 %[r], %[s]\n\t"
               : [r] "=v"(slot_addr)
               : [s] "v"((unsigned int)(uintptr_t)slot_ptr));
  asm volatile("ds_store_b16_d16_hi %[a], %[v]\n\t"
               :
               : [a] "v"(slot_addr), [v] "v"(packed));
  if (tid == 0) {
    *slot_ptr = (unsigned int)slot[0] | ((unsigned int)slot[1] << 16);
  }
}
