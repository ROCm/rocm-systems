// ARCH: gfx1250
//
// Lift fixture for the `flat_store_short_d16_hi` half-register
// store — FLAT-addressing sibling of `global_store_d16_hi_b16`
// (pinned by `global_store_short_d16_hi/`) and of
// `ds_store_b16_d16_hi` (pinned by `ds_store_b16_d16_hi/`).  All
// three families have structurally identical "upper 16 bits of
// source VGPR" semantics; the `handle_flat.cpp` emission routes
// GLOBAL and FLAT through the shared `emitD16HiHalfTruncI16`
// helper.
//
// Without the FLAT fixture, a regression that breaks ONLY the
// FLAT branch (e.g. a future refactor that adds a condition gate
// specific to `GLOBAL_STORE_*`) would pass `global_store_short_d16_hi`
// and silently miscompile every FLAT bf16-cast-to-store kernel.
// The pair `flat_store_short_d16_hi` + `global_store_short_d16_hi`
// closes the addrspace-symmetry regression-guard surface.
//
// We force emission of `flat_store_short_d16_hi` via inline asm —
// same forcing approach as the GLOBAL sibling.

#include <hip/hip_runtime.h>

extern "C" __global__ void
flat_store_short_d16_hi_kernel(unsigned short *out, unsigned int packed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned short *slot = out + tid;
  asm volatile("flat_store_short_d16_hi %[a], %[v]\n\t"
               :
               : [a] "v"(slot), [v] "v"(packed)
               : "memory");
}
