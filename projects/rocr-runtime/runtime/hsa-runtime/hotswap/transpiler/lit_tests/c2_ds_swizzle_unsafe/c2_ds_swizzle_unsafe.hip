// ARCH: gfx1250
//
// Negative-case fixture for the Class-2 ds_swizzle_b32 sub-category
// (lit_tests/c2_ds_swizzle_unsafe).
//
// The P6 rewrite (see the ds_swizzle_b32 row of hotswap/docs/
// wave-size-translation.md §5.3) lifted ds_swizzle_b32 through
// `llvm.amdgcn.ds.swizzle`, and after the audit follow-up that
// widened the safe set with empirical evidence (gfx942 wave64
// preserves bit 5 of lane id across all four valid swizzle-mode
// envelopes), QUAD_PERM / BITMASK_PERM / FFT_MODE / ROTATE_MODE imms
// all lift cleanly.
//
// What remains refused is the RESERVED top-nibble envelope (top
// nibble in {0x9, 0xA, 0xB, 0xD, 0xF}) where AMDGPU SIDefines.h
// `Swizzle::EncBits` assigns no semantics. Hardware behavior is
// undefined; refusing loudly here prevents the source kernel's
// otherwise-meaningful imm from silently mapping to whatever the
// wave64 backend happens to do.
//
// The immediate `offset:0x9000` selects a RESERVED top-nibble (0x9).
// No GPT-OSS corpus kernel uses this; the fixture exists solely to
// pin the unsafe-imm refusal path so a future widening of
// `dsSwizzleSafeForModRep` that accidentally accepts the RESERVED
// envelope is caught.

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_ds_swizzle_unsafe_kernel(int *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[tid];
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0x9000\n\t"
      "s_wait_dscnt 0\n\t"
      : [v] "+v"(v));
  a[tid] = v;
}
