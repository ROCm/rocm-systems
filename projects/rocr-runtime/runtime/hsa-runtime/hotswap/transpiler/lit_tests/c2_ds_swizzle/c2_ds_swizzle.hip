// ARCH: gfx1250
//
// Fixture for the Class-2 ds_swizzle_b32 sub-category
// (lit_tests/c2_ds_swizzle).
//
// Class 2 (hotswap/docs/wave-size-translation.md §6) and
// hotswap/docs/gpt-oss-derisking.md §5 flag ds_swizzle_b32 in the
// `sum_bitmatrix_rows` kernel of GPT-OSS. The swizzle descriptor
// (encoded in the instruction's offset immediate) selects a cross-
// lane pattern whose semantics depend on the source wave size. The
// P6 rewrite path (lift through llvm.amdgcn.ds.swizzle; see the
// ds_swizzle_b32 row of wave-size-translation.md §5.3) covers the
// structurally safe swizzle modes; anything outside that envelope
// the classifier must refuse.
//
// The immediate `offset:0x041f` here encodes the 32-wide
// fft-butterfly-style swizzle pattern that
// triton_kernels' sum_bitmatrix_rows emits. The classifier keys on
// the SemOp/mnemonic, not the swizzle semantics, so any well-formed
// swizzle immediate is sufficient.

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_ds_swizzle_kernel(int *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[tid];
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0x041f\n\t"
      "s_wait_dscnt 0\n\t"
      : [v] "+v"(v));
  a[tid] = v;
}
