// ARCH: gfx1250
//
// Fixture for the Class-2 "wave-size-baked cross-lane op" obstruction,
// rewritable-subclass sub-category (lit_tests/c2_permlane_swap).
//
// Class 2 (hotswap/docs/wave-size-translation.md §6) catalogues
// cross-lane primitives whose semantics are parameterised by the
// source wave size. Some variants (e.g. v_permlane64_b32) have no
// wave32 meaning at all and are unrewritable. Others — the
// permlane16_swap / permlanex16 family listed as P2–P4 in wave-
// size-translation.md §5.3 — do have a clean
// intrinsic lift that preserves the source's cross-lane data
// movement once the raiser lowers them through
// llvm.amdgcn.permlane16. The classifier therefore flags this class
// as "C2-rewritable-pending" until those lifts land, graduating to
// "C2-handled" once the handler code is in.
//
// This fixture uses v_permlane16_swap_b32 — the permlane variant
// GPT-OSS's attention forward path emits for its softmax row-reduce
// (see gpt-oss-derisking.md §5 and §8.4). Pinning it via inline asm
// keeps the probe independent of hipcc codegen choices.

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_permlane_swap_kernel(int *a, int *b) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int va = a[tid];
  int vb = b[tid];
  // v_permlane16_swap_b32 exchanges values between lane groups
  // 0..15 and 16..31 of the source wave. This is the obstruction
  // the classifier should identify and tag with rewrite P2/P3/P4.
  asm volatile(
      "v_permlane16_swap_b32 %[va], %[vb]\n\t"
      : [va] "+v"(va), [vb] "+v"(vb));
  a[tid] = va;
  b[tid] = vb;
}
