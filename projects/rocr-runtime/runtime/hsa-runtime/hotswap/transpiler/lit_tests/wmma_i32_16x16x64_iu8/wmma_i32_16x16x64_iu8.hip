// ARCH: gfx1250
//
// Lift fixture for v_wmma_i32_16x16x64_iu8 (gfx1250 RDNA4 VOP3P
// opcode 0x072). The kernel issues exactly one WMMA via the clang
// builtin `__builtin_amdgcn_wmma_i32_16x16x64_iu8`, which compiles
// to v_wmma_i32_16x16x64_iu8 on gfx1250 — the same MC opcode the
// corpus tensilelite I8I8 GEMMs emit.
//
// This is the single integer-accumulator member of the WMMA family
// (everything else uses an f32 accumulator). The gfx942 lowering
// path (`emitWMMAtoMFMA(..., IU8)`) reuses the lane-redistribution
// scaffolding shared with the FP8/BF8 siblings (same per-lane A/B
// fragment shape — 8 VGPRs of <8 x i32> = 32 packed i8 bytes per
// lane), but dispatches to `mfma_i32_16x16x32_i8` and packs the
// accumulator as `<4 x i32>` instead of `<4 x float>`. The K=64 is
// decomposed into 2 chained MFMA(K=32) calls per Wave32 group.
//
// The fixture's purpose is to pin (a) the IU8 SemOp dispatched
// correctly through the parameterised emitWMMAtoMFMA helper, (b)
// the integer-accumulator MFMA intrinsic choice, and (c) the
// `<4 x i32>` accumulator pack type — distinct from the
// `<4 x float>` used by every other WMMA variant. The IU8 row is
// the ONLY non-f32-accumulator WMMA in the corpus, so this fixture
// also pins the accumulator-type fork in the helper itself.

#include <hip/hip_runtime.h>

typedef int v8i  __attribute__((ext_vector_type(8)));

extern "C" __global__ void
wmma_i32_16x16x64_iu8_kernel(const v8i *a_ptr,
                              const v8i *b_ptr,
                              v8i *c_ptr) {
  v8i a = a_ptr[0];
  v8i b = b_ptr[0];
  v8i c = c_ptr[0];
  c = __builtin_amdgcn_wmma_i32_16x16x64_iu8(
      /*a_sign=*/false, a,
      /*b_sign=*/false, b,
      c,
      /*matrix_a_reuse=*/false,
      /*matrix_b_reuse=*/false);
  c_ptr[0] = c;
}
