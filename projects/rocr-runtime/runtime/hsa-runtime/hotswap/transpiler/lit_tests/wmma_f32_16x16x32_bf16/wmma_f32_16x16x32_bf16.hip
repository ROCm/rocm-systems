// ARCH: gfx1250
//
// Lift fixture for v_wmma_f32_16x16x32_bf16 (gfx1250 RDNA4 VOP3P
// opcode 0x062). The kernel issues exactly one WMMA via the clang
// builtin, which compiles to v_wmma_f32_16x16x32_bf16 on gfx1250
// — the same MC opcode the corpus tensilelite GEMMs emit.
//
// On the gfx942 lowering path (`emitWMMAtoMFMA(..., BF16)`), the
// lifted IR for this single WMMA expands to the standard 16x16x32 →
// 2× MFMA(16x16x16) decomposition documented in
// transpiler/wmma_lowering.cpp:
//
//   * lane redistribution via amdgcn.ds.bpermute (shared with the
//     F16 path — element-type-agnostic for 16-bit elements);
//   * per-MFMA-call bitcast of the source A/B fragments from
//     <4 x bfloat>-equivalent dwords to <4 x i16> (CDNA's bf16
//     MFMA intrinsic was defined before bfloat became a first-class
//     LLVM type);
//   * two `amdgcn.mfma.f32.16x16x16bf16.1k` calls chaining the
//     accumulator (mfma1 → mfma2);
//   * collect step gathering the 4-VGPR MFMA result back to the
//     8-VGPR WMMA layout.
//
// The fixture's purpose is to pin (a) the bf16 SemOp dispatch
// landed correctly and (b) the bf16-specific intrinsic choice.
// The shared lane-redistribution math is covered transitively
// (any regression in it would also break the existing F16 gtest
// path Gfx1250Gpu.Matmul128x128).

#include <hip/hip_runtime.h>

typedef __bf16 v16bf16 __attribute__((ext_vector_type(16)));
typedef float  v8f     __attribute__((ext_vector_type(8)));

extern "C" __global__ void
wmma_f32_16x16x32_bf16_kernel(const v16bf16 *a_ptr,
                               const v16bf16 *b_ptr,
                               v8f *c_ptr) {
  v16bf16 a = a_ptr[0];
  v16bf16 b = b_ptr[0];
  v8f c = c_ptr[0];
  c = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
      /*a_neg=*/0, a,
      /*b_neg=*/0, b,
      /*c_mod=*/0, c,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  c_ptr[0] = c;
}
