// ARCH: gfx1250
//
// Lift fixture for v_wmma_f32_16x16x64_fp8_fp8 (gfx1250 RDNA4 VOP3P
// opcode 0x06a). The kernel issues exactly one WMMA via the clang
// builtin `__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8`, which compiles
// to v_wmma_f32_16x16x64_fp8_fp8 on gfx1250 — the same MC opcode the
// corpus tensilelite F8 GEMMs emit.
//
// On the gfx942 lowering path (`emitWMMAtoMFMA(..., FP8_FP8)`), the
// lifted IR for this single WMMA expands to a 16x16x64 → 2× MFMA
// (16x16x32) decomposition documented in transpiler/wmma_lowering.cpp:
//
//   * lane redistribution via amdgcn.ds.bpermute (shared with the
//     16-bit WMMA paths — the redistribution operates at dword
//     granularity and is element-type-agnostic for any 32-byte-per-
//     lane fragment);
//   * per-MFMA-call bitcast of the source A/B fragments from packed
//     2 × i32 dwords to a single i64 (CDNA3's fp8/bf8 MFMA intrinsic
//     was defined before fp8 became a first-class LLVM type, so it
//     takes 8 packed fp8/bf8 bytes as i64 directly);
//   * two `amdgcn.mfma.f32.16x16x32.fp8.fp8` calls chaining the
//     accumulator (mfma1 → mfma2);
//   * collect step gathering the 4-VGPR MFMA result back to the
//     8-VGPR WMMA layout.
//
// The fixture's purpose is to pin (a) the FP8_FP8 SemOp dispatch
// landed correctly through the parameterised emitWMMAtoMFMA helper
// and (b) the fp8_fp8-specific intrinsic choice and i64 pack type.
// The four AB combinations (fp8_fp8, fp8_bf8, bf8_fp8, bf8_bf8) share
// the same dispatch shape — only the trailing intrinsic name differs
// — so this fixture covers the entire family transitively.
// Regression in the underlying lane redistribution math would also
// surface in the existing F16/BF16 lit + GPU-test coverage.

#include <hip/hip_runtime.h>

typedef int   v8i  __attribute__((ext_vector_type(8)));
typedef float v8f  __attribute__((ext_vector_type(8)));

extern "C" __global__ void
wmma_f32_16x16x64_fp8_fp8_kernel(const v8i *a_ptr,
                                  const v8i *b_ptr,
                                  v8f *c_ptr) {
  v8i a = a_ptr[0];
  v8i b = b_ptr[0];
  v8f c = c_ptr[0];
  c = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
      a, b,
      /*c_mod=*/0, c,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  c_ptr[0] = c;
}
