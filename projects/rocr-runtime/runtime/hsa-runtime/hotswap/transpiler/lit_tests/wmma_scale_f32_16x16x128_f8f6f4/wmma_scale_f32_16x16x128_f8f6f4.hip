// ARCH: gfx1250
//
// Lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4 (gfx1250 RDNA4
// VOP3PX2 opcode 0x033, ScaledWMMA family). The kernel issues exactly
// one scaled WMMA via the clang builtin, which compiles to the
// `_f8_f8_w32_threeaddr` MC pseudo with `matrix_a_fmt:MATRIX_FMT_BF8`
// (matrix_b_fmt defaulted to `MATRIX_FMT_FP8` = 0). Same shape as the
// failing kerneldex GEMMs:
//
//   tensilelite__TensileLibrary_B8F8_SB8F8_UA_Type_B8S_HPA_Contraction_
//     l_Alik_Bljk_Cijk_Dijk_ID73f0_gfx1250.co
//   tensilelite__TensileLibrary_F8B8_SF8B8_UA_Type_F8S_HPA_Contraction_
//     l_Alik_Bljk_Cijk_Dijk_ID73f0_gfx1250.co
//
// 18 MC pseudos collapse onto the single
// `SemOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4` (9 mantissa pairs
// `{f4,f6,f8} A × {f4,f6,f8} B` × `_twoaddr`/`_threeaddr`), per
// `WMMA_F8F6F4_Profiles` in VOP3PInstructions.td:1908. The per-matrix
// dword count is determined by the opcode suffix (f8 → 16, f6 → 12,
// f4 → 8); the in-family element distinction (BF8 vs FP8 within f8;
// BF6 vs FP6 within f6) lives in the `matrix_a_fmt` / `matrix_b_fmt`
// named-immediate operands (`enum MatrixFMT { FP8=0, BF8=1, FP6=2,
// BF6=3, FP4=4 }`, SIDefines.h:1052-1058).
//
// The native intrinsic
// `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4` is declared inside
// `AMDGPUWMMAIntrinsicsGFX1250` (gated by `isGFX125xOnly` in
// IntrinsicsAMDGPU.td:4138, class `AMDGPUWmmaScaleIntrinsicModsC
// <llvm_i32_ty>`) and therefore is NOT in the gfx12 RDNA4-base WMMA
// family. Same-target lift therefore keys on
// `ISAProfile::hasTensorOps` (FeatureGFX1250Insts) — matching the
// LLVM intrinsic's actual subtarget gating. Cross-target lift to
// gfx942 has no scaled-WMMA analog: the closest gfx942 sibling is
// `mfma_scale_f32_16x16x128_f8f6f4`, but the WMMA → MFMA lane
// redistribution + per-matrix-fmt selection + scale-exponent
// application is not modelled in `wmma_lowering.cpp`, so the handler
// refuses loudly via `RaiseFailure::unsupportedShape` to surface the
// gap immediately rather than silently degrade.
//
// MAINTENANCE: the kernel must NEVER be dispatched — only the raiser
// pre-translation contract is exercised. The clang builtin signature
// is fixed in BuiltinsAMDGPU.td:1154:
//
//   _ExtVector<8, float> __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
//       _Constant int matrix_a_fmt,
//       _ExtVector<16, int> a,
//       _Constant int matrix_b_fmt,
//       _ExtVector<16, int> b,
//       _Constant short c_mod,
//       _ExtVector<8, float> c,
//       _Constant int matrix_a_scale,
//       _Constant int matrix_a_scale_fmt,
//       int scale_src0,
//       _Constant int matrix_b_scale,
//       _Constant int matrix_b_scale_fmt,
//       int scale_src1,
//       _Constant bool a_reuse,
//       _Constant bool b_reuse);

#include <hip/hip_runtime.h>

typedef int v16i __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

extern "C" __global__ void
wmma_scale_f32_16x16x128_f8f6f4_kernel(const v16i *a_ptr,
                                        const v16i *b_ptr,
                                        v8f *c_ptr,
                                        int scale_src0,
                                        int scale_src1) {
  v16i a = a_ptr[0];
  v16i b = b_ptr[0];
  v8f c = c_ptr[0];
  c = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
      /*matrix_a_fmt=*/1,           // MATRIX_FMT_BF8
      a,
      /*matrix_b_fmt=*/0,           // MATRIX_FMT_FP8
      b,
      /*c_mod=*/(short)0,
      c,
      /*matrix_a_scale=*/0,
      /*matrix_a_scale_fmt=*/0,
      scale_src0,
      /*matrix_b_scale=*/0,
      /*matrix_b_scale_fmt=*/0,
      scale_src1,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  c_ptr[0] = c;
}
