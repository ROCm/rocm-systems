// ARCH: gfx1250
//
// Lift fixture for v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4 VOP3P
// opcode 0x05D). The kernel issues exactly one WMMA via the clang
// builtin, which compiles to v_wmma_f32_16x16x4_f32 on gfx1250 —
// the same MC opcode the corpus tensilelite f32 GEMMs emit (e.g.
// `tensilelite__TensileLibrary_..._gfx1250_f19386caaa7b.co` and
// the three sibling `_gfx1250_*.co` ID73f0 SAV/UA contractions
// that surfaced this gap in the kerneldex sweep).
//
// This SemOp stands ALONE from the K=32 (16-bit) and K=64 (8-bit)
// WMMA family (`SemOp::V_WMMA_F32_16x16x32_*`,
// `SemOp::V_WMMA_F32_16x16x64_*`, `SemOp::V_WMMA_I32_16x16x64_IU8`)
// because (a) the per-Wave32-lane A/B fragment is `<2 x f32>` —
// only 2 dwords — versus `<16 x t>` (16-bit) or `<8 x i32>`
// (8-bit), and (b) the shared `emitWMMAtoMFMA` decomposition path
// (transpiler/wmma_lowering.cpp) is parameterised on 16-/8-bit
// element packing and does NOT cover the K=4 f32 case.
//
// The native intrinsic `int_amdgcn_wmma_f32_16x16x4_f32` is
// declared inside `AMDGPUWMMAIntrinsicsGFX1250` (gated by
// `isGFX125xOnly` in IntrinsicsAMDGPU.td:4113-4114) and therefore
// is NOT in the gfx12 RDNA4-base WMMA family
// (`AMDGPUWMMAIntrinsicsGFX12`, FeatureWMMA{128,256}bInsts).
// Same-target lift therefore keys on `ISAProfile::hasTensorOps`
// (FeatureGFX1250Insts) — matching the LLVM intrinsic's actual
// subtarget gating — rather than `hasWMMA12`. Cross-target lift
// to gfx942 goes through `emitWMMAtoMFMA_F32_16x16x4`
// (transpiler/wmma_lowering.cpp), a dedicated K=4 helper that
// decomposes the Wave32 WMMA into one `mfma_f32_16x16x4f32` call
// per virtual Wave32 group via ds.bpermute lane redistribution
// (gfx942 has a direct K=4 MFMA, so the K-dimension matches
// exactly and no K-tiling is required inside a group — a
// structural simplification versus the K=32/K=64 family's 2×
// chained-MFMA decomposition). Targets with neither `hasTensorOps`
// nor `hasMFMA` refuse loudly via
// `RaiseFailure::unsupportedShape` (no K=4 f32 matrix path is
// available on e.g. gfx12 RDNA4 base).
//
// MAINTENANCE: the kernel must NEVER be dispatched — only the
// raiser pre-translation contract is exercised. The clang builtin
// signature is fixed in BuiltinsAMDGPU.td:1066:
//
//   _ExtVector<8, float> __builtin_amdgcn_wmma_f32_16x16x4_f32(
//       _Constant bool a_neg, _ExtVector<2, float> a,
//       _Constant bool b_neg, _ExtVector<2, float> b,
//       _Constant short c_mod, _ExtVector<8, float> c,
//       _Constant bool a_reuse, _Constant bool b_reuse);

#include <hip/hip_runtime.h>

typedef float v2f __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

extern "C" __global__ void
wmma_f32_16x16x4_f32_kernel(const v2f *a_ptr,
                             const v2f *b_ptr,
                             v8f *c_ptr) {
  v2f a = a_ptr[0];
  v2f b = b_ptr[0];
  v8f c = c_ptr[0];
  c = __builtin_amdgcn_wmma_f32_16x16x4_f32(
      /*a_neg=*/false, a,
      /*b_neg=*/false, b,
      /*c_mod=*/(short)0, c,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  c_ptr[0] = c;
}
