// ARCH: gfx1250
//
// Lit fixture for the gfx1250-only VOP3 scaled packed-8 FP4 -> BF16
// conversion (V_CVT_SCALE_PK8_BF16_FP4_e64, VOP3Instructions.td:1788).
// Closes the kerneldex corpus handler gap observed on
// scope_discovery `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1`.
//
// Shape pinned:
//   dst   : <8 x bfloat>   (4 consecutive VGPRs, 128 bits)
//   src0  : i32            (packed 8xFP4 — 4 bits per lane)
//   src1  : i32            (E8M0-encoded scale value, VSrc_b32)
//   sel   : i32 immarg     (byte_sel; 0 is the default / only value
//                            the corpus emits today)
//
// The LLVM intrinsic
// `@llvm.amdgcn.cvt.scale.pk8.bf16.fp4` is declared inside the
// gfx1250-only block of IntrinsicsAMDGPU.td:686 (class
// `AMDGPUCvtScaleIntrinsic<llvm_v8bf16_ty, llvm_i32_ty, ...>` guarded
// by `isGFX125xOnly`), so same-target lift gates on
// `ctx.targetIsa.hasTensorOps`.  The fixture therefore runs the lift
// same-target (`--isa=gfx1250` + default target); the cross-target
// refusal path is exercised by the `.cross.ll` companion below.
//
// We use a clang builtin rather than inline asm because the exact
// modifier surface (scale_sel named immediate, VReg_128 aligned dest)
// is stabler through the builtin than the inline-asm assembler, and
// the corpus kernel itself was emitted from the same builtin path.

#include <hip/hip_runtime.h>

typedef __attribute__((ext_vector_type(8))) __bf16 v8bf16;

// scale_sel = 0: the corpus-observed value across all 128 captured
// instances; both handler arms (same-target and cross-target) lower
// this case.
extern "C" __global__ void
v_cvt_scale_pk8_bf16_fp4_kernel(unsigned int *out) {
  unsigned int packedFP4 = (unsigned int)(uintptr_t)out;
  unsigned int scale     = (unsigned int)(uintptr_t)(out + 1);

  v8bf16 r = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(packedFP4, scale, 0);

  unsigned short *dst = (unsigned short *)out;
  for (int i = 0; i < 8; ++i) {
    dst[i] = __builtin_bit_cast(unsigned short, r[i]);
  }
}

// scale_sel = 2: a value OUTSIDE the handler's declared support set.
// The 4-bit `scale_sel` field's semantics for the packed-8 FP4 shape
// aren't pinned in-tree today (no corpus kernel uses scale_sel != 0;
// AMD ISA spec definition not reproduced).  Both handler arms MUST
// refuse this loudly.  Pins the refusal diagnostic on a kernel that
// exercises it — complementing the non-refusal pin in
// `v_cvt_scale_pk8_bf16_fp4_kernel` above.  See
// `hotswap/docs/matrix-translation.md §7.4` for the declared support
// set and why we refuse instead of guessing the semantics.
extern "C" __global__ void
v_cvt_scale_pk8_bf16_fp4_kernel_sel2(unsigned int *out) {
  unsigned int packedFP4 = (unsigned int)(uintptr_t)out;
  unsigned int scale     = (unsigned int)(uintptr_t)(out + 1);

  v8bf16 r = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(packedFP4, scale, 2);

  unsigned short *dst = (unsigned short *)out;
  for (int i = 0; i < 8; ++i) {
    dst[i] = __builtin_bit_cast(unsigned short, r[i]);
  }
}
