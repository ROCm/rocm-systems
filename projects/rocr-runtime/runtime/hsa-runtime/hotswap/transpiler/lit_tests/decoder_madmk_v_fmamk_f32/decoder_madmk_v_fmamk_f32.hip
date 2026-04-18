// ARCH: gfx1250
//
// Decoder regression fixture for VOP2 MADMK form (lit_tests/
// decoder_madmk_v_fmamk_f32).
//
// VOP_MADMK (VOP2Instructions.td) defines Ins32 = (src0, K-imm,
// src1) — the 32-bit literal sits at MCInst index 2 BETWEEN src0
// (index 1) and src1 (index 3). The natural positional walk in
// `decode.cpp::buildSrcMap` produces srcMap = [src0, K-imm, src1],
// which is the convention the V_FMAMK_F32 handler in handle_valu.
// cpp expects (`srcF(0)=src0, srcF(1)=K, srcF(2)=src2`).
//
// Before commit ab25d0257a, `driftCheckSrcN` would abort the raise
// with "srcMap disagrees with OpName::srcN table for v_fmamk_f32"
// because OpName::src1 lives at MCInst index 3, not srcMap[1] = 2.
// The fix relaxes the check at k=1 for the MADMK signature
// (src0Idx < immIdx < src1Idx).
//
// This fixture pins the MADMK acceptance independently of corpus
// state. The GPT-OSS `attn_fwd` kernel uses v_fmamk_f32 and was
// the original trigger; isolating the regression-detection here
// means a future change to driftCheckSrcN that re-introduces the
// strict check would be caught even if the corpus sweep wasn't
// run.

#include <hip/hip_runtime.h>

extern "C" __global__ void decoder_madmk_v_fmamk_f32_kernel(float *out,
                                                              float *a,
                                                              float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float va = a[tid];
  float vc = c[tid];
  float r;
  // v_fmamk_f32 dst, src0, K, src2: dst = src0 * K + src2
  // The K immediate is encoded inline in the instruction word.
  // Inline asm with a literal K constant forces hipcc to emit
  // v_fmamk_f32 (rather than v_fma_f32 or v_fmac_f32 or
  // v_fmaak_f32 which has a different operand order).
  asm volatile(
      "v_fmamk_f32 %[r], %[va], 0x40490fdb, %[vc]\n\t"
      : [r] "=v"(r)
      : [va] "v"(va), [vc] "v"(vc));
  out[tid] = r;
}
