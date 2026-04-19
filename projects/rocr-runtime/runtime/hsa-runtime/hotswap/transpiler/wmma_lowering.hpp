#ifndef HOTSWAP_TRANSPILER_WMMA_LOWERING_HPP
#define HOTSWAP_TRANSPILER_WMMA_LOWERING_HPP

namespace llvm {
class Value;
} // namespace llvm

namespace transpiler {

struct RaiseContext;

/// Element type / AB combination of the A/B input fragments to a
/// 16x16x{32,64} WMMA that we are lowering to gfx942 MFMA. Selects
/// the per-MFMA intrinsic AND the per-MFMA bitcast shape; the lane-
/// redistribution math itself is BYTE-IDENTICAL across all variants
/// because every supported WMMA input keeps the same per-Wave32-lane
/// fragment size — 8 VGPRs (= 32 bytes) on the A side, 8 VGPRs on the
/// B side, and 8 VGPRs of f32 on the C/D side. We factor on the
/// operation, not the opcode.
///
/// Per-variant divergence inside `runGroupPass`:
///
///   F16    → 2× mfma_f32_16x16x16f16,        per-MFMA pack `<4 x half>`
///   BF16   → 2× mfma_f32_16x16x16bf16_1k,    per-MFMA pack `<4 x i16>`
///   FP8_*  → 2× mfma_f32_16x16x32_<a>_<b>,   per-MFMA pack `i64`
///   BF8_*  → 2× mfma_f32_16x16x32_<a>_<b>,   per-MFMA pack `i64`
///
/// The 2× MFMA decomposition is invariant across variants: a WMMA
/// computes a 16x16 result for K=32 (16-bit) or K=64 (8-bit), and we
/// always split that K dimension in half so each MFMA covers K=16
/// (16-bit MFMA) or K=32 (8-bit MFMA) per call. Per-Wave64-lane MFMA
/// fragment is exactly 2 dwords either way.
///
/// The bf8/fp8 16-bit→i16 / fp8→i64 bitcasts are principled: the CDNA
/// MFMA intrinsics for these element types were defined before the
/// matching first-class LLVM types existed, and the storage containers
/// (i16 for bf16, i64 for 8 packed fp8/bf8) are bit-for-bit identical.
enum class WMMAInputType {
  F16,
  BF16,
  FP8_FP8,
  FP8_BF8,
  BF8_FP8,
  BF8_BF8,
};

/// Lower a Wave32 WMMA to Wave64 MFMA using ds_bpermute lane
/// redistribution.
///
/// This ports the proven approach from the original hotswap prototype
/// (hotswap/transpiler.cpp, WMMA→MFMA translation block) to principled
/// LLVM IR using two intrinsic families:
///
///   @llvm.amdgcn.ds.bpermute  — cross-lane data reads (EXEC-independent)
///   @llvm.amdgcn.mfma.*       — hardware matrix multiply (reads all 64 lanes)
///
/// No strict_wwm wrapper is needed because both ds_bpermute and MFMA read
/// from all lanes regardless of EXEC, and matmul kernels execute with a
/// non-divergent control flow (full EXEC mask).
///
/// Source fragment shapes (Wave32, 8 VGPRs per side either way):
///   16-bit (F16 / BF16):  A/B = <16 x {half|bfloat}> per lane
///   8-bit  (FP8 / BF8 ):  A/B = <8 x i32> per lane (32 packed fp8/bf8 bytes)
/// C/D is always <8 x float> per lane.
///
/// \param a  WMMA source A fragment (8 VGPRs in Wave32, layout per
///           inputType)
/// \param b  WMMA source B fragment (8 VGPRs in Wave32, layout per
///           inputType)
/// \param c  <8 x float> — WMMA accumulator fragment (8 VGPRs in Wave32)
/// \param inputType  selects MFMA intrinsic + per-MFMA bitcast type.
/// \returns  <8 x float> — result in Wave32 layout
llvm::Value *emitWMMAtoMFMA(RaiseContext &ctx,
                            llvm::Value *a,
                            llvm::Value *b,
                            llvm::Value *c,
                            WMMAInputType inputType);

} // namespace transpiler

#endif
