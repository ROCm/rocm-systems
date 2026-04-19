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
///   F16    → 2× mfma_f32_16x16x16f16,        AB pack `<4 x half>`,  acc f32
///   BF16   → 2× mfma_f32_16x16x16bf16_1k,    AB pack `<4 x i16>`,   acc f32
///   FP8_*  → 2× mfma_f32_16x16x32_<a>_<b>,   AB pack `i64`,         acc f32
///   BF8_*  → 2× mfma_f32_16x16x32_<a>_<b>,   AB pack `i64`,         acc f32
///   IU8    → 2× mfma_i32_16x16x32_i8,        AB pack `i64`,         acc i32
///
/// The 2× MFMA decomposition is invariant across variants: a WMMA
/// computes a 16x16 result for K=32 (16-bit) or K=64 (8-bit), and we
/// always split that K dimension in half so each MFMA covers K=16
/// (16-bit MFMA) or K=32 (8-bit MFMA) per call. Per-Wave64-lane MFMA
/// fragment is exactly 2 dwords either way.
///
/// The bf8/fp8 16-bit→i16 / fp8/i8→i64 bitcasts are principled: the
/// CDNA MFMA intrinsics for these element types were defined before
/// the matching first-class LLVM types existed (or, for i8, before
/// AMDGCN had a packed i8 vector type at all), and the storage
/// containers (i16 for bf16, i64 for 8 packed fp8/bf8/i8 bytes) are
/// bit-for-bit identical to the per-lane source dwords.
///
/// The IU8 variant is the only one with a non-f32 accumulator —
/// `runGroupPass` carries the accumulator IR pack type through the
/// dispatch alongside the AB pack type so the MFMA call signature
/// matches the i32-accumulator MFMA intrinsic exactly.
enum class WMMAInputType {
  F16,
  BF16,
  FP8_FP8,
  FP8_BF8,
  BF8_FP8,
  BF8_BF8,
  IU8,
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
///   8-bit  (IU8        ):  A/B = <8 x i32> per lane (32 packed i8/u8 bytes)
/// C/D fragment shape:
///   F32-accumulator variants (everything except IU8): <8 x float>
///   IU8 variant:                                       <8 x i32>
///
/// The C and result IR types must match the accumulator side of the
/// dispatched MFMA intrinsic (CDNA distinguishes mfma_f32_* and
/// mfma_i32_* by accumulator element type). Caller is expected to
/// pre-load `c` with the correct vector type for the inputType — the
/// helper does NOT bitcast; the redistribution and pack code paths
/// operate on dword-extracted scalars and only the final vector
/// reassembly cares about the element type.
///
/// \param a  WMMA source A fragment (8 VGPRs in Wave32, layout per
///           inputType)
/// \param b  WMMA source B fragment (8 VGPRs in Wave32, layout per
///           inputType)
/// \param c  WMMA accumulator fragment (8 VGPRs in Wave32, vector
///           element type matches the dispatched MFMA accumulator —
///           f32 for everything except IU8 which uses i32)
/// \param inputType  selects MFMA intrinsic, per-MFMA AB pack type,
///                   AND the accumulator pack/result element type.
/// \returns  <8 x t> — result in Wave32 layout, where t = float for
///           every variant except IU8 which returns <8 x i32>.
llvm::Value *emitWMMAtoMFMA(RaiseContext &ctx,
                            llvm::Value *a,
                            llvm::Value *b,
                            llvm::Value *c,
                            WMMAInputType inputType);

} // namespace transpiler

#endif
