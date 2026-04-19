#ifndef HOTSWAP_TRANSPILER_WMMA_LOWERING_HPP
#define HOTSWAP_TRANSPILER_WMMA_LOWERING_HPP

namespace llvm {
class Value;
} // namespace llvm

namespace transpiler {

struct RaiseContext;

/// Element type of the A/B input fragments to a 16x16x32 K=32 WMMA
/// that we are lowering to gfx942 MFMA. Selects the per-MFMA intrinsic
/// AND the per-MFMA bitcast shape; the lane-redistribution math itself
/// is identical between F16 and BF16 because both are 16-bit element
/// types — we factor on the operation, not the opcode.
enum class WMMAInputType {
  /// v_wmma_f32_16x16x32_f16  (gfx1250) → 2× mfma_f32_16x16x16f16
  /// MFMA per-call element type: <4 x half>.
  F16,
  /// v_wmma_f32_16x16x32_bf16 (gfx1250) → 2× mfma_f32_16x16x16bf16_1k
  /// MFMA per-call element type: <4 x i16> (CDNA bf16 is encoded as
  /// i16 in the MFMA intrinsic signature; the bitcast is principled —
  /// IEEE 754 binary16 and bfloat16 share an i16 storage container).
  BF16,
};

/// Lower a Wave32 WMMA `v_wmma_f32_16x16x32_<f16|bf16>` to Wave64
/// MFMA using ds_bpermute lane redistribution.
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
/// \param a  <16 x {half|bfloat}> — WMMA source A fragment (8 VGPRs in Wave32)
/// \param b  <16 x {half|bfloat}> — WMMA source B fragment (8 VGPRs in Wave32)
/// \param c  <8 x float>         — WMMA accumulator fragment (8 VGPRs in Wave32)
/// \param inputType  selects MFMA intrinsic + per-MFMA bitcast type.
/// \returns  <8 x float>          — result in Wave32 layout
llvm::Value *emitWMMAtoMFMA(RaiseContext &ctx,
                            llvm::Value *a,
                            llvm::Value *b,
                            llvm::Value *c,
                            WMMAInputType inputType);

} // namespace transpiler

#endif
