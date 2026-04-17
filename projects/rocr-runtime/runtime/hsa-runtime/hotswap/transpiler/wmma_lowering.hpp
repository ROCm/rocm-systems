#ifndef HOTSWAP_TRANSPILER_WMMA_LOWERING_HPP
#define HOTSWAP_TRANSPILER_WMMA_LOWERING_HPP

namespace llvm {
class Value;
} // namespace llvm

namespace transpiler {

struct RaiseContext;

/// Lower a Wave32 WMMA v_wmma_f32_16x16x32_f16 to Wave64 MFMA using
/// ds_bpermute lane redistribution.
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
/// \param a  <16 x half>  — WMMA source A fragment (8 VGPRs in Wave32)
/// \param b  <16 x half>  — WMMA source B fragment (8 VGPRs in Wave32)
/// \param c  <8 x float>  — WMMA accumulator fragment (8 VGPRs in Wave32)
/// \returns  <8 x float>  — result in Wave32 layout
llvm::Value *emitWMMAtoMFMA(RaiseContext &ctx,
                            llvm::Value *a,
                            llvm::Value *b,
                            llvm::Value *c);

} // namespace transpiler

#endif
