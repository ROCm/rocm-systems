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
/// LLVM IR using three intrinsic families:
///
///   @llvm.amdgcn.ds.bpermute  — cross-lane data reads
///   @llvm.amdgcn.mfma.*       — hardware matrix multiply (reads all 64 lanes)
///   @llvm.amdgcn.strict.wwm   — run the redistribute / MFMA / collect
///                               chain with EXEC = -1 so partial-wave
///                               launches (e.g. blockDim == 32 on gfx942)
///                               still update the upper-half lanes' VGPRs
///                               that MFMA and the collect-time bpermute
///                               read back. See wmma_lowering.cpp,
///                               file-header "Whole-wave mode".
///
/// `ds_bpermute` and `v_mfma_*` both READ from all 64 source lanes
/// regardless of EXEC, but their per-lane WRITE is EXEC-gated — hence
/// the whole-wave-mode wrapper, which forces EXEC = -1 across the
/// collective so every lane's MFMA input / MFMA output / collected
/// result VGPR is written and subsequent cross-lane reads see fresh
/// data rather than stale / poison bits. We emit the WWM marker as
/// eight `strict.wwm.i32` calls (one per result dword), NOT one
/// `strict.wwm.v8i32` call on the packed vector: the backend's
/// `SIPreAllocateWWMRegs` pass reserves an aligned physical VGPR
/// per WWM virtual register, and a `<8 x i32>` WWM operand cannot
/// always be satisfied (`physreg not found for WWM expression`
/// llvm_unreachable) in WMMA-heavy kernels. Per-dword wrapping
/// only needs single-VGPR slots. See the "Whole-wave mode" header
/// in wmma_lowering.cpp for the full argument.
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

/// Lower a Wave32 v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4 VOP3P opcode
/// 0x05D) to Wave64 mfma_f32_16x16x4f32 (gfx942 CDNA3) using
/// ds_bpermute lane redistribution.
///
/// This K=4 f32 variant is STRUCTURALLY DISTINCT from the K=32 / K=64
/// family covered by `emitWMMAtoMFMA` above:
///
///   * A/B fragment per Wave32 lane is `<2 x f32>` (2 VGPRs) — not
///     the <16 x t> (16-bit element) or <8 x i32> (8-bit element)
///     8-VGPR fragments used by the K=32 / K=64 variants.
///   * Only ONE MFMA call per Wave32 group — not 2 chained. The
///     source WMMA is already K=4, which exactly matches the target
///     `mfma_f32_16x16x4f32` K dimension; there is no K-tiling to do
///     inside a group.
///   * The MFMA A/B input is a single `float` per Wave64 lane — not
///     `<4 x t>` (16-bit) or `i64` (8-bit).
///
/// The C/D fragment shape (<8 x f32> per Wave32 lane, Wave32 C-layout
/// equation `i = 8*floor(lane/16) + GPR`) is IDENTICAL to the K=32 /
/// K=64 variants, so the accumulator redistribution and the final
/// result collection reuse the same internal helpers as `emitWMMAtoMFMA`.
///
/// LAYOUT NOTE. The gfx1250 V_WMMA_F32_16x16x4_F32 per-lane A/B
/// (i, k) layout is extrapolated from the documented K=32 / K=64
/// pattern — the AMD Matrix Instruction Calculator does not yet list
/// the K=4 f32 variant. We assume:
///
///   A/B input (<2 x f32> per lane):
///     i = lane % 16
///     k = 2*floor(lane/16) + GPR
///   Per-lane:
///     Lanes 0-15, GPR 0 → k=0    GPR 1 → k=1
///     Lanes 16-31, GPR 0 → k=2   GPR 1 → k=3
///
/// This is the unique natural extension of the K=32/K=64 layout
/// documented in wmma_lowering.cpp (lower half holds lower k-range,
/// upper half holds upper k-range; GPR indexes along k within a
/// lane-half). The layout is validated out-of-band by the hipBLASLt
/// Tensile `SS_SS_HA_Bias_SAV_UA` f32 GEMM kernels (macro-tile
/// `MT32x32x16`, WMMA shape `MI16x16x1`, wave32) — see
/// `tools/tensile_gold_verify/verify_transpile.py`, which compares
/// the lifted gfx942 output against the gfx1250 reference. A layout
/// mismatch would surface as a numerical regression there, not a
/// silent wrong answer.
///
/// \param a  WMMA source A fragment (<2 x f32> in Wave32)
/// \param b  WMMA source B fragment (<2 x f32> in Wave32)
/// \param c  WMMA accumulator fragment (<8 x f32> in Wave32)
/// \returns  `<8 x float>` — result in Wave32 C-layout.
llvm::Value *emitWMMAtoMFMA_F32_16x16x4(RaiseContext &ctx,
                                         llvm::Value *a,
                                         llvm::Value *b,
                                         llvm::Value *c);

} // namespace transpiler

#endif
