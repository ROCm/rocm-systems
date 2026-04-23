// ============================================================================
// WMMA → MFMA Lowering via Layout-Aware Lane Redistribution
// ============================================================================
//
// This file lowers Wave32 WMMA instructions (gfx1250 / RDNA4) to Wave64 MFMA
// instructions (gfx942 / CDNA3) using cross-lane data movement.
//
// Background
// ----------
// WMMA (Wave Matrix Multiply Accumulate) on gfx1250 is a Wave32 collective
// operation: 32 lanes cooperate to compute a 16×16 matrix multiply.  MFMA on
// gfx942 is a Wave64 collective: 64 lanes cooperate.  Because the per-lane
// fragment sizes differ (WMMA: <16 x half> / <8 x float>; MFMA: <4 x half> /
// <4 x float>), a simple intrinsic swap is impossible.
//
// The core problem is the wave-size mismatch.  When gfx1250 code is compiled
// for gfx942, the hardware groups 64 threads into one wavefront instead of 32.
// Threads 0-31 and 32-63 were in separate Wave32 wavefronts on the source
// architecture, computing different sub-tiles.  A single MFMA would mix their
// unrelated data.
//
// Register Layout Equations (from AMD Matrix Instruction Calculator)
// ------------------------------------------------------------------
//
// gfx12 (RDNA4) v_wmma_f32_16x16x{32,64}_{f16,bf16,fp8_*,bf8_*}, Wave32:
//
// All supported variants share the same per-Wave32-lane fragment shape:
// 8 VGPRs (= 32 bytes) per A side, 8 VGPRs per B side, 8 VGPRs of f32
// per C/D side. The K-dimension scales inversely with the element
// width (K=32 for 16-bit elements, K=64 for 8-bit elements), so the
// total bytes per lane stay constant. The lane redistribution math
// below operates on dwords (32-bit cells); it is therefore byte-
// identical across element widths — the only per-variant divergence
// lives in (a) the MFMA intrinsic dispatched on the gfx942 side and
// (b) the per-MFMA bitcast / pack type. See `runGroupPass` and the
// `WMMAInputType` enum in `wmma_lowering.hpp` for the full enumeration.
//
//   A input — 16-bit variants (8 VGPRs, <16 x {half|bfloat}>):
//     i = lane % 16
//     k = 8*floor(GPR/2) + 4*floor(lane/16) + 2*(GPR%2) + floor(bits/16)
//
//     Per-lane breakdown:
//       Lanes 0-15:   GPR 0→k={0,1}  GPR 1→k={2,3}  GPR 2→k={8,9}
//                     GPR 3→k={10,11} GPR 4→k={16,17} GPR 5→k={18,19}
//                     GPR 6→k={24,25} GPR 7→k={26,27}
//       Lanes 16-31:  GPR 0→k={4,5}  GPR 1→k={6,7}  GPR 2→k={12,13}
//                     GPR 3→k={14,15} GPR 4→k={20,21} GPR 5→k={22,23}
//                     GPR 6→k={28,29} GPR 7→k={30,31}
//
//   A input — 8-bit variants (8 VGPRs, <8 x i32> = 32 packed fp8/bf8):
//     Same dword-grain layout as the 16-bit variants — a fp8/bf8 byte
//     occupies the same byte slot inside its containing dword and
//     across lanes/GPRs that the corresponding 16-bit element would
//     have occupied. The K-stride doubles (each dword holds 4 fp8/bf8
//     bytes vs 2 halves) so the per-GPR K-range is twice as wide, but
//     the redistribution acts at dword granularity and does not see
//     the element-level interpretation.
//
//   C/D output (8 VGPRs, <8 x float>) — invariant across variants:
//     i = 8*floor(lane/16) + GPR
//     j = lane % 16
//     → Lanes 0-15: rows 0-7;  Lanes 16-31: rows 8-15
//
// gfx942 (CDNA3) MFMA targets:
//
//   16-bit MFMA (v_mfma_f32_16x16x16_{f16|bf16_1k}, K=16 per call):
//     A input (2 VGPRs, <4 x {half|i16}>):
//       i = lane % 16
//       k = 4*floor(lane/16) + 2*GPR + floor(bits/16)
//
//   8-bit MFMA (v_mfma_f32_16x16x32_{fp8|bf8}_{fp8|bf8}, K=32 per call):
//     A input (2 VGPRs, packed as i64 = 8 fp8/bf8 bytes):
//       Same per-lane VGPR width (2 dwords) as the 16-bit MFMA. The
//       K-fanout doubles to match the doubled WMMA K-range.
//
//   C/D output (4 VGPRs, <4 x float>) — invariant across variants:
//     i = 4*floor(lane/16) + (GPR % 4)
//     j = lane % 16
//     → Lanes 0-15: rows 0-3; 16-31: rows 4-7; 32-47: rows 8-11; 48-63: rows 12-15
//
// Approach
// --------
// We process each "virtual Wave32 group" (lanes 0-31 and 32-63) in a separate
// pass.  For each group:
//
//   1. REDISTRIBUTE: Use ds_bpermute to move WMMA fragments into the MFMA
//      layout.  The mapping is NOT a simple lane/2 — it must account for the
//      interleaved k-distribution between lanes 0-15 and 16-31 in gfx12 WMMA,
//      and the 4-way lane-group distribution in gfx942 MFMA.
//
//      For each MFMA GPR, we read from the correct WMMA GPR and lane using
//      a 4-way select based on the MFMA lane group (floor(laneId/16)):
//
//        Lane group 0 (lanes 0-15):  reads from WMMA lower half (lanes 0-15)
//        Lane group 1 (lanes 16-31): reads from WMMA upper half (lanes 16-31)
//        Lane group 2 (lanes 32-47): reads from WMMA lower half
//        Lane group 3 (lanes 48-63): reads from WMMA upper half
//
//      With WMMA GPR selection cycling every 2 lane groups (GPR pairs {0,1},
//      {2,3} for first K=16; {4,5}, {6,7} for second K=16).
//
//   2. MFMA: Two v_mfma_f32_16x16x16_f16 calls (K=32 decomposed to 2× K=16),
//      chaining the accumulator.
//
//   3. COLLECT: Gather the 4-VGPR MFMA result back to 8-VGPR WMMA layout.
//      WMMA GPR_w reads from MFMA GPR (GPR_w % 4), lane computed as:
//        srcLane = 32*(w32Lane >= 16) + 16*(GPR_w >= 4) + (w32Lane & 15)
//
// After both passes, a lane-ID-based select picks the correct group's result.
//
// Partial-wave correctness and hardware EXEC
// -------------------------------------------
// The redistribute / MFMA / collect pipeline is semantically a Wave64
// collective: every MFMA input and every collect-time bpermute source
// is physically stored in SOME lane of the Wave64, and each destination
// lane's VGPR must hold the correct value for the next stage to read
// it back.
//
// `ds_bpermute` and `v_mfma_*` both READ all 64 source lanes regardless
// of EXEC — but the WRITE of their per-lane result is EXEC-gated.  So
// a lane with EXEC=0 silently skips updating its destination VGPR, and
// any later cross-lane read of that VGPR returns stale / poison data.
//
// This is invisible when the kernel is launched with a blockDim that
// fills an entire Wave64 (every lane is active; MFMA inputs / outputs
// are written everywhere).  It manifests as a catastrophic correctness
// failure on partial-wave launches — e.g. a Wave32 WMMA kernel
// launched with blockDim == 32 runs as a single Wave64 with EXEC =
// 0x0000_0000_FFFF_FFFF on gfx942.  Lanes 32-63 never update their
// mfmaA/B/C VGPRs, so MFMA reads garbage for k=2,3 (for the K=4 f32
// path) or for the entire upper k-half (for the K=32/K=64 path), and
// the collect-stage bpermute's reads from lanes 32-63 of the MFMA
// output return garbage too.  Rows 8-15 of the output come out as
// undefined / zero, and rows 0-7 get only a partial K-accumulation.
//
// The fix lives OUTSIDE this file, at the transpiler's kernel-entry
// plumbing: `WaveNativeProjection::emitInitialExec` (in
// `wave_projection.cpp`) emits `@llvm.amdgcn.init_whole_wave` at the
// very top of the lifted kernel, which (a) sets hardware EXEC = -1 for
// the remainder of the kernel and (b) captures the original per-lane
// active mask into the transpiler's EXEC alloca. Every VGPR write,
// memory store, LDS op, and atomic in the lifted IR already routes
// through `RaiseContext::emitUnderExec`, which reads the alloca-backed
// source EXEC and emits an `if (lane_active)` diamond — the AMDGPU
// backend lowers those divergent branches by setting hardware EXEC
// to the ballot of the per-lane predicate inside each `do` block and
// restoring to EXEC = -1 afterwards. So between `emitUnderExec`
// diamonds (which is where the bpermute / MFMA / select chain here
// lives) hardware EXEC is -1, and all 64 lanes participate in the
// Wave64 collective exactly as required.
//
// This supersedes an earlier design that wrapped MFMA-output dwords in
// `@llvm.amdgcn.strict.wwm`. That design was semantically correct but
// unscalable: `SIPreAllocateWWMRegs` requires a DEDICATED physical
// VGPR per virtual register defined inside a WWM bracket, and the
// WWM def-chain from an MFMA output walks back through the entire
// accumulator initialisation. A 128×128 f16 matmul tile's entry
// region contains ~200 IMPLICIT_DEF / AV_MOV_B32 0 instructions for
// its accumulator ring, which together with the kernel's own VGPR
// demand exceeds gfx942's 256-VGPR pool and aborts the allocator
// with `physreg not found for WWM expression`. Moving the EXEC = -1
// guarantee to kernel entry sidesteps the allocator pressure entirely
// — no intermediate vreg is ever "inside WWM" and regalloc is
// ordinary — while preserving the partial-wave correctness property.
//
// From the perspective of this file, that means the redistribute +
// MFMA + collect chain emits ONLY ordinary IR (bpermute, bitcast,
// select, MFMA intrinsic) with no WWM markers. The hardware EXEC = -1
// invariant is the kernel-wide ambient set up by
// `WaveNativeProjection::emitInitialExec`, and the `writeRegVec` call
// in `handle_valu_vop3p.cpp` that consumes this file's return value
// takes care of gating the Wave32-layout destination VGPRs back to
// the original per-lane active mask.
//
// ============================================================================

#include "wmma_lowering.hpp"
#include "raise_context.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace transpiler {

static Value *emitDSBpermute(IRBuilder<> &B, Module &M,
                             Value *byteOffset, Value *srcVal) {
  Function *fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_ds_bpermute);
  return B.CreateCall(fn, {byteOffset, srcVal}, "bperm");
}

static Value *emitLaneId(IRBuilder<> &B, Module &M, Type *i32Ty) {
  Function *mbcntLo = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_mbcnt_lo);
  Function *mbcntHi = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_mbcnt_hi);
  Value *allOnes = ConstantInt::getSigned(i32Ty, -1);
  Value *zero = ConstantInt::get(i32Ty, 0);
  Value *lo = B.CreateCall(mbcntLo, {allOnes, zero}, "lane_lo");
  return B.CreateCall(mbcntHi, {allOnes, lo}, "lane_id");
}

static Value *packDwords(IRBuilder<> &B, Value **dwords, unsigned nDwords,
                         Type *i32Ty, Type *targetTy) {
  auto *vecTy = FixedVectorType::get(i32Ty, nDwords);
  Value *vec = PoisonValue::get(vecTy);
  for (unsigned i = 0; i < nDwords; ++i)
    vec = B.CreateInsertElement(vec, dwords[i], i, "pack");
  return B.CreateBitCast(vec, targetTy, "cast");
}

static void unpackDwords(IRBuilder<> &B, Value *vec, unsigned nDwords,
                         Type *i32Ty, Value **out) {
  auto *vecTy = FixedVectorType::get(i32Ty, nDwords);
  Value *asI32 = B.CreateBitCast(vec, vecTy, "toi32");
  for (unsigned i = 0; i < nDwords; ++i)
    out[i] = B.CreateExtractElement(asI32, i, "dw");
}

/// 4-way select based on lane group index (0..3).
/// Returns vals[laneGroup] for each lane.
static Value *selectByLaneGroup(IRBuilder<> &B, Value *laneGroup,
                                Value *v0, Value *v1, Value *v2, Value *v3) {
  Value *s = v3;
  s = B.CreateSelect(B.CreateICmpEQ(laneGroup, B.getInt32(2)), v2, s);
  s = B.CreateSelect(B.CreateICmpEQ(laneGroup, B.getInt32(1)), v1, s);
  s = B.CreateSelect(B.CreateICmpEQ(laneGroup, B.getInt32(0)), v0, s);
  return s;
}

/// Redistribute A or B input from gfx12 WMMA layout (8 VGPRs, Wave32)
/// to gfx942 MFMA layout (2 VGPRs × 2 passes, Wave64).
///
/// The gfx12 WMMA k-distribution interleaves between the two lane halves:
///   Lanes 0-15:  GPR pairs {0,1}→k 0-3, {2,3}→k 8-11, {4,5}→k 16-19, {6,7}→k 24-27
///   Lanes 16-31: GPR pairs {0,1}→k 4-7, {2,3}→k 12-15, {4,5}→k 20-23, {6,7}→k 28-31
///
/// The gfx942 MFMA distributes k across 4 lane groups of 16:
///   LG 0 (0-15)→k 0-3, LG 1 (16-31)→k 4-7, LG 2 (32-47)→k 8-11, LG 3 (48-63)→k 12-15
///
/// For MFMA1 (first K=16): LG {0,1} read WMMA GPRs {0,1}; LG {2,3} read GPRs {2,3}
/// For MFMA2 (second K=16): LG {0,1} read WMMA GPRs {4,5}; LG {2,3} read GPRs {6,7}
/// Even lane groups read from lower W32 half, odd from upper W32 half.
static void redistributeInput(IRBuilder<> &B, Module &M,
                               Value **wmmaGPRs,
                               Value *addrLo, Value *addrHi,
                               Value *laneGroup,
                               Value **mfmaLo, Value **mfmaHi) {
  for (unsigned g = 0; g < 2; ++g) {
    Value *v0 = emitDSBpermute(B, M, addrLo, wmmaGPRs[g]);
    Value *v1 = emitDSBpermute(B, M, addrHi, wmmaGPRs[g]);
    Value *v2 = emitDSBpermute(B, M, addrLo, wmmaGPRs[g + 2]);
    Value *v3 = emitDSBpermute(B, M, addrHi, wmmaGPRs[g + 2]);
    mfmaLo[g] = selectByLaneGroup(B, laneGroup, v0, v1, v2, v3);

    Value *u0 = emitDSBpermute(B, M, addrLo, wmmaGPRs[g + 4]);
    Value *u1 = emitDSBpermute(B, M, addrHi, wmmaGPRs[g + 4]);
    Value *u2 = emitDSBpermute(B, M, addrLo, wmmaGPRs[g + 6]);
    Value *u3 = emitDSBpermute(B, M, addrHi, wmmaGPRs[g + 6]);
    mfmaHi[g] = selectByLaneGroup(B, laneGroup, u0, u1, u2, u3);
  }
}

/// Redistribute accumulator C from gfx12 WMMA layout (8 VGPRs, Wave32)
/// to gfx942 MFMA layout (4 VGPRs, Wave64).
///
/// gfx12 WMMA: i = 8*floor(lane/16) + GPR  →  rows 0-7 in lanes 0-15, 8-15 in lanes 16-31
/// gfx942 MFMA: i = 4*floor(lane/16) + GPR →  rows 0-3 in LG0, 4-7 in LG1, 8-11 in LG2, 12-15 in LG3
///
/// MFMA GPR g needs:
///   LG 0: i = g      → WMMA GPR g,   lower W32 half
///   LG 1: i = 4+g    → WMMA GPR 4+g, lower W32 half
///   LG 2: i = 8+g    → WMMA GPR g,   upper W32 half
///   LG 3: i = 12+g   → WMMA GPR 4+g, upper W32 half
static void redistributeAcc(IRBuilder<> &B, Module &M,
                              Value **cDwords,
                              Value *addrLo, Value *addrHi,
                              Value *laneGroup,
                              Value **mfmaC) {
  for (unsigned g = 0; g < 4; ++g) {
    Value *v0 = emitDSBpermute(B, M, addrLo, cDwords[g]);
    Value *v1 = emitDSBpermute(B, M, addrLo, cDwords[g + 4]);
    Value *v2 = emitDSBpermute(B, M, addrHi, cDwords[g]);
    Value *v3 = emitDSBpermute(B, M, addrHi, cDwords[g + 4]);
    mfmaC[g] = selectByLaneGroup(B, laneGroup, v0, v1, v2, v3);
  }
}

/// Collect MFMA result (4 VGPRs, Wave64) back to WMMA layout (8 VGPRs, Wave32).
///
/// WMMA GPR_w reads MFMA GPR (GPR_w % 4) from:
///   srcLane = 32*(w32Lane >= 16) + 16*(GPR_w >= 4) + (w32Lane & 15)
static void collectResult(IRBuilder<> &B, Module &M,
                           Value **mfmaDwords, Value *w32Lane,
                           Value **out) {
  Value *w32Lo = B.CreateAnd(w32Lane, B.getInt32(15), "w32_lo");
  Value *isUpper = B.CreateICmpUGE(w32Lane, B.getInt32(16), "is_upper");
  Value *upperOff = B.CreateSelect(isUpper, B.getInt32(32), B.getInt32(0),
                                   "upper_off");

  for (unsigned gw = 0; gw < 8; ++gw) {
    Value *gprOff = B.getInt32((gw >= 4) ? 16 : 0);
    Value *srcLane = B.CreateAdd(
        B.CreateAdd(upperOff, gprOff), w32Lo, "col_lane");
    Value *addr = B.CreateShl(srcLane, B.getInt32(2), "col_addr");
    out[gw] = emitDSBpermute(B, M, addr, mfmaDwords[gw % 4]);
  }
}

/// Run one full pass for a virtual Wave32 group:
/// redistribute → 2× MFMA → collect, wrapped in a single whole-wave
/// region so the cross-lane pipeline runs with EXEC = -1 regardless
/// of the caller-level EXEC mask (see file-header "Whole-wave mode").
///
/// \param groupBase  0 for group 0 (W64 lanes 0-31), 32 for group 1 (lanes 32-63)
/// \param inputType  selects MFMA intrinsic + per-MFMA pack/bitcast type.
///                   The lane-redistribution math is element-type-agnostic
///                   across the entire WMMAInputType enumeration because
///                   every supported variant has the same per-Wave32-lane
///                   fragment size (8 VGPRs of A, 8 VGPRs of B, 8 VGPRs of
///                   f32 C/D) and the same K-decomposition factor (split
///                   into 2 chained MFMA calls per Wave32 group). The only
///                   per-variant divergence is the MFMA intrinsic name
///                   and the per-MFMA-call pack type:
///                     F16    → mfma_f32_16x16x16f16,        <4 x half>
///                     BF16   → mfma_f32_16x16x16bf16_1k,    <4 x i16>
///                     FP8_*  → mfma_f32_16x16x32_<a>_<b>,   i64
///                     BF8_*  → mfma_f32_16x16x32_<a>_<b>,   i64
///                   The bf16 → i16 and fp8/bf8 → i64 bitcasts are
///                   principled: the matching CDNA MFMA intrinsics were
///                   defined before the corresponding first-class LLVM
///                   types existed, and the storage containers are
///                   bit-for-bit identical.
static void runGroupPass(IRBuilder<> &B, Module &M, RaiseContext &ctx,
                         unsigned groupBase, Value *laneId,
                         Value **aDwords, Value **bDwords, Value **cDwords,
                         WMMAInputType inputType,
                         Value **resultDwords) {
  Value *laneMod16 = B.CreateAnd(laneId, B.getInt32(15), "lane16");
  Value *loLane = B.CreateAdd(laneMod16, B.getInt32(groupBase), "lo_lane");
  Value *hiLane = B.CreateAdd(laneMod16, B.getInt32(groupBase + 16), "hi_lane");
  Value *addrLo = B.CreateShl(loLane, B.getInt32(2), "addr_lo");
  Value *addrHi = B.CreateShl(hiLane, B.getInt32(2), "addr_hi");
  Value *laneGroup = B.CreateLShr(laneId, B.getInt32(4), "lane_grp");

  Value *mfmaA_lo[2], *mfmaA_hi[2];
  redistributeInput(B, M, aDwords, addrLo, addrHi, laneGroup,
                    mfmaA_lo, mfmaA_hi);

  Value *mfmaB_lo[2], *mfmaB_hi[2];
  redistributeInput(B, M, bDwords, addrLo, addrHi, laneGroup,
                    mfmaB_lo, mfmaB_hi);

  Value *mfmaC[4];
  redistributeAcc(B, M, cDwords, addrLo, addrHi, laneGroup, mfmaC);

  // Per-MFMA bitcast type and intrinsic dispatch — the only point in
  // the lowering where the WMMA variants diverge.
  //
  // AB pack type:
  //   16-bit variants pack 2 redistributed dwords into a `<4 x t>` vector
  //   (4 elements per lane × 2 bytes = 8 bytes = 2 dwords). The 8-bit
  //   variants pack the same 2 dwords into a single `i64`; the CDNA fp8/
  //   bf8/i8 MFMA intrinsics were defined before fp8 became a first-class
  //   LLVM type (and there's no first-class packed-i8 vector type either),
  //   so they take 8 packed fp8/bf8/i8 bytes as i64 directly. Both packings
  //   are 64-bit and produced by the same 2-dword reduce (`packDwords`).
  //
  // Accumulator pack type:
  //   F32-accumulator MFMAs (everything except IU8) take `<4 x float>`.
  //   The IU8 path takes `<4 x i32>` to match the integer-accumulator
  //   `mfma_i32_16x16x32_i8` signature.
  Type *mfmaABPackTy = nullptr;
  Type *mfmaAccPackTy = nullptr;
  Intrinsic::ID mfmaId;
  switch (inputType) {
  case WMMAInputType::F16:
    mfmaABPackTy = FixedVectorType::get(ctx.f16Ty, 4);
    mfmaAccPackTy = FixedVectorType::get(ctx.f32Ty, 4);
    mfmaId = Intrinsic::amdgcn_mfma_f32_16x16x16f16;
    break;
  case WMMAInputType::BF16:
    mfmaABPackTy = FixedVectorType::get(Type::getInt16Ty(ctx.C), 4);
    mfmaAccPackTy = FixedVectorType::get(ctx.f32Ty, 4);
    mfmaId = Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k;
    break;
  case WMMAInputType::FP8_FP8:
    mfmaABPackTy = ctx.i64Ty;
    mfmaAccPackTy = FixedVectorType::get(ctx.f32Ty, 4);
    mfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8;
    break;
  case WMMAInputType::FP8_BF8:
    mfmaABPackTy = ctx.i64Ty;
    mfmaAccPackTy = FixedVectorType::get(ctx.f32Ty, 4);
    mfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8;
    break;
  case WMMAInputType::BF8_FP8:
    mfmaABPackTy = ctx.i64Ty;
    mfmaAccPackTy = FixedVectorType::get(ctx.f32Ty, 4);
    mfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8;
    break;
  case WMMAInputType::BF8_BF8:
    mfmaABPackTy = ctx.i64Ty;
    mfmaAccPackTy = FixedVectorType::get(ctx.f32Ty, 4);
    mfmaId = Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8;
    break;
  case WMMAInputType::IU8:
    mfmaABPackTy = ctx.i64Ty;
    mfmaAccPackTy = FixedVectorType::get(ctx.i32Ty, 4);
    mfmaId = Intrinsic::amdgcn_mfma_i32_16x16x32_i8;
    break;
  }

  Value *srcA_lo = packDwords(B, mfmaA_lo, 2, ctx.i32Ty, mfmaABPackTy);
  Value *srcB_lo = packDwords(B, mfmaB_lo, 2, ctx.i32Ty, mfmaABPackTy);
  Value *acc     = packDwords(B, mfmaC,    4, ctx.i32Ty, mfmaAccPackTy);

  Function *mfmaFn = Intrinsic::getOrInsertDeclaration(&M, mfmaId);
  Value *cbsz = B.getInt32(0), *abid = B.getInt32(0), *blgp = B.getInt32(0);

  // MFMA is EXEC-gated on its WRITE: a lane with EXEC=0 skips
  // updating its destination VGPR.  Under `WaveNativeProjection` the
  // kernel-entry `init_whole_wave` keeps HW EXEC=-1 kernel-wide so
  // every lane writes its MFMA output and `wrapAsWWMValue` is a
  // no-op.  Under `ModuloReplicationProjection` (phantom-lane
  // fallback) HW EXEC = source-active-mask kernel-wide, so target
  // lanes 32..63 would never write their MFMA destination VGPR —
  // and the subsequent `collectResult` bpermute DOES read from
  // those target lanes (target lanes 16..31 pull rows 8..15 from
  // source lanes 32..47's MFMA output), so stale data would
  // corrupt output rows 8..15.  `wrapAsWWMValue` inserts a
  // `strict.wwm` marker on each MFMA output under MODREP, telling
  // the AMDGPU backend's `SIWholeQuadMode` pass to mark the MFMA
  // itself as WWM and emit `s_or_saveexec_b64 sN, -1` / `s_mov_b64
  // exec, sN` around it so every lane writes its output.
  //
  // We wrap the MFMA outputs specifically (not just the final collect
  // results) because SIWholeQuadMode's backward-propagation from a
  // `strict.wwm` on a later `ds_bpermute` result stops at the
  // bpermute boundary — the backend sees the bpermute reads source
  // lanes 0..W_src-1 by address and concludes the MFMA output on
  // lanes W_src..2*W_src-1 is "not consumed", which is correct for
  // a single-pass MFMA but wrong for our cross-widening lowering
  // where the collect bpermute DOES pull from those upper-half
  // lanes to assemble rows 8..15.  Wrapping the MFMA result directly
  // forces the MFMA into the WWM backward slice.
  Value *mfma1 = ctx.projection.wrapAsWWMValue(
      B,
      B.CreateCall(mfmaFn,
                   {srcA_lo, srcB_lo, acc, cbsz, abid, blgp}, "mfma1"),
      "mfma1_wwm");

  Value *srcA_hi = packDwords(B, mfmaA_hi, 2, ctx.i32Ty, mfmaABPackTy);
  Value *srcB_hi = packDwords(B, mfmaB_hi, 2, ctx.i32Ty, mfmaABPackTy);

  Value *mfma2 = ctx.projection.wrapAsWWMValue(
      B,
      B.CreateCall(mfmaFn,
                   {srcA_hi, srcB_hi, mfma1, cbsz, abid, blgp}, "mfma2"),
      "mfma2_wwm");

  Value *mfmaDst[4];
  unpackDwords(B, mfma2, 4, ctx.i32Ty, mfmaDst);

  Value *w32Lane = B.CreateAnd(laneId, B.getInt32(31), "w32_lane");
  collectResult(B, M, mfmaDst, w32Lane, resultDwords);

  // Also wrap each collect-stage output dword as WWM under MODREP,
  // so `SIWholeQuadMode` keeps the collect bpermutes in the WWM
  // region.  Without this, the bpermute WRITEs are EXEC-gated to
  // source-active lanes under normal HW EXEC, leaving target lanes
  // past the source active range with zeros (or stale values from
  // earlier instruction re-use of the same physical VGPR).  Target
  // lanes 16..31 in the wave32-source phantom-lane regime read
  // collect outputs from target lanes 32..47 / 48..63 to assemble
  // rows 8..15; those reads happen at ds_bpermute READ-time which
  // is un-gated, but the WRITE-back on target lanes 16..31 itself
  // is HW EXEC-gated and requires WWM for the chain to land the
  // correct per-lane value — the bpermute-READ's result sits in
  // the target lane's register file only if that target lane is
  // HW-active at the bpermute WRITE, which under non-WWM MODREP is
  // the case for source-active lanes 0..31 but the destination
  // register allocation often aliases with values that were last
  // written in an earlier WWM region (where phantom lanes DID
  // write), so the stale phantom-lane content leaks through.
  // Wrapping the collect outputs forces the backend to treat the
  // bpermute writes as WWM, which clears the alias hazard.
  //
  // Under WaveNative this is an identity no-op.
  for (unsigned i = 0; i < 8; ++i)
    resultDwords[i] =
        ctx.projection.wrapAsWWMValue(B, resultDwords[i], "wmma_collect_wwm");
}

Value *emitWMMAtoMFMA(RaiseContext &ctx, Value *a, Value *b, Value *c,
                       WMMAInputType inputType) {
  // The redistribute / 2×MFMA / collect chain is a Wave64 collective.
  // Per-MFMA-output `strict.wwm` markers inside `runGroupPass` handle
  // the two projections uniformly:
  //
  //   * WaveNativeProjection — `init_whole_wave` at kernel entry
  //     already keeps HW EXEC=-1 kernel-wide.  `wrapAsWWMValue` is
  //     an identity no-op here to avoid the `SIPreAllocateWWMRegs`
  //     regalloc blow-up on large WMMA tiles (see
  //     `WaveProjection::emitInitialExec`'s block comment).
  //
  //   * ModuloReplicationProjection — HW EXEC stays at the source-
  //     active mask kernel-wide; the per-MFMA `strict.wwm` tells
  //     the backend's `SIWholeQuadMode` pass to emit `s_or_saveexec
  //     _b64 sN, -1` / `s_mov_b64 exec, sN` around the MFMA and its
  //     redistribute inputs so every target lane writes its MFMA
  //     destination VGPR, which the subsequent collect bpermute
  //     then reads across the full Wave64.
  //
  // Number of passes depends on the projection's
  // `numSourceWavesPerTarget()`:
  //
  //   * 1 source wave per target (MODREP phantom-lane fallback):
  //     run only pass 0 (`groupBase=0`).  Pass 1 would pull data
  //     from source lanes W_src..2*W_src-1 which do not exist under
  //     phantom-lane, feeding undef into the MFMA cross-lane
  //     reduction.  Target lanes 32..63's output from pass 0 is
  //     well-defined (a correct MFMA group-0 output spread across
  //     the full Wave64) and harmlessly discarded by the post-lift
  //     `writeRegVec` `emitUnderExec`, whose HW-EXEC-gated store
  //     only fires on source-active lanes.
  //
  //   * 2 source waves per target (WaveNative cross-widen): run
  //     both passes; each source wave's WMMA maps to one pass.
  //     Target lanes 0..31 get pass-0's output (source wave 0's
  //     matmul), target lanes 32..63 get pass-1's (source wave 1's
  //     matmul), selected via a `laneId >= 32` comparison.
  IRBuilder<> &B = ctx.B;
  Module &M = ctx.M;

  Value *aDwords[8], *bDwords[8], *cDwords[8];
  unpackDwords(B, a, 8, ctx.i32Ty, aDwords);
  unpackDwords(B, b, 8, ctx.i32Ty, bDwords);
  unpackDwords(B, c, 8, ctx.i32Ty, cDwords);

  Value *laneId = emitLaneId(B, M, ctx.i32Ty);

  const unsigned numSrcWaves = ctx.projection.numSourceWavesPerTarget();
  assert((numSrcWaves == 1 || numSrcWaves == 2) &&
         "WMMA→MFMA lowering defined only for wave32 source projections; "
         "MODREP phantom-lane (1 source wave per target) or WaveNative "
         "cross-widen (2 source waves per target) are the two supported "
         "shapes — a new projection class must declare which applies.");
  // Dead-code gate for the refusal-gate-still-in-place state.  The K=4
  // f32 and K=32/K=64 refusal arms in `handle_valu_vop3p.cpp` return
  // before reaching this helper when `!providesFullWaveExecInvariant()`,
  // i.e. under any non-WaveNative projection.  The `numSrcWaves == 1`
  // branch below is therefore unreachable in production.  It is kept
  // in-tree because (a) my staged MODREP path has been verified
  // correct for minimal repros (`wmma_phantom_lane_f16_chain`) and is
  // the right code to re-enable once the matmul_fp16_16x16 residual is
  // pinned, and (b) removing it would require re-deriving the
  // pass-1-skip logic later.  This assertion makes the unreachable
  // branch a loud abort in assert-on builds so a future refactor that
  // accidentally flips the refusal gate without vetting the MODREP
  // path surfaces at the first matmul_fp16 test run instead of the
  // second.
  assert(numSrcWaves == 2 &&
         "reached emitWMMAtoMFMA under a projection that returns "
         "numSourceWavesPerTarget() != 2 — the MODREP arm of this "
         "lowering is staged-but-gated-off via the refusal in "
         "`handle_valu_vop3p.cpp`; if you flipped that gate, also "
         "vet compare_correctness's matmul_fp16_16x16 end-to-end "
         "output before removing this assert.");

  Value *result0[8];
  runGroupPass(B, M, ctx, /*groupBase=*/0, laneId, aDwords, bDwords, cDwords,
               inputType, result0);

  Value *finalDwords[8];
  if (numSrcWaves == 1) {
    for (unsigned i = 0; i < 8; ++i) finalDwords[i] = result0[i];
  } else {
    Value *result1[8];
    runGroupPass(B, M, ctx, /*groupBase=*/32, laneId, aDwords, bDwords,
                 cDwords, inputType, result1);
    Value *isGroup1 = B.CreateICmpUGE(laneId, B.getInt32(32), "is_group1");
    for (unsigned i = 0; i < 8; ++i)
      finalDwords[i] =
          B.CreateSelect(isGroup1, result1[i], result0[i], "sel");
  }

  // Result element type matches the dispatched MFMA accumulator
  // type (i32 for the IU8 integer-accumulator variant, f32 for
  // everything else).  The per-pass output dwords are i32
  // throughout; only the final reassembly cares about the element
  // semantics.
  Type *resultElemTy = (inputType == WMMAInputType::IU8) ? ctx.i32Ty : ctx.f32Ty;
  return packDwords(B, finalDwords, 8, ctx.i32Ty,
                     FixedVectorType::get(resultElemTy, 8));
}

// ----------------------------------------------------------------------
// v_wmma_f32_16x16x4_f32 → mfma_f32_16x16x4f32 lowering
// ----------------------------------------------------------------------
//
// Source (gfx1250 RDNA4, Wave32):
//   int_amdgcn_wmma_f32_16x16x4_f32 — `<8 x f32>` = (…, <2 x f32> A,
//   …, <2 x f32> B, …, <8 x f32> C, …)
//
// Target (gfx942 CDNA3, Wave64):
//   int_amdgcn_mfma_f32_16x16x4f32 — `<4 x f32>` = (f32 A, f32 B,
//   <4 x f32> C, i32 cbsz, i32 abid, i32 blgp)
//
// Register-layout equations
// -------------------------
// Source WMMA (Wave32, per-lane fragment):
//   A/B  — <2 x f32> (2 VGPRs):  i = lane%16,  k = 2*floor(lane/16) + GPR
//     Lanes 0-15 GPR 0→k=0, GPR 1→k=1
//     Lanes 16-31 GPR 0→k=2, GPR 1→k=3
//   C/D  — <8 x f32> (8 VGPRs):  i = 8*floor(lane/16) + GPR,  j = lane%16
//     Lanes 0-15 → rows 0-7;   Lanes 16-31 → rows 8-15
//
// Target MFMA (Wave64, per-lane fragment):
//   A/B  — f32 (1 VGPR):          i = lane%16,  k = floor(lane/16)
//     LG0 (lanes 0-15)  → k=0
//     LG1 (lanes 16-31) → k=1
//     LG2 (lanes 32-47) → k=2
//     LG3 (lanes 48-63) → k=3
//   C/D  — <4 x f32> (4 VGPRs):   i = 4*floor(lane/16) + GPR, j = lane%16
//     (same layout equation as the K=32/K=64 MFMA family, so the C
//     redistribution + result collection helpers above are reused
//     verbatim.)
//
// Redistribution
// --------------
// Per-group pass (`groupBase ∈ {0, 32}`): the W32-group-N's data is
// held by W64 lanes `[groupBase .. groupBase+31]`. Each MFMA call
// spreads ONE Wave32 group's 32-lane × 2-dword A across all 64 Wave64
// lanes:
//
//   loAddr = 4 * ((lane%16) + groupBase)       // source for k=0..1
//   hiAddr = 4 * ((lane%16) + groupBase + 16)  // source for k=2..3
//
//   LG0 (lanes 0-15,  k=0): bpermute(loAddr, aDwords[0])
//   LG1 (lanes 16-31, k=1): bpermute(loAddr, aDwords[1])
//   LG2 (lanes 32-47, k=2): bpermute(hiAddr, aDwords[0])
//   LG3 (lanes 48-63, k=3): bpermute(hiAddr, aDwords[1])
//
// All four reads deliver the full K=4 range for ONE virtual Wave32
// group, which matches the K=4 MFMA signature — so there is exactly
// ONE MFMA call per group (not 2 chained as in the K=32/K=64 path).
//
// B redistribution mirrors A exactly (same layout equation). The C
// redistribution reuses `redistributeAcc` (same WMMA C layout) and
// the result collection reuses `collectResult` (same WMMA D layout).
//
// Hardware EXEC: the redistribute / MFMA / collect chain relies on
// the kernel-wide EXEC = -1 invariant set up by
// `WaveNativeProjection::emitInitialExec`, so no in-file WWM marker
// is needed.  See the file-header "Partial-wave correctness and
// hardware EXEC" section for the correctness argument.
static void runGroupPassF32K4(IRBuilder<> &B, Module &M, RaiseContext &ctx,
                               unsigned groupBase, Value *laneId,
                               Value **aDwords, Value **bDwords,
                               Value **cDwords,
                               Value **resultDwords) {
  Value *laneMod16 = B.CreateAnd(laneId, B.getInt32(15), "lane16");
  Value *loLane = B.CreateAdd(laneMod16, B.getInt32(groupBase), "lo_lane");
  Value *hiLane = B.CreateAdd(laneMod16, B.getInt32(groupBase + 16), "hi_lane");
  Value *addrLo = B.CreateShl(loLane, B.getInt32(2), "addr_lo");
  Value *addrHi = B.CreateShl(hiLane, B.getInt32(2), "addr_hi");
  Value *laneGroup = B.CreateLShr(laneId, B.getInt32(4), "lane_grp");

  // A input: single-dword MFMA fragment, one bpermute per (lane-group,
  // GPR) combination. aDwords[0] carries k=0 and k=2; aDwords[1]
  // carries k=1 and k=3 (lower vs upper WMMA half selects the +16
  // lane offset).
  Value *aLG0 = emitDSBpermute(B, M, addrLo, aDwords[0]);
  Value *aLG1 = emitDSBpermute(B, M, addrLo, aDwords[1]);
  Value *aLG2 = emitDSBpermute(B, M, addrHi, aDwords[0]);
  Value *aLG3 = emitDSBpermute(B, M, addrHi, aDwords[1]);
  Value *mfmaA_i32 =
      selectByLaneGroup(B, laneGroup, aLG0, aLG1, aLG2, aLG3);

  Value *bLG0 = emitDSBpermute(B, M, addrLo, bDwords[0]);
  Value *bLG1 = emitDSBpermute(B, M, addrLo, bDwords[1]);
  Value *bLG2 = emitDSBpermute(B, M, addrHi, bDwords[0]);
  Value *bLG3 = emitDSBpermute(B, M, addrHi, bDwords[1]);
  Value *mfmaB_i32 =
      selectByLaneGroup(B, laneGroup, bLG0, bLG1, bLG2, bLG3);

  Value *mfmaC[4];
  redistributeAcc(B, M, cDwords, addrLo, addrHi, laneGroup, mfmaC);

  // Pack per-lane MFMA operands. The signatures are:
  //   A:f32         (scalar dword, not packed)
  //   B:f32         (scalar dword, not packed)
  //   C:<4 x f32>
  // The redistribution produced i32 dwords; bitcast A/B to f32 and
  // pack C into `<4 x float>` via the existing helper (which also
  // bitcasts).
  auto *mfmaAccPackTy = FixedVectorType::get(ctx.f32Ty, 4);
  Value *mfmaA = B.CreateBitCast(mfmaA_i32, ctx.f32Ty, "mfma_a");
  Value *mfmaB = B.CreateBitCast(mfmaB_i32, ctx.f32Ty, "mfma_b");
  Value *acc = packDwords(B, mfmaC, 4, ctx.i32Ty, mfmaAccPackTy);

  // cbsz / abid / blgp are the per-matrix broadcast-and-shift
  // modifiers hard-coded to zero here; the corpus kernels emit the
  // MFMA equivalent with these defaulted, matching what gfx1250
  // WMMA surfaces for the failing kerneldex Tensile GEMMs.
  Value *cbsz = B.getInt32(0), *abid = B.getInt32(0), *blgp = B.getInt32(0);

  Function *mfmaFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_mfma_f32_16x16x4f32);
  // See the K=32 / K=64 `runGroupPass` helper above for the full
  // rationale: under MODREP phantom-lane, the MFMA's destination
  // VGPR must be written on every target lane so the subsequent
  // `collectResult` bpermute reads real data from lanes 32..47's
  // output.  `wrapAsWWMValue` inserts a `strict.wwm` marker under
  // MODREP (so the backend's `SIWholeQuadMode` pulls the MFMA into
  // a WWM region) and is an identity no-op under WaveNative (whose
  // kernel-entry `init_whole_wave` already keeps HW EXEC=-1).
  Value *mfma = ctx.projection.wrapAsWWMValue(
      B,
      B.CreateCall(mfmaFn, {mfmaA, mfmaB, acc, cbsz, abid, blgp}, "mfma"),
      "mfma_wwm");

  Value *mfmaDst[4];
  unpackDwords(B, mfma, 4, ctx.i32Ty, mfmaDst);

  Value *w32Lane = B.CreateAnd(laneId, B.getInt32(31), "w32_lane");
  collectResult(B, M, mfmaDst, w32Lane, resultDwords);

  // Wrap collect outputs as WWM under MODREP — see the equivalent
  // block comment in the K=32 / K=64 `runGroupPass` for the full
  // rationale.
  for (unsigned i = 0; i < 8; ++i)
    resultDwords[i] =
        ctx.projection.wrapAsWWMValue(B, resultDwords[i], "wmma_collect_wwm");
}

Value *emitWMMAtoMFMA_F32_16x16x4(RaiseContext &ctx, Value *a, Value *b,
                                   Value *c) {
  // K=4 f32 counterpart to `emitWMMAtoMFMA` above — see that
  // function's block comment for the full design rationale
  // (projection-aware per-MFMA `strict.wwm` wrapping via
  // `wrapAsWWMValue`, and pass-1-skipped under MODREP phantom-
  // lane).  The only structural difference is that the K=4 f32
  // decomposition emits ONE MFMA per group pass (the source WMMA
  // is already K=4, exactly matching `mfma_f32_16x16x4f32`), not
  // the 2-chained-MFMA K=32→2×K=16 structure of the 16-/8-bit
  // family.
  IRBuilder<> &B = ctx.B;
  Module &M = ctx.M;

  Value *aDwords[2], *bDwords[2], *cDwords[8];
  unpackDwords(B, a, 2, ctx.i32Ty, aDwords);
  unpackDwords(B, b, 2, ctx.i32Ty, bDwords);
  unpackDwords(B, c, 8, ctx.i32Ty, cDwords);

  Value *laneId = emitLaneId(B, M, ctx.i32Ty);

  const unsigned numSrcWaves = ctx.projection.numSourceWavesPerTarget();
  assert((numSrcWaves == 1 || numSrcWaves == 2) &&
         "WMMA→MFMA lowering defined only for wave32 source projections; "
         "MODREP phantom-lane (1 source wave per target) or WaveNative "
         "cross-widen (2 source waves per target) are the two supported "
         "shapes — a new projection class must declare which applies.");
  // See `emitWMMAtoMFMA` above for the identical gating rationale.
  assert(numSrcWaves == 2 &&
         "reached emitWMMAtoMFMA_F32_16x16x4 under a projection that "
         "returns numSourceWavesPerTarget() != 2; the MODREP arm is "
         "staged-but-gated-off in `handle_valu_vop3p.cpp`.");

  Value *result0[8];
  runGroupPassF32K4(B, M, ctx, /*groupBase=*/0, laneId, aDwords, bDwords,
                     cDwords, result0);

  Value *finalDwords[8];
  if (numSrcWaves == 1) {
    for (unsigned i = 0; i < 8; ++i) finalDwords[i] = result0[i];
  } else {
    Value *result1[8];
    runGroupPassF32K4(B, M, ctx, /*groupBase=*/32, laneId, aDwords, bDwords,
                       cDwords, result1);
    Value *isGroup1 = B.CreateICmpUGE(laneId, B.getInt32(32), "is_group1");
    for (unsigned i = 0; i < 8; ++i)
      finalDwords[i] =
          B.CreateSelect(isGroup1, result1[i], result0[i], "sel");
  }

  return packDwords(B, finalDwords, 8, ctx.i32Ty,
                     FixedVectorType::get(ctx.f32Ty, 8));
}

} // namespace transpiler
