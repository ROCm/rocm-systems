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
// gfx12 (RDNA4) v_wmma_f32_16x16x32_f16, Wave32:
//
//   A input (8 VGPRs, <16 x half>):
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
//   C/D output (8 VGPRs, <8 x float>):
//     i = 8*floor(lane/16) + GPR
//     j = lane % 16
//     → Lanes 0-15: rows 0-7;  Lanes 16-31: rows 8-15
//
// gfx942 (CDNA3) v_mfma_f32_16x16x16_f16, Wave64:
//
//   A input (2 VGPRs, <4 x half>):
//     i = lane % 16
//     k = 4*floor(lane/16) + 2*GPR + floor(bits/16)
//     → Lanes 0-15: k=0..3;  16-31: k=4..7;  32-47: k=8..11;  48-63: k=12..15
//
//   C/D output (4 VGPRs, <4 x float>):
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
// Why no strict_wwm wrapper
// -------------------------
// ds_bpermute reads from ANY lane regardless of EXEC.  MFMA on CDNA reads
// source operands from ALL 64 lanes regardless of EXEC (EXEC only gates the
// destination write).  WMMA operations occur in non-divergent code (EXEC is
// all-ones).  mbcnt_lo/hi with mask=-1 computes a pure bit-count independent
// of EXEC.
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
/// redistribute → 2× MFMA(K=16) → collect.
///
/// \param groupBase  0 for group 0 (W64 lanes 0-31), 32 for group 1 (lanes 32-63)
static void runGroupPass(IRBuilder<> &B, Module &M, RaiseContext &ctx,
                         unsigned groupBase, Value *laneId,
                         Value **aDwords, Value **bDwords, Value **cDwords,
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

  auto *v4f16Ty = FixedVectorType::get(ctx.f16Ty, 4);
  auto *v4f32Ty = FixedVectorType::get(ctx.f32Ty, 4);

  Value *srcA_lo = packDwords(B, mfmaA_lo, 2, ctx.i32Ty, v4f16Ty);
  Value *srcB_lo = packDwords(B, mfmaB_lo, 2, ctx.i32Ty, v4f16Ty);
  Value *acc     = packDwords(B, mfmaC,    4, ctx.i32Ty, v4f32Ty);

  Function *mfmaFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_mfma_f32_16x16x16f16);
  Value *cbsz = B.getInt32(0), *abid = B.getInt32(0), *blgp = B.getInt32(0);

  Value *mfma1 = B.CreateCall(mfmaFn,
      {srcA_lo, srcB_lo, acc, cbsz, abid, blgp}, "mfma1");

  Value *srcA_hi = packDwords(B, mfmaA_hi, 2, ctx.i32Ty, v4f16Ty);
  Value *srcB_hi = packDwords(B, mfmaB_hi, 2, ctx.i32Ty, v4f16Ty);

  Value *mfma2 = B.CreateCall(mfmaFn,
      {srcA_hi, srcB_hi, mfma1, cbsz, abid, blgp}, "mfma2");

  Value *mfmaDst[4];
  unpackDwords(B, mfma2, 4, ctx.i32Ty, mfmaDst);

  Value *w32Lane = B.CreateAnd(laneId, B.getInt32(31), "w32_lane");
  collectResult(B, M, mfmaDst, w32Lane, resultDwords);
}

Value *emitWMMAtoMFMA(RaiseContext &ctx, Value *a, Value *b, Value *c) {
  IRBuilder<> &B = ctx.B;
  Module &M = ctx.M;


  Value *aDwords[8], *bDwords[8], *cDwords[8];
  unpackDwords(B, a, 8, ctx.i32Ty, aDwords);
  unpackDwords(B, b, 8, ctx.i32Ty, bDwords);
  unpackDwords(B, c, 8, ctx.i32Ty, cDwords);

  Value *laneId = emitLaneId(B, M, ctx.i32Ty);

  Value *result0[8], *result1[8];
  runGroupPass(B, M, ctx, 0,  laneId, aDwords, bDwords, cDwords, result0);
  runGroupPass(B, M, ctx, 32, laneId, aDwords, bDwords, cDwords, result1);

  Value *isGroup1 = B.CreateICmpUGE(laneId, B.getInt32(32), "is_group1");
  auto *v8f32Ty = FixedVectorType::get(ctx.f32Ty, 8);
  Value *finalDwords[8];
  for (unsigned i = 0; i < 8; ++i)
    finalDwords[i] = B.CreateSelect(isGroup1, result1[i], result0[i], "sel");

  return packDwords(B, finalDwords, 8, ctx.i32Ty, v8f32Ty);
}

} // namespace transpiler
