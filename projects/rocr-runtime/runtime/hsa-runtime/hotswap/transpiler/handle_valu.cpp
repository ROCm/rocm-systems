#include "handle_valu_internal.hpp"
#include "handlers.hpp"
#include "opcode_map.hpp"
#include "wmma_lowering.hpp"

#include "semop.hpp"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>
#include <tuple>

using namespace llvm;

namespace transpiler {

namespace {

// ============================================================================
// Carry-chain scalar-operand routing for VOP2-CI / VOP3B-CI instructions.
//
// The V_{ADD,SUB,SUBREV}_CO_(CI_)U32 family exists in two encodings:
//
//   * e32: implicit VCC for both carry-in (ci variants) and carry-out.
//     The MC operand table has no scalar operand; `op.nSrcs() == 2`
//     (vsrc0, vsrc1) and `di.numDefs == 1` (vdst only).
//
//   * e64 / VOP3B: EXPLICIT scalar operand for both carry-in (ci
//     variants, MC src index 2) and carry-out (MC def index 1). The
//     scalar can be `vcc_lo` / `vcc` OR an arbitrary `sN` — the
//     compiler picks based on SGPR pressure. `op.nSrcs() == 3` on ci
//     variants and `di.numDefs == 2` on every co variant (ci or not).
//
// Pre-2026-04-22 the six carry-chain handlers in this file hardcoded
// `ctx.regs.loadVCC` / `ctx.regs.storeVCC` for both endpoints,
// silently ignoring the explicit scalar operand on e64 forms. That
// matches the VOPD `v_dual_cndmask_b32` SGPR-condition bug that
// miscompiled `canary_bpermute_scan_fp32` and `corpus_layernorm_fp32`
// (hotswap/docs/modrep-predicate-chain.md §6.4). The current corpus
// (Triton on gfx1250 / gfx942, AITER TensileLite) does not exercise
// the non-VCC SGPR form of these instructions — Triton emits
// `v_add_nc_u32` / `v_add_nc_u64` (no-carry) on gfx1250 and the
// fused `v_lshl_add_u64` on gfx942, AITER emits `v_add_co_u32 ...,
// vcc_lo, ...` exclusively. But the latent silent-miscompile is
// strictly worse than the VOPD bug it mirrors, because it would
// miscompile address arithmetic rather than a single predicate, and
// the principled project rule is "never do silent fallbacks". The
// helpers below mirror `V_CNDMASK_B32`'s SGPR-aware routing in
// `handle_valu_vop3p.cpp` so these six handlers now share the
// exact same scalar-operand semantics.
// ============================================================================

// Read the per-lane i1 carry-in for a carry-chain instruction.
//
// For e64 forms whose MC src at `srcIndex` is an explicit scalar register:
//
//   * `vcc_lo` / `vcc` → `loadVCC` (the same path e32 would take).
//   * `sN` → prefer `lookupSgprWaveMaskI1(N)`'s fresh per-BB V_CMP
//     shadow `i1` (populated by V_CMP_*_e64 writers in the same BB);
//     fall back to `projection.extractLaneBitFromWaveMask` on the
//     raw SGPR alloca (lossy under wave32 → wave64 cross-widening
//     if the producer truncated to source width — same residual as
//     the non-VOPD V_CNDMASK_B32 handler, see
//     hotswap/docs/sgpr-wave-mask-translation.md §3.1).
//   * NOREG (null ssrc2) → zero carry-in (hardware semantics for
//     null scalar source; defensive — AMDGPU backends don't emit
//     this in practice, but an i1 zero is the least-surprising
//     interpretation if it ever appears).
//
// For e32 forms (no explicit scalar operand — `op.nSrcs() <= srcIndex`
// or the operand is not a register) → `loadVCC` (the e32 implicit
// VCC semantics).
Value *readCarryInI1(RaiseContext &ctx, const DecodedInst &di,
                      OpResolver &op, unsigned srcIndex) {
  if (op.nSrcs() > srcIndex && di.isReg(op.srcIdx(srcIndex))) {
    ParsedReg carryReg =
        ctx.parseReg(di.getReg(op.srcIdx(srcIndex)), op.srcIdx(srcIndex));
    switch (carryReg.kind) {
    case ParsedReg::VCC:
      return ctx.regs.loadVCC(ctx.B);
    case ParsedReg::SGPR:
      if (carryReg.baseIdx >= 0) {
        if (Value *freshCmp = ctx.lookupSgprWaveMaskI1(carryReg.baseIdx))
          return freshCmp;
        Value *condVal = ctx.isa.isWave32()
                             ? ctx.regs.loadSGPR32(ctx.B, carryReg.baseIdx)
                             : ctx.regs.loadSGPR64(ctx.B, carryReg.baseIdx);
        return ctx.projection.extractLaneBitFromWaveMask(ctx.B, condVal);
      }
      break;
    case ParsedReg::NOREG:
      return ConstantInt::getFalse(ctx.B.getInt1Ty());
    default:
      break;
    }
  }
  return ctx.regs.loadVCC(ctx.B);
}

// Write the per-lane i1 carry-out for a carry-chain instruction.
//
// For e64 forms whose MC def at index 1 is an explicit scalar register:
//
//   * `vcc_lo` / `vcc` → `storeVCC` (the same path e32 would take).
//   * `sN` → ballot the per-lane i1 up to source-wave-mask width via
//     `projection.ballotI1ToWidth`, store the narrow mask to the
//     SGPR via `writeRegExecWidth` (wave32-source single SGPR;
//     wave64-source SGPR pair), AND record the fresh per-lane `i1`
//     shadow via `recordSgprWaveMaskI1` so a same-BB consumer (e.g.
//     a following V_CNDMASK_B32 or V_ADD_CO_CI_U32) can look it up
//     without the lossy extract round-trip. Mirrors the V_CMP_*_e64
//     SGPR-write path in `handle_valu_vcmp.cpp`.
//   * NOREG (null sdst) → discard the carry-out (hardware semantics
//     for null scalar destination).
//
// For e32 forms (no explicit destination — `di.numDefs < 2` or the
// def is not a register) → `storeVCC` (the e32 implicit VCC
// semantics).
void writeCarryOutI1(RaiseContext &ctx, const DecodedInst &di,
                      OpResolver &op, Value *carryI1) {
  if (di.numDefs >= 2 && di.isReg(1)) {
    ParsedReg carryDst = op.dst(1);
    switch (carryDst.kind) {
    case ParsedReg::VCC:
      ctx.regs.storeVCC(ctx.B, carryI1);
      return;
    case ParsedReg::SGPR:
      if (carryDst.baseIdx >= 0) {
        Type *sourceWidth = ctx.projection.sourceWaveMaskTy();
        Value *mask = ctx.projection.ballotI1ToWidth(
            ctx.B, carryI1, sourceWidth, "carry_ballot");
        ctx.writeRegExecWidth(carryDst, mask);
        ctx.recordSgprWaveMaskI1(carryDst.baseIdx, carryI1,
                                  /*isPair=*/carryDst.width >= 2);
      }
      return;
    case ParsedReg::NOREG:
      return;
    default:
      break;
    }
  }
  ctx.regs.storeVCC(ctx.B, carryI1);
}

// Emit the cross-target (gfx1250 -> gfx94x) dequantisation expansion
// of `v_cvt_scale_pk8_bf16_fp4` for `scale_sel == 0` as pure IR.
// Returns a `<8 x bfloat>` Value; the caller hands it to
// `writeRegVec` exactly like the same-target intrinsic path does.
//
// see hotswap/docs/matrix-translation.md §7.4 — MXFP4 dequant primitive.
//
// Algorithm (per lane i in 0..7; matches
// `mxfp4::mxfp4BitAlgebraBf16Bits` in mxfp4_dequant.cpp step-for-step
// so the cpp-level unit test pins the algorithm and the canary on
// gfx1250 pins it end-to-end against the hardware primitive):
//
//   1. Extract 4-bit nibble:   %n_i  = (%src >> (i*4)) & 0xF
//   2. Decompose FP4 E2M1:     sign=n_i[3], exp_fp4=n_i[2:1], mant_fp4=n_i[0]
//   3. FP4 -> BF16 fields:
//        // Normal FP4 (exp_fp4 >= 1): bf16_exp = exp_fp4 + 126,
//        // bf16_mant = mant_fp4 ? 0x40 : 0.
//        // Subnormal FP4 (exp_fp4 == 0): bf16_exp = mant_fp4 ? 126 : 0,
//        // bf16_mant = 0.  (±0 stays ±0; ±0.5 becomes normal BF16 exp=126.)
//   4. Scale byte:             scale_byte = %scale & 0xFF    (scale_sel==0 only)
//   5. Apply scale via exp add:
//        new_exp = (signed i32) bf16_exp + scale_byte - 127
//        result =
//          (scale_byte == 0xFF)         ? 0x7FC0                         // qNaN
//          : (bf16_magnitude == 0)       ? (sign << 15)                   // ±0
//          : (new_exp >= 0xFF)           ? (sign << 15) | 0x7F80          // ±Inf
//          : (new_exp >= 1)              ? (sign<<15) | (new_exp<<7) | bf16_mant   // normal
//          : subnormal_shift(new_exp, sign, 0x80 | bf16_mant)             // subnormal / ±0
//   6. Insert i16 bits -> bfloat -> <8 x bfloat> lane i.
//
// Corner-case summary (all bit-exact against the OCP MXFP spec + what
// the hardware primitive emits on bit-valid inputs; see
// `tests/mxfp4_dequant_test.cpp` for the full 4096-point sweep):
//   * FP4 ±0 × NaN scale  -> NaN (IEEE 0 × NaN = NaN).  The NaN-scale
//     branch short-circuits before the magnitude-zero check.
//   * FP4 ±0 × finite scale -> ±0 preserving sign.
//   * Overflow (new_exp >= 0xFF): saturate to BF16 ±Inf.  BF16
//     supports Inf even though FP4 does not — destination-format
//     semantics apply after the scale add.
//   * Underflow (new_exp <= 0): compute BF16 subnormal via right-shift
//     of the (implicit-1).mant field; zero when the shift drops all
//     bits past count 7 (BF16 subnormal range floor is 2^-133).
//   * Rounding mode: N/A.  The multiplication by 2^(scale_byte - 127)
//     is exact in floating-point for any power-of-2 scale; we emit
//     integer field manipulation instead of an fmul so the lowering
//     is bit-exact regardless of the target's float-mode register
//     state (FTZ / DAZ bits are irrelevant because no fmul actually
//     runs).
static llvm::Value *emitCvtScalePk8Bf16Fp4CrossTargetExpansion(
    RaiseContext &ctx, llvm::Value *srcI32, llvm::Value *scaleI32) {
  llvm::IRBuilder<> &B = ctx.B;
  llvm::Type *i32Ty = ctx.i32Ty;
  llvm::Type *i16Ty = llvm::Type::getInt16Ty(ctx.C);
  llvm::Type *bf16Ty = llvm::Type::getBFloatTy(ctx.C);

  // Constants used across all 8 lanes.  Factored out so the emitted
  // IR reads cleanly in lit / FileCheck output.
  llvm::Constant *c0xF   = llvm::ConstantInt::get(i32Ty, 0xF);
  llvm::Constant *c0xFF  = llvm::ConstantInt::get(i32Ty, 0xFF);
  llvm::Constant *c127   = llvm::ConstantInt::get(i32Ty, 127);
  llvm::Constant *c126   = llvm::ConstantInt::get(i32Ty, 126);
  llvm::Constant *c1     = llvm::ConstantInt::get(i32Ty, 1);
  llvm::Constant *c3     = llvm::ConstantInt::get(i32Ty, 3);
  llvm::Constant *c7     = llvm::ConstantInt::get(i32Ty, 7);
  llvm::Constant *c8     = llvm::ConstantInt::get(i32Ty, 8);
  llvm::Constant *c0x40  = llvm::ConstantInt::get(i32Ty, 0x40);
  llvm::Constant *c0x80  = llvm::ConstantInt::get(i32Ty, 0x80);
  llvm::Constant *c0x7F80 = llvm::ConstantInt::get(i32Ty, 0x7F80);

  // Scale-byte extraction: low byte of the i32 scale register.  All 8
  // lanes share the same scale byte for scale_sel == 0 per the
  // declared support set.
  llvm::Value *scaleByte = B.CreateAnd(scaleI32, c0xFF, "mxfp4_scale_byte");
  llvm::Value *isScaleNaN =
      B.CreateICmpEQ(scaleByte, c0xFF, "mxfp4_is_scale_nan");

  // BF16 canonical qNaN (0x7FC0), used when scale_byte == 0xFF; stored
  // as i32 so it merges with the select chain's other i32 branches.
  llvm::Constant *bf16NaN = llvm::ConstantInt::get(i32Ty, 0x7FC0);

  // Per-lane result accumulator: <8 x bfloat>, starts as undef (none
  // of the 8 lanes are fully-defined until every insertelement has
  // fired).  Mirrors the shape `writeRegVec` expects from the
  // same-target intrinsic arm.
  llvm::Type *v8bf16Ty = llvm::FixedVectorType::get(bf16Ty, 8);
  llvm::Value *vec = llvm::UndefValue::get(v8bf16Ty);

  llvm::Constant *c0 = llvm::ConstantInt::get(i32Ty, 0);

  for (unsigned lane = 0; lane < 8; ++lane) {
    // Nibble extraction.  Low nibble (lane 0) is in src bits [3:0],
    // matching hardware's "nibble 0 = lane 0" contract (documented on
    // the same-target arm above).
    llvm::Value *shamt = llvm::ConstantInt::get(i32Ty, lane * 4);
    llvm::Value *nibble = B.CreateAnd(
        B.CreateLShr(srcI32, shamt, "mxfp4_src_shr"),
        c0xF, "mxfp4_nibble");

    // FP4 E2M1 field decomposition.
    llvm::Value *signBit =
        B.CreateAnd(B.CreateLShr(nibble, c3), c1, "mxfp4_sign");
    llvm::Value *expFp4 =
        B.CreateAnd(B.CreateLShr(nibble, c1), c3, "mxfp4_exp_fp4");
    llvm::Value *mantFp4 =
        B.CreateAnd(nibble, c1, "mxfp4_mant_fp4");
    llvm::Value *signField =
        B.CreateShl(signBit, llvm::ConstantInt::get(i32Ty, 15),
                    "mxfp4_sign_field");

    // Normal-FP4 BF16 fields: exp_fp4 + 126 and (mant_fp4 ? 0x40 : 0).
    llvm::Value *bf16ExpNorm =
        B.CreateAdd(expFp4, c126, "mxfp4_bf16_exp_norm");
    llvm::Value *mantFp4NZ =
        B.CreateICmpNE(mantFp4, c0, "mxfp4_mant_fp4_nz");
    llvm::Value *bf16MantNorm =
        B.CreateSelect(mantFp4NZ, c0x40, c0, "mxfp4_bf16_mant_norm");

    // Subnormal-FP4 BF16 fields: if mant_fp4 = 1 (FP4 ±0.5) use
    // bf16_exp = 126; otherwise (FP4 ±0) bf16_exp = 0.  bf16_mant is
    // always 0 in this branch.
    llvm::Value *bf16ExpSub =
        B.CreateSelect(mantFp4NZ, c126, c0, "mxfp4_bf16_exp_sub");
    llvm::Value *isFp4Sub =
        B.CreateICmpEQ(expFp4, c0, "mxfp4_is_fp4_sub");
    llvm::Value *bf16Exp = B.CreateSelect(isFp4Sub, bf16ExpSub, bf16ExpNorm,
                                           "mxfp4_bf16_exp");
    llvm::Value *bf16Mant = B.CreateSelect(isFp4Sub, c0, bf16MantNorm,
                                            "mxfp4_bf16_mant");

    // Magnitude (exp || mant in low 15 bits).  Used only to detect
    // the FP4-±0 shortcut; no rounding implication.
    llvm::Value *magnitude = B.CreateOr(
        B.CreateShl(bf16Exp, c7), bf16Mant, "mxfp4_magnitude");
    llvm::Value *isFp4Zero =
        B.CreateICmpEQ(magnitude, c0, "mxfp4_is_fp4_zero");

    // Scaled exponent: bf16_exp + scale_byte - 127.  Signed i32 so
    // subnormal / zero decay is captured by new_exp < 1 rather than
    // by unsigned wrap.
    llvm::Value *expPlusScale =
        B.CreateAdd(bf16Exp, scaleByte, "mxfp4_exp_plus_scale");
    llvm::Value *newExp =
        B.CreateSub(expPlusScale, c127, "mxfp4_new_exp");

    // Overflow branch: new_exp >= 0xFF -> BF16 ±Inf.  Comparison is
    // signed because new_exp may underflow negative; anything >=
    // 0xFF is overflow regardless.
    llvm::Value *isOverflow =
        B.CreateICmpSGE(newExp, c0xFF, "mxfp4_is_overflow");
    llvm::Value *infBits =
        B.CreateOr(signField, c0x7F80, "mxfp4_inf_bits");

    // Normal branch: new_exp in [1, 0xFE] -> (sign<<15) | (new_exp<<7)
    // | bf16_mant.  We mask new_exp to 8 bits to keep the field
    // width correct when the branch is dead (the select's other arm
    // handles that case, but we still want a clean IR shape).
    llvm::Value *newExpMasked =
        B.CreateAnd(newExp, c0xFF, "mxfp4_new_exp_masked");
    llvm::Value *normalBits = B.CreateOr(
        B.CreateOr(signField,
                   B.CreateShl(newExpMasked, c7, "mxfp4_new_exp_shl"),
                   "mxfp4_sign_or_exp"),
        bf16Mant, "mxfp4_normal_bits");

    // Subnormal branch: new_exp <= 0 -> shift (implicit-1).mant right
    // by (1 - new_exp).  If shift >= 8 the BF16 representation loses
    // every bit and we flush to ±0.  This defensive clamp is
    // unreachable today — for FP4 exp >= 1 + scale_byte = 0 the
    // minimum new_exp is 127 + 0 - 127 = 0, giving shift_amt = 1; we
    // keep the clamp so widening the declared support set (e.g. a
    // future scale_sel handling that exposes smaller FP4 exponents)
    // doesn't silently miscompile.
    llvm::Value *implicit1Mant =
        B.CreateOr(c0x80, bf16Mant, "mxfp4_implicit_1_mant");
    llvm::Value *shiftAmt =
        B.CreateSub(c1, newExp, "mxfp4_shift_amt");
    llvm::Value *shiftedMant =
        B.CreateLShr(implicit1Mant, shiftAmt, "mxfp4_shifted_mant");
    llvm::Value *shiftTooBig =
        B.CreateICmpSGE(shiftAmt, c8, "mxfp4_shift_too_big");
    llvm::Value *subMant = B.CreateSelect(shiftTooBig, c0, shiftedMant,
                                           "mxfp4_sub_mant");
    llvm::Value *subBits =
        B.CreateOr(signField, subMant, "mxfp4_sub_bits");

    // new_exp >= 1 selects the normal bits; otherwise subnormal.
    llvm::Value *newExpGe1 =
        B.CreateICmpSGE(newExp, c1, "mxfp4_new_exp_ge_1");
    llvm::Value *normalOrSub =
        B.CreateSelect(newExpGe1, normalBits, subBits,
                       "mxfp4_normal_or_sub");

    // Priority-ordered merge, matching the C++ reference's control flow:
    //   result = is_scale_nan ? qNaN
    //          : is_fp4_zero  ? sign_field
    //          : is_overflow  ? inf_bits
    //          :                normal_or_sub
    //
    // LLVM's `select` is bottom-up (inner selects evaluated last), so
    // build the chain from the default case outward.
    llvm::Value *afterOverflow = B.CreateSelect(
        isOverflow, infBits, normalOrSub, "mxfp4_after_overflow");
    llvm::Value *afterZero = B.CreateSelect(
        isFp4Zero, signField, afterOverflow, "mxfp4_after_zero");
    llvm::Value *laneI32 = B.CreateSelect(
        isScaleNaN, bf16NaN, afterZero, "mxfp4_lane_i32");

    // i32 -> i16 -> bfloat insertion.  Trunc drops the zero-padded
    // upper bits; every result branch above produces a value in
    // [0, 0xFFFF] (qNaN=0x7FC0, Inf|sign ≤ 0xFF80, normal|sign|mant
    // ≤ 0xFFC0, sub|sign ≤ 0x80C0), so trunc is information-
    // preserving.
    llvm::Value *laneI16 = B.CreateTrunc(laneI32, i16Ty, "mxfp4_lane_i16");
    llvm::Value *laneBf = B.CreateBitCast(laneI16, bf16Ty, "mxfp4_lane_bf16");
    vec = B.CreateInsertElement(vec, laneBf,
                                 llvm::ConstantInt::get(i32Ty, lane),
                                 "mxfp4_vec_insert");
  }
  return vec;
}

} // namespace

HandlerResult handleVALU(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op) {
  HandlerResult hr;
  // `mn` is retained for diagnostic messages only; dispatch is driven entirely
  // by `sop`, which the OpcodeMap canonicalizer resolves from the DPP/SDWA/e32
  // encoding to the base pseudo before the handler sees it.
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;
  if (sop == SemOp::V_NOP) {
    hr.handled = true;
    return hr;
  }
  // ---- v_mov_b32 ----
  if (sop == SemOp::V_MOV_B32) {
    ctx.writeReg32(op.dst(), op.src(0));
    hr.handled = true;
    return hr;
  }
  // ---- Cross-lane primitives (readlane/writelane/permlane/mbcnt/
  //      readfirstlane) — extracted to handle_valu_cross_lane.cpp ----
  {
    HandlerResult sub = handleVALU_CrossLane(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  // ---- Small ops (conversions, F16 arith, single-src F32 transcendentals,
  //      16-bit shifts, V_BFREV_B32 / V_NOT_B32, byte pack) ----
  // Extracted to handle_valu_small_ops.cpp.
  {
    HandlerResult sub = handleVALU_SmallOps(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  // ---- Simple 2-src integer ALU ----
  if (sop == SemOp::V_ADD_NC_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(op.src(0), op.src(1), "vadd"));
    hr.handled = true;
    return hr;
  }
  // GFX9 VOP3-only v_add_i32 / v_sub_i32: plain add/sub when clamp=0,
  // signed saturation (saddsat/ssubsat) when clamp=1.
  if (sop == SemOp::V_ADD_I32 || sop == SemOp::V_SUB_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    int clampIdx = AMDGPU::getNamedOperandIdx(
        di.inst.getOpcode(), AMDGPU::OpName::clamp);
    bool clamped = clampIdx >= 0 && di.isImm(clampIdx) &&
                   di.getImm(clampIdx) != 0;
    if (clamped) {
      Intrinsic::ID satID = (sop == SemOp::V_ADD_I32)
                                ? Intrinsic::sadd_sat
                                : Intrinsic::ssub_sat;
      Value *res = ctx.B.CreateBinaryIntrinsic(satID, s0, s1);
      ctx.writeReg32(op.dst(), res);
    } else {
      Value *res = (sop == SemOp::V_ADD_I32)
                       ? ctx.B.CreateAdd(s0, s1, "vadd_i32")
                       : ctx.B.CreateSub(s0, s1, "vsub_i32");
      ctx.writeReg32(op.dst(), res);
    }
    hr.handled = true;
    return hr;
  }
  // Vector add with carry-out (GFX12: v_add_co_u32; VCC or sN = carry).
  // See `writeCarryOutI1` above for the SGPR-vs-VCC routing rationale.
  if (sop == SemOp::V_ADD_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateAdd(s0, s1, "vadd_co");
    ctx.writeReg32(op.dst(), res);
    auto *ov = ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {ctx.i32Ty}, {s0, s1});
    writeCarryOutI1(ctx, di, op, ctx.B.CreateExtractValue(ov, 1));
    hr.handled = true;
    return hr;
  }
  // Vector sub with carry-out (GFX9: v_sub_u32; GFX10+: v_sub_co_u32).
  if (sop == SemOp::V_SUB_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateSub(s0, s1, "vsub_co");
    ctx.writeReg32(op.dst(), res);
    writeCarryOutI1(ctx, di, op, ctx.B.CreateICmpULT(s0, s1));
    hr.handled = true;
    return hr;
  }
  // Vector reversed sub with carry-out (GFX9: v_subrev_u32; GFX10+: v_subrev_co_u32).
  if (sop == SemOp::V_SUBREV_CO_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *res = ctx.B.CreateSub(s1, s0, "vsubrev_co");
    ctx.writeReg32(op.dst(), res);
    writeCarryOutI1(ctx, di, op, ctx.B.CreateICmpULT(s1, s0));
    hr.handled = true;
    return hr;
  }
  // Vector sub with borrow-in/borrow-out (GFX9: v_subb_u32; GFX10+:
  // v_sub_co_ci_u32). See `readCarryInI1` / `writeCarryOutI1` above for
  // the SGPR-vs-VCC routing on both endpoints; the e64 form can bind
  // either (or both!) of ssrc2 and sdst to an arbitrary `sN`.
  if (sop == SemOp::V_SUB_CO_CI_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *bin = ctx.B.CreateZExt(readCarryInI1(ctx, di, op, /*srcIndex=*/2),
                                   ctx.i32Ty);
    Value *diff1 = ctx.B.CreateSub(s0, s1);
    Value *diff2 = ctx.B.CreateSub(diff1, bin, "vsub_ci");
    Value *b1 = ctx.B.CreateICmpULT(s0, s1);
    Value *b2 = ctx.B.CreateICmpULT(diff1, bin);
    ctx.writeReg32(op.dst(), diff2);
    writeCarryOutI1(ctx, di, op, ctx.B.CreateOr(b1, b2));
    hr.handled = true;
    return hr;
  }
  // Vector reversed sub with borrow-in/borrow-out (v_subbrev_co_u32).
  if (sop == SemOp::V_SUBREV_CO_CI_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *bin = ctx.B.CreateZExt(readCarryInI1(ctx, di, op, /*srcIndex=*/2),
                                   ctx.i32Ty);
    Value *diff1 = ctx.B.CreateSub(s1, s0);
    Value *diff2 = ctx.B.CreateSub(diff1, bin, "vsubrev_ci");
    Value *b1 = ctx.B.CreateICmpULT(s1, s0);
    Value *b2 = ctx.B.CreateICmpULT(diff1, bin);
    ctx.writeReg32(op.dst(), diff2);
    writeCarryOutI1(ctx, di, op, ctx.B.CreateOr(b1, b2));
    hr.handled = true;
    return hr;
  }
  // Vector add with carry-in/carry-out (GFX12: v_add_co_ci_u32).
  if (sop == SemOp::V_ADD_CO_CI_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *cin = ctx.B.CreateZExt(readCarryInI1(ctx, di, op, /*srcIndex=*/2),
                                   ctx.i32Ty);
    Function *uaddOv = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::uadd_with_overflow, {ctx.i32Ty});
    Value *step1 = ctx.B.CreateCall(uaddOv, {s0, s1});
    Value *sum1 = ctx.B.CreateExtractValue(step1, 0);
    Value *c1   = ctx.B.CreateExtractValue(step1, 1);
    Value *step2 = ctx.B.CreateCall(uaddOv, {sum1, cin});
    Value *res   = ctx.B.CreateExtractValue(step2, 0, "vadd_ci");
    Value *c2    = ctx.B.CreateExtractValue(step2, 1);
    ctx.writeReg32(op.dst(), res);
    writeCarryOutI1(ctx, di, op, ctx.B.CreateOr(c1, c2));
    hr.handled = true;
    return hr;
  }
  // v_mad_co_u64_u32: D.u64 = S0.u32 * S1.u32 + S2.u64, VCC = carry
  if (sop == SemOp::V_MAD_CO_U64_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src64(2), "vmad_co64");
    ctx.writeReg64(op.dst(0), res);
    hr.handled = true;
    return hr;
  }
  // v_mad_nc_{u64_u32,i64_i32} (gfx1250, VOP3Only_Realtriple_gfx1250
  // @ 0x2fa / 0x2fb):
  //   U form: D.u64 = zext(S0.u32) * zext(S1.u32) + S2.u64
  //   I form: D.i64 = sext(S0.i32) * sext(S1.i32) + S2.i64
  // No carry output (hence the "nc" suffix), single dst.  Shared
  // clamp / widening dispatch below.
  //
  // The backend's `SelectMad64_32()` pattern matcher
  // (AMDGPUISelDAGToDAG.cpp:1220) matches the canonical
  // `add(mul(widen s0, widen s1), s2_i64)` and re-emits
  // V_MAD_NC_{U,I}64_{U,I}32 on gfx1250 targets or the legacy
  // V_MAD_{CO_,}U64_U32 / V_MAD_I64_I32 (with VCC allocated to a
  // scratch SGPR and discarded) on gfx942 — so the same IR here
  // is correct for both same-target and cross-target lift paths.
  //
  // Clamp handling: the `VOP_I32_I32_I64_DPP` profile sets
  // `HasClamp = 1` (VOP3Instructions.td:196 profile body), so the
  // hardware instruction CAN saturate the 64-bit sum (to
  // INT64_{MIN,MAX} for the signed form, to UINT64_MAX for the
  // unsigned form) when the encoding's clamp bit is set.  The
  // corpus producers we've seen so far (`downcast_to_mxfp_*`,
  // which use the signed MAD as part of pointer/offset widening
  // arithmetic) all emit `clamp = 0`, relying on natural
  // wraparound.  If a future corpus kernel surfaces with
  // `clamp = 1`, the principled fix is to extend this handler to
  // emit `llvm.{s,u}add.sat.i64` for the final accumulator add
  // (the widening product is exact — `i32 * i32` fits in `i64`
  // without overflow — so saturation reduces to the sum step
  // only).  Until such a producer exists, refuse loudly rather
  // than silently emit a wraparound that the source kernel's
  // `clamp = 1` intent would not tolerate.  The refusal mirrors
  // the `V_ADD_I32` / `V_SUB_I32` GFX9 handler up-file which
  // already toggles between plain-add and `sadd_sat.i32` /
  // `ssub_sat.i32` on the clamp bit — treating the clamp bit as
  // raise-time-authoritative, not "observably ignorable".
  if (sop == SemOp::V_MAD_NC_U64_U32 || sop == SemOp::V_MAD_NC_I64_I32) {
    const bool isSigned = (sop == SemOp::V_MAD_NC_I64_I32);
    const int clampIdx = AMDGPU::getNamedOperandIdx(
        di.inst.getOpcode(), AMDGPU::OpName::clamp);
    const bool clamped = clampIdx >= 0 && di.isImm(clampIdx) &&
                         di.getImm(clampIdx) != 0;
    if (clamped) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          isSigned
              ? "v_mad_nc_i64_i32 with clamp=1 (signed 64-bit saturating MAD) "
                "is not yet lifted: no corpus producer exercises this encoding, "
                "and emitting the plain `add i64` form would silently drop the "
                "saturation semantics the source kernel's clamp bit requests.  "
                "Principled upgrade path when a producer surfaces: wrap the "
                "accumulator add in `llvm.sadd.sat.i64` (the widening product "
                "is exact in i64 so saturation reduces to the sum step only). "
                "See the block comment above this refusal for the full audit."
              : "v_mad_nc_u64_u32 with clamp=1 (unsigned 64-bit saturating "
                "MAD) is not yet lifted: same rationale as the signed sibling "
                "above — no corpus producer, and the upgrade path is "
                "`llvm.uadd.sat.i64`.  See the V_MAD_NC_* block comment.");
      return hr;
    }
    Value *a = isSigned
                   ? ctx.B.CreateSExt(op.src(0), ctx.i64Ty)
                   : ctx.B.CreateZExt(op.src(0), ctx.i64Ty);
    Value *b = isSigned
                   ? ctx.B.CreateSExt(op.src(1), ctx.i64Ty)
                   : ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src64(2),
                                 isSigned ? "vmad_nc_i64" : "vmad_nc_u64");
    ctx.writeReg64(op.dst(), res);
    hr.handled = true;
    return hr;
  }
  // v_mad_u32: D.u32 = S0.u32 * S1.u32 + S2.u32 (no carry)
  if (sop == SemOp::V_MAD_U32) {
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(op.src(0), op.src(1)), op.src(2), "vmad_u32");
    ctx.writeReg32(op.dst(), res);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_OR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(op.src(0), op.src(1), "vor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_AND_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAnd(op.src(0), op.src(1), "vand"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_LO_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateMul(op.src(0), op.src(1), "vmul"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUB_NC_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateSub(op.src(0), op.src(1), "vsub"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUBREV_NC_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateSub(op.src(1), op.src(0), "vsubrev"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_XOR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateXor(op.src(0), op.src(1), "vxor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_XNOR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateNot(ctx.B.CreateXor(op.src(0), op.src(1)), "vxnor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpUGT(s0, s1), s0, s1, "vmax"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpULT(s0, s1), s0, s1, "vmin"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpSGT(s0, s1), s0, s1, "vmax"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    ctx.writeReg32(op.dst(), ctx.B.CreateSelect(ctx.B.CreateICmpSLT(s0, s1), s0, s1, "vmin"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateLShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "vmulhi"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_I32) {
    Value *a = ctx.B.CreateSExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateSExt(op.src(1), ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateAShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "vmulhi"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_U32_U24) {
    Value *a = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    Value *b = ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    ctx.writeReg32(op.dst(), ctx.B.CreateMul(a, b, "mul24"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_I32_I24) {
    Value *a = ctx.B.CreateShl(op.src(0), 8);
    a = ctx.B.CreateAShr(a, 8);
    Value *b = ctx.B.CreateShl(op.src(1), 8);
    b = ctx.B.CreateAShr(b, 8);
    ctx.writeReg32(op.dst(), ctx.B.CreateMul(a, b, "mul24i"));
    hr.handled = true;
    return hr;
  }
  // v_dot8c_i32_i4: dst += sum of 8 signed 4-bit lane products
  if (sop == SemOp::V_DOT8C_I32_I4) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Value *acc = ctx.regs.readReg32(ctx.B, op.dst());
    for (int i = 0; i < 8; i++) {
      Value *shift = ConstantInt::get(ctx.i32Ty, i * 4);
      Value *a = ctx.B.CreateAnd(ctx.B.CreateLShr(s0, shift), ConstantInt::get(ctx.i32Ty, 0xF));
      Value *b = ctx.B.CreateAnd(ctx.B.CreateLShr(s1, shift), ConstantInt::get(ctx.i32Ty, 0xF));
      // Sign-extend 4-bit: shift left 28, arithmetic shift right 28
      a = ctx.B.CreateAShr(ctx.B.CreateShl(a, 28), 28);
      b = ctx.B.CreateAShr(ctx.B.CreateShl(b, 28), 28);
      acc = ctx.B.CreateAdd(acc, ctx.B.CreateMul(a, b));
    }
    ctx.writeReg32(op.dst(), acc);
    hr.handled = true;
    return hr;
  }
  // v_dot4c_i32_i8: dst += sum of 4 signed 8-bit lane products
  if (sop == SemOp::V_DOT4C_I32_I8) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Type *i8Ty = Type::getInt8Ty(ctx.C);
    Value *acc = ctx.regs.readReg32(ctx.B, op.dst());
    for (int i = 0; i < 4; i++) {
      Value *shift = ConstantInt::get(ctx.i32Ty, i * 8);
      Value *a = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s0, shift), i8Ty), ctx.i32Ty);
      Value *b = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s1, shift), i8Ty), ctx.i32Ty);
      acc = ctx.B.CreateAdd(acc, ctx.B.CreateMul(a, b));
    }
    ctx.writeReg32(op.dst(), acc);
    hr.handled = true;
    return hr;
  }
  // v_dot2c_i32_i16: dst += src0.lo16 * src1.lo16 + src0.hi16 * src1.hi16
  if (sop == SemOp::V_DOT2C_I32_I16) {
    Value *s0 = op.src(0), *s1 = op.src(1);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *a_lo = ctx.B.CreateSExt(ctx.B.CreateTrunc(s0, i16Ty), ctx.i32Ty);
    Value *a_hi = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s0, 16), i16Ty), ctx.i32Ty);
    Value *b_lo = ctx.B.CreateSExt(ctx.B.CreateTrunc(s1, i16Ty), ctx.i32Ty);
    Value *b_hi = ctx.B.CreateSExt(ctx.B.CreateTrunc(ctx.B.CreateLShr(s1, 16), i16Ty), ctx.i32Ty);
    Value *dot = ctx.B.CreateAdd(ctx.B.CreateMul(a_lo, b_lo), ctx.B.CreateMul(a_hi, b_hi));
    Value *acc = ctx.regs.readReg32(ctx.B, op.dst());
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(acc, dot, "dot2c"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_U32_U24) {
    Value *a = ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF)), ctx.i64Ty);
    Value *b = ctx.B.CreateZExt(ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF)), ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateLShr(ctx.B.CreateMul(a, b), 32), ctx.i32Ty, "mulhi24"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_HI_I32_I24) {
    Value *a = ctx.B.CreateAShr(ctx.B.CreateShl(op.src(0), 8), 8);
    Value *b = ctx.B.CreateAShr(ctx.B.CreateShl(op.src(1), 8), 8);
    Value *a64 = ctx.B.CreateSExt(a, ctx.i64Ty), *b64 = ctx.B.CreateSExt(b, ctx.i64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateTrunc(ctx.B.CreateAShr(ctx.B.CreateMul(a64, b64), 32), ctx.i32Ty, "mulhi24i"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAD_U32_U24) {
    Value *a = ctx.B.CreateAnd(op.src(0), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    Value *b = ctx.B.CreateAnd(op.src(1), ConstantInt::get(ctx.i32Ty, 0xFFFFFF));
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src(2), "mad24"));
    hr.handled = true;
    return hr;
  }
  // v_writelane_b32 / v_readlane_b32 are handled in handle_valu_cross_lane.cpp.

  // v_bfe_u32: Bit Field Extract Unsigned
  // D.u = (S0.u >> S1.u[4:0]) & ((1 << S2.u[4:0]) - 1)
  if (sop == SemOp::V_BFE_U32) {
    Value *base = op.src(0), *offset = op.src(1), *width = op.src(2);
    offset = ctx.B.CreateAnd(offset, ConstantInt::get(ctx.i32Ty, 31));
    width = ctx.B.CreateAnd(width, ConstantInt::get(ctx.i32Ty, 31));
    Value *shifted = ctx.B.CreateLShr(base, offset);
    // width is 0-31 after masking, so shl i32 1, width is always valid
    Value *mask = ctx.B.CreateSub(ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), width),
                              ConstantInt::get(ctx.i32Ty, 1));
    Value *isZeroWidth = ctx.B.CreateICmpEQ(width, ConstantInt::get(ctx.i32Ty, 0));
    Value *result = ctx.B.CreateAnd(shifted, mask, "bfe");
    result = ctx.B.CreateSelect(isZeroWidth, ConstantInt::get(ctx.i32Ty, 0), result);
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }
  // v_bfe_i32: signed Bit Field Extract.
  //   D.i = signext(bits [off+width-1 : off] of src), treating src as if
  //         it had been sign-extended past bit 31 first.
  // Implementation: arithmetic right shift by off (fills high bits with
  // src's sign), then mask to `width` low bits and sign-extend from bit
  // (width-1).  Using LShr instead of AShr here would give mask-and-sx
  // only when off+width <= 32; hardware diverges in the wraparound case,
  // so we must use AShr to stay bit-identical to native v_bfe_i32.
  //
  // Note: this is NOT the same formula as s_bfe_i32 — the scalar form
  // uses a shift-trick (`(src << (32-off-w)) >> (32-w)`), the vector
  // form uses mask-and-sign-extend.  The two hardware blocks differ on
  // the wraparound case; do not "unify" them.
  if (sop == SemOp::V_BFE_I32) {
    Value *base = op.src(0), *offset = op.src(1), *width = op.src(2);
    Value *c31 = ConstantInt::get(ctx.i32Ty, 0x1F);
    offset = ctx.B.CreateAnd(offset, c31);
    width = ctx.B.CreateAnd(width, c31);
    Value *shifted = ctx.B.CreateAShr(base, offset);
    // Build a mask of `width` low bits.  For width == 0 the result is 0
    // (nothing to extract), so we special-case that below and use a
    // safe shift amount (1) here to avoid UB in the mask computation.
    Value *widthNonZero = ctx.B.CreateICmpNE(width,
                                             ConstantInt::get(ctx.i32Ty, 0));
    Value *maskShift = ctx.B.CreateSelect(widthNonZero, width,
                                          ConstantInt::get(ctx.i32Ty, 1));
    Value *mask = ctx.B.CreateSub(
        ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), maskShift),
        ConstantInt::get(ctx.i32Ty, 1));
    Value *field = ctx.B.CreateAnd(shifted, mask);
    Value *widthMinus1 = ctx.B.CreateSub(maskShift,
                                         ConstantInt::get(ctx.i32Ty, 1));
    Value *signBit = ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1),
                                     widthMinus1);
    Value *sx = ctx.B.CreateSub(ctx.B.CreateXor(field, signBit), signBit,
                                "vbfe_i");
    Value *result = ctx.B.CreateSelect(widthNonZero, sx,
                                       ConstantInt::get(ctx.i32Ty, 0));
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }
  // v_bfi_b32: VOP3 bit-field insert.
  //   D.u = (S0.u & S1.u) | (~S0.u & S2.u)
  // S0 is the mask: bits set in S0 take the corresponding bit from
  // S1, bits clear in S0 take the corresponding bit from S2.
  // VOP3Instructions.td emits `AMDGPUbfiPattern` which is exactly
  // the mask-and-merge formula; no hardware masking of any operand,
  // no modifiers beyond the standard B32 set. gfx942 has the same
  // opcode, so the AMDGPU backend will isel this pair of and/or
  // back to a single v_bfi_b32 on the way down.
  if (sop == SemOp::V_BFI_B32) {
    Value *mask = op.src(0), *one = op.src(1), *zero = op.src(2);
    Value *picked_one  = ctx.B.CreateAnd(mask, one);
    Value *picked_zero = ctx.B.CreateAnd(ctx.B.CreateNot(mask), zero);
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateOr(picked_one, picked_zero, "vbfi"));
    hr.handled = true;
    return hr;
  }
  // v_mbcnt_{lo,hi}_u32_b32 are handled in handle_valu_cross_lane.cpp.
  // ---- 64-bit float ops ----
  if (sop == SemOp::V_ADD_F64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "vadd_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_F64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "vmul_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  // v_rcp_f64: VOP1 transcendental, single F64 source → F64 result.
  // The hardware op is a ~26-bit accurate reciprocal approximation
  // (TRANS-class, WriteTrans64). Lift to `llvm.amdgcn.rcp.f64` so
  // the AMDGPU backend isels straight back to v_rcp_f64 on gfx942
  // (no Newton-Raphson refinement is added). A generic `fdiv 1.0,
  // x` would lower to a software divide sequence here unless `arcp`
  // / fast-math flags are present, which would be a silent
  // semantics change versus the source op. See the V_RCP_F64
  // SemOp comment in semop.hpp for the rationale.
  if (sop == SemOp::V_RCP_F64) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *s = ctx.B.CreateBitCast(op.src64(0), f64Ty);
    Function *rcp = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_rcp, {f64Ty});
    Value *r = ctx.B.CreateCall(rcp, {s}, "vrcp_f64");
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(r, ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FMA_F64 || sop == SemOp::V_FMAC_F64) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *s0, *s1, *s2;
    if (sop == SemOp::V_FMA_F64) {
      s0 = op.src64(0); s1 = op.src64(1); s2 = op.src64(2);
    } else {
      s0 = op.src64(0); s1 = op.src64(1);
      s2 = ctx.B.CreateBitCast(ctx.regs.readReg64(ctx.B, op.dst()), f64Ty);
    }
    s0 = ctx.B.CreateBitCast(s0, f64Ty); s1 = ctx.B.CreateBitCast(s1, f64Ty);
    if (s2->getType() != f64Ty) s2 = ctx.B.CreateBitCast(s2, f64Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {f64Ty});
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "vfma_f64"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F64_U32) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateUIToFP(op.src(0), f64Ty, "cvt_f64_u32"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_F64_I32) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateSIToFP(op.src(0), f64Ty, "cvt_f64_i32"), ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_U32_F64) {
    auto *f64Ty = Type::getDoubleTy(ctx.C);
    Value *v = ctx.B.CreateBitCast(op.src64(0), f64Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateFPToUI(v, ctx.i32Ty, "cvt_u32_f64"));
    hr.handled = true;
    return hr;
  }

  // ---- Reversed-operand shifts ----
  if (sop == SemOp::V_LSHRREV_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateLShr(op.src(1), op.src(0), "vlshr"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHLREV_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateShl(op.src(1), op.src(0), "vlshl"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ASHRREV_I32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAShr(op.src(1), op.src(0), "vashr"));
    hr.handled = true;
    return hr;
  }

  // ---- FP ALU (srcF applies VOP3 neg/abs modifiers) ----
  if (sop == SemOp::V_ADD_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "fadd"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MUL_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "fmul"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUB_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFSub(s0, s1, "fsub"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_SUBREV_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateFSub(s1, s0, "fsubrev"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX_F32 || sop == SemOp::V_MAX_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "fmax"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN_F32 || sop == SemOp::V_MIN_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "fmin"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAXIMUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maximum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "fmaximum"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MINIMUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minimum, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "fminimum"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FMA_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "fma"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_FMAC_F32) {
    ParsedReg dstReg = op.dst();
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *dv = ctx.regs.readReg32(ctx.B, dstReg);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (dv->getType() != ctx.f32Ty) dv = ctx.B.CreateBitCast(dv, ctx.f32Ty);
    Function *fmuladd = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fmuladd, {ctx.f32Ty});
    ctx.writeReg32(dstReg, ctx.B.CreateBitCast(ctx.B.CreateCall(fmuladd, {s0, s1, dv}, "fmac"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_fmamk_f32 dst, src0, K, src2: dst = src0 * K + src2
  // Operand order from MC disassembler: srcF(0)=src0, srcF(1)=K (literal),
  // srcF(2)=src2. Same ordering applies to v_fmaak_f32 below.
  if (sop == SemOp::V_FMAMK_F32) {
    Value *s0 = op.srcF(0), *k = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (k->getType() != ctx.f32Ty) k = ctx.B.CreateBitCast(k, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, k, s2}, "fmamk"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_fmaak_f32 dst, src0, src1, K: dst = src0 * src1 + K
  if (sop == SemOp::V_FMAAK_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *k = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (k->getType() != ctx.f32Ty) k = ctx.B.CreateBitCast(k, ctx.f32Ty);
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, k}, "fmaak"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_madmk_f16 dst, src0, K, src2: dst = src0 * K + src2
  // v_madak_f16 dst, src0, src1, K: dst = src0 * src1 + K
  // F16 mirror of V_FMAMK_F32 / V_FMAAK_F32. Same operand ordering
  // convention: srcF(0..2) follow the disassembler's order, and the
  // 16-bit literal K lives in the slot named in the mnemonic. Both
  // lower to llvm.fma.f16 (no rounding of the intermediate product),
  // matching VOP2Instructions.td:1206-1210.
  if (sop == SemOp::V_MADMK_F16 || sop == SemOp::V_MADAK_F16) {
    Type *f16Ty = Type::getHalfTy(ctx.C);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    auto toF16 = [&](Value *v) -> Value * {
      Value *truncated = ctx.B.CreateTrunc(v, i16Ty);
      return ctx.B.CreateBitCast(truncated, f16Ty);
    };
    Value *s0 = toF16(op.srcF(0));
    Value *s1 = toF16(op.srcF(1));
    Value *s2 = toF16(op.srcF(2));
    Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma,
                                                     {f16Ty});
    Value *res = ctx.B.CreateCall(
        fma, {s0, s1, s2},
        sop == SemOp::V_MADMK_F16 ? "madmk_f16" : "madak_f16");
    Value *bits = ctx.B.CreateBitCast(res, i16Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateZExt(bits, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- Division helpers (VOP3) ----
  if (sop == SemOp::V_DIV_SCALE_F32) {
    // `v_div_scale_f32 dst, vcc, src0, src1, src2` scales one operand
    // of a numerator/denominator pair for a subsequent IEEE-conformant
    // divide (rcp + Newton + div_fixup).  The hardware encodes the
    // divide via operand-identity equality in the (src0, src1, src2)
    // triple — src2 duplicates either src0 or src1 to name which
    // operand is being scaled:
    //
    //   (n, d, n)  — src0 == src2, both carry the numerator       → scale numerator.
    //   (d, d, n)  — src0 == src1, both carry the denominator     → scale denominator.
    //
    // The LLVM intrinsic `@llvm.amdgcn.div.scale.f32(numer, denom, flag)`
    // takes canonical (numer, denom) and an i1 flag whose convention is
    // documented in `include/llvm/IR/IntrinsicsAMDGPU.td`:
    //   `0 = Denominator, 1 = Numerator`
    // — the flag selects which of (numer, denom) is the scaling target,
    // and the corresponding bit-pattern is what the hardware backend
    // re-emits as src2 of the lowered instruction.
    //
    // Identity is at the operand level (same register slot, or same
    // literal bit pattern), not at the runtime-value level — Triton's
    // AMDGPU codegen emits the literal-numerator variant of `1.0 / x`
    // as `(x, x, 1.0) + (1.0, x, 1.0)`, two scale calls whose
    // numerator is a `1.0` inline constant in src2.  Before this
    // handler knew about literal equality, both scale calls fell
    // through the register-only `isSrcReg(2) && isSrcReg(0)` check,
    // decoded as `selectNumerator = false`, and the backend's
    // re-lowering chain collapsed every `1.0 / sqrt(...)` in the
    // translated HSACO to a constant 1.0 via div_fixup's undefined-
    // scale-flag special-case (observable as layer-norm's rstd
    // deterministically reading `0x3f800000` regardless of input).
    // Detecting the literal-matching shape closes that gap without
    // changing the flag convention for the all-register case.
    //
    // Canonical `(numer, denom)` extraction from the hardware triple.
    // The denom always lives in src1.  The numer lives wherever the
    // matched shape identifies it: src0 in scale-numer (where
    // src0 == src2), src2 in scale-denom (where src0 == src1 and
    // src2 is the lone numer-bearing slot).  We pull the numer from
    // the slot that lexically exists in the matched shape — src0
    // for scale-numer to keep IR identity with the pre-audit all-
    // register handler, src2 for scale-denom to route the numer
    // correctly rather than silently duplicating the denom through
    // s0/s1 (the pre-audit shape).  Modifier symmetry on the
    // matched-identity pair is asserted below before either pick
    // becomes observable: asymmetric modifiers on the duplicated
    // slot are an emitter-ambiguity shape the lifted IR cannot
    // represent faithfully, and we refuse rather than guess.
    auto sameOperand = [&](unsigned a, unsigned b) -> bool {
      bool aIsReg = op.isSrcReg(a), bIsReg = op.isSrcReg(b);
      if (aIsReg != bIsReg) return false;
      if (aIsReg) {
        ParsedReg ra = op.srcReg(a), rb = op.srcReg(b);
        return ra.kind == rb.kind && ra.baseIdx == rb.baseIdx;
      }
      // Both are non-register operands.  Only compare when both
      // are plain immediates — other non-register kinds (special
      // encodings that parseReg would map to VCC / EXEC / SRC_*)
      // are not carried through the OpResolver as literals today and
      // the `isSrcReg` check above would have returned true for
      // them, so reaching here guarantees the isImm check is safe.
      //
      // Literal identity is compared via `MCOperand::getImm`, which
      // returns AMDGPU's raw encoded bit pattern — inline constants
      // come through their special-index encoding (246 for `1.0`,
      // etc.) and 32-bit literals come through their IEEE bit
      // pattern.  Two representations of the same value (e.g. `1.0`
      // as inline-const vs as a 32-bit literal `0x3f800000`) would
      // compare as UNEQUAL at this layer.  That is a pre-condition
      // refuse, not a silent miscompile — the handler falls through
      // to the three-arm match's `else` arm below and surfaces a
      // diagnostic.  In practice every corpus emitter (Triton's
      // AMDGPU backend, hipcc, libdevice) uses the canonical
      // inline-const encoding for the `1.0 / x` fdiv expansion, so
      // this representation assumption doesn't trip anywhere today;
      // tightening to semantic-value equality is the follow-up if a
      // future emitter surfaces the long-literal form.
      unsigned ai = op.srcIdx(a), bi = op.srcIdx(b);
      if (!op.di.isImm(ai) || !op.di.isImm(bi)) return false;
      return op.di.getImm(ai) == op.di.getImm(bi);
    };

    bool src0EqSrc2 = sameOperand(0, 2);
    bool src0EqSrc1 = sameOperand(0, 1);
    bool scaleNumerator;
    if (src0EqSrc2 && !src0EqSrc1) {
      scaleNumerator = true;    // (n, d, n)
    } else if (src0EqSrc1 && !src0EqSrc2) {
      scaleNumerator = false;   // (d, d, n)
    } else {
      // All three sources matching is the degenerate `x/x` shape
      // (ambiguous between scale-numer and scale-denom); src2 not
      // matching either of src0/src1 would break the hardware's own
      // divide-protocol and is unreachable from any known codegen
      // emitter.  Refuse loudly rather than guess — consistent with
      // the "refuse when uncertain" rule in
      // hotswap/docs/wave-size-translation.md.
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          "v_div_scale_f32 operand triple does not match a known "
          "divide-scaling shape: expected (numer, denom, numer) with "
          "src0 == src2 for scale-numerator, or (denom, denom, numer) "
          "with src0 == src1 for scale-denominator.  See handle_valu.cpp "
          "for the decode rule.");
      return hr;
    }

    // FP-modifier symmetry check on the matched-identity pair.  The
    // hardware's operand-identity protocol makes `src0 == src<M>`
    // (for M = 1 in scale-denom, M = 2 in scale-numer) tell the
    // scale unit "these two slots carry the same operand value."
    // But VOP3 modifiers (abs/neg bits in the `modMap` entry per
    // source index) can be set independently on each slot, which
    // would make the two slots semantically different operands (one
    // `v`, the other `-v` or `abs(v)`).  The hardware's behaviour
    // in that case is undocumented / effectively undefined for the
    // divide protocol — no known codegen emitter (Triton AMD
    // backend, hipcc, libdevice) produces asymmetric modifiers on
    // the duplicated slot — and the lifted IR would silently drop
    // one modifier set because we can only thread a single
    // `(numer, denom)` pair through `@llvm.amdgcn.div.scale.f32`.
    // Refuse loudly rather than guess.
    unsigned peer = scaleNumerator ? 2u : 1u;
    if (op.srcMod(0) != op.srcMod(peer)) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          scaleNumerator
              ? "v_div_scale_f32 scale-numerator shape (src0 == src2) "
                "has asymmetric FP modifiers on src0 and src2; the "
                "hardware's operand-identity protocol treats both as "
                "the same numerator operand, but the lifted IR can "
                "only carry one modifier set.  No known codegen "
                "emitter produces this shape; refusing rather than "
                "dropping a modifier silently."
              : "v_div_scale_f32 scale-denominator shape (src0 == src1) "
                "has asymmetric FP modifiers on src0 and src1; the "
                "hardware's operand-identity protocol treats both as "
                "the same denominator operand, but the lifted IR can "
                "only carry one modifier set.  No known codegen "
                "emitter produces this shape; refusing rather than "
                "dropping a modifier silently.");
      return hr;
    }

    // Canonical (numer, denom) sourced from the operand slot that
    // holds each value in the matched shape.  Modifier symmetry was
    // just asserted above, so for scale-numer picking src0 or src2
    // is equivalent — we take src0 to keep the all-register scale-
    // numer case IR-identical to the pre-fix handler (which also
    // used srcF(0) for numer); for scale-denom, src2 is the only
    // slot that carries the numer at all, so there is no choice.
    // The denom always lives in src1 (src0 aliases it in the
    // scale-denom shape, and src1 is the natural anchor in both).
    Value *numer = ctx.B.CreateBitCast(
        scaleNumerator ? op.srcF(0) : op.srcF(2), ctx.f32Ty);
    Value *denom = ctx.B.CreateBitCast(op.srcF(1), ctx.f32Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_div_scale,
                                                     {ctx.f32Ty});
    Value *r = ctx.B.CreateCall(fn, {numer, denom,
                 scaleNumerator ? ctx.B.getTrue() : ctx.B.getFalse()}, "divscale");
    ctx.writeReg32(op.dst(0), ctx.B.CreateBitCast(ctx.B.CreateExtractValue(r, 0), ctx.i32Ty));
    // Write the boolean flag to the actual SDST destination (operand 1):
    // vcc_lo, sN, or null. The kernel saves flags to SGPRs and later
    // restores them to VCC via s_mov_b32 before each v_div_fmas_f32.
    Value *flag = ctx.B.CreateExtractValue(r, 1);
    if (di.numDefs >= 2 && di.isReg(1)) {
      ParsedReg flagDst = op.dst(1);
      if (flagDst.kind == ParsedReg::VCC)
        ctx.regs.storeVCC(ctx.B, flag);
      else if (flagDst.kind == ParsedReg::SGPR && flagDst.baseIdx >= 0)
        ctx.regs.storeSGPR32(ctx.B, flagDst.baseIdx, ctx.B.CreateZExt(flag, ctx.i32Ty));
      // NOREG (null) or unrecognized → discard the flag
    } else {
      ctx.regs.storeVCC(ctx.B, flag);
    }
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_DIV_FIXUP_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.srcF(1), ctx.f32Ty);
    Value *s2 = ctx.B.CreateBitCast(op.srcF(2), ctx.f32Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_div_fixup,
                                                     {ctx.f32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {s0, s1, s2}, "divfixup"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_DIV_FMAS_F32) {
    Value *s0 = ctx.B.CreateBitCast(op.srcF(0), ctx.f32Ty);
    Value *s1 = ctx.B.CreateBitCast(op.srcF(1), ctx.f32Ty);
    Value *s2 = ctx.B.CreateBitCast(op.srcF(2), ctx.f32Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_div_fmas,
                                                     {ctx.f32Ty});
    Value *vcc = ctx.regs.loadVCC(ctx.B);
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(fn, {s0, s1, s2, vcc}, "divfmas"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- 3-source integer VOP3 ----
  if (sop == SemOp::V_ADD3_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(ctx.B.CreateAdd(op.src(0), op.src(1)), op.src(2), "vadd3"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_ADD_U32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateAdd(ctx.B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshl_add"));
    hr.handled = true;
    return hr;
  }
  // v_add_lshl_u32: fused three-input "add then shift".
  //   D.u = (S0.u + S1.u) << S2.u[4:0]
  // Unsigned wrap on the add is well-defined (CreateAdd defaults to
  // "may wrap"), matching hardware.  The shift amount must be masked to
  // 5 bits up front — AMDGPU shifts only consume S2[4:0], but LLVM's
  // `shl` with a shift >= bit-width is poison, so an un-masked `op.src(2)`
  // containing any high bits would silently corrupt the IR.  V_ADD_LSHL
  // has no carry-out and writes no SCC/VCC, so this is the whole op.
  if (sop == SemOp::V_ADD_LSHL_U32) {
    Value *sum = ctx.B.CreateAdd(op.src(0), op.src(1));
    Value *shamt = ctx.B.CreateAnd(op.src(2),
                                   ConstantInt::get(ctx.i32Ty, 0x1F));
    ctx.writeReg32(op.dst(), ctx.B.CreateShl(sum, shamt, "vadd_lshl"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_OR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(ctx.B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshlor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_AND_OR_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(ctx.B.CreateAnd(op.src(0), op.src(1)), op.src(2), "vandor"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_OR3_B32) {
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(ctx.B.CreateOr(op.src(0), op.src(1)), op.src(2), "vor3"));
    hr.handled = true;
    return hr;
  }
  // VOP3 v_xad_u32: dst = (src0 ^ src1) + src2. The .td iselect
  // pattern (VOP3Instructions.td:831) is
  // `ThreeOp_i32_Pats<xor, add, V_XAD_U32_e64>`. Same skeleton
  // as V_AND_OR_B32 / V_LSHL_OR_B32 above with xor+add in place
  // of the inner+outer ops.
  if (sop == SemOp::V_XAD_U32) {
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateAdd(ctx.B.CreateXor(op.src(0), op.src(1)),
                                   op.src(2), "vxad"));
    hr.handled = true;
    return hr;
  }
  // VOP3 v_alignbit_b32: funnel-shift right.
  //   dst = ((src0 << 32) | src1) >> (src2 & 0x1F))[31:0]
  // The .td uses the SDAG `fshr` node directly
  // (VOP3Instructions.td:222), so the lift maps 1:1 to
  // `llvm.fshr.i32`. The shift amount is masked to 5 bits in
  // hardware before dispatch — we mirror that explicit mask
  // here (although LLVM's fshr semantics already implement
  // modulo-bitwidth shifts, the explicit AND keeps the IR
  // shape pinnable and makes the bit-width assumption local).
  if (sop == SemOp::V_ALIGNBIT_B32) {
    Value *hi = op.src(0);
    Value *lo = op.src(1);
    Value *shamt = ctx.B.CreateAnd(op.src(2),
        ConstantInt::get(ctx.i32Ty, 0x1F), "valign_shamt");
    Function *fshr = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fshr, {ctx.i32Ty});
    ctx.writeReg32(op.dst(), ctx.B.CreateCall(fshr, {hi, lo, shamt}, "valignbit"));
    hr.handled = true;
    return hr;
  }
  // VOP3 v_xor3_b32: 3-way xor. Direct mirror of V_OR3_B32 above
  // — the .td iselect pattern is `(xor (xor a, b), c)` (see
  // VOP3Instructions.td:1350); both nested and outer xor lift to
  // plain CreateXor with no source modifiers (B32 ops carry only
  // ABS/NEG-style modifiers on the FP forms, not the bitwise
  // ones).
  if (sop == SemOp::V_XOR3_B32) {
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateXor(ctx.B.CreateXor(op.src(0), op.src(1)),
                                   op.src(2), "vxor3"));
    hr.handled = true;
    return hr;
  }
  // VOP3 v_add_nc_u16: 16-bit no-carry add with op_sel half
  // selection on src0/src1/dst. Defined at
  // VOP3Instructions.td:1362; gfx10/gfx11/gfx12 share the same
  // 0x303 opcode (lines :1852, :2016).
  //
  // op_sel layout in the disassembly is `op_sel:[s0,s1,dst]`
  // where each entry picks the lo (0) or hi (1) 16-bit half of
  // the corresponding 32-bit VGPR. The unselected half of the
  // destination register is preserved per the RDNA3+ ISA — that
  // is the *only* reason this handler reads the prior dst value
  // and merges, distinguishing it from the existing V_MAX_U16 /
  // V_MIN_U16 family which assume default op_sel and
  // zero-extend.
  if (sop == SemOp::V_ADD_NC_U16) {
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    int opSel[3] = {0, 0, 0};
    StringRef text(di.fullText);
    auto pos = text.find("op_sel:");
    if (pos != StringRef::npos) {
      auto brk = text.find('[', pos);
      auto end = text.find(']', brk);
      if (brk != StringRef::npos && end != StringRef::npos) {
        StringRef inner = text.slice(brk + 1, end);
        SmallVector<StringRef, 3> parts;
        inner.split(parts, ',');
        for (unsigned i = 0; i < parts.size() && i < 3; i++) {
          int val = 0;
          if (!parts[i].trim().getAsInteger(10, val))
            opSel[i] = val;
        }
      }
    }
    auto half = [&](Value *v, int sel) -> Value * {
      if (sel) v = ctx.B.CreateLShr(v, 16);
      return ctx.B.CreateTrunc(v, i16Ty);
    };
    Value *a = half(op.src(0), opSel[0]);
    Value *b = half(op.src(1), opSel[1]);
    Value *sum = ctx.B.CreateAdd(a, b, "vadd_nc_u16");
    Value *sumZ = ctx.B.CreateZExt(sum, ctx.i32Ty);
    if (opSel[2] == 0) {
      // Write low half, preserve high half. The default-opsel
      // case is the dominant one (matches the corpus instances
      // we've seen) and would lift identically to a plain
      // trunc+add+zext if the prior dst high bits were known to
      // be zero — the explicit OR with the masked old value
      // makes the merge semantics observable in the IR shape.
      Value *old = ctx.regs.readReg32(ctx.B, op.dst());
      Value *high = ctx.B.CreateAnd(old,
          ConstantInt::get(ctx.i32Ty, 0xFFFF0000u));
      ctx.writeReg32(op.dst(), ctx.B.CreateOr(high, sumZ, "vadd_u16_merge_lo"));
    } else {
      Value *old = ctx.regs.readReg32(ctx.B, op.dst());
      Value *low = ctx.B.CreateAnd(old,
          ConstantInt::get(ctx.i32Ty, 0x0000FFFFu));
      Value *shifted = ctx.B.CreateShl(sumZ, 16);
      ctx.writeReg32(op.dst(), ctx.B.CreateOr(low, shifted, "vadd_u16_merge_hi"));
    }
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_BITOP3_B32 || sop == SemOp::V_BITOP3_B16) {
    // v_bitop3 dst, src0, src1, src2, imm8
    // For each bit position i, dst[i] = LUT[4*src0[i] + 2*src1[i] + src2[i]]
    // Expand as: result = OR of (minterm_i AND expand(LUT[i])) for i in 0..7
    Value *a = op.src(0), *b = op.src(1), *c = op.src(2);
    Value *imm = op.src(3);
    uint64_t lutConst = 0;
    bool lutIsConst = false;
    if (auto *CI = dyn_cast<ConstantInt>(imm)) {
      lutConst = CI->getZExtValue() & 0xFF;
      lutIsConst = true;
    }
    Value *na = ctx.B.CreateNot(a), *nb = ctx.B.CreateNot(b), *nc = ctx.B.CreateNot(c);
    Value *allOnes = ConstantInt::get(ctx.i32Ty, ~0U);
    Value *result = ConstantInt::get(ctx.i32Ty, 0);
    Value *minterms[8] = {
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), nc),  // 0: ~a & ~b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), c),   // 1: ~a & ~b &  c
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), nc),   // 2: ~a &  b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), c),    // 3: ~a &  b &  c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), nc),   // 4:  a & ~b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), c),    // 5:  a & ~b &  c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), nc),    // 6:  a &  b & ~c
      ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), c),     // 7:  a &  b &  c
    };
    if (lutIsConst) {
      for (int i = 0; i < 8; i++)
        if (lutConst & (1 << i))
          result = ctx.B.CreateOr(result, minterms[i]);
    } else {
      for (int i = 0; i < 8; i++) {
        Value *bit = ctx.B.CreateAnd(ctx.B.CreateLShr(imm, ConstantInt::get(ctx.i32Ty, i)),
                                 ConstantInt::get(ctx.i32Ty, 1));
        Value *mask = ctx.B.CreateSub(ConstantInt::get(ctx.i32Ty, 0), bit);
        result = ctx.B.CreateOr(result, ctx.B.CreateAnd(minterms[i], mask));
      }
    }
    if (sop == SemOp::V_BITOP3_B16)
      result = ctx.B.CreateAnd(result, ConstantInt::get(ctx.i32Ty, 0xFFFF));
    ctx.writeReg32(op.dst(), result);
    hr.handled = true;
    return hr;
  }
  // VOP3 v_max3_u32: 3-way unsigned max, ternary. The .td pattern
  // is `AMDGPUumax3` = `umax(umax(a,b), c)`; we lift it as the
  // direct ICmp+Select chain (matches the V_MAX_U32 idiom one
  // block above so the intermediate pair `vmax3_lo` reuses the
  // same shape and a future refactor that switches V_MAX_U32 to
  // an llvm.umax intrinsic can lift this in lockstep).
  if (sop == SemOp::V_MAX3_U32) {
    Value *s0 = op.src(0), *s1 = op.src(1), *s2 = op.src(2);
    Value *m01 =
        ctx.B.CreateSelect(ctx.B.CreateICmpUGT(s0, s1), s0, s1, "vmax3_lo");
    ctx.writeReg32(
        op.dst(),
        ctx.B.CreateSelect(ctx.B.CreateICmpUGT(m01, s2), m01, s2, "vmax3"));
    hr.handled = true;
    return hr;
  }
  // VOP3 v_med3_i32: signed-integer median-of-three.
  // Hardware semantic (VOP3Instructions.td:1796 via AMDGPUsmed3
  // SDAG node) is the standard sort-and-pick-middle for three i32
  // values:
  //   med3_i32(a, b, c) = smax(smin(a, b), smin(smax(a, b), c))
  // We emit it as a pair of `llvm.smin.i32` + `llvm.smax.i32`
  // intrinsics — these are the canonical IR forms, and the AMDGPU
  // backend pattern-matches the exact `smax(smin(...),
  // smin(smax(...), ...))` shape back to V_MED3_I32 via
  // AMDGPUInstructions.td so the round-trip is structure-preserving
  // (no codegen quality loss). We deliberately do not depend on
  // `llvm.amdgcn.smed3` because: (a) the LLVM IR-level intrinsic
  // is already the most compact lowering for the same pattern;
  // (b) the smin/smax form composes with peephole IR optimisations
  // that smed3 does not (e.g. constant folding when one source is
  // a known bound).
  if (sop == SemOp::V_MED3_I32) {
    Value *s0 = op.src(0), *s1 = op.src(1), *s2 = op.src(2);
    Function *sminFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::smin, {ctx.i32Ty});
    Function *smaxFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::smax, {ctx.i32Ty});
    Value *lo = ctx.B.CreateCall(sminFn, {s0, s1}, "vmed3_lo");
    Value *hi = ctx.B.CreateCall(smaxFn, {s0, s1}, "vmed3_hi");
    Value *clamped = ctx.B.CreateCall(sminFn, {hi, s2}, "vmed3_clamp");
    ctx.writeReg32(
        op.dst(),
        ctx.B.CreateCall(smaxFn, {lo, clamped}, "vmed3"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MAX3_F32 || sop == SemOp::V_MAX3_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    Value *m01 = ctx.B.CreateCall(maxFn, {s0, s1});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {m01, s2}, "max3"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // VOP3 v_minmax_num_f32: dst = minnum(maxnum(s0, s1), s2).
  // gfx11 emitted this as v_minmax_f32; gfx12 renamed it to
  // v_minmax_num_f32 once the IEEE-2019 NaN-propagating
  // V_MINIMUMMAXIMUM_F32 (opcode 0x26c) needed an unambiguous
  // namesake. The .NUM suffix is the IEEE-754 2008 minNum
  // semantic — NaN-pruning, exactly what `llvm.maxnum` /
  // `llvm.minnum` model. Same shape as V_MAX3_F32 above with
  // minnum as the outer reduction.
  if (sop == SemOp::V_MINMAX_NUM_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    Function *minFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    Value *mx = ctx.B.CreateCall(maxFn, {s0, s1}, "vminmax_inner");
    Value *r = ctx.B.CreateCall(minFn, {mx, s2}, "vminmax_num");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(r, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MIN3_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    Value *m01 = ctx.B.CreateCall(minFn, {s0, s1});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {m01, s2}, "min3"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_MED3_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (s2->getType() != ctx.f32Ty) s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
    Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
    Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
    Value *mn01 = ctx.B.CreateCall(minFn, {s0, s1});
    Value *mx01 = ctx.B.CreateCall(maxFn, {s0, s1});
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {mn01, ctx.B.CreateCall(minFn, {mx01, s2})}, "med3"), ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_cvt_pkrtz_f16_f32: pack two f32 into <2 x f16> with round-to-zero.
  // Maps directly onto the dedicated hardware intrinsic so the backend
  // keeps the RTZ rounding mode (a plain FPTrunc uses round-to-nearest).
  if (sop == SemOp::V_CVT_PKRTZ_F16_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Function *fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_cvt_pkrtz);
    Value *v2h = ctx.B.CreateCall(fn, {s0, s1}, "pkrtz");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(v2h, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_cvt_pk_f16_f32: pack two f32 into <2 x f16> with round-to-nearest-even
  // (the default IEEE rounding). No dedicated intrinsic exists; a pair of
  // FPTrunc operations followed by a packed i32 assembly is the canonical
  // lowering and the backend recognises the pattern.
  if (sop == SemOp::V_CVT_PK_F16_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Type *halfTy = Type::getHalfTy(ctx.C);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *h0 = ctx.B.CreateFPTrunc(s0, halfTy, "pk_h0");
    Value *h1 = ctx.B.CreateFPTrunc(s1, halfTy, "pk_h1");
    Value *b0 = ctx.B.CreateZExt(ctx.B.CreateBitCast(h0, i16Ty), ctx.i32Ty);
    Value *b1 = ctx.B.CreateZExt(ctx.B.CreateBitCast(h1, i16Ty), ctx.i32Ty);
    ctx.writeReg32(op.dst(),
        ctx.B.CreateOr(b0, ctx.B.CreateShl(b1, 16), "pk_f16"));
    hr.handled = true;
    return hr;
  }
  // v_cvt_scalef32_pk_fp4_f32 vdst, src0_f32, src1_f32, scale_f32 op_sel:[..]
  //
  // Converts two f32 sources to FP4 and packs them into one of the four 8-bit
  // slots of vdst (selected by op_sel bits 0..3), using a scalar f32 scale.
  // The remaining slots of the old vdst value are preserved — this is
  // captured by the intrinsic's tied `old_vdst` argument.
  if (sop == SemOp::V_CVT_SCALEF32_PK_FP4_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1), *scale = op.srcF(2);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    if (scale->getType() != ctx.f32Ty)
      scale = ctx.B.CreateBitCast(scale, ctx.f32Ty);
    // Extract the destination nibble index from op_sel. LLVM disasm prints
    // op_sel:[0,0,0,0] with the 4th entry being the slot selector.
    int opSel[4] = {0, 0, 0, 0};
    StringRef text(di.fullText);
    auto pos = text.find("op_sel:");
    if (pos != StringRef::npos) {
      auto brk = text.find('[', pos);
      auto end = text.find(']', brk);
      if (brk != StringRef::npos && end != StringRef::npos) {
        StringRef inner = text.slice(brk + 1, end);
        SmallVector<StringRef, 4> parts;
        inner.split(parts, ',');
        for (unsigned i = 0; i < parts.size() && i < 4; i++) {
          int val = 0;
          if (!parts[i].trim().getAsInteger(10, val))
            opSel[i] = val;
        }
      }
    }
    // Dest-slot index is packed as bits[3:2]+bit[0] per the HW op_sel
    // layout (see LLVM's SIInstrInfo::lowerScaleCvt for reference); for the
    // common `op_sel:[0,0,0,0]` form the selector is simply 0.
    unsigned dstSel = static_cast<unsigned>(opSel[3]);
    Value *oldVdst = ctx.regs.readReg32(ctx.B, op.dst());
    Function *fn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::amdgcn_cvt_scalef32_pk_fp4_f32);
    Value *r = ctx.B.CreateCall(
        fn, {oldVdst, s0, s1, scale, ConstantInt::get(ctx.i32Ty, dstSel)},
        "scalef32_pk_fp4");
    ctx.writeReg32(op.dst(), r);
    hr.handled = true;
    return hr;
  }
  // v_mov_b64 vdst:64, src:64
  if (sop == SemOp::V_MOV_B64) {
    ctx.writeReg64(op.dst(), op.src64(0));
    hr.handled = true;
    return hr;
  }
  // v_swap_b32 vdstA, vdstB / uses vdstA, vdstB - exchange two VGPRs.
  // MC encoding has two defs (vdst, vdst_in) and two uses (src0, src0_in).
  // The old values of both registers swap: src0 -> vdst and old-vdst ->
  // vdst_in.
  if (sop == SemOp::V_SWAP_B32) {
    // vdst = old src0; vdst_in(== src0's slot) = old vdst.
    ParsedReg dstA = op.dst(0);
    ParsedReg dstB = (di.numDefs >= 2) ? op.dst(1) : op.srcReg(0);
    Value *vA = ctx.regs.readReg32(ctx.B, dstA);
    Value *vB = ctx.regs.readReg32(ctx.B, dstB);
    ctx.writeReg32(dstA, vB);
    ctx.writeReg32(dstB, vA);
    hr.handled = true;
    return hr;
  }
  // v_cvt_f32_bf16: low 16 bits of src are interpreted as bfloat16.
  if (sop == SemOp::V_CVT_F32_BF16) {
    Type *bfTy = Type::getBFloatTy(ctx.C);
    Type *i16Ty = Type::getInt16Ty(ctx.C);
    Value *bits = ctx.B.CreateTrunc(op.src(0), i16Ty);
    Value *bf = ctx.B.CreateBitCast(bits, bfTy);
    Value *f = ctx.B.CreateFPExt(bf, ctx.f32Ty, "cvt_bf16");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(f, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // v_bfm_b32: D = ((1 << src0[4:0]) - 1) << src1[4:0]
  if (sop == SemOp::V_BFM_B32) {
    Value *width  = ctx.B.CreateAnd(op.src(0),
        ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *offset = ctx.B.CreateAnd(op.src(1),
        ConstantInt::get(ctx.i32Ty, 0x1F));
    Value *ones   = ctx.B.CreateSub(
        ctx.B.CreateShl(ConstantInt::get(ctx.i32Ty, 1), width),
        ConstantInt::get(ctx.i32Ty, 1));
    // width==0 must yield 0 — the 1<<0 base case would otherwise leave a
    // single bit set. Mask it out explicitly rather than relying on the
    // subtraction underflow.
    Value *isZero = ctx.B.CreateICmpEQ(width, ConstantInt::get(ctx.i32Ty, 0));
    ones = ctx.B.CreateSelect(isZero, ConstantInt::get(ctx.i32Ty, 0), ones);
    ctx.writeReg32(op.dst(), ctx.B.CreateShl(ones, offset, "bfm"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_PK_BF16_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    auto *bfTy = Type::getBFloatTy(ctx.C);
    Value *bf0 = ctx.B.CreateFPTrunc(s0, bfTy, "tobf16_0");
    Value *bf1 = ctx.B.CreateFPTrunc(s1, bfTy, "tobf16_1");
    Value *bits0 = ctx.B.CreateZExt(ctx.B.CreateBitCast(bf0, Type::getInt16Ty(ctx.C)), ctx.i32Ty);
    Value *bits1 = ctx.B.CreateZExt(ctx.B.CreateBitCast(bf1, Type::getInt16Ty(ctx.C)), ctx.i32Ty);
    ctx.writeReg32(op.dst(), ctx.B.CreateOr(bits0, ctx.B.CreateShl(bits1, 16), "pk_bf16"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_PK_FP8_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    // v_cvt_pk_fp8_f32 packs two f32 into two fp8 values in the low 16 bits.
    // The "old" value and word_sel determine where in the dest the result goes.
    // src2 = old value, src3 (imm) = word_sel.
    // Use the LLVM intrinsic which handles this correctly.
    Value *oldVal = (op.nSrcs() >= 3) ? op.src(2) : ConstantInt::get(ctx.i32Ty, 0);
    bool wordSel = (op.nSrcs() >= 4 && di.isImm(op.srcIdx(3))) ? (op.srcImm(3) != 0) : false;
    Function *cvtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_cvt_pk_fp8_f32);
    ctx.writeReg32(op.dst(), ctx.B.CreateCall(cvtFn,
        {s0, s1, oldVal, ConstantInt::get(ctx.i1Ty, wordSel)}, "pk_fp8"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_CVT_PK_BF8_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);
    if (s0->getType() != ctx.f32Ty) s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty);
    if (s1->getType() != ctx.f32Ty) s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
    Value *oldVal = (op.nSrcs() >= 3) ? op.src(2) : ConstantInt::get(ctx.i32Ty, 0);
    bool wordSel = (op.nSrcs() >= 4 && di.isImm(op.srcIdx(3))) ? (op.srcImm(3) != 0) : false;
    Function *cvtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_cvt_pk_bf8_f32);
    ctx.writeReg32(op.dst(), ctx.B.CreateCall(cvtFn,
        {s0, s1, oldVal, ConstantInt::get(ctx.i1Ty, wordSel)}, "pk_bf8"));
    hr.handled = true;
    return hr;
  }
  // VOP1 read-side companions: v_cvt_pk_f32_{fp8,bf8} expand 16 bits
  // of the i32 src into a v2f32 written to the dst VGPR pair. The
  // word selector (which 16-bit half of src to decode) lives in
  // op_sel:[0] for the e64 / VOP3 form and is parsed from di.fullText
  // — we do not have a first-class modifier channel in OperandView.
  // The dst op_sel slot (op_sel:[1]) is irrelevant: the destination
  // is a v2f32 pair, not a half-register, so the assembler always
  // prints `0` there. We refuse loudly if op_sel parsing produces a
  // value outside {0,1} so corpus drift surfaces immediately rather
  // than silently flipping the word selector. Lowering selects the
  // matching `llvm.amdgcn.cvt.pk.f32.{fp8,bf8}` intrinsic and
  // bitcasts its v2f32 result to i64 before writeReg64.
  if (sop == SemOp::V_CVT_PK_F32_FP8 || sop == SemOp::V_CVT_PK_F32_BF8) {
    int wordSelInt = 0;
    StringRef text(di.fullText);
    auto pos = text.find("op_sel:");
    if (pos != StringRef::npos) {
      auto brk = text.find('[', pos);
      auto end = text.find(']', brk);
      if (brk != StringRef::npos && end != StringRef::npos) {
        StringRef inner = text.slice(brk + 1, end);
        SmallVector<StringRef, 4> parts;
        inner.split(parts, ',');
        if (!parts.empty()) {
          int parsed = 0;
          if (parts[0].trim().getAsInteger(10, parsed) ||
              (parsed != 0 && parsed != 1)) {
            hr.failure = RaiseFailure::unsupportedShape(
                di, "VOP3",
                "unparseable or out-of-range op_sel[0] (expected 0 or 1)");
            return hr;
          }
          wordSelInt = parsed;
        }
      }
    }
    Value *src = op.src(0);
    if (src->getType() != ctx.i32Ty)
      src = ctx.B.CreateBitOrPointerCast(src, ctx.i32Ty);
    Intrinsic::ID iid = (sop == SemOp::V_CVT_PK_F32_FP8)
                            ? Intrinsic::amdgcn_cvt_pk_f32_fp8
                            : Intrinsic::amdgcn_cvt_pk_f32_bf8;
    Function *cvtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, iid);
    Value *v2 = ctx.B.CreateCall(cvtFn,
        {src, ConstantInt::get(ctx.i1Ty, wordSelInt != 0)},
        sop == SemOp::V_CVT_PK_F32_FP8 ? "cvt_pk_f32_fp8"
                                       : "cvt_pk_f32_bf8");
    ctx.writeReg64(op.dst(), ctx.B.CreateBitCast(v2, ctx.i64Ty));
    hr.handled = true;
    return hr;
  }
  // VOP1 single-lane v_cvt_f32_{fp8,bf8}: decode one 8-bit lane of
  // src into f32. The corpus only ever emits the e64 form with no
  // op_sel (byte_sel=0) — the SDWA / op_sel-bearing encodings, which
  // would let LLVM's isel pick byte 1/2/3, are not present in any
  // gfx1250 kernel today. We refuse loudly if disassembly carries an
  // op_sel: marker so corpus drift surfaces instead of a silent
  // byte-0 collapse.
  if (sop == SemOp::V_CVT_F32_FP8 || sop == SemOp::V_CVT_F32_BF8) {
    StringRef text(di.fullText);
    if (text.contains("op_sel:") || text.contains("_sdwa")) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP1",
          "non-default op_sel/sdwa byte_sel on v_cvt_f32_{fp8,bf8} "
          "(only the byte_sel=0 e64 form is wired today)");
      return hr;
    }
    Value *src = op.src(0);
    if (src->getType() != ctx.i32Ty)
      src = ctx.B.CreateBitOrPointerCast(src, ctx.i32Ty);
    Intrinsic::ID iid = (sop == SemOp::V_CVT_F32_FP8)
                            ? Intrinsic::amdgcn_cvt_f32_fp8
                            : Intrinsic::amdgcn_cvt_f32_bf8;
    Function *cvtFn = Intrinsic::getOrInsertDeclaration(&ctx.M, iid);
    Value *f = ctx.B.CreateCall(cvtFn,
        {src, ConstantInt::get(ctx.i32Ty, 0)},
        sop == SemOp::V_CVT_F32_FP8 ? "cvt_f32_fp8" : "cvt_f32_bf8");
    ctx.writeReg32(op.dst(), ctx.B.CreateBitCast(f, ctx.i32Ty));
    hr.handled = true;
    return hr;
  }
  // VOP3 gfx1250-only scaled packed-8 FP4 -> BF16 convert.
  //
  // Hardware shape (AMDGPUGenInstrInfo.inc / VOP3Instructions.td:1873,
  // opcode V_CVT_SCALE_PK8_BF16_FP4_e64, VOP3 opcode 0x2a0):
  //     0: vdst        (VReg_128 aligned — 4 consecutive VGPRs,
  //                     written as <8 x bfloat> / 128 bits)
  //     1: src0        (VGPR_32 — 1 VGPR, packed 8xFP4 in the i32
  //                     bits, nibble 0 = lane 0, nibble 7 = lane 7)
  //     2: src1        (VSrc_b32 — scale, E8M0 encoded in an i32)
  //     3: scale_sel   (4-bit ImmArg, range 0..15 per
  //                     `AMDGPUCvtScaleIntrinsic` in
  //                     IntrinsicsAMDGPU.td:686.  The AMD ISA spec
  //                     definition of scale_sel's 4-bit semantics
  //                     for the packed-8 FP4 shape is not currently
  //                     reproduced in this tree; the captured gfx1250
  //                     corpus (`scope_discovery/kernels/
  //                     _matmul_ogs_{06d912ce88af,0af655e6ea2b}.hsaco`)
  //                     uses only `scale_sel == 0` across 128
  //                     instances combined, which has the unambiguous
  //                     reading "the scale byte is the low byte of
  //                     the 32-bit scale register".  Both handler
  //                     arms REFUSE scale_sel != 0 loudly until the
  //                     spec is pinned.).
  //
  // LLVM lowering (IntrinsicsAMDGPU.td:688):
  //   declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(
  //       i32 %src, i32 %scale, i32 immarg %scale_sel)
  //
  // Dispatch
  // --------
  //
  //   * `ctx.targetIsa.hasTensorOps` (gfx1250 or any future target
  //     that ships the same VOP3 family): emit the native intrinsic
  //     directly and let the backend select the hardware instruction.
  //
  //   * Otherwise (cross-target: gfx942 / gfx950): emit the per-nibble
  //     bit-algebra dequantisation expansion from
  //     `emitCvtScalePk8Bf16Fp4CrossTargetExpansion` above.  Bit-exact
  //     against the hardware primitive on bit-valid inputs within the
  //     declared support set (see `hotswap/docs/matrix-translation.md
  //     §7.4`).
  //
  // Both arms share the same operand-shape validation (src/scale i32,
  // `scale_sel` immediate, `scale_sel == 0` or refuse) so a corpus
  // drift surfaces on both paths rather than only on whichever one
  // happened to run.
  if (sop == SemOp::V_CVT_SCALE_PK8_BF16_FP4) {
    unsigned opc = di.inst.getOpcode();
    int selIdx = AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::scale_sel);
    if (selIdx < 0 || !di.isImm(selIdx)) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          "v_cvt_scale_pk8_bf16_fp4 missing OpName::scale_sel "
          "immediate operand — operand table mismatch");
      return hr;
    }
    int64_t scaleSel = di.getImm(selIdx);

    // Declared support set: scale_sel == 0 only.  The 4-bit
    // scale_sel field's semantics for the packed-8 FP4 shape aren't
    // pinned in any doc in-tree (see comment block above); the
    // captured corpus uses only scale_sel == 0 across both blobs
    // that emit this primitive.  Refusing other values is the
    // "fail loud on declared-support-set boundary" discipline —
    // same shape as the refusal-of-non-default-op_sel check on
    // V_CVT_F32_{FP8,BF8} higher in this file.
    if (scaleSel != 0) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          "v_cvt_scale_pk8_bf16_fp4 scale_sel != 0 is outside the "
          "declared support set (AMD ISA spec semantics for the "
          "4-bit scale_sel field on packed-8 FP4 are not pinned "
          "in-tree today; captured corpus uses only scale_sel=0 "
          "across every instance) — see hotswap/docs/"
          "matrix-translation.md §7.4");
      return hr;
    }

    Value *src = op.src(0);
    if (src->getType() != ctx.i32Ty)
      src = ctx.B.CreateBitOrPointerCast(src, ctx.i32Ty);
    Value *scale = op.src(1);
    if (scale->getType() != ctx.i32Ty)
      scale = ctx.B.CreateBitOrPointerCast(scale, ctx.i32Ty);

    Value *result;
    if (ctx.targetIsa.hasTensorOps) {
      // Same-target arm: emit the LLVM intrinsic that lowers 1:1 to
      // the hardware opcode.  The write-back path bitcasts the
      // <8 x bfloat> to i128 before handing to writeRegVec.
      Function *cvtFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp4);
      result = ctx.B.CreateCall(
          cvtFn,
          {src, scale, ConstantInt::get(ctx.i32Ty, scaleSel)},
          "cvt_scale_pk8_bf16_fp4");
    } else {
      // Cross-target arm: bit-algebra per-nibble dequantisation,
      // bit-exact against the hardware primitive's output for
      // scale_sel == 0 on every (packed_fp4, scale) input in the
      // declared support set.  See
      // `emitCvtScalePk8Bf16Fp4CrossTargetExpansion` above.
      result = emitCvtScalePk8Bf16Fp4CrossTargetExpansion(ctx, src, scale);
    }
    ctx.writeRegVec(op.dst(), result);
    hr.handled = true;
    return hr;
  }

  if (sop == SemOp::V_PERM_B32) {
    Function *permFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::amdgcn_perm);
    ctx.writeReg32(op.dst(), ctx.B.CreateCall(permFn, {op.src(0), op.src(1), op.src(2)}, "perm"));
    hr.handled = true;
    return hr;
  }

  // ---- 64-bit vector ops ----
  if (sop == SemOp::V_LSHLREV_B64) {
    Value *shamt = op.src(0);
    Value *src = op.src64(1);
    if (src->getType() != ctx.i64Ty) src = ctx.B.CreateBitOrPointerCast(src, ctx.i64Ty);
    Value *shamtExt = ctx.B.CreateZExt(shamt, ctx.i64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateShl(src, shamtExt, "shl"));
    hr.handled = true;
    return hr;
  }
  // gfx8+ V_LSHRREV_B64 / V_ASHRREV_I64 — same operand shape as
  // V_LSHLREV_B64: `dst = src1 >> src0`. Logical right shift fills with
  // zero (lshr) and arithmetic right shift fills with the sign bit (ashr).
  // The hardware masks the shift count to 6 bits; LLVM treats shifts >=
  // bitwidth as poison, so we don't paper over the difference. Corpus
  // shifts always carry a finite immediate or a producer that already
  // masks (the `_upcast_from_mxfp` blocker is `>> 16` over an i64 that
  // packs two i32 lanes — well-defined for both ISA and LLVM).
  if (sop == SemOp::V_LSHRREV_B64 || sop == SemOp::V_ASHRREV_I64) {
    Value *shamt = op.src(0);
    Value *src = op.src64(1);
    if (src->getType() != ctx.i64Ty) src = ctx.B.CreateBitOrPointerCast(src, ctx.i64Ty);
    Value *shamtExt = ctx.B.CreateZExt(shamt, ctx.i64Ty);
    Value *res = (sop == SemOp::V_LSHRREV_B64)
                     ? ctx.B.CreateLShr(src, shamtExt, "lshr")
                     : ctx.B.CreateAShr(src, shamtExt, "ashr");
    ctx.writeReg64(op.dst(), res);
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_LSHL_ADD_U64) {
    Value *src0 = op.src64(0);
    Value *shamt = op.src(1);
    Value *src2 = op.src64(2);
    if (src0->getType()->isPointerTy()) src0 = ctx.B.CreatePtrToInt(src0, ctx.i64Ty);
    if (src0->getType() != ctx.i64Ty) src0 = ctx.B.CreateBitOrPointerCast(src0, ctx.i64Ty);
    if (src2->getType() != ctx.i64Ty) src2 = ctx.B.CreateBitOrPointerCast(src2, ctx.i64Ty);
    Value *shamtExt = ctx.B.CreateZExt(shamt, ctx.i64Ty);
    Value *shifted = ctx.B.CreateShl(src0, shamtExt);
    ctx.writeReg64(op.dst(), ctx.B.CreateAdd(shifted, src2, "lshl_add"));
    hr.handled = true;
    return hr;
  }
  if (sop == SemOp::V_ADD_NC_U64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    if (s0->getType() != ctx.i64Ty) s0 = ctx.B.CreateBitOrPointerCast(s0, ctx.i64Ty);
    if (s1->getType() != ctx.i64Ty) s1 = ctx.B.CreateBitOrPointerCast(s1, ctx.i64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateAdd(s0, s1, "vadd64"));
    hr.handled = true;
    return hr;
  }
  // gfx1250 V_MUL_U64: VOP2 64-bit unsigned multiply producing the low
  // 64 bits of (s0 * s1). Mirrors the V_ADD_NC_U64 shape.
  if (sop == SemOp::V_MUL_U64) {
    Value *s0 = op.src64(0), *s1 = op.src64(1);
    if (s0->getType() != ctx.i64Ty) s0 = ctx.B.CreateBitOrPointerCast(s0, ctx.i64Ty);
    if (s1->getType() != ctx.i64Ty) s1 = ctx.B.CreateBitOrPointerCast(s1, ctx.i64Ty);
    ctx.writeReg64(op.dst(), ctx.B.CreateMul(s0, s1, "vmul64"));
    hr.handled = true;
    return hr;
  }

  // ---- v_mad_u64_u32 (2 defs: VDST + SDST, firstSrcIdx=2) ----
  if (sop == SemOp::V_MAD_U64_U32) {
    Value *a = ctx.B.CreateZExt(op.src(0), ctx.i64Ty), *b = ctx.B.CreateZExt(op.src(1), ctx.i64Ty);
    Value *res = ctx.B.CreateAdd(ctx.B.CreateMul(a, b), op.src64(2), "vmad64");
    ctx.writeReg64(op.dst(0), res);
    ctx.writeReg64(op.dst(1), ConstantInt::get(ctx.i64Ty, 0));
    hr.handled = true;
    return hr;
  }

  // ---- Vector compares (V_CMP / V_CMPX) ----
  // Extracted to handle_valu_vcmp.cpp.
  {
    HandlerResult sub = handleVALU_Vcmp(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  // ---- VOP3P / WMMA / v_fma_mix_f32 / v_cndmask_b32 ----
  // Extracted to handle_valu_vop3p.cpp.
  {
    HandlerResult sub = handleVALU_VOP3P(ctx, di, op);
    if (sub.handled || sub.failure.hasFailed())
      return sub;
  }

  return hr;
}

} // namespace transpiler
