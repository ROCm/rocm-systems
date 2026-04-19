#include "handle_valu_internal.hpp"

#include "semop.hpp"
#include "wmma_lowering.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

namespace {

// Parse a `key:[a,b,c]` bracketed int list out of the disassembled
// instruction text. Used for VOP3P modifiers (op_sel_hi, neg_lo,
// neg_hi, op_sel) that LLVM's MC layer doesn't surface as first-class
// operands. Leaves `out[]` untouched for indices whose parse fails.
void parseBracketList3(StringRef text, StringRef key, int out[3]) {
  auto pos = text.find(key);
  if (pos == StringRef::npos) return;
  auto brk = text.find('[', pos);
  if (brk == StringRef::npos) return;
  auto end = text.find(']', brk);
  if (end == StringRef::npos) return;
  StringRef inner = text.slice(brk + 1, end);
  SmallVector<StringRef, 3> parts;
  inner.split(parts, ',');
  for (unsigned i = 0; i < parts.size() && i < 3; i++) {
    int val = 0;
    if (!parts[i].trim().getAsInteger(10, val))
      out[i] = val;
  }
}

} // namespace

HandlerResult handleVALU_VOP3P(RaiseContext &ctx, const DecodedInst &di,
                                OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;
  StringRef mn(di.mnemonic);

  switch (sop) {
  // ---- VOP3P packed ops (2x fp32 in 2 dwords) ----
  // Handle op_sel_hi, neg_lo, neg_hi modifiers.
  case SemOp::V_PK_MOV_B32: {
    ctx.writeReg64(op.dst(), op.src64(0));
    hr.handled = true;
    return hr;
  }
  case SemOp::V_PK_ADD_F32:
  case SemOp::V_PK_MUL_F32:
  case SemOp::V_PK_FMA_F32:
  case SemOp::V_PK_MAX_F32:
  case SemOp::V_PK_MIN_F32: {
    auto *v2f32 = FixedVectorType::get(ctx.f32Ty, 2);

    int opSelHi[3] = {1, 1, 1};  // default: high lane reads high element
    int negLo[3] = {0, 0, 0};
    int negHi[3] = {0, 0, 0};
    StringRef text(di.fullText);
    parseBracketList3(text, "op_sel_hi:", opSelHi);
    parseBracketList3(text, "neg_lo:", negLo);
    parseBracketList3(text, "neg_hi:", negHi);

    // Read each source as <2 x f32>, apply element selection and negation.
    //
    // Two operand shapes are accepted:
    //   * Register (the common case): reads a 64-bit VGPR pair as
    //     `<2 x f32>`; lo/hi extract index the two packed lanes.
    //   * Immediate / inline literal: VOP3P encodes a single 32-bit
    //     literal per source slot which the hardware broadcasts to
    //     both packed lanes (the `op_sel_hi` modifier is ignored on
    //     scalar literals because there's only one element to choose).
    //     The swiglu tensilelite kernel exercises this path with
    //     `v_pk_add_f32 vN, vM, 0x...` where the literal is a packed
    //     bias constant.  We model it by reading the i32, bit-casting
    //     to f32, and constructing a 2-lane vector with both lanes
    //     equal to the literal — the high-lane source will then be
    //     `lit_f32` regardless of `opSelHi[i]` (broadcast is
    //     idempotent).  `negLo` / `negHi` still apply per-lane.
    auto readPkSrc = [&](unsigned i) -> Value * {
      Value *lo, *hi;
      if (op.isSrcReg(i)) {
        Value *vec = ctx.regs.readRegVec(ctx.B, op.srcReg(i), v2f32);
        lo = ctx.B.CreateExtractElement(vec, (uint64_t)0);
        hi = ctx.B.CreateExtractElement(vec, (uint64_t)1);
        // op_sel_hi: if 0, high lane reads low element (broadcast).
        if (opSelHi[i] == 0)
          hi = lo;
      } else {
        // Inline 32-bit literal broadcast to both packed lanes.
        Value *lit = ctx.B.CreateBitCast(op.src(i), ctx.f32Ty);
        lo = lit;
        hi = lit;
      }
      if (negLo[i])
        lo = ctx.B.CreateFNeg(lo);
      if (negHi[i])
        hi = ctx.B.CreateFNeg(hi);
      Value *r = UndefValue::get(v2f32);
      r = ctx.B.CreateInsertElement(r, lo, (uint64_t)0);
      r = ctx.B.CreateInsertElement(r, hi, (uint64_t)1);
      return r;
    };

    Value *s0 = readPkSrc(0);
    Value *s1 = readPkSrc(1);
    if (!s0 || !s1) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VALU", "V_PK_* missing packed operand");
      return hr;
    }

    Value *res = nullptr;
    switch (sop) {
    case SemOp::V_PK_ADD_F32:
      res = ctx.B.CreateFAdd(s0, s1, "pk_add");
      break;
    case SemOp::V_PK_MUL_F32:
      res = ctx.B.CreateFMul(s0, s1, "pk_mul");
      break;
    case SemOp::V_PK_MAX_F32: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::maxnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_max");
      break;
    }
    case SemOp::V_PK_MIN_F32: {
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::minnum, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1}, "pk_min");
      break;
    }
    case SemOp::V_PK_FMA_F32: {
      Value *s2 = readPkSrc(2);
      if (!s2) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "VALU", "V_PK_FMA_F32 missing src2");
        return hr;
      }
      Function *fn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::fma, {v2f32});
      res = ctx.B.CreateCall(fn, {s0, s1, s2}, "pk_fma");
      break;
    }
    default: llvm_unreachable("filtered by outer switch");
    }
    ctx.writeRegVec(op.dst(), res);
    hr.handled = true;
    return hr;
  }

  // ---- WMMA (gfx1250 RDNA4, VOP3P encoding) ----
  // 16x16xK WMMA family with f32 accumulator. Two K-families:
  //   * 16-bit elements, K=32 (8 VGPRs of <16 x t> per A/B side):
  //       v_wmma_f32_16x16x32_f16,  v_wmma_f32_16x16x32_bf16
  //   * 8-bit elements,  K=64 (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_f32_16x16x64_<a>_<b>  for a,b ∈ {fp8, bf8}
  //
  // All share the per-Wave32-lane fragment shape on both A/B (8 VGPRs,
  // 32 bytes) and C/D (<8 x f32>) sides. The WMMA12 native-intrinsic
  // path (when target supports it) and the gfx942 MFMA lowering path
  // (`emitWMMAtoMFMA`, parameterised on `WMMAInputType`) are common
  // across the entire family — the local A/B IR vector type +
  // native-WMMA intrinsic ID + WMMAInputType triple is the only delta
  // between variants. "Design the operation, not the opcode."
  //
  // Native WMMA12 intrinsic-call shapes diverge between the 16-bit and
  // 8-bit families: the 16-bit form uses `AMDGPUWmmaIntrinsicModsAllReuse`
  // (8 args: A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b) and the
  // 8-bit form uses `AMDGPUWmmaIntrinsicModsC` (6 args: A, B, C_mod, C,
  // reuse_a, reuse_b). The MFMA fallback path is uniform across both.
  case SemOp::V_WMMA_F32_16x16x32_F16:
  case SemOp::V_WMMA_F32_16x16x32_BF16:
  case SemOp::V_WMMA_F32_16x16x64_FP8_FP8:
  case SemOp::V_WMMA_F32_16x16x64_FP8_BF8:
  case SemOp::V_WMMA_F32_16x16x64_BF8_FP8:
  case SemOp::V_WMMA_F32_16x16x64_BF8_BF8: {
    const bool is8bit =
        (sop == SemOp::V_WMMA_F32_16x16x64_FP8_FP8) ||
        (sop == SemOp::V_WMMA_F32_16x16x64_FP8_BF8) ||
        (sop == SemOp::V_WMMA_F32_16x16x64_BF8_FP8) ||
        (sop == SemOp::V_WMMA_F32_16x16x64_BF8_BF8);
    const bool isBF16 = (sop == SemOp::V_WMMA_F32_16x16x32_BF16);

    Type *abIRTy = nullptr;
    if (is8bit) {
      abIRTy = FixedVectorType::get(ctx.i32Ty, 8);
    } else {
      Type *elemTy = isBF16 ? Type::getBFloatTy(ctx.C)
                            : Type::getHalfTy(ctx.C);
      abIRTy = FixedVectorType::get(elemTy, 16);
    }
    Type *v8f32Ty = FixedVectorType::get(ctx.f32Ty, 8);

    ParsedReg dest = op.dst();
    ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
    ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;

    Value *a = ctx.regs.readRegVec(ctx.B, srcA, abIRTy);
    Value *b = ctx.regs.readRegVec(ctx.B, srcB, abIRTy);
    Value *c = ctx.regs.readRegVec(ctx.B, srcC, v8f32Ty);

    auto wmmaInputType = [&]() -> WMMAInputType {
      switch (sop) {
      case SemOp::V_WMMA_F32_16x16x32_F16:    return WMMAInputType::F16;
      case SemOp::V_WMMA_F32_16x16x32_BF16:   return WMMAInputType::BF16;
      case SemOp::V_WMMA_F32_16x16x64_FP8_FP8:return WMMAInputType::FP8_FP8;
      case SemOp::V_WMMA_F32_16x16x64_FP8_BF8:return WMMAInputType::FP8_BF8;
      case SemOp::V_WMMA_F32_16x16x64_BF8_FP8:return WMMAInputType::BF8_FP8;
      case SemOp::V_WMMA_F32_16x16x64_BF8_BF8:return WMMAInputType::BF8_BF8;
      default:
        report_fatal_error("transpiler: WMMA SemOp not in dispatch table");
      }
    }();

    Value *result_val;
    if (ctx.targetIsa.hasWMMA12) {
      Intrinsic::ID wmmaId;
      switch (sop) {
      case SemOp::V_WMMA_F32_16x16x32_F16:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_f16; break;
      case SemOp::V_WMMA_F32_16x16x32_BF16:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x32_bf16; break;
      case SemOp::V_WMMA_F32_16x16x64_FP8_FP8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_fp8; break;
      case SemOp::V_WMMA_F32_16x16x64_FP8_BF8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_bf8; break;
      case SemOp::V_WMMA_F32_16x16x64_BF8_FP8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_fp8; break;
      case SemOp::V_WMMA_F32_16x16x64_BF8_BF8:
        wmmaId = Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_bf8; break;
      default:
        report_fatal_error("transpiler: WMMA SemOp not in WMMA12 dispatch");
      }
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, wmmaId, {v8f32Ty, abIRTy});
      if (is8bit) {
        // AMDGPUWmmaIntrinsicModsC: (A, B, C_mod, C, reuse_a, reuse_b)
        result_val = ctx.B.CreateCall(wmmaFn, {
            a, b,
            ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
            ctx.B.getFalse(), ctx.B.getFalse()
        }, "wmma");
      } else {
        // AMDGPUWmmaIntrinsicModsAllReuse:
        //   (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b)
        result_val = ctx.B.CreateCall(wmmaFn, {
            ctx.B.getFalse(), a,
            ctx.B.getFalse(), b,
            ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
            ctx.B.getFalse(), ctx.B.getFalse()
        }, "wmma");
      }
    } else {
      result_val = emitWMMAtoMFMA(ctx, a, b, c, wmmaInputType);
    }

    ctx.writeRegVec(dest, result_val);
    hr.handled = true;
    return hr;
  }

  // ---- v_fma_mix_f32: mixed-precision FMA (VOP3P) ----
  // dst = fma(cvt_f32(src0_part), cvt_f32(src1_part), cvt_f32(src2_part))
  // op_sel_hi[i]==1 → source i is f16 (lo/hi selected by op_sel[i])
  // op_sel_hi[i]==0 → source i is full f32
  case SemOp::V_FMA_MIX_F32: {
    int opSel[3] = {0, 0, 0};
    int opSelHi[3] = {0, 0, 0};
    StringRef text(di.fullText);
    parseBracketList3(text, "op_sel:", opSel);
    parseBracketList3(text, "op_sel_hi:", opSelHi);

    auto readMixSrc = [&](unsigned i) -> Value * {
      Value *raw = op.srcF(i);
      if (opSelHi[i] == 0) {
        if (raw->getType() != ctx.f32Ty) raw = ctx.B.CreateBitCast(raw, ctx.f32Ty);
        return raw;
      }
      if (raw->getType() == ctx.f32Ty) raw = ctx.B.CreateBitCast(raw, ctx.i32Ty);
      Value *bits;
      if (opSel[i] == 0)
        bits = ctx.B.CreateTrunc(raw, Type::getInt16Ty(ctx.C));
      else
        bits = ctx.B.CreateTrunc(ctx.B.CreateLShr(raw, 16),
                                  Type::getInt16Ty(ctx.C));
      Value *f16Val = ctx.B.CreateBitCast(bits, ctx.f16Ty);
      return ctx.B.CreateFPExt(f16Val, ctx.f32Ty, "mix_cvt");
    };

    Value *s0 = readMixSrc(0);
    Value *s1 = readMixSrc(1);
    Value *s2 = readMixSrc(2);
    Function *fmaFn = Intrinsic::getOrInsertDeclaration(
        &ctx.M, Intrinsic::fma, {ctx.f32Ty});
    ctx.writeReg32(op.dst(),
                   ctx.B.CreateBitCast(
                       ctx.B.CreateCall(fmaFn, {s0, s1, s2}, "fma_mix"),
                       ctx.i32Ty));
    hr.handled = true;
    return hr;
  }

  // ---- v_cndmask_b32 (VOP2 or VOP3 — srcMap skips modifiers) ----
  case SemOp::V_CNDMASK_B32: {
    ParsedReg dest = op.dst();
    Value *src0 = op.src(0);
    Value *src1 = op.src(1);
    Value *cond = nullptr;
    if (op.nSrcs() >= 3 && di.isReg(op.srcIdx(2))) {
      ParsedReg condReg =
          ctx.parseReg(di.getReg(op.srcIdx(2)), op.srcIdx(2));
      if (condReg.kind == ParsedReg::SGPR) {
        Value *condVal = ctx.isa.isWave32()
                              ? (Value *)ctx.regs.loadSGPR32(ctx.B,
                                                               condReg.baseIdx)
                              : (Value *)ctx.regs.loadSGPR64(ctx.B,
                                                               condReg.baseIdx);
        cond = ctx.B.CreateICmpNE(condVal,
                                   Constant::getNullValue(condVal->getType()));
      } else {
        cond = ctx.regs.loadVCC(ctx.B);
      }
    }
    if (!cond) cond = ctx.regs.loadVCC(ctx.B);
    ctx.writeReg32(dest, ctx.B.CreateSelect(cond, src1, src0, "cndmask"));
    hr.handled = true;
    return hr;
  }

  default:
    break;
  }
  return hr;
}

} // namespace transpiler
