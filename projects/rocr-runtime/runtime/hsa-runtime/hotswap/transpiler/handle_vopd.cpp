#include "handlers.hpp"
#include "raiser.hpp"

#include "semop.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

namespace transpiler {
HandlerResult handleVOPD(RaiseContext &ctx, const DecodedInst &di,
                        OpResolver &op, RaiseResult &result) {
  HandlerResult hr;
  StringRef mn(di.mnemonic);
  SemOp sop = di.semOp;

  // VOPD instructions are printed as: "v_dual_X dst, src... :: v_dual_Y dst, src..."
  // We parse the full text to decompose both operations and handle each
  // by mapping v_dual_X → v_X and dispatching to the VOP handler.
  StringRef text(di.fullText);
  auto [xPart, yPart] = text.split(" :: ");
  if (yPart.empty()) {
    llvm::errs() << "transpiler: VOPD: cannot split dual instruction: " << text << "\n";
    result.failMnemonic = di.mnemonic;
        result.failFormat = "VOPD";
        hr.handled = false;
        return hr;
  }

  // Parse "v_dual_<op> vDST, vSRC0, vSRC1" for each half.
  // Extract mnemonic and register operands from printed text.
  //
  // Both halves share the same 8-bit s_set_vgpr_msb state: LLVM's
  // AMDGPULowerVGPREncoding pass asserts that the X-op and Y-op of a VOPD
  // pair must carry identical MSB values per operand slot (see
  // "Invalid VOPD pair was created").  So we apply `ctx.vgprMSBs` directly.
  auto parseVOPDHalf = [&](StringRef part) -> bool {
    part = part.ltrim();
    auto [mnPart, argsPart] = part.split(' ');

    // Map v_dual_X → v_X
    StringRef baseMn = mnPart;
    if (baseMn.starts_with("v_dual_"))
      baseMn = baseMn.drop_front(7); // "v_dual_" = 7 chars
    std::string vopMn = ("v_" + baseMn).str();

    // Parse comma-separated operands, then further split on spaces
    // to separate modifiers like "bitop3:0x40" from register names
    SmallVector<StringRef, 8> operands;
    StringRef remaining = argsPart.ltrim();
    while (!remaining.empty()) {
      if (remaining.starts_with("//")) break;
      auto [tok, rest] = remaining.split(',');
      tok = tok.trim();
      if (!tok.empty()) {
        // Split further on spaces to handle "v0 bitop3:0x40" → "v0", "bitop3:0x40"
        SmallVector<StringRef, 4> subToks;
        tok.split(subToks, ' ', -1, false);
        for (auto &st : subToks)
          operands.push_back(st);
      }
      remaining = rest.ltrim();
    }
    if (operands.empty()) return false;

    // operands[0] = dst, operands[1..] = srcs
    auto parseVRegIdx = [](StringRef name) -> int {
      if (name.starts_with("v") && !name.starts_with("vcc")) {
        int idx = -1;
        if (!name.drop_front(1).getAsInteger(10, idx)) return idx;
      }
      return -1;
    };

    // s_set_vgpr_msb offset for a given slot (0=src0, 1=src1, 2=src2, 3=dst).
    auto msbOffset = [&](unsigned slot) -> int {
      return ((ctx.vgprMSBs >> (slot * 2)) & 0x3) * 256;
    };

    // dst — apply DST MSB (slot 3)
    int dstIdx = parseVRegIdx(operands[0]);
    if (dstIdx < 0) return false;
    dstIdx += msbOffset(3);

    // Generic VOPD operand reader: VGPR, SGPR, or literal immediate.
    // Handles source modifiers: -v0 (fneg), |v0| (fabs), -|v0| (fneg+fabs)
    // srcSlot: MSB slot for this source (0=src0, 1=src1, 2=src2)
    auto readVOPDSrc = [&](StringRef name, unsigned srcSlot = 0) -> Value * {
      bool neg = false, absmod = false;
      if (name.starts_with("-")) { neg = true; name = name.drop_front(1); }
      if (name.starts_with("|") && name.ends_with("|")) {
        absmod = true; name = name.drop_front(1).drop_back(1);
      }
      Value *v = nullptr;
      int vidx = parseVRegIdx(name);
      if (vidx >= 0) { v = ctx.regs.loadVGPR32(ctx.B, vidx + msbOffset(srcSlot)); }
      else if (name.starts_with("s")) {
        int sidx = -1;
        if (!name.drop_front(1).getAsInteger(10, sidx))
          v = ctx.regs.loadSGPR32(ctx.B, sidx);
      }
      if (!v) {
        int64_t imm;
        if (!name.getAsInteger(0, imm))
          v = ConstantInt::get(ctx.i32Ty, (uint32_t)(imm & 0xFFFFFFFF));
      }
      if (!v) return nullptr;
      if (neg || absmod) {
        v = ctx.B.CreateBitCast(v, ctx.f32Ty);
        if (absmod) v = ctx.B.CreateCall(Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fabs, {ctx.f32Ty}), {v});
        if (neg) v = ctx.B.CreateFNeg(v);
        v = ctx.B.CreateBitCast(v, ctx.i32Ty);
      }
      return v;
    };

    if (vopMn == "v_mov_b32") {
      if (operands.size() < 2) return false;
      Value *srcVal = readVOPDSrc(operands[1]);
      if (!srcVal) return false;
      ctx.regs.storeVGPR32(ctx.B, dstIdx, srcVal);
      return true;
    }

    if (vopMn == "v_cndmask_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1], 0);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      Value *cond = ctx.regs.loadVCC(ctx.B);
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateSelect(cond, s1, s0, "vopd_cndmask"));
      return true;
    }

    if (vopMn == "v_add_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "vopd_fadd"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_mul_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "vopd_fmul"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_sub_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateBitCast(ctx.B.CreateFSub(s0, s1, "vopd_fsub"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_fmac_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      Value *dv = ctx.B.CreateBitCast(ctx.regs.loadVGPR32(ctx.B, dstIdx), ctx.f32Ty);
      Function *fmuladd = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fmuladd, {ctx.f32Ty});
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(fmuladd, {s0, s1, dv}, "vopd_fmac"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_add_nc_u32" || vopMn == "v_add_u32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateAdd(s0, s1, "vopd_add"));
      return true;
    }

    if (vopMn == "v_sub_nc_u32" || vopMn == "v_subrev_nc_u32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      if (vopMn == "v_subrev_nc_u32") std::swap(s0, s1);
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateSub(s0, s1, "vopd_sub"));
      return true;
    }

    if (vopMn == "v_lshlrev_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateShl(s1, s0, "vopd_shl"));
      return true;
    }

    if (vopMn == "v_and_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateAnd(s0, s1, "vopd_and"));
      return true;
    }

    if (vopMn == "v_lshrrev_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateLShr(s1, s0, "vopd_lshr"));
      return true;
    }

    if (vopMn == "v_fma_f32") {
      if (operands.size() < 4) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      Value *s2 = readVOPDSrc(operands[3], 2);
      if (!s0 || !s1 || !s2) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      s2 = ctx.B.CreateBitCast(s2, ctx.f32Ty);
      Function *fma = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::fma, {ctx.f32Ty});
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "vopd_fma"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_max_num_f32" || vopMn == "v_max_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "vopd_fmax"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_min_num_f32" || vopMn == "v_min_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
      ctx.regs.storeVGPR32(ctx.B, dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "vopd_fmin"), ctx.i32Ty));
      return true;
    }

    // v_bitop2_b32 / v_bitop3_b32 in VOPD context
    // The truth table immediate is appended as "bitop3:0xNN"
    if (vopMn == "v_bitop2_b32" || vopMn == "v_bitop3_b32") {
      if (operands.size() < 3) return false;
      Value *a = readVOPDSrc(operands[1]);
      Value *b = readVOPDSrc(operands[2], 1);
      if (!a || !b) return false;
      uint32_t lut = 0;
      for (unsigned k = 3; k < operands.size(); k++) {
        if (operands[k].starts_with("bitop3:")) {
          StringRef hex = operands[k].drop_front(7);
          hex.getAsInteger(0, lut);
          break;
        }
      }
      // bitop2: src2 = 0, so only even-indexed LUT entries (0,2,4,6) matter
      Value *c = ConstantInt::get(ctx.i32Ty, 0);
      Value *na = ctx.B.CreateNot(a), *nb = ctx.B.CreateNot(b), *nc = ctx.B.CreateNot(c);
      Value *result = ConstantInt::get(ctx.i32Ty, 0);
      Value *minterms[8] = {
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, nb), c),
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(na, b), c),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, nb), c),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), nc),
        ctx.B.CreateAnd(ctx.B.CreateAnd(a, b), c),
      };
      for (int i = 0; i < 8; i++)
        if (lut & (1 << i))
          result = ctx.B.CreateOr(result, minterms[i]);
      ctx.regs.storeVGPR32(ctx.B, dstIdx, result);
      return true;
    }

    llvm::errs() << "transpiler: VOPD: unhandled sub-operation '" << vopMn << "'\n";
    return false;
  };

  bool xOk = parseVOPDHalf(xPart);
  bool yOk = xOk && parseVOPDHalf(yPart);
  if (!xOk || !yOk) {
    result.failMnemonic = di.mnemonic;
    result.failFormat = "VOPD";
    llvm::errs() << "transpiler: VOPD decomposition failed: " << text << "\n";
    hr.handled = false;
        return hr;
  }
  hr.handled = true;
  return hr;
}

} // namespace transpiler
