#include "handlers.hpp"

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
                        OpResolver &op) {
  HandlerResult hr;
  SemOp sop = di.semOp;

  // VOPD instructions are printed as: "v_dual_X dst, src... :: v_dual_Y dst, src..."
  // We parse the full text to decompose both operations and handle each
  // by mapping v_dual_X → v_X and dispatching to the VOP handler.
  StringRef text(di.fullText);
  auto [xPart, yPart] = text.split(" :: ");
  if (yPart.empty()) {
    llvm::errs() << "transpiler: VOPD: cannot split dual instruction: " << text << "\n";
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOPD", "cannot split dual instruction");
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
      // TTMP SGPRs ("trap temps") — Triton-emitted kernels on gfx1250 use
      // `ttmp9` in VOPD source slots to read the workgroup-id-X value the
      // SPE prelude seeded into `regs.ttmp[9]` (see raiser.cpp's Phase 4
      // `fnWorkgroupIdX` store).  TTMP lives in its own alloca bank and
      // must be checked before the plain SGPR branch below, because
      // `ttmp<N>` does not start with the letter 's' but the next
      // `name.starts_with("s")` catch-all would otherwise let this fall
      // through to the integer-literal parser (which fails loudly) rather
      // than the register load that was intended.
      else if (name.starts_with("ttmp")) {
        int tidx = -1;
        if (!name.drop_front(4).getAsInteger(10, tidx) &&
            tidx >= 0 &&
            static_cast<unsigned>(tidx) < ctx.regs.ttmp.size())
          v = ctx.B.CreateLoad(ctx.i32Ty, ctx.regs.ttmp[tidx], "vopd_ttmp");
      }
      // VCC.lo as a scalar source — present in ~8 VOPD sites across the
      // current corpus (workgroup-id distribution prologues and similar
      // SPE fixtures).  VOPD is a wave32-only family, so `vcc_lo` is the
      // entire VCC bitmask in the source ISA; we route through
      // `readVCCAsWaveMask` so the wave-projection layer sees a
      // principled VCC-as-scalar read and can re-project to the target
      // wave width, instead of a width-specific half-slice that would
      // silently miscompile under cross-widening.
      else if (name == "vcc_lo") {
        v = ctx.regs.readVCCAsWaveMask(ctx.B, ctx.i32Ty);
      }
      else if (name.starts_with("s")) {
        int sidx = -1;
        if (!name.drop_front(1).getAsInteger(10, sidx))
          v = ctx.regs.loadSGPR32(ctx.B, sidx);
      }
      if (!v) {
        int64_t imm;
        if (!name.getAsInteger(0, imm))
          v = ConstantInt::get(ctx.i32Ty,
                                static_cast<uint32_t>(imm & 0xFFFFFFFF));
      }
      // AMDGPU's instruction printer surfaces the f32 inline-constant
      // pool (see `printImmediateFloat32` in
      // llvm/lib/Target/AMDGPU/MCTargetDesc/AMDGPUInstPrinter.cpp) as
      // their decimal float spelling rather than as the hex bit
      // pattern.  The integer-literal path above can't parse them, so
      // we map each printed literal back to its canonical IEEE-754
      // 32-bit encoding here.  This is the same finite enumeration
      // the printer uses; anything outside it surfaces as `0xNNNN...`
      // and is already handled by the integer parse.  The negation /
      // abs source modifiers were already stripped before this point.
      if (!v) {
        static const struct {
          const char *txt;
          uint32_t bits;
        } kF32Inline[] = {
            {"0.0", 0x00000000u},
            {"1.0", 0x3f800000u},
            {"-1.0", 0xbf800000u},
            {"0.5", 0x3f000000u},
            {"-0.5", 0xbf000000u},
            {"2.0", 0x40000000u},
            {"-2.0", 0xc0000000u},
            {"4.0", 0x40800000u},
            {"-4.0", 0xc0800000u},
            {"0.15915494", 0x3e22f983u},
        };
        for (auto &e : kF32Inline) {
          if (name == e.txt) {
            v = ConstantInt::get(ctx.i32Ty, e.bits);
            break;
          }
        }
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
      ctx.storeVGPR32(dstIdx, srcVal);
      return true;
    }

    if (vopMn == "v_cndmask_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1], 0);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;

      // VOPD on gfx1250 encodes an EXPLICIT scalar condition operand
      // (operands[3]) for each `v_dual_cndmask_b32` half. Ignoring it and
      // defaulting to VCC (the previous behaviour) is a silent
      // miscompile whenever the paired instruction writes `vcc_lo` and
      // the cndmask's real condition source is a separate SGPR — which
      // is exactly the shape Triton's `tl.cumsum` Kogge-Stone scan
      // emits at distance-8 and distance-16: `v_dual_cndmask_b32 v<sel>,
      // v<sel>, v<next_sel>, s<stage_guard> :: v_dual_cndmask_b32
      // v<val>, v<val>, v<fadd>, vcc_lo` pairs the selector advance
      // (guarded by `sN = (tid < 2^s)`) with the value update (guarded
      // by `vcc = (tid > 2^s - 1)`). Hardcoding VCC for both halves
      // conflates the two predicates and produces
      // `canary_bpermute_scan_fp32`'s silent-WRONG scan output (the
      // root-cause finding in hotswap/docs/modrep-predicate-chain.md
      // §6.4: stage-3 lanes >= 8 read themselves instead of
      // lane-8 partner, doubling their accumulator).
      //
      // Mirrors the non-VOPD `V_CNDMASK_B32` handler in
      // handle_valu_vop3p.cpp — if the 3rd operand is an SGPR, prefer
      // the fresh V_CMP shadow `i1` from the per-BB cache; else route
      // through the projection's `extractLaneBitFromWaveMask` (lossy
      // under wave32 → wave64 cross-widening if the producer truncated
      // to source width, per the documented gap in
      // hotswap/docs/sgpr-wave-mask-translation.md §3.1). Only if no
      // scalar condition is specified (or the 3rd operand is
      // `vcc_lo`/`vcc`) do we fall back to `loadVCC`.
      Value *cond = nullptr;
      if (operands.size() >= 4) {
        StringRef condName = operands[3];
        if (condName == "vcc_lo" || condName == "vcc") {
          cond = ctx.regs.loadVCC(ctx.B);
        } else if (condName.starts_with("s") && !condName.starts_with("scc")) {
          int sidx = -1;
          if (!condName.drop_front(1).getAsInteger(10, sidx) && sidx >= 0) {
            if (Value *freshCmp = ctx.lookupSgprWaveMaskI1(sidx)) {
              cond = freshCmp;
            } else {
              Value *condVal = ctx.isa.isWave32()
                                   ? ctx.regs.loadSGPR32(ctx.B, sidx)
                                   : ctx.regs.loadSGPR64(ctx.B, sidx);
              cond = ctx.projection.extractLaneBitFromWaveMask(ctx.B,
                                                                condVal);
            }
          }
        }
      }
      if (!cond) cond = ctx.regs.loadVCC(ctx.B);
      ctx.storeVGPR32(dstIdx, ctx.B.CreateSelect(cond, s1, s0, "vopd_cndmask"));
      return true;
    }

    if (vopMn == "v_add_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      ctx.storeVGPR32(dstIdx, ctx.B.CreateBitCast(ctx.B.CreateFAdd(s0, s1, "vopd_fadd"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_mul_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      ctx.storeVGPR32(dstIdx, ctx.B.CreateBitCast(ctx.B.CreateFMul(s0, s1, "vopd_fmul"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_sub_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      ctx.storeVGPR32(dstIdx, ctx.B.CreateBitCast(ctx.B.CreateFSub(s0, s1, "vopd_fsub"), ctx.i32Ty));
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
      ctx.storeVGPR32(dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(fmuladd, {s0, s1, dv}, "vopd_fmac"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_add_nc_u32" || vopMn == "v_add_u32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.storeVGPR32(dstIdx, ctx.B.CreateAdd(s0, s1, "vopd_add"));
      return true;
    }

    if (vopMn == "v_sub_nc_u32" || vopMn == "v_subrev_nc_u32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      if (vopMn == "v_subrev_nc_u32") std::swap(s0, s1);
      ctx.storeVGPR32(dstIdx, ctx.B.CreateSub(s0, s1, "vopd_sub"));
      return true;
    }

    if (vopMn == "v_lshlrev_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.storeVGPR32(dstIdx, ctx.B.CreateShl(s1, s0, "vopd_shl"));
      return true;
    }

    if (vopMn == "v_and_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.storeVGPR32(dstIdx, ctx.B.CreateAnd(s0, s1, "vopd_and"));
      return true;
    }

    if (vopMn == "v_lshrrev_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.storeVGPR32(dstIdx, ctx.B.CreateLShr(s1, s0, "vopd_lshr"));
      return true;
    }

    // v_ashrrev_i32 mirrors v_lshrrev_b32 but with arithmetic (sign-
    // preserving) right shift.  The "rev" suffix means the operand
    // ordering in the printed text is `(shift_amount, value)` — i.e.
    // operands[1] = shift amount, operands[2] = value to shift.
    if (vopMn == "v_ashrrev_i32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.storeVGPR32(dstIdx, ctx.B.CreateAShr(s1, s0, "vopd_ashr"));
      return true;
    }

    // v_max_i32 (signed max).  Use llvm.smax for symmetry with the
    // float min/max paths above; LLVM lowers it to the canonical
    // `select (icmp sgt) ...` shape on AMDGPU.
    if (vopMn == "v_max_i32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      Function *smaxFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::smax, {ctx.i32Ty});
      ctx.storeVGPR32(dstIdx,
                      ctx.B.CreateCall(smaxFn, {s0, s1}, "vopd_smax"));
      return true;
    }

    // v_min_i32 (signed min).  Mirror of v_max_i32 above; included
    // for completeness of the signed-integer dual-issue family even
    // though the current corpus only exercises the smax variant.
    if (vopMn == "v_min_i32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      Function *sminFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::smin, {ctx.i32Ty});
      ctx.storeVGPR32(dstIdx,
                      ctx.B.CreateCall(sminFn, {s0, s1}, "vopd_smin"));
      return true;
    }

    // v_max_u32 / v_min_u32 (unsigned max/min).  Same shape as the
    // signed variants but using llvm.umax / llvm.umin.  Included for
    // completeness alongside the signed siblings.
    if (vopMn == "v_max_u32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      Function *umaxFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::umax, {ctx.i32Ty});
      ctx.storeVGPR32(dstIdx,
                      ctx.B.CreateCall(umaxFn, {s0, s1}, "vopd_umax"));
      return true;
    }
    if (vopMn == "v_min_u32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      Function *uminFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::umin, {ctx.i32Ty});
      ctx.storeVGPR32(dstIdx,
                      ctx.B.CreateCall(uminFn, {s0, s1}, "vopd_umin"));
      return true;
    }

    // v_xor_b32 / v_or_b32 — bitwise siblings of v_and_b32 above.
    // VOPD is a Wave32 dual-issue family for the most common simple
    // VALU ops; rounding out the bitwise trio costs nothing and
    // forecloses the next likely "unhandled sub-operation" surprise.
    if (vopMn == "v_xor_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.storeVGPR32(dstIdx, ctx.B.CreateXor(s0, s1, "vopd_xor"));
      return true;
    }
    if (vopMn == "v_or_b32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      ctx.storeVGPR32(dstIdx, ctx.B.CreateOr(s0, s1, "vopd_or"));
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
      ctx.storeVGPR32(dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(fma, {s0, s1, s2}, "vopd_fma"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_max_num_f32" || vopMn == "v_max_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      Function *maxFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::maxnum, {ctx.f32Ty});
      ctx.storeVGPR32(dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(maxFn, {s0, s1}, "vopd_fmax"), ctx.i32Ty));
      return true;
    }

    if (vopMn == "v_min_num_f32" || vopMn == "v_min_f32") {
      if (operands.size() < 3) return false;
      Value *s0 = readVOPDSrc(operands[1]);
      Value *s1 = readVOPDSrc(operands[2], 1);
      if (!s0 || !s1) return false;
      s0 = ctx.B.CreateBitCast(s0, ctx.f32Ty); s1 = ctx.B.CreateBitCast(s1, ctx.f32Ty);
      Function *minFn = Intrinsic::getOrInsertDeclaration(&ctx.M, Intrinsic::minnum, {ctx.f32Ty});
      ctx.storeVGPR32(dstIdx, ctx.B.CreateBitCast(ctx.B.CreateCall(minFn, {s0, s1}, "vopd_fmin"), ctx.i32Ty));
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
      ctx.storeVGPR32(dstIdx, result);
      return true;
    }

    llvm::errs() << "transpiler: VOPD: unhandled sub-operation '" << vopMn << "'\n";
    return false;
  };

  bool xOk = parseVOPDHalf(xPart);
  bool yOk = xOk && parseVOPDHalf(yPart);
  if (!xOk || !yOk) {
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOPD", "VOPD decomposition failed");
    llvm::errs() << "transpiler: VOPD decomposition failed: " << text << "\n";
    return hr;
  }
  hr.handled = true;
  return hr;
}

} // namespace transpiler
