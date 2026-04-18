#include "raise_context.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::VCC, AMDGPU::EXEC, ...
#include "SIDefines.h"                        // AMDGPU::HWEncoding::*
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace llvm;

namespace transpiler {

BasicBlock *RaiseContext::lookupBB(uint64_t addr) {
  auto it = offsetToBB.find(addr);
  if (it != offsetToBB.end())
    return it->second;
  errs() << "transpiler: missing basic block for offset 0x" << utohexstr(addr)
         << " — creating fallback\n";
  BasicBlock *bb =
      BasicBlock::Create(C, "bb_fallback_0x" + utohexstr(addr), kernel);
  offsetToBB[addr] = bb;
  return bb;
}

void RaiseContext::computeVGPRAdjust(const DecodedInst &di) {
  std::memset(currentVGPRAdjust, 0, sizeof(currentVGPRAdjust));
  if (vgprMSBs == 0)
    return;

  // vgprMSBs is an 8-bit state shared by single-issue instructions and both
  // halves of a VOPD pair.  Layout: src0[1:0], src1[3:2], src2[5:4], dst[7:6].
  unsigned dstMsb = ((unsigned)(vgprMSBs >> 6) & 0x3u) * 256u;
  unsigned srcMsb[3] = {
      ((unsigned)(vgprMSBs >> 0) & 0x3u) * 256u,
      ((unsigned)(vgprMSBs >> 2) & 0x3u) * 256u,
      ((unsigned)(vgprMSBs >> 4) & 0x3u) * 256u,
  };

  for (unsigned i = 0; i < di.numDefs && i < kMaxOps; i++)
    currentVGPRAdjust[i] = dstMsb;

  for (unsigned i = 0; i < di.numSrcs && i < 3; i++) {
    unsigned opIdx = di.srcMap[i];
    if (opIdx < kMaxOps)
      currentVGPRAdjust[opIdx] = srcMsb[i];
  }
}

// Count how many 32-bit sub-registers make up `reg`. A 32-bit register has
// no sub0 (getSubReg returns 0) and is reported as width 1. Tuples
// (SGPR0_SGPR1, VReg_128, ...) walk sub0, sub1, ... until exhausted.
// The loop is bounded by the target's declared sub-reg index count so it
// terminates even if a future TableGen change introduced a cycle in the
// sub-reg graph.
static int computeRegWidth32(const MCRegisterInfo &MRI, MCRegister reg) {
  const unsigned maxSubIdx = MRI.getNumSubRegIndices();
  int w = 0;
  for (unsigned subIdx = AMDGPU::sub0; subIdx < maxSubIdx; ++subIdx) {
    if (!MRI.getSubReg(reg, subIdx))
      break;
    ++w;
  }
  return w ? w : 1;
}

// Locate `reg` inside `RC` and return its 0-based position. Used where the
// hardware encoding is not what we want (TTMPs live at generation-specific
// HW slots 108+ or 112+, but downstream we index a logical `ttmp[16]`
// array). Relies only on TableGen's declared class membership — no enum
// arithmetic assumptions.
static int findIndexInClass(const MCRegisterClass &RC, MCRegister reg) {
  for (unsigned i = 0, e = RC.getNumRegs(); i != e; ++i)
    if (RC.getRegister(i) == reg)
      return static_cast<int>(i);
  return -1;
}

ParsedReg RaiseContext::parseReg(MCRegister reg, int mciOpIdx) const {
  ParsedReg pr;
  if (!reg) {
    pr.kind = ParsedReg::NOREG;
    return pr;
  }

  const MCRegisterInfo &MRI = *mc.regInfo;

  // Width is computed on the as-decoded register: only the subtarget-
  // specific aliases (TTMPx_gfx9plus, FLAT_SCR_vi, ...) carry the correct
  // sub0/sub1/... chain from the disassembler.
  const int width = computeRegWidth32(MRI, reg);

  // Reduce everything to a canonical 32-bit pseudo for class/enum lookups:
  //   * sub0 on the as-decoded register picks the first 32-bit lane out
  //     of a tuple (sub-reg graph is authoritative on the real MC reg).
  //   * mc2PseudoReg then strips any subtarget suffix:
  //       TTMP8_gfx9plus         -> TTMP8
  //       FLAT_SCR_LO_vi         -> FLAT_SCR_LO
  //       SGPR_NULL64_gfx11plus  -> SGPR_NULL
  //       M0_gfx11plus           -> M0
  MCRegister lane = MRI.getSubReg(reg, AMDGPU::sub0);
  if (!lane)
    lane = reg;
  lane = AMDGPU::mc2PseudoReg(lane);

  switch (lane) {
  // Wave-mask registers. The ``_LO`` / ``_HI`` halves get the same
  // classification as the full pair; downstream VCC/EXEC handling routes
  // through loadVCC/storeVCC (which already respects wave size), so the
  // ``width`` field is informational here rather than load-bearing.
  case AMDGPU::VCC_LO:
  case AMDGPU::VCC_HI:
    pr.kind = ParsedReg::VCC;
    pr.width = isa.isWave32() ? 1 : 2;
    return pr;
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
    pr.kind = ParsedReg::EXEC;
    // baseIdx discriminates between the two 32-bit halves of wave64 EXEC
    // (0 = EXEC_LO, 1 = EXEC_HI). The full 64-bit pair also resolves here
    // via `sub0(EXEC) = EXEC_LO`, but `width = 2` tags it distinctly so
    // storeExec partial-write logic can route correctly.
    pr.baseIdx = (lane == AMDGPU::EXEC_HI) ? 1 : 0;
    pr.width = width;
    return pr;
  case AMDGPU::SCC:
    pr.kind = ParsedReg::SCC;
    pr.width = 1;
    return pr;
  case AMDGPU::MODE:
    pr.kind = ParsedReg::MODE;
    pr.width = 1;
    return pr;
  case AMDGPU::M0:
    pr.kind = ParsedReg::M0;
    pr.width = 1;
    return pr;
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
    pr.kind = ParsedReg::FLAT_SCR;
    pr.width = width;
    return pr;
  // GFX11+ uses SGPR_NULL / SGPR_NULL_HI (and the 64-bit pair SGPR_NULL64)
  // as carry-discard sinks, e.g. `v_mad_co_u64_u32 ..., null, ...`. They
  // have no backing slot — treat writes to them as no-ops.
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
    pr.kind = ParsedReg::NOREG;
    return pr;
  // XNACK_MASK controls page-fault retry masking per lane. On data-center
  // GPUs (MI300/MI350) XNACK is typically disabled and the register has no
  // effect on compute semantics. We treat it as NOREG (reads→zero,
  // writes→nop). If a kernel compiled with XNACK enabled relies on the
  // mask for correctness, this approximation is wrong — but such kernels
  // are not expected in practice.
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
    pr.kind = ParsedReg::NOREG;
    return pr;
  // LDS_DIRECT (src_lds_direct, enc 254): reads a dword from LDS at the
  // byte offset held in M0. Used as a VALU source after buffer_load_*_lds.
  case AMDGPU::LDS_DIRECT:
    pr.kind = ParsedReg::LDS_DIRECT;
    pr.width = 1;
    return pr;
  default:
    break;
  }

  // Family classification via the HW encoding flag bits. getEncodingValue
  // returns the correct HWEncoding payload for both pseudos and subtarget-
  // specific aliases. IS_VGPR (bit 10) and IS_AGPR (bit 11) are defined as
  // disjoint in SIRegisterInfo.td, so checking either first is correct;
  // AGPR goes first only because it is the more specific case.
  unsigned enc = MRI.getEncodingValue(reg);
  unsigned hwIdx = enc & AMDGPU::HWEncoding::REG_IDX_MASK;

  if (enc & AMDGPU::HWEncoding::IS_AGPR) {
    pr.kind = ParsedReg::AGPR;
    pr.baseIdx = hwIdx;
    pr.width = width;
    if (mciOpIdx >= 0 && (unsigned)mciOpIdx < kMaxOps)
      pr.baseIdx += currentVGPRAdjust[mciOpIdx];
    return pr;
  }
  if (enc & AMDGPU::HWEncoding::IS_VGPR) {
    pr.kind = ParsedReg::VGPR;
    pr.baseIdx = hwIdx;
    pr.width = width;
    if (mciOpIdx >= 0 && (unsigned)mciOpIdx < kMaxOps)
      pr.baseIdx += currentVGPRAdjust[mciOpIdx];
    return pr;
  }

  // TTMPs live at a generation-specific HW encoding (108+ on gfx9+ vs 112+
  // on gfx8), so we cannot use the raw encoding as the logical 0..15
  // index. Locate the lane inside TTMP_32RegClass instead; the class is
  // defined as `(add (sequence "TTMP%u", 0, 15))` so position == index.
  const MCRegisterClass &TTMP32 =
      MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  if (int idx = findIndexInClass(TTMP32, lane); idx >= 0) {
    pr.kind = ParsedReg::TTMP;
    pr.baseIdx = idx;
    pr.width = width;
    return pr;
  }

  // SGPR_32 is the narrow class for `SGPR0..SGPR105`; SReg_32 would also
  // include VCC_LO, EXEC_LO, FLAT_SCR_LO, M0, TTMP_32, SGPR_NULL, and the
  // SRC_* inline-value registers, which we have already ruled out above.
  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(lane)) {
    pr.kind = ParsedReg::SGPR;
    pr.baseIdx = hwIdx;
    pr.width = width;
    return pr;
  }

  report_fatal_error(Twine("transpiler: parseReg could not classify '") +
                     MRI.getName(reg) + "' (enc=0x" +
                     Twine::utohexstr(enc) + ")");
}

Value *RaiseContext::readOp32(const DecodedInst &di, unsigned opIdx) {
  if (di.isReg(opIdx)) {
    ParsedReg pr = parseReg(di.getReg(opIdx), opIdx);
    if (pr.kind == ParsedReg::VCC) {
      // Reading VCC as an i32 (wave32 wave-mask, or low 32 bits on
      // wave64) is a cross-lane collection: emit amdgcn.ballot so each
      // lane gets the same bit-mask assembled from all lanes' per-lane
      // VCC bits. On wave64 this is the low 32 lanes; upper-half reads
      // are separately materialised via readOp64.
      return regs.readVCCAsWaveMask(B, i32Ty);
    }
    if (pr.kind == ParsedReg::EXEC) {
      Value *v = regs.loadExec(B);
      if (v->getType() == i32Ty)
        return v;
      if (pr.width < 2 && pr.baseIdx == 1)
        v = B.CreateLShr(v, 32, "exec_hi_shr");
      return B.CreateTrunc(v, i32Ty,
                            (pr.width < 2 && pr.baseIdx == 1) ? "exec_hi"
                                                               : "exec_lo");
    }
    if (pr.kind == ParsedReg::SCC)
      return B.CreateZExt(regs.loadSCC(B), i32Ty);
    if (pr.kind == ParsedReg::NOREG)
      return ConstantInt::get(i32Ty, 0);
    if (pr.kind == ParsedReg::MODE)
      return ConstantInt::get(i32Ty, 0);
    Value *v = regs.readReg32(B, pr);
    if (!v) {
      errs() << "transpiler: unreadable register '"
             << mc.regInfo->getName(di.getReg(opIdx)) << "' in " << di.mnemonic
             << "\n";
      return UndefValue::get(i32Ty);
    }
    return v;
  }
  if (di.isImm(opIdx))
    return ConstantInt::get(i32Ty,
                            (uint32_t)(di.getImm(opIdx) & 0xFFFFFFFF));
  if (opIdx < di.numOps() && di.inst.getOperand(opIdx).isExpr()) {
    int64_t val = 0;
    if (di.inst.getOperand(opIdx).getExpr()->evaluateAsAbsolute(val))
      return ConstantInt::get(i32Ty, (uint32_t)(val & 0xFFFFFFFF));
    return ConstantInt::get(i32Ty, 0);
  }
  errs() << "transpiler: readOp32 unresolvable operand " << opIdx << " in "
         << di.mnemonic << "\n";
  return UndefValue::get(i32Ty);
}

Value *RaiseContext::readOp64(const DecodedInst &di, unsigned opIdx) {
  if (di.isReg(opIdx)) {
    ParsedReg pr = parseReg(di.getReg(opIdx), opIdx);
    if (pr.kind == ParsedReg::VCC)
      return regs.readVCCAsWaveMask(B, i64Ty);
    if (pr.kind == ParsedReg::EXEC) {
      Value *v = regs.loadExec(B);
      if (v->getType() != i64Ty)
        v = B.CreateZExt(v, i64Ty, "exec_ext");
      return v;
    }
    Value *v = regs.readReg64(B, pr);
    if (!v) {
      errs() << "transpiler: unreadable register64 '"
             << mc.regInfo->getName(di.getReg(opIdx)) << "' in " << di.mnemonic
             << "\n";
      return UndefValue::get(i64Ty);
    }
    return v;
  }
  if (di.isImm(opIdx))
    return ConstantInt::getSigned(i64Ty, di.getImm(opIdx));
  if (opIdx < di.numOps() && di.inst.getOperand(opIdx).isExpr()) {
    int64_t val = 0;
    di.inst.getOperand(opIdx).getExpr()->evaluateAsAbsolute(val);
    return ConstantInt::getSigned(i64Ty, val);
  }
  errs() << "transpiler: readOp64 unresolvable operand " << opIdx << " in "
         << di.mnemonic << "\n";
  return UndefValue::get(i64Ty);
}

Value *RaiseContext::emitUpdateDpp(Value *oldVal, Value *src, uint16_t ctrl,
                                    uint8_t rowMask, uint8_t bankMask,
                                    bool boundCtrl) {
  // CROSS_LANE_SURVEY.md P5: lift the DPP src-pathway modifier through
  // `llvm.amdgcn.update.dpp`. The intrinsic is type-overloaded
  // (`llvm_any_ty`). We route through integer overloads sized to match
  // the input's bit-width and bitcast through when the input is a
  // same-width non-integer type (f32 / <2 x i16> / etc.); codegen
  // picks the same DPP lowering for all same-width overloads.
  //
  // Widths supported today: 32-bit and 64-bit, matching the hardware
  // DPP encoding families (VOP_DPP and VOP_DPP_64). Other widths
  // `report_fatal_error` — the tsFlags DPP bit can only be set on
  // VOP1/VOP2/VOP3 classes whose operands are always 32-bit or 64-bit
  // in AMDGPU's ISA, so any other width is a decoder/tblgen drift
  // situation worth surfacing loudly rather than silently downgrading.
  assert(oldVal->getType() == src->getType() &&
         "emitUpdateDpp: old and src must have matching types");
  Type *origTy = src->getType();
  const unsigned bits = origTy->getPrimitiveSizeInBits();
  Type *intTy = nullptr;
  if (bits == 32)
    intTy = i32Ty;
  else if (bits == 64)
    intTy = i64Ty;
  else
    report_fatal_error(
        "emitUpdateDpp: unsupported DPP operand width (expected 32 or 64 "
        "bits). Extend the bit-width dispatch below when a new DPP "
        "operand width lands in AMDGPU's ISA.");
  auto toIntTy = [&](Value *v) {
    return v->getType() == intTy ? v : B.CreateBitCast(v, intTy);
  };
  Value *oldInt = toIntTy(oldVal);
  Value *srcInt = toIntTy(src);
  Function *fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_update_dpp, {intTy});
  Value *result =
      B.CreateCall(fn, {oldInt, srcInt, B.getInt32(ctrl),
                         B.getInt32(rowMask), B.getInt32(bankMask),
                         B.getInt1(boundCtrl)},
                   "dpp");
  if (result->getType() != origTy)
    result = B.CreateBitCast(result, origTy);
  return result;
}

Value *RaiseContext::emitLaneIdx() {
  // Per-BB memoisation, mirroring `emitLaneActiveBit` below — the
  // `mbcnt_lo` (and on wave64, `mbcnt_hi`) pair WaveProjection emits
  // is invariant within a basic block for a given EXEC value, and
  // the cached SSA value dominates anything emitted later in the
  // same BB.
  //
  // Invalidation: shared with the lane_active cache via
  // `resetLaneActiveCache`, which the main raiser loop fires at
  // every source-instruction boundary. See `emitLaneActiveBit`'s
  // comment block below for the dominance argument that keeps reuse
  // safe across `emitUnderExec` diamonds within a single source
  // instruction's emission.
  if (cachedLaneIdx && B.GetInsertBlock() == cachedLaneIdxBB)
    return cachedLaneIdx;
  cachedLaneIdx = projection.emitLaneIdx(B);
  cachedLaneIdxBB = B.GetInsertBlock();
  return cachedLaneIdx;
}

Value *RaiseContext::emitLaneActiveBit() {
  // Memoisation (see RaiseContext::resetLaneActiveCache docs).
  //
  // Dominance argument for reusing a cached i1 across blocks within a
  // single source instruction's emission:
  //
  //   Each emitUnderExec diamond is structurally linear:
  //
  //     preBB ──┬─▶ doBB ──▶ skipBB
  //             └────────────▶ skipBB   (the conditional skip edge)
  //
  //   Chaining N emitUnderExecs yields
  //     preBB → (doBB1 → skipBB1) → (doBB2 → skipBB2) → … → skipBBN
  //
  //   where every successor BB has preBB on its dominator path. So an
  //   i1 defined in preBB dominates every subsequent doBB/skipBB emitted
  //   by the same source-instruction handler.
  //
  // The reuse invariant is therefore maintained by the invalidation
  // contract alone:
  //   * The raiser main loop calls `resetLaneActiveCache` at every
  //     source-instruction boundary, covering (a) possible EXEC writes
  //     by the prior instruction and (b) jumps into offset-keyed
  //     named BBs where the prior `active` no longer dominates.
  //   * `ctx.storeExec` resets on EXEC mutation.
  //   * Any handler that writes EXEC via a lower-level path is
  //     responsible for calling `resetLaneActiveCache` itself (the
  //     allow-list audit in raiser.cpp documents that all such sites
  //     route through `ctx.storeExec` or `writeReg*`→`storeExec`).
  //
  // Consequently, a cache *hit* is valid regardless of whether the
  // current BB equals `cachedLaneActiveBB`.
  if (cachedLaneActive)
    return cachedLaneActive;

  // The projection owns the modulo-replication math; this context only
  // handles the cache + EXEC load.
  Value *active = projection.emitLaneActiveBit(B, regs.loadExec(B));
  cachedLaneActive = active;
  cachedLaneActiveBB = B.GetInsertBlock();
  return active;
}

void RaiseContext::writeReg32(ParsedReg pr, Value *v) {
  if (pr.kind == ParsedReg::VGPR || pr.kind == ParsedReg::AGPR) {
    emitUnderExec([&] { regs.writeReg32(B, pr, v); });
  } else {
    regs.writeReg32(B, pr, v);
    // regs.writeReg32 dispatches to storeExec when pr.kind == EXEC. The
    // lane_active memo must be invalidated so subsequent emitUnderExec
    // calls recompute against the new EXEC value rather than the pre-
    // write snapshot. See resetLaneActiveCache docs in raise_context.hpp.
    if (pr.kind == ParsedReg::EXEC)
      resetLaneActiveCache();
  }
}

void RaiseContext::writeReg64(ParsedReg pr, Value *v) {
  if (pr.kind == ParsedReg::VGPR || pr.kind == ParsedReg::AGPR) {
    emitUnderExec([&] { regs.writeReg64(B, pr, v); });
  } else {
    regs.writeReg64(B, pr, v);
    if (pr.kind == ParsedReg::EXEC)
      resetLaneActiveCache();
  }
}

void RaiseContext::writeRegVec(ParsedReg pr, Value *v) {
  if (pr.kind == ParsedReg::VGPR || pr.kind == ParsedReg::AGPR) {
    emitUnderExec([&] { regs.writeRegVec(B, pr, v); });
  } else {
    // Vector SGPR writes can't target EXEC (EXEC is scalar/pair, never
    // vector), so no cache invalidation is needed.
    regs.writeRegVec(B, pr, v);
  }
}

void RaiseContext::writeRegExecWidth(ParsedReg pr, Value *v) {
  // Wave-level commit. SGPR-pair / VCC / EXEC writes carry the wave mask
  // itself and are computed cross-lane (ballot / sext-i1 today), so they
  // must not be predicated on the per-lane EXEC bit.
  regs.writeRegExecWidth(B, pr, v);
  if (pr.kind == ParsedReg::EXEC)
    resetLaneActiveCache();
}

void RaiseContext::storeVGPR32(int idx, Value *v) {
  emitUnderExec([&] { regs.storeVGPR32(B, idx, v); });
}

void RaiseContext::storeVGPR64(int idx, Value *v) {
  emitUnderExec([&] { regs.storeVGPR64(B, idx, v); });
}

void RaiseContext::storeAGPR32(int idx, Value *v) {
  emitUnderExec([&] { regs.storeAGPR32(B, idx, v); });
}

void RaiseContext::emitUnderExec(llvm::function_ref<void()> body) {
  Value *active = emitLaneActiveBit();
  BasicBlock *preBB = B.GetInsertBlock();
  Function *F = preBB->getParent();
  BasicBlock *doBB = BasicBlock::Create(C, "spe_do", F);
  BasicBlock *skipBB = BasicBlock::Create(C, "spe_skip", F);
  B.CreateCondBr(active, doBB, skipBB);

  B.SetInsertPoint(doBB);
  body();
  // `body()` normally falls through without terminating. If a handler ever
  // ends its emission with an unconditional control-flow op (shouldn't
  // happen for the side-effectful ops we wrap, but defensively handled),
  // don't double-terminate doBB.
  if (!B.GetInsertBlock()->getTerminator())
    B.CreateBr(skipBB);

  B.SetInsertPoint(skipBB);
}


Value *RaiseContext::readOpExecWidth(const DecodedInst &di, unsigned opIdx) {
  if (di.isReg(opIdx)) {
    ParsedReg pr = parseReg(di.getReg(opIdx), opIdx);
    if (pr.kind == ParsedReg::VCC)
      return regs.readVCCAsWaveMask(B, regs.execTy);
    if (pr.kind == ParsedReg::EXEC)
      return regs.loadExec(B);
    if (pr.kind == ParsedReg::SGPR) {
      if (isa.isWave32())
        return regs.loadSGPR32(B, pr.baseIdx);
      return regs.loadSGPR64(B, pr.baseIdx);
    }
    errs() << "transpiler: readOpExecWidth unresolvable register '"
           << mc.regInfo->getName(di.getReg(opIdx)) << "' in " << di.mnemonic
           << "\n";
    return UndefValue::get(regs.execTy);
  }
  if (di.isImm(opIdx))
    return ConstantInt::getSigned(regs.execTy, di.getImm(opIdx));
  if (opIdx < di.numOps() && di.inst.getOperand(opIdx).isExpr()) {
    int64_t val = 0;
    di.inst.getOperand(opIdx).getExpr()->evaluateAsAbsolute(val);
    return ConstantInt::getSigned(regs.execTy, val);
  }
  errs() << "transpiler: readOpExecWidth unresolvable operand " << opIdx
         << " in " << di.mnemonic << "\n";
  return UndefValue::get(regs.execTy);
}

} // namespace transpiler
