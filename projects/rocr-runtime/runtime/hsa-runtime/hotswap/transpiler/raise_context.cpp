#include "raise_context.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::VCC, AMDGPU::EXEC, ...
#include "SIDefines.h"                        // AMDGPU::HWEncoding::*
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
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

// Count how many 32-bit sub-registers make up `reg`. A plain 32-bit register
// has no sub0 (getSubReg returns 0) and is reported as width 1. Tuples
// (SGPR0_SGPR1, VReg_128, ...) walk sub0, sub1, ... until exhausted.
static int computeRegWidth32(const MCRegisterInfo &MRI, MCRegister reg) {
  int w = 0;
  for (unsigned subIdx = AMDGPU::sub0; /**/; ++subIdx) {
    if (!MRI.getSubReg(reg, subIdx))
      break;
    ++w;
  }
  return w ? w : 1;
}

ParsedReg RaiseContext::parseReg(unsigned reg, int mciOpIdx) const {
  ParsedReg pr;
  if (reg == 0) {
    pr.kind = ParsedReg::NOREG;
    return pr;
  }

  const MCRegisterInfo &MRI = *mc.regInfo;

  // Width is computed on the as-decoded register: only the subtarget-
  // specific aliases (TTMPx_gfx9plus, FLAT_SCR_vi, ...) carry the correct
  // sub0/sub1/... chain from the disassembler.
  const int width = computeRegWidth32(MRI, reg);

  // Normalise away subtarget-specific aliases so the switch below can use
  // target-agnostic register constants:
  //   TTMP8_gfx9plus            -> TTMP8
  //   FLAT_SCR_LO_vi            -> FLAT_SCR_LO
  //   SGPR_NULL64_gfx11plus     -> SGPR_NULL
  //   M0_gfx11plus              -> M0
  MCRegister pseudo = AMDGPU::mc2PseudoReg(MCRegister(reg));

  // A tuple (e.g. VReg_64) classifies as whatever its first 32-bit lane is.
  // For a 32-bit register getSubReg returns 0 and we use the register
  // itself. mc2PseudoReg on a tuple keeps it a tuple, so we need sub0 here.
  MCRegister lane = MRI.getSubReg(pseudo, AMDGPU::sub0);
  if (!lane)
    lane = pseudo;
  lane = AMDGPU::mc2PseudoReg(lane);

  switch (lane) {
  case AMDGPU::VCC_LO:
  case AMDGPU::VCC_HI:
    pr.kind = ParsedReg::VCC;
    pr.width = isa.isWave32() ? 1 : 2;
    return pr;
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
    pr.kind = ParsedReg::EXEC;
    pr.width = isa.isWave32() ? 1 : 2;
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
    pr.width = width; // 2 for the pair, 1 for a single half
    return pr;
  // GFX11+ uses SGPR_NULL / SGPR_NULL_HI (and the 64-bit pair SGPR_NULL64)
  // as carry-discard sinks, e.g. `v_mad_co_u64_u32 ..., null, ...`. They
  // have no backing slot — treat writes to them as no-ops.
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
    pr.kind = ParsedReg::NOREG;
    return pr;
  default:
    break;
  }

  // Family classification via the HW encoding flag bits. `getEncodingValue`
  // on the as-decoded register returns the correct HWEncoding payload even
  // for subtarget-specific aliases.
  unsigned enc = MRI.getEncodingValue(reg);
  unsigned hwIdx = enc & AMDGPU::HWEncoding::REG_IDX_MASK;

  // AGPR must be checked before VGPR: on gfx90a the AGPR encoding can set
  // both IS_VGPR and IS_AGPR (the shared VGPR/AGPR operand slot), so
  // checking IS_AGPR first disambiguates.
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

  // TTMPs live at a generation-specific HW encoding (108+ or 112+). Use
  // the AMDGPU::TTMP<N> pseudo values as a source of truth so we report the
  // 0..15 logical index expected by the TTMP register file.
  if (lane >= AMDGPU::TTMP0 && lane <= AMDGPU::TTMP15) {
    pr.kind = ParsedReg::TTMP;
    pr.baseIdx = lane - AMDGPU::TTMP0;
    pr.width = width;
    return pr;
  }

  // SReg_32RegClass is the canonical 32-bit scalar set (SGPR0..SGPR105).
  // We reach here only after ruling out special scalars above.
  if (MRI.getRegClass(AMDGPU::SReg_32RegClassID).contains(lane)) {
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
      Value *v = regs.loadVCC(B);
      return B.CreateSExt(v, i32Ty);
    }
    if (pr.kind == ParsedReg::EXEC) {
      Value *v = regs.loadExec(B);
      if (v->getType() != i32Ty)
        v = B.CreateTrunc(v, i32Ty, "exec_lo");
      return v;
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
      return B.CreateSExt(regs.loadVCC(B), i64Ty);
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

Value *RaiseContext::readOpExecWidth(const DecodedInst &di, unsigned opIdx) {
  if (di.isReg(opIdx)) {
    ParsedReg pr = parseReg(di.getReg(opIdx), opIdx);
    if (pr.kind == ParsedReg::VCC)
      return B.CreateSExt(regs.loadVCC(B), regs.execTy);
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
