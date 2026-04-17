#include "raise_context.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/StringExtras.h"
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

ParsedReg RaiseContext::parseReg(unsigned reg, int mciOpIdx) const {
  ParsedReg pr;
  if (reg == 0) {
    pr.kind = ParsedReg::NOREG;
    return pr;
  }
  StringRef name = mc.regInfo->getName(reg);

  if (name.starts_with("AGPR")) {
    pr.kind = ParsedReg::AGPR;
    name.substr(4).split('_').first.getAsInteger(10, pr.baseIdx);
    pr.width = name.count("AGPR");
    if (mciOpIdx >= 0 && (unsigned)mciOpIdx < kMaxOps)
      pr.baseIdx += currentVGPRAdjust[mciOpIdx];
    return pr;
  }
  if (name.starts_with("SGPR")) {
    // `SGPR_NULL` / `SGPR_NULL_HI` are the sink registers GFX11+ uses for
    // carry-discard destinations (e.g. `v_mad_co_u64_u32 ..., null, ...`).
    // They are not regular SGPR files and have no backing slot; treat them
    // as NOREG so handlers skip the write.
    if (name.starts_with("SGPR_NULL")) {
      pr.kind = ParsedReg::NOREG;
      return pr;
    }
    pr.kind = ParsedReg::SGPR;
    if (name.substr(4).split('_').first.getAsInteger(10, pr.baseIdx))
      llvm::report_fatal_error(llvm::Twine("transpiler: unparseable SGPR "
                                           "register name '") +
                               name + "'");
    pr.width = name.count("SGPR");
    return pr;
  }
  if (name.starts_with("VGPR")) {
    pr.kind = ParsedReg::VGPR;
    name.substr(4).split('_').first.getAsInteger(10, pr.baseIdx);
    pr.width = name.count("VGPR");
    if (mciOpIdx >= 0 && (unsigned)mciOpIdx < kMaxOps)
      pr.baseIdx += currentVGPRAdjust[mciOpIdx];
    return pr;
  }
  if (name.starts_with("VCC")) {
    pr.kind = ParsedReg::VCC;
    pr.width = isa.isWave32() ? 1 : 2;
    return pr;
  }
  if (name.starts_with("EXEC")) {
    pr.kind = ParsedReg::EXEC;
    pr.width = isa.isWave32() ? 1 : 2;
    return pr;
  }
  if (name == "SCC") {
    pr.kind = ParsedReg::SCC;
    pr.width = 1;
    return pr;
  }
  if (name == "MODE") {
    pr.kind = ParsedReg::MODE;
    pr.width = 1;
    return pr;
  }
  if (name.starts_with("M0")) {
    pr.kind = ParsedReg::M0;
    pr.baseIdx = 0;
    pr.width = 1;
    return pr;
  }
  if (name.starts_with("FLAT_SCR")) {
    pr.kind = ParsedReg::FLAT_SCR;
    pr.baseIdx = 0;
    pr.width = name.contains("_") ? 2 : 1;
    return pr;
  }
  if (name.starts_with("TTMP")) {
    pr.kind = ParsedReg::TTMP;
    StringRef numStr = name.substr(4).split('_').first;
    numStr.getAsInteger(10, pr.baseIdx);
    pr.width = name.count("TTMP");
    return pr;
  }
  return pr;
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
