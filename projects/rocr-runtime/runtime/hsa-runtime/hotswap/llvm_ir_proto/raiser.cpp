#include "raiser.hpp"
#include "amdgpu_formats.hpp"
#include "code_object_utils.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"

#include <map>
#include <set>

using namespace llvm;

namespace ir_proto {

namespace {

// ============================================================================
// MC State
// ============================================================================

struct MCState {
  const Target *target = nullptr;
  std::unique_ptr<MCInstrInfo> instrInfo;
  std::unique_ptr<MCRegisterInfo> regInfo;
  std::unique_ptr<MCSubtargetInfo> subtargetInfo;
  std::unique_ptr<const MCAsmInfo> asmInfo;
  std::unique_ptr<MCContext> ctx;
  std::unique_ptr<MCDisassembler> disasm;
  std::unique_ptr<MCInstPrinter> printer;
};

bool initMCState(MCState &state, const std::string &targetISA) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  Triple triple("amdgcn-amd-amdhsa");
  std::string error;
  state.target = TargetRegistry::lookupTarget(triple, error);
  if (!state.target) {
    errs() << "ir_proto: Target lookup failed: " << error << "\n";
    return false;
  }

  state.instrInfo.reset(state.target->createMCInstrInfo());
  state.regInfo.reset(state.target->createMCRegInfo(triple));
  state.subtargetInfo.reset(
      state.target->createMCSubtargetInfo(triple, targetISA, ""));
  state.asmInfo.reset(state.target->createMCAsmInfo(
      *state.regInfo, triple, MCTargetOptions()));
  state.ctx = std::make_unique<MCContext>(triple, state.asmInfo.get(),
                                         state.regInfo.get(),
                                         state.subtargetInfo.get());
  state.disasm.reset(
      state.target->createMCDisassembler(*state.subtargetInfo, *state.ctx));
  state.printer.reset(state.target->createMCInstPrinter(
      triple, 0, *state.asmInfo, *state.instrInfo, *state.regInfo));
  state.printer->setPrintImmHex(true);
  return true;
}

std::string getMnemonic(const MCState &mc, const MCInst &inst) {
  std::string s;
  raw_string_ostream os(s);
  mc.printer->printInst(&inst, 0, "", *mc.subtargetInfo, os);
  StringRef sr(s);
  sr = sr.ltrim();
  return sr.split('\t').first.split(' ').first.str();
}

StringRef stripEncoding(StringRef mn) {
  for (StringRef suffix : {"_e32", "_e64", "_vi"})
    if (mn.ends_with(suffix))
      return mn.drop_back(suffix.size());
  return mn;
}

// ============================================================================
// Decoded instruction
// ============================================================================

struct DecodedInst {
  std::string mnemonic;
  std::string rawMnemonic;
  MCInst inst;
  unsigned numDefs = 0;
  bool isBranch = false;
  bool isConditionalBranch = false;
  uint64_t offset = 0;
  uint64_t size = 0;

  // Phase 2: MC metadata
  uint64_t tsFlags = 0;
  FormatKind format = FormatKind::Unknown;
  bool defsSCC = false;
  bool defsVCC = false;
  bool defsEXEC = false;
  unsigned firstSrcIdx = 0;

  static constexpr unsigned kMaxSrcs = 16;
  unsigned srcMap[kMaxSrcs] = {};
  // Index of the INPUT_MODS operand preceding each srcMap entry, or UINT_MAX if none.
  unsigned modMap[kMaxSrcs] = {};
  unsigned numSrcs = 0;

  unsigned numOps() const { return inst.getNumOperands(); }
  bool isReg(unsigned i) const {
    return i < numOps() && inst.getOperand(i).isReg();
  }
  bool isImm(unsigned i) const {
    return i < numOps() && inst.getOperand(i).isImm();
  }
  unsigned getReg(unsigned i) const { return inst.getOperand(i).getReg(); }
  int64_t getImm(unsigned i) const { return inst.getOperand(i).getImm(); }
};

// ============================================================================
// Register classification
// ============================================================================

struct ParsedReg {
  enum Kind { SGPR, VGPR, AGPR, VCC, EXEC, SCC, MODE, M0, FLAT_SCR, NOREG, OTHER };
  Kind kind = OTHER;
  int baseIdx = -1;
  int width = 1;
};

ParsedReg parseReg(const MCState &mc, unsigned reg) {
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
    return pr;
  }
  if (name.starts_with("SGPR")) {
    pr.kind = ParsedReg::SGPR;
    name.substr(4).split('_').first.getAsInteger(10, pr.baseIdx);
    pr.width = name.count("SGPR");
    return pr;
  }
  if (name.starts_with("VGPR")) {
    pr.kind = ParsedReg::VGPR;
    name.substr(4).split('_').first.getAsInteger(10, pr.baseIdx);
    pr.width = name.count("VGPR");
    return pr;
  }
  if (name.starts_with("VCC")) {
    pr.kind = ParsedReg::VCC;
    pr.width = 2;
    return pr;
  }
  if (name.starts_with("EXEC")) {
    pr.kind = ParsedReg::EXEC;
    pr.width = 2;
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
  return pr;
}

// ============================================================================
// Alloca-based register file for SSA construction via PromoteMemToReg
// ============================================================================

struct AllocaRegFile {
  static constexpr int MAX_SGPR = 106;
  static constexpr int MAX_VGPR = 256;
  AllocaInst *sgpr[106] = {};
  AllocaInst *vgpr[256] = {};
  AllocaInst *agpr[256] = {};
  AllocaInst *vcc = nullptr;
  AllocaInst *scc = nullptr;
  AllocaInst *m0 = nullptr;
  AllocaInst *flatScr[2] = {};

  void init(IRBuilder<> &B, Type *i32Ty, Type *i1Ty) {
    for (int i = 0; i < MAX_SGPR; i++)
      sgpr[i] = B.CreateAlloca(i32Ty, nullptr, "sgpr" + std::to_string(i));
    for (int i = 0; i < MAX_VGPR; i++)
      vgpr[i] = B.CreateAlloca(i32Ty, nullptr, "vgpr" + std::to_string(i));
    for (int i = 0; i < MAX_VGPR; i++)
      agpr[i] = B.CreateAlloca(i32Ty, nullptr, "agpr" + std::to_string(i));
    vcc = B.CreateAlloca(i1Ty, nullptr, "vcc");
    scc = B.CreateAlloca(i1Ty, nullptr, "scc");
    m0 = B.CreateAlloca(i32Ty, nullptr, "m0");
    flatScr[0] = B.CreateAlloca(i32Ty, nullptr, "flat_scr_lo");
    flatScr[1] = B.CreateAlloca(i32Ty, nullptr, "flat_scr_hi");
  }

  void storeSGPR32(IRBuilder<> &B, int idx, Value *v) {
    Type *i32Ty = B.getInt32Ty();
    if (v->getType() != i32Ty)
      v = B.CreateBitCast(v, i32Ty);
    B.CreateStore(v, sgpr[idx]);
  }
  Value *loadSGPR32(IRBuilder<> &B, int idx) {
    return B.CreateLoad(B.getInt32Ty(), sgpr[idx]);
  }
  void storeSGPR64(IRBuilder<> &B, int idx, Value *v) {
    Type *i32Ty = B.getInt32Ty();
    Type *i64Ty = B.getInt64Ty();
    if (v->getType()->isPointerTy())
      v = B.CreatePtrToInt(v, i64Ty);
    if (v->getType() != i64Ty)
      v = B.CreateBitCast(v, i64Ty);
    Value *lo = B.CreateTrunc(v, i32Ty);
    Value *hi = B.CreateTrunc(B.CreateLShr(v, 32), i32Ty);
    B.CreateStore(lo, sgpr[idx]);
    B.CreateStore(hi, sgpr[idx + 1]);
  }
  Value *loadSGPR64(IRBuilder<> &B, int idx) {
    Type *i32Ty = B.getInt32Ty();
    Type *i64Ty = B.getInt64Ty();
    Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, sgpr[idx]), i64Ty);
    Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, sgpr[idx + 1]), i64Ty);
    return B.CreateOr(lo, B.CreateShl(hi, 32));
  }
  void storeVGPR32(IRBuilder<> &B, int idx, Value *v) {
    Type *i32Ty = B.getInt32Ty();
    if (v->getType() != i32Ty) {
      if (v->getType()->isPointerTy())
        v = B.CreatePtrToInt(v, B.getInt64Ty());
      if (v->getType() == B.getFloatTy())
        v = B.CreateBitCast(v, i32Ty);
      else if (v->getType() != i32Ty)
        v = B.CreateTrunc(v, i32Ty);
    }
    B.CreateStore(v, vgpr[idx]);
  }
  Value *loadVGPR32(IRBuilder<> &B, int idx) {
    return B.CreateLoad(B.getInt32Ty(), vgpr[idx]);
  }
  void storeVGPR64(IRBuilder<> &B, int idx, Value *v) {
    Type *i32Ty = B.getInt32Ty();
    Type *i64Ty = B.getInt64Ty();
    if (v->getType()->isPointerTy())
      v = B.CreatePtrToInt(v, i64Ty);
    if (v->getType() != i64Ty)
      v = B.CreateBitCast(v, i64Ty);
    Value *lo = B.CreateTrunc(v, i32Ty);
    Value *hi = B.CreateTrunc(B.CreateLShr(v, 32), i32Ty);
    B.CreateStore(lo, vgpr[idx]);
    B.CreateStore(hi, vgpr[idx + 1]);
  }
  Value *loadVGPR64(IRBuilder<> &B, int idx) {
    Type *i32Ty = B.getInt32Ty();
    Type *i64Ty = B.getInt64Ty();
    Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, vgpr[idx]), i64Ty);
    Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, vgpr[idx + 1]), i64Ty);
    return B.CreateOr(lo, B.CreateShl(hi, 32));
  }
  void storeAGPR32(IRBuilder<> &B, int idx, Value *v) {
    Type *i32Ty = B.getInt32Ty();
    if (v->getType() != i32Ty)
      v = B.CreateBitCast(v, i32Ty);
    B.CreateStore(v, agpr[idx]);
  }
  Value *loadAGPR32(IRBuilder<> &B, int idx) {
    return B.CreateLoad(B.getInt32Ty(), agpr[idx]);
  }
  void storeVCC(IRBuilder<> &B, Value *v) {
    if (v->getType() != B.getInt1Ty())
      v = B.CreateICmpNE(v, Constant::getNullValue(v->getType()));
    B.CreateStore(v, vcc);
  }
  Value *loadVCC(IRBuilder<> &B) {
    return B.CreateLoad(B.getInt1Ty(), vcc);
  }
  void storeSCC(IRBuilder<> &B, Value *v) {
    if (v->getType() != B.getInt1Ty())
      v = B.CreateICmpNE(v, Constant::getNullValue(v->getType()));
    B.CreateStore(v, scc);
  }
  Value *loadSCC(IRBuilder<> &B) {
    return B.CreateLoad(B.getInt1Ty(), scc);
  }

  Value *readReg32(IRBuilder<> &B, ParsedReg pr) {
    if (pr.kind == ParsedReg::SGPR) return loadSGPR32(B, pr.baseIdx);
    if (pr.kind == ParsedReg::VGPR) return loadVGPR32(B, pr.baseIdx);
    if (pr.kind == ParsedReg::AGPR) return loadAGPR32(B, pr.baseIdx);
    if (pr.kind == ParsedReg::M0) return B.CreateLoad(B.getInt32Ty(), m0, "m0_val");
    if (pr.kind == ParsedReg::FLAT_SCR) return B.CreateLoad(B.getInt32Ty(), flatScr[0], "fscr_val");
    return nullptr;
  }
  Value *readReg64(IRBuilder<> &B, ParsedReg pr) {
    if (pr.kind == ParsedReg::SGPR) return loadSGPR64(B, pr.baseIdx);
    if (pr.kind == ParsedReg::VGPR) return loadVGPR64(B, pr.baseIdx);
    if (pr.kind == ParsedReg::VCC)
      return B.CreateSExt(loadVCC(B), B.getInt64Ty());
    if (pr.kind == ParsedReg::EXEC)
      return ConstantInt::getSigned(B.getInt64Ty(), -1);
    if (pr.kind == ParsedReg::M0)
      return B.CreateZExt(B.CreateLoad(B.getInt32Ty(), m0, "m0_val"), B.getInt64Ty());
    if (pr.kind == ParsedReg::FLAT_SCR) {
      Type *i32Ty = B.getInt32Ty();
      Type *i64Ty = B.getInt64Ty();
      Value *lo = B.CreateZExt(B.CreateLoad(i32Ty, flatScr[0]), i64Ty);
      Value *hi = B.CreateZExt(B.CreateLoad(i32Ty, flatScr[1]), i64Ty);
      return B.CreateOr(lo, B.CreateShl(hi, 32), "fscr64");
    }
    return nullptr;
  }
  void writeReg32(IRBuilder<> &B, ParsedReg pr, Value *v) {
    if (pr.kind == ParsedReg::SGPR) storeSGPR32(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::VGPR) storeVGPR32(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::AGPR) storeAGPR32(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::M0) {
      if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
      B.CreateStore(v, m0);
    }
    else if (pr.kind == ParsedReg::FLAT_SCR) {
      if (v->getType() != B.getInt32Ty()) v = B.CreateBitCast(v, B.getInt32Ty());
      B.CreateStore(v, flatScr[0]);
    }
  }
  void writeReg64(IRBuilder<> &B, ParsedReg pr, Value *v) {
    if (pr.kind == ParsedReg::SGPR) storeSGPR64(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::VGPR) storeVGPR64(B, pr.baseIdx, v);
    else if (pr.kind == ParsedReg::VCC) {
      Type *i64Ty = B.getInt64Ty();
      if (v->getType() != i64Ty)
        v = B.CreateBitOrPointerCast(v, i64Ty);
      storeVCC(B, B.CreateICmpNE(v, ConstantInt::get(i64Ty, 0)));
    }
    else if (pr.kind == ParsedReg::EXEC) { /* exec writes are no-ops in scalar model */ }
    else if (pr.kind == ParsedReg::FLAT_SCR) {
      Type *i32Ty = B.getInt32Ty();
      Type *i64Ty = B.getInt64Ty();
      if (v->getType() != i64Ty) v = B.CreateBitOrPointerCast(v, i64Ty);
      B.CreateStore(B.CreateTrunc(v, i32Ty), flatScr[0]);
      B.CreateStore(B.CreateTrunc(B.CreateLShr(v, 32), i32Ty), flatScr[1]);
    }
  }

  // Read/write N dwords as a vector from contiguous VGPRs/AGPRs
  Value *readRegVec(IRBuilder<> &B, ParsedReg pr, Type *vecTy) {
    unsigned n = vecTy->isVectorTy()
        ? cast<FixedVectorType>(vecTy)->getNumElements()
        : 1;
    Type *elemTy = vecTy->isVectorTy()
        ? cast<FixedVectorType>(vecTy)->getElementType()
        : vecTy;
    unsigned dwordsPerElem = elemTy->getPrimitiveSizeInBits() / 32;
    if (dwordsPerElem == 0) dwordsPerElem = 1;

    if (n == 1 && !vecTy->isVectorTy() && vecTy->getPrimitiveSizeInBits() <= 32) {
      Value *v = readReg32(B, pr);
      if (v->getType() != vecTy) v = B.CreateBitCast(v, vecTy);
      return v;
    }

    unsigned totalDwords = 0;
    if (elemTy->isFloatTy()) totalDwords = n;
    else if (elemTy->isIntegerTy(32)) totalDwords = n;
    else if (elemTy->isHalfTy()) totalDwords = (n + 1) / 2;
    else totalDwords = (n * elemTy->getPrimitiveSizeInBits() + 31) / 32;

    // Load all dwords
    SmallVector<Value *, 16> dwords;
    for (unsigned i = 0; i < totalDwords; i++) {
      ParsedReg sub = pr;
      sub.baseIdx = pr.baseIdx + i;
      sub.width = 1;
      dwords.push_back(readReg32(B, sub));
    }

    // Bitcast the dwords into the target vector type
    Type *i32Ty = B.getInt32Ty();
    unsigned totalBits = totalDwords * 32;
    Type *intTy = Type::getIntNTy(B.getContext(), totalBits);

    Value *packed = ConstantInt::get(intTy, 0);
    for (unsigned i = 0; i < totalDwords; i++) {
      Value *ext = B.CreateZExt(dwords[i], intTy);
      if (i > 0) ext = B.CreateShl(ext, i * 32);
      packed = B.CreateOr(packed, ext);
    }
    return B.CreateBitCast(packed, vecTy);
  }

  void writeRegVec(IRBuilder<> &B, ParsedReg pr, Value *v) {
    Type *ty = v->getType();
    unsigned totalBits = ty->getPrimitiveSizeInBits();
    unsigned totalDwords = (totalBits + 31) / 32;

    Type *intTy = Type::getIntNTy(B.getContext(), totalDwords * 32);
    Type *i32Ty = B.getInt32Ty();
    Value *packed = B.CreateBitCast(v, intTy);

    for (unsigned i = 0; i < totalDwords; i++) {
      Value *dw;
      if (i == 0)
        dw = B.CreateTrunc(packed, i32Ty);
      else
        dw = B.CreateTrunc(B.CreateLShr(packed, i * 32), i32Ty);
      ParsedReg sub = pr;
      sub.baseIdx = pr.baseIdx + i;
      sub.width = 1;
      writeReg32(B, sub, dw);
    }
  }

  void collectAllocas(SmallVectorImpl<AllocaInst *> &out) {
    for (auto *a : sgpr) if (a) out.push_back(a);
    for (auto *a : vgpr) if (a) out.push_back(a);
    for (auto *a : agpr) if (a) out.push_back(a);
    if (vcc) out.push_back(vcc);
    if (scc) out.push_back(scc);
    if (m0) out.push_back(m0);
    for (auto *a : flatScr) if (a) out.push_back(a);
  }
};

// ============================================================================
// Kernarg layout
// ============================================================================

struct KernargParam {
  int byteOffset;
  int byteSize;
  int paramIdx;
  bool isPointer;
};

struct KernargLayout {
  std::vector<KernargParam> params;
  int implicitArgsBase = 0;

  void resolveLoad(
      int byteOffset, int loadBytes,
      std::vector<std::pair<int, int>> &out) const {
    int loadEnd = byteOffset + loadBytes;
    for (auto &p : params) {
      int pEnd = p.byteOffset + p.byteSize;
      if (p.byteOffset >= byteOffset && pEnd <= loadEnd)
        out.push_back({p.byteSize / 4, p.paramIdx});
    }
  }
};

} // anonymous namespace

// ============================================================================
// Main raising function
// ============================================================================

RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &targetISA,
                      const std::string &kernelName,
                      const KernelMeta &meta) {
  RaiseResult result;

  MCState mc;
  if (!initMCState(mc, targetISA))
    return result;

  // ==== Phase 1: Disassemble + identify block boundaries ====
  ArrayRef<uint8_t> bytes(textBytes.data(), textBytes.size());
  uint64_t totalSize = textBytes.size();
  std::vector<DecodedInst> insts;
  std::set<uint64_t> blockStarts;
  blockStarts.insert(0);

  {
    uint64_t off = 0;
    while (off < totalSize) {
      MCInst inst;
      uint64_t instSize = 0;
      auto status = mc.disasm->getInstruction(inst, instSize,
                                               bytes.slice(off), off, nulls());
      if (status != MCDisassembler::Success) {
        off += 4;
        continue;
      }
      const MCInstrDesc &desc = mc.instrInfo->get(inst.getOpcode());
      DecodedInst di;
      di.rawMnemonic = getMnemonic(mc, inst);
      di.mnemonic = stripEncoding(StringRef(di.rawMnemonic)).str();
      di.inst = inst;
      di.numDefs = desc.getNumDefs();
      di.isBranch = desc.isBranch();
      di.isConditionalBranch = desc.isConditionalBranch();
      di.offset = off;
      di.size = instSize;

      di.tsFlags = desc.TSFlags;
      di.format = classifyFormat(desc.TSFlags);
      di.firstSrcIdx = desc.getNumDefs();

      auto opInfos = desc.operands();
      // DPP/SDWA instructions have a tied "old" operand as the first source
      // (fallback value for inactive lanes). In our scalar model all lanes are
      // active, so "old" is never used — skip it so srcMap aligns with the
      // base VOP encoding.
      unsigned srcStart = di.firstSrcIdx;
      if ((di.format == FormatKind::DPP || di.format == FormatKind::SDWA) &&
          srcStart < inst.getNumOperands())
        srcStart++;
      unsigned pendingModIdx = UINT_MAX;
      for (unsigned i = srcStart; i < inst.getNumOperands(); ++i) {
        if (i < opInfos.size() &&
            opInfos[i].OperandType == OPERAND_INPUT_MODS) {
          pendingModIdx = i;
          continue;
        }
        if (di.numSrcs < DecodedInst::kMaxSrcs) {
          di.srcMap[di.numSrcs] = i;
          di.modMap[di.numSrcs] = pendingModIdx;
          di.numSrcs++;
        }
        pendingModIdx = UINT_MAX;
      }

      for (MCPhysReg r : desc.implicit_defs()) {
        StringRef rn = mc.regInfo->getName(r);
        if (rn == "SCC") di.defsSCC = true;
        else if (rn.starts_with("VCC")) di.defsVCC = true;
        else if (rn.starts_with("EXEC")) di.defsEXEC = true;
      }

      if (di.isBranch) {
        for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
          if (inst.getOperand(i).isImm()) {
            int64_t raw = inst.getOperand(i).getImm();
            int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
            blockStarts.insert(off + 4 + brOff * 4);
          }
        }
        if (di.isConditionalBranch)
          blockStarts.insert(off + instSize);
      }

      bool isEnd = (di.mnemonic == "s_endpgm");
      insts.push_back(std::move(di));
      if (isEnd)
        break;
      off += instSize;
    }
  }

  result.totalCount = (int)insts.size();

  // ==== Phase 2: Build LLVM IR module + function ====
  result.ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *result.ctx;
  result.module = std::make_unique<Module>("ir_proto_module", C);
  Module &M = *result.module;
  M.setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  TargetOptions opts;
  std::unique_ptr<TargetMachine> tm(mc.target->createTargetMachine(
      Triple("amdgcn-amd-amdhsa"), targetISA, "", opts, Reloc::PIC_));
  if (!tm) {
    errs() << "ir_proto: Failed to create TargetMachine\n";
    return result;
  }
  M.setDataLayout(tm->createDataLayout());

  auto *voidTy = Type::getVoidTy(C);
  auto *i1Ty = Type::getInt1Ty(C);
  auto *i8Ty = Type::getInt8Ty(C);
  auto *i32Ty = Type::getInt32Ty(C);
  auto *i64Ty = Type::getInt64Ty(C);
  auto *f32Ty = Type::getFloatTy(C);
  auto *ptrGlobalTy = PointerType::get(C, 1);

  // Build function signature dynamically from kernel metadata
  SmallVector<Type *, 8> paramTypes;
  KernargLayout kernargs;
  int paramIdx = 0;
  for (auto &arg : meta.args) {
    if (arg.valueKind == "hidden_global_offset_x" ||
        arg.valueKind == "hidden_global_offset_y" ||
        arg.valueKind == "hidden_global_offset_z" ||
        arg.valueKind.rfind("hidden_", 0) == 0)
      continue;
    bool isPtr = (arg.valueKind == "global_buffer");
    Type *ty;
    if (isPtr) {
      ty = ptrGlobalTy;
    } else if (arg.size == 8) {
      ty = i64Ty;
    } else {
      ty = i32Ty;
    }
    paramTypes.push_back(ty);
    kernargs.params.push_back(
        {arg.offset, arg.size, paramIdx, isPtr});
    paramIdx++;
  }
  kernargs.implicitArgsBase = meta.implicitArgsBase();

  auto *funcTy = FunctionType::get(voidTy, paramTypes, false);
  Function *F =
      Function::Create(funcTy, GlobalValue::ExternalLinkage, kernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1024");

  for (int i = 0; i < paramIdx; i++)
    F->getArg(i)->setName("arg" + std::to_string(i));

  errs() << "ir_proto: Kernel '" << kernelName << "' has " << paramIdx
         << " args (kernarg_segment_size=" << meta.kernargSegmentSize << ")\n";

  Function *fnWorkgroupIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_x);
  Function *fnWorkitemIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workitem_id_x);
  Function *fnImplicitArgPtr =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_implicitarg_ptr);

  // ==== Phase 3: Create basic blocks ====
  std::map<uint64_t, BasicBlock *> offsetToBB;
  for (uint64_t addr : blockStarts)
    offsetToBB[addr] = BasicBlock::Create(C, "bb_0x" + utohexstr(addr), F);

  // ==== Phase 4: Init entry registers ====
  IRBuilder<> B(offsetToBB[0]);

  AllocaRegFile regs;
  regs.init(B, i32Ty, i1Ty);

  // s[0:1] = kernarg segment pointer (sentinel)
  regs.storeSGPR64(B, 0, Constant::getNullValue(PointerType::get(C, 4)));
  // s2 = workgroup_id_x
  regs.storeSGPR32(B, 2, B.CreateCall(fnWorkgroupIdX, {}, "wg_id_x"));
  // s3 = workgroup_id_y
  Function *fnWorkgroupIdY =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_y);
  regs.storeSGPR32(B, 3, B.CreateCall(fnWorkgroupIdY, {}, "wg_id_y"));
  // v0 = workitem_id_x
  regs.storeVGPR32(B, 0, B.CreateCall(fnWorkitemIdX, {}, "tid"));
  // Init VCC/SCC to false
  regs.storeVCC(B, ConstantInt::getFalse(i1Ty));
  regs.storeSCC(B, ConstantInt::getFalse(i1Ty));

  // ==== Phase 5: Raise each instruction ====
  BasicBlock *currentBB = offsetToBB[0];
  int raisedCount = 0;

  auto readOp32 = [&](const DecodedInst &di, unsigned opIdx) -> Value * {
    if (di.isReg(opIdx)) {
      ParsedReg pr = parseReg(mc, di.getReg(opIdx));
      if (pr.kind == ParsedReg::VCC) {
        Value *v = regs.loadVCC(B);
        return B.CreateSExt(v, i32Ty);
      }
      if (pr.kind == ParsedReg::EXEC)
        return ConstantInt::get(i32Ty, -1);
      if (pr.kind == ParsedReg::SCC)
        return B.CreateZExt(regs.loadSCC(B), i32Ty);
      if (pr.kind == ParsedReg::NOREG)
        return ConstantInt::get(i32Ty, 0);
      if (pr.kind == ParsedReg::MODE)
        return ConstantInt::get(i32Ty, 0);
      Value *v = regs.readReg32(B, pr);
      if (!v) {
        errs() << "ir_proto: unreadable register '"
               << mc.regInfo->getName(di.getReg(opIdx)) << "' in " << di.mnemonic << "\n";
      }
      return v;
    }
    if (di.isImm(opIdx))
      return ConstantInt::get(i32Ty, (uint32_t)(di.getImm(opIdx) & 0xFFFFFFFF));
    return nullptr;
  };

  auto readOp64 = [&](const DecodedInst &di, unsigned opIdx) -> Value * {
    if (di.isReg(opIdx)) {
      ParsedReg pr = parseReg(mc, di.getReg(opIdx));
      if (pr.kind == ParsedReg::VCC)
        return B.CreateSExt(regs.loadVCC(B), i64Ty);
      if (pr.kind == ParsedReg::EXEC)
        return ConstantInt::getSigned(i64Ty, -1);
      Value *v = regs.readReg64(B, pr);
      if (!v) {
        errs() << "ir_proto: unreadable register64 '"
               << mc.regInfo->getName(di.getReg(opIdx)) << "' in " << di.mnemonic << "\n";
      }
      return v;
    }
    if (di.isImm(opIdx))
      return ConstantInt::getSigned(i64Ty, di.getImm(opIdx));
    return nullptr;
  };

  // OpResolver: read source operands via srcMap, which skips VOP3 modifiers.
  struct OpResolver {
    const DecodedInst &di;
    const MCState &mc;
    IRBuilder<> &B;
    AllocaRegFile &regs;
    Type *i32Ty, *i64Ty, *f32Ty;
    decltype(readOp32) &rawRead32;
    decltype(readOp64) &rawRead64;

    unsigned srcIdx(unsigned i) const {
      assert(i < di.numSrcs && "source index out of range");
      return di.srcMap[i];
    }
    unsigned nSrcs() const { return di.numSrcs; }

    // Raw modifier bits for source i: bit 0 = neg, bit 1 = abs.
    unsigned srcMod(unsigned i) const {
      unsigned modIdx = di.modMap[i];
      if (modIdx == UINT_MAX) return 0;
      if (!di.isImm(modIdx)) return 0;
      return (unsigned)(di.getImm(modIdx) & 0xF);
    }

    // Apply VOP3 neg/abs modifiers to an FP value.
    Value *applyMods(unsigned i, Value *v) {
      unsigned mods = srcMod(i);
      if (mods == 0) return v;
      bool isI32 = (v->getType() == i32Ty);
      if (isI32) v = B.CreateBitCast(v, f32Ty);
      if (mods & 2) v = B.CreateUnaryIntrinsic(Intrinsic::fabs, v, nullptr, "abs");
      if (mods & 1) v = B.CreateFNeg(v, "neg");
      if (isI32) v = B.CreateBitCast(v, i32Ty);
      return v;
    }

    Value *src(unsigned i) { return rawRead32(di, srcIdx(i)); }
    // Read source and apply FP modifiers.
    Value *srcF(unsigned i) { return applyMods(i, rawRead32(di, srcIdx(i))); }
    Value *src64(unsigned i) { return rawRead64(di, srcIdx(i)); }
    int64_t srcImm(unsigned i) { return di.getImm(srcIdx(i)); }
    ParsedReg dst(unsigned i = 0) { return parseReg(mc, di.getReg(i)); }
    bool isSrcReg(unsigned i) {
      return di.isReg(srcIdx(i));
    }
    ParsedReg srcReg(unsigned i) {
      unsigned idx = srcIdx(i);
      if (!di.isReg(idx)) {
        ParsedReg pr;
        pr.kind = ParsedReg::OTHER;
        return pr;
      }
      return parseReg(mc, di.getReg(idx));
    }
  };

  for (size_t instIdx = 0; instIdx < insts.size(); ++instIdx) {
    const DecodedInst &di = insts[instIdx];

    auto bbIt = offsetToBB.find(di.offset);
    if (bbIt != offsetToBB.end() && bbIt->second != currentBB) {
      if (!currentBB->getTerminator())
        B.CreateBr(bbIt->second);
      currentBB = bbIt->second;
      B.SetInsertPoint(currentBB);
    }

    StringRef mn(di.mnemonic);
    OpResolver op{di, mc, B, regs, i32Ty, i64Ty, f32Ty, readOp32, readOp64};

    // Auto-writeback: handlers set sccResult to enable auto SCC = (result != 0).
    // Handlers with special SCC semantics (carry, explicit compare) set
    // sccHandled=true and write SCC themselves.
    bool handled = false;
    bool sccHandled = false;
    Value *sccResult = nullptr;

    // ================================================================
    // Format-based dispatch
    // ================================================================
    switch (di.format) {

    // ================================================================
    // SOPP: branches, control flow, s_endpgm, waitcnt, nop
    // ================================================================
    case FormatKind::SOPP: {
      if (mn == "s_waitcnt" || mn == "s_nop" || mn == "s_code_end" ||
          mn == "s_waitcnt_vscnt" || mn == "s_waitcnt_vmcnt" ||
          mn == "s_waitcnt_expcnt" || mn == "s_waitcnt_lgkmcnt" ||
          mn == "s_wait_idle" || mn == "s_barrier" ||
          mn == "s_set_inst_prefetch_distance" ||
          mn == "s_inst_prefetch" ||
          mn == "s_sched_barrier" || mn == "s_sleep" ||
          mn == "s_setprio" || mn == "s_sendmsg" ||
          mn == "s_wait_storecnt" || mn == "s_wait_loadcnt" ||
          mn == "s_wait_samplecnt" || mn == "s_wait_bvhcnt" ||
          mn == "s_wait_dscnt" || mn == "s_wait_kmcnt" ||
          mn == "s_set_gpr_idx_off") {
        handled = true; break;
      }
      if (mn == "s_endpgm") {
        B.CreateRetVoid(); handled = true; break;
      }
      if (mn == "s_branch") {
        int64_t raw = di.getImm(0);
        int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
        B.CreateBr(offsetToBB[di.offset + 4 + brOff * 4]);
        handled = true; break;
      }
      if (mn == "s_cbranch_execz" || mn == "s_cbranch_execnz") {
        // Scalar model: EXEC is always all-ones.
        // execz (branch if EXEC==0) → never taken → always fall through.
        // execnz (branch if EXEC!=0) → always taken → unconditional branch.
        int64_t raw = di.getImm(0);
        int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
        uint64_t target = di.offset + 4 + brOff * 4;
        BasicBlock *targetBB = offsetToBB[target];
        BasicBlock *fallthroughBB = offsetToBB[di.offset + di.size];
        if (mn == "s_cbranch_execz")
          B.CreateBr(fallthroughBB);
        else
          B.CreateBr(targetBB);
        handled = true; break;
      }
      if (mn == "s_cbranch_scc0" || mn == "s_cbranch_scc1") {
        int64_t raw = di.getImm(0);
        int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
        uint64_t target = di.offset + 4 + brOff * 4;
        BasicBlock *targetBB = offsetToBB[target];
        BasicBlock *fallthroughBB = offsetToBB[di.offset + di.size];
        Value *sccV = regs.loadSCC(B);
        if (mn == "s_cbranch_scc0") sccV = B.CreateNot(sccV, "not_scc");
        B.CreateCondBr(sccV, targetBB, fallthroughBB);
        handled = true; break;
      }
      if (mn == "s_cbranch_vccnz" || mn == "s_cbranch_vccz") {
        int64_t raw = di.getImm(0);
        int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
        uint64_t target = di.offset + 4 + brOff * 4;
        BasicBlock *targetBB = offsetToBB[target];
        BasicBlock *fallthroughBB = offsetToBB[di.offset + di.size];
        Value *vccV = regs.loadVCC(B);
        if (mn == "s_cbranch_vccz") vccV = B.CreateNot(vccV, "not_vcc");
        B.CreateCondBr(vccV, targetBB, fallthroughBB);
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // SMEM: s_load_dword[xN]
    // ================================================================
    case FormatKind::SMEM: {
      if (mn.starts_with("s_load_dword")) {
        int loadDwords = 1;
        if (mn.contains("dwordx2")) loadDwords = 2;
        else if (mn.contains("dwordx4")) loadDwords = 4;
        else if (mn.contains("dwordx8")) loadDwords = 8;
        int loadBytes = loadDwords * 4;

        ParsedReg dest = op.dst();
        ParsedReg base = op.srcReg(0);
        int64_t byteOffset = op.srcImm(1);
        bool isKernarg = (base.kind == ParsedReg::SGPR && base.baseIdx == 0);

        if (isKernarg && byteOffset < kernargs.implicitArgsBase) {
          std::vector<std::pair<int, int>> resolved;
          kernargs.resolveLoad(byteOffset, loadBytes, resolved);
          if (resolved.empty()) {
            errs() << "ir_proto: Cannot resolve kernarg at offset " << byteOffset << "\n";
            return result;
          }
          int regOff = 0;
          for (auto &[regWidth, pIdx] : resolved) {
            Value *arg = F->getArg(pIdx);
            if (regWidth == 1)      regs.storeSGPR32(B, dest.baseIdx + regOff, arg);
            else if (regWidth == 2) regs.storeSGPR64(B, dest.baseIdx + regOff, arg);
            regOff += regWidth;
          }
        } else if (isKernarg) {
          int implOffset = byteOffset - kernargs.implicitArgsBase;
          Value *implPtr = B.CreateCall(fnImplicitArgPtr, {}, "implicitarg_ptr");
          Value *gep = B.CreateInBoundsGEP(i8Ty, implPtr, B.getInt64(implOffset), "impl_gep");
          regs.storeSGPR32(B, dest.baseIdx, B.CreateLoad(i32Ty, gep, "impl_load"));
        } else {
          Value *baseAddr = regs.loadSGPR64(B, base.baseIdx);
          Value *ptr = B.CreateIntToPtr(baseAddr, ptrGlobalTy);
          if (byteOffset != 0)
            ptr = B.CreateInBoundsGEP(i8Ty, ptr, B.getInt64(byteOffset));
          for (int d = 0; d < loadDwords; d++) {
            Value *ep = (d == 0) ? ptr : B.CreateInBoundsGEP(i8Ty, ptr, B.getInt64(d * 4));
            regs.storeSGPR32(B, dest.baseIdx + d, B.CreateLoad(i32Ty, ep, "smem_load"));
          }
        }
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // SOPC: scalar compares — SCC = compare result (explicit)
    // ================================================================
    case FormatKind::SOPC: {
      // s_set_gpr_idx_on enables GPR dynamic indexing via M0.
      // In scalar model, we store the index value to M0 and treat this as a
      // control-flow nop. The actual VGPR indexing effect is not modeled.
      if (mn == "s_set_gpr_idx_on") {
        ParsedReg m0reg;
        m0reg.kind = ParsedReg::M0;
        m0reg.baseIdx = 0;
        regs.writeReg32(B, m0reg, op.src(0));
        handled = true; break;
      }
      if (mn == "s_set_gpr_idx_off") {
        handled = true; break;
      }
      if (mn == "s_setvskip") {
        handled = true; break;
      }
      Value *src0 = op.src(0);
      Value *src1 = op.src(1);
      Value *cmp = nullptr;
      if      (mn == "s_cmp_gt_i32") cmp = B.CreateICmpSGT(src0, src1, "scmp");
      else if (mn == "s_cmp_lt_i32") cmp = B.CreateICmpSLT(src0, src1, "scmp");
      else if (mn == "s_cmp_ge_i32") cmp = B.CreateICmpSGE(src0, src1, "scmp");
      else if (mn == "s_cmp_le_i32") cmp = B.CreateICmpSLE(src0, src1, "scmp");
      else if (mn == "s_cmp_eq_u32") cmp = B.CreateICmpEQ(src0, src1, "scmp");
      else if (mn == "s_cmp_lg_u32") cmp = B.CreateICmpNE(src0, src1, "scmp");
      else if (mn == "s_cmp_ge_u32") cmp = B.CreateICmpUGE(src0, src1, "scmp");
      else if (mn == "s_cmp_gt_u32") cmp = B.CreateICmpUGT(src0, src1, "scmp");
      else if (mn == "s_cmp_lt_u32") cmp = B.CreateICmpULT(src0, src1, "scmp");
      else if (mn == "s_cmp_le_u32") cmp = B.CreateICmpULE(src0, src1, "scmp");
      else if (mn == "s_cmp_eq_i32") cmp = B.CreateICmpEQ(src0, src1, "scmp");
      else if (mn == "s_cmp_lg_i32") cmp = B.CreateICmpNE(src0, src1, "scmp");
      if (cmp) {
        regs.storeSCC(B, cmp);
        sccHandled = true;
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // SOP1: single-source scalar ops
    // ================================================================
    case FormatKind::SOP1: {
      if (mn == "s_mov_b32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }
      if (mn == "s_mov_b64") {
        regs.writeReg64(B, op.dst(), op.src64(0));
        handled = true; break;
      }
      if (mn == "s_and_saveexec_b64") {
        Value *src = op.src64(0);
        regs.writeReg64(B, op.dst(), ConstantInt::getSigned(i64Ty, -1));
        regs.storeVCC(B, B.CreateICmpNE(src, ConstantInt::get(i64Ty, 0)));
        sccHandled = true;
        handled = true; break;
      }
      if (mn == "s_or_saveexec_b64") {
        regs.writeReg64(B, op.dst(), ConstantInt::getSigned(i64Ty, -1));
        sccHandled = true;
        handled = true; break;
      }
      if (mn == "s_xor_saveexec_b64") {
        regs.writeReg64(B, op.dst(), ConstantInt::getSigned(i64Ty, -1));
        sccHandled = true;
        handled = true; break;
      }
      if (mn == "s_not_b64") {
        sccResult = B.CreateNot(op.src64(0), "not64");
        regs.writeReg64(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_not_b32") {
        sccResult = B.CreateNot(op.src(0), "not32");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_brev_b32") {
        Function *brev = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::bitreverse, {i32Ty});
        regs.writeReg32(B, op.dst(), B.CreateCall(brev, {op.src(0)}, "sbrev"));
        handled = true; break;
      }
      if (mn == "s_ff1_i32_b32") {
        Function *cttz = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::cttz, {i32Ty});
        regs.writeReg32(B, op.dst(), B.CreateCall(cttz, {op.src(0), ConstantInt::getTrue(i1Ty)}, "ff1"));
        handled = true; break;
      }
      if (mn == "s_ff1_i32_b64") {
        Function *cttz64 = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::cttz, {i64Ty});
        Value *r = B.CreateCall(cttz64, {op.src64(0), ConstantInt::getTrue(i1Ty)}, "ff1_64");
        regs.writeReg32(B, op.dst(), B.CreateTrunc(r, i32Ty));
        handled = true; break;
      }
      if (mn == "s_flbit_i32_b64") {
        Function *ctlz64 = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ctlz, {i64Ty});
        Value *r = B.CreateCall(ctlz64, {op.src64(0), ConstantInt::getTrue(i1Ty)}, "flbit64");
        regs.writeReg32(B, op.dst(), B.CreateTrunc(r, i32Ty));
        handled = true; break;
      }
      if (mn == "s_flbit_i32_b32") {
        Function *ctlz = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ctlz, {i32Ty});
        regs.writeReg32(B, op.dst(), B.CreateCall(ctlz, {op.src(0), ConstantInt::getTrue(i1Ty)}, "flbit"));
        handled = true; break;
      }
      if (mn == "s_sext_i32_i8") {
        Value *v = B.CreateTrunc(op.src(0), Type::getInt8Ty(C));
        regs.writeReg32(B, op.dst(), B.CreateSExt(v, i32Ty, "sext8"));
        handled = true; break;
      }
      if (mn == "s_sext_i32_i16") {
        Value *v = B.CreateTrunc(op.src(0), Type::getInt16Ty(C));
        regs.writeReg32(B, op.dst(), B.CreateSExt(v, i32Ty, "sext16"));
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // SOPK: scalar with 16-bit inline constant
    // ================================================================
    case FormatKind::SOPK: {
      // SOPK format: dst = SDST, src(0) = sign-extended 16-bit imm
      if (mn == "s_movk_i32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }
      if (mn == "s_mulk_i32") {
        Value *dst = regs.readReg32(B, op.dst());
        regs.writeReg32(B, op.dst(), B.CreateMul(dst, op.src(0), "mulk"));
        handled = true; break;
      }
      if (mn == "s_addk_i32") {
        Value *dst = regs.readReg32(B, op.dst());
        Value *imm = op.src(0);
        Value *res = B.CreateAdd(dst, imm, "addk");
        regs.writeReg32(B, op.dst(), res);
        auto *ov = B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {i32Ty}, {dst, imm});
        regs.storeSCC(B, B.CreateExtractValue(ov, 1));
        sccHandled = true; handled = true; break;
      }
      // SOPK compares: s_cmpk_XX_i32 / s_cmpk_XX_u32
      if (mn.starts_with("s_cmpk_")) {
        Value *sdst = regs.readReg32(B, op.dst());
        Value *imm = op.src(0);
        Value *cmp = nullptr;
        if      (mn == "s_cmpk_eq_i32" || mn == "s_cmpk_eq_u32") cmp = B.CreateICmpEQ(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_lg_i32" || mn == "s_cmpk_lg_u32") cmp = B.CreateICmpNE(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_gt_i32") cmp = B.CreateICmpSGT(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_ge_i32") cmp = B.CreateICmpSGE(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_lt_i32") cmp = B.CreateICmpSLT(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_le_i32") cmp = B.CreateICmpSLE(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_gt_u32") cmp = B.CreateICmpUGT(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_ge_u32") cmp = B.CreateICmpUGE(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_lt_u32") cmp = B.CreateICmpULT(sdst, imm, "scmpk");
        else if (mn == "s_cmpk_le_u32") cmp = B.CreateICmpULE(sdst, imm, "scmpk");
        if (cmp) {
          regs.storeSCC(B, cmp);
          sccHandled = true;
          handled = true; break;
        }
      }
      break;
    }

    // ================================================================
    // SOP2: two-source scalar ALU
    // Auto SCC writeback: handlers set sccResult, post-switch emits
    // SCC = (sccResult != 0) when di.defsSCC is true.
    // ================================================================
    case FormatKind::SOP2: {
      // 32-bit binary ops — auto SCC via sccResult
      if (mn == "s_and_b32") {
        sccResult = B.CreateAnd(op.src(0), op.src(1), "and");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_or_b32") {
        sccResult = B.CreateOr(op.src(0), op.src(1), "or");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_lshl_b32") {
        sccResult = B.CreateShl(op.src(0), op.src(1), "shl");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_lshr_b32") {
        sccResult = B.CreateLShr(op.src(0), op.src(1), "lshr");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_ashr_i32") {
        sccResult = B.CreateAShr(op.src(0), op.src(1), "ashr");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_add_i32") {
        Value *src0 = op.src(0), *src1 = op.src(1);
        Value *res = B.CreateAdd(src0, src1, "add");
        regs.writeReg32(B, op.dst(), res);
        auto *ov = B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {i32Ty}, {src0, src1});
        regs.storeSCC(B, B.CreateExtractValue(ov, 1));
        sccHandled = true; handled = true; break;
      }
      if (mn == "s_sub_i32") {
        Value *src0 = op.src(0), *src1 = op.src(1);
        Value *res = B.CreateSub(src0, src1, "sub");
        regs.writeReg32(B, op.dst(), res);
        regs.storeSCC(B, B.CreateICmpULT(src0, src1));
        sccHandled = true; handled = true; break;
      }

      // Special SCC semantics — handler writes SCC explicitly
      if (mn == "s_add_u32") {
        Value *src0 = op.src(0), *src1 = op.src(1);
        Value *res = B.CreateAdd(src0, src1, "add");
        regs.writeReg32(B, op.dst(), res);
        auto *ov = B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {i32Ty}, {src0, src1});
        regs.storeSCC(B, B.CreateExtractValue(ov, 1));
        sccHandled = true; handled = true; break;
      }
      if (mn == "s_addc_u32") {
        Value *src0 = op.src(0), *src1 = op.src(1);
        Value *carry = B.CreateZExt(regs.loadSCC(B), i32Ty);
        Value *res = B.CreateAdd(B.CreateAdd(src0, src1), carry, "addc");
        regs.writeReg32(B, op.dst(), res);
        regs.storeSCC(B, B.CreateOr(B.CreateICmpULT(res, src0), B.CreateICmpULT(res, src1)));
        sccHandled = true; handled = true; break;
      }
      if (mn == "s_sub_u32") {
        Value *src0 = op.src(0), *src1 = op.src(1);
        Value *res = B.CreateSub(src0, src1, "sub");
        regs.writeReg32(B, op.dst(), res);
        regs.storeSCC(B, B.CreateICmpULT(src0, src1));
        sccHandled = true; handled = true; break;
      }
      if (mn == "s_subb_u32") {
        Value *src0 = op.src(0), *src1 = op.src(1);
        Value *borrow = B.CreateZExt(regs.loadSCC(B), i32Ty);
        Value *res = B.CreateSub(B.CreateSub(src0, src1), borrow, "subb");
        regs.writeReg32(B, op.dst(), res);
        regs.storeSCC(B, B.CreateOr(B.CreateICmpULT(src0, src1),
            B.CreateAnd(B.CreateICmpEQ(src0, src1), regs.loadSCC(B))));
        sccHandled = true; handled = true; break;
      }

      // No SCC side-effect (di.defsSCC=false for these)
      if (mn == "s_mul_i32") {
        regs.writeReg32(B, op.dst(), B.CreateMul(op.src(0), op.src(1), "mul"));
        handled = true; break;
      }
      if (mn == "s_mul_hi_u32") {
        Value *a = B.CreateZExt(op.src(0), i64Ty), *b = B.CreateZExt(op.src(1), i64Ty);
        regs.writeReg32(B, op.dst(), B.CreateTrunc(B.CreateLShr(B.CreateMul(a, b, "mulhi_wide"), 32), i32Ty, "mulhi"));
        handled = true; break;
      }
      if (mn == "s_min_u32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        sccResult = B.CreateSelect(B.CreateICmpULT(s0, s1), s0, s1, "smin");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_max_u32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        sccResult = B.CreateSelect(B.CreateICmpUGT(s0, s1), s0, s1, "smax");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_min_i32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        sccResult = B.CreateSelect(B.CreateICmpSLT(s0, s1), s0, s1, "smin");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_max_i32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        sccResult = B.CreateSelect(B.CreateICmpSGT(s0, s1), s0, s1, "smax");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_lshl1_add_u32") {
        sccResult = B.CreateAdd(B.CreateShl(op.src(0), 1), op.src(1), "lshl1add");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_lshl2_add_u32") {
        sccResult = B.CreateAdd(B.CreateShl(op.src(0), 2), op.src(1), "lshl2add");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_lshl3_add_u32") {
        sccResult = B.CreateAdd(B.CreateShl(op.src(0), 3), op.src(1), "lshl3add");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_lshl4_add_u32") {
        sccResult = B.CreateAdd(B.CreateShl(op.src(0), 4), op.src(1), "lshl4add");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_xor_b32") {
        sccResult = B.CreateXor(op.src(0), op.src(1), "xor");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_xor_b64") {
        sccResult = B.CreateXor(op.src64(0), op.src64(1), "xor64");
        regs.writeReg64(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_bfm_b64") {
        // s_bfm_b64 dst, width, offset: creates a 64-bit mask with `width` ones starting at `offset`
        Value *width = B.CreateZExt(B.CreateAnd(op.src(0), ConstantInt::get(i32Ty, 0x3F)), i64Ty);
        Value *offset = B.CreateZExt(B.CreateAnd(op.src(1), ConstantInt::get(i32Ty, 0x3F)), i64Ty);
        Value *mask = B.CreateSub(B.CreateShl(ConstantInt::get(i64Ty, 1), width), ConstantInt::get(i64Ty, 1));
        sccResult = B.CreateShl(mask, offset, "bfm64");
        regs.writeReg64(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_bfm_b32") {
        Value *width = B.CreateAnd(op.src(0), ConstantInt::get(i32Ty, 0x1F));
        Value *offset = B.CreateAnd(op.src(1), ConstantInt::get(i32Ty, 0x1F));
        Value *mask = B.CreateSub(B.CreateShl(ConstantInt::get(i32Ty, 1), width), ConstantInt::get(i32Ty, 1));
        sccResult = B.CreateShl(mask, offset, "bfm32");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_bfe_u32") {
        Value *src = op.src(0), *ctrl = op.src(1);
        Value *offset = B.CreateAnd(ctrl, ConstantInt::get(i32Ty, 0x1F));
        Value *width = B.CreateAnd(B.CreateLShr(ctrl, 16), ConstantInt::get(i32Ty, 0x7F));
        Value *shifted = B.CreateLShr(src, offset);
        Value *mask = B.CreateSub(B.CreateShl(ConstantInt::get(i32Ty, 1), width), ConstantInt::get(i32Ty, 1));
        sccResult = B.CreateAnd(shifted, mask, "bfe");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_pack_ll_b32_b16") {
        Value *lo = B.CreateAnd(op.src(0), ConstantInt::get(i32Ty, 0xFFFF));
        Value *hi = B.CreateShl(B.CreateAnd(op.src(1), ConstantInt::get(i32Ty, 0xFFFF)), 16);
        regs.writeReg32(B, op.dst(), B.CreateOr(lo, hi, "pack_ll"));
        handled = true; break;
      }
      if (mn == "s_pack_lh_b32_b16") {
        Value *lo = B.CreateAnd(op.src(0), ConstantInt::get(i32Ty, 0xFFFF));
        Value *hi = B.CreateAnd(op.src(1), ConstantInt::get(i32Ty, 0xFFFF0000u));
        regs.writeReg32(B, op.dst(), B.CreateOr(lo, hi, "pack_lh"));
        handled = true; break;
      }
      if (mn == "s_cselect_b32") {
        regs.writeReg32(B, op.dst(), B.CreateSelect(regs.loadSCC(B), op.src(0), op.src(1), "csel"));
        handled = true; break;
      }
      if (mn == "s_cselect_b64") {
        regs.writeReg64(B, op.dst(), B.CreateSelect(regs.loadSCC(B), op.src64(0), op.src64(1), "csel"));
        handled = true; break;
      }

      // 64-bit SOP2 — auto SCC via sccResult
      if (mn == "s_lshl_b64") {
        sccResult = B.CreateShl(op.src64(0), op.src64(1), "shl64");
        regs.writeReg64(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_or_b64") {
        ParsedReg dest = op.dst();
        Value *res = B.CreateOr(op.src64(0), op.src64(1), "or64");
        if (dest.kind != ParsedReg::EXEC)
          regs.writeReg64(B, dest, res);
        sccResult = res;
        handled = true; break;
      }
      if (mn == "s_and_b64") {
        ParsedReg dest = op.dst();
        Value *res = B.CreateAnd(op.src64(0), op.src64(1), "and64");
        if (dest.kind != ParsedReg::EXEC)
          regs.writeReg64(B, dest, res);
        sccResult = res;
        handled = true; break;
      }
      if (mn == "s_andn2_b64") {
        sccResult = B.CreateAnd(op.src64(0), B.CreateNot(op.src64(1)), "andn2_64");
        regs.writeReg64(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_orn2_b64") {
        sccResult = B.CreateOr(op.src64(0), B.CreateNot(op.src64(1)), "orn2_64");
        regs.writeReg64(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_andn2_b32") {
        sccResult = B.CreateAnd(op.src(0), B.CreateNot(op.src(1)), "andn2");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      if (mn == "s_orn2_b32") {
        sccResult = B.CreateOr(op.src(0), B.CreateNot(op.src(1)), "orn2");
        regs.writeReg32(B, op.dst(), sccResult);
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // VALU: VOP1 / VOP2 / VOP3 / VOPC / VOP3P / DPP / SDWA
    // DPP/SDWA are base VOP ops with lane permutation / sub-dword
    // addressing — in scalar model these are identity, so we strip
    // the suffix and handle them as the base VOP operation.
    // The srcMap was already adjusted during decode to skip the tied
    // "old" operand, so op.src(0)/op.src(1) are the real data sources.
    // ================================================================
    case FormatKind::DPP:
    case FormatKind::SDWA:
    case FormatKind::VOP1:
    case FormatKind::VOP2:
    case FormatKind::VOP3:
    case FormatKind::VOPC:
    case FormatKind::VOP3P: {
      // Strip DPP/SDWA suffix so handlers match the base mnemonic.
      if (di.format == FormatKind::DPP || di.format == FormatKind::SDWA) {
        for (const char *suffix : {"_dpp", "_sdwa"}) {
          if (mn.ends_with(suffix)) {
            mn = mn.drop_back(strlen(suffix));
            break;
          }
        }
      }
      if (mn == "v_nop") { handled = true; break; }
      // ---- v_mov_b32 ----
      if (mn == "v_mov_b32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }
      // Lane permutation instructions — in scalar model, identity
      if (mn.starts_with("v_permlane") || mn == "v_permlanex16_b32") {
        // v_permlane16_swap_b32 has dst = src0, in-place swap is identity in scalar
        if (di.numDefs >= 1 && di.numSrcs >= 1) {
          regs.writeReg32(B, op.dst(), op.src(0));
        }
        handled = true; break;
      }
      // In our scalar model v_readfirstlane_b32 is a VGPR→SGPR move
      if (mn == "v_readfirstlane_b32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }

      // ---- Type conversions ----
      if (mn == "v_cvt_f32_u32") {
        Value *r = B.CreateUIToFP(op.src(0), f32Ty, "cvt");
        regs.writeReg32(B, op.dst(), B.CreateBitCast(r, i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_f32_i32") {
        Value *r = B.CreateSIToFP(op.src(0), f32Ty, "cvt");
        regs.writeReg32(B, op.dst(), B.CreateBitCast(r, i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_u32_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        regs.writeReg32(B, op.dst(), B.CreateFPToUI(s, i32Ty, "cvt"));
        handled = true; break;
      }
      if (mn == "v_cvt_i32_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        regs.writeReg32(B, op.dst(), B.CreateFPToSI(s, i32Ty, "cvt"));
        handled = true; break;
      }
      if (mn == "v_cvt_f16_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Value *h = B.CreateFPTrunc(s, Type::getHalfTy(C), "cvt");
        Value *bits = B.CreateBitCast(h, Type::getInt16Ty(C));
        regs.writeReg32(B, op.dst(), B.CreateZExt(bits, i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_f32_f16") {
        Value *bits = B.CreateTrunc(op.src(0), Type::getInt16Ty(C));
        Value *h = B.CreateBitCast(bits, Type::getHalfTy(C));
        Value *f = B.CreateFPExt(h, f32Ty, "cvt");
        regs.writeReg32(B, op.dst(), B.CreateBitCast(f, i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_f32_ubyte0") {
        Value *byte = B.CreateAnd(op.src(0), ConstantInt::get(i32Ty, 0xFF));
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateUIToFP(byte, f32Ty), i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_f32_ubyte1") {
        Value *byte = B.CreateAnd(B.CreateLShr(op.src(0), 8), ConstantInt::get(i32Ty, 0xFF));
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateUIToFP(byte, f32Ty), i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_f32_ubyte2") {
        Value *byte = B.CreateAnd(B.CreateLShr(op.src(0), 16), ConstantInt::get(i32Ty, 0xFF));
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateUIToFP(byte, f32Ty), i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_f32_ubyte3") {
        Value *byte = B.CreateLShr(op.src(0), 24);
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateUIToFP(byte, f32Ty), i32Ty));
        handled = true; break;
      }
      if (mn == "v_bfrev_b32") {
        Function *brev = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::bitreverse, {i32Ty});
        regs.writeReg32(B, op.dst(), B.CreateCall(brev, {op.src(0)}, "bfrev"));
        handled = true; break;
      }
      if (mn == "v_not_b32") {
        regs.writeReg32(B, op.dst(), B.CreateNot(op.src(0), "vnot"));
        handled = true; break;
      }
      if (mn == "v_rcp_f32" || mn == "v_rcp_iflag_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Value *r = B.CreateFDiv(ConstantFP::get(f32Ty, 1.0), s, "rcp");
        regs.writeReg32(B, op.dst(), B.CreateBitCast(r, i32Ty));
        handled = true; break;
      }
      if (mn == "v_exp_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *exp2Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::exp2, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(exp2Fn, {s}, "exp"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_log_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *log2Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::log2, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(log2Fn, {s}, "log"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_sqrt_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *sqrtFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::sqrt, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(sqrtFn, {s}, "sqrt"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_rsq_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *sqrtFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::sqrt, {f32Ty});
        Value *sq = B.CreateCall(sqrtFn, {s}, "sqrt");
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateFDiv(ConstantFP::get(f32Ty, 1.0), sq, "rsq"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_floor_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *floorFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::floor, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(floorFn, {s}, "floor"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_ceil_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *ceilFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ceil, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(ceilFn, {s}, "ceil"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_trunc_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *truncFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::trunc, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(truncFn, {s}, "trunc"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_fract_f32") {
        Value *s = B.CreateBitCast(op.srcF(0), f32Ty);
        Function *floorFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::floor, {f32Ty});
        Value *fl = B.CreateCall(floorFn, {s}, "floor");
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateFSub(s, fl, "fract"), i32Ty));
        handled = true; break;
      }

      // ---- Simple 2-src integer ALU ----
      if (mn == "v_add_u32" || mn == "v_add_nc_u32" || mn == "v_add_i32") {
        regs.writeReg32(B, op.dst(), B.CreateAdd(op.src(0), op.src(1), "vadd"));
        handled = true; break;
      }
      if (mn == "v_or_b32") {
        regs.writeReg32(B, op.dst(), B.CreateOr(op.src(0), op.src(1), "vor"));
        handled = true; break;
      }
      if (mn == "v_and_b32") {
        regs.writeReg32(B, op.dst(), B.CreateAnd(op.src(0), op.src(1), "vand"));
        handled = true; break;
      }
      if (mn == "v_mul_lo_u32") {
        regs.writeReg32(B, op.dst(), B.CreateMul(op.src(0), op.src(1), "vmul"));
        handled = true; break;
      }
      if (mn == "v_sub_u32" || mn == "v_sub_nc_u32" || mn == "v_sub_i32") {
        regs.writeReg32(B, op.dst(), B.CreateSub(op.src(0), op.src(1), "vsub"));
        handled = true; break;
      }
      if (mn == "v_subrev_u32" || mn == "v_subrev_nc_u32") {
        regs.writeReg32(B, op.dst(), B.CreateSub(op.src(1), op.src(0), "vsubrev"));
        handled = true; break;
      }
      if (mn == "v_xor_b32") {
        regs.writeReg32(B, op.dst(), B.CreateXor(op.src(0), op.src(1), "vxor"));
        handled = true; break;
      }
      if (mn == "v_max_u32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        regs.writeReg32(B, op.dst(), B.CreateSelect(B.CreateICmpUGT(s0, s1), s0, s1, "vmax"));
        handled = true; break;
      }
      if (mn == "v_min_u32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        regs.writeReg32(B, op.dst(), B.CreateSelect(B.CreateICmpULT(s0, s1), s0, s1, "vmin"));
        handled = true; break;
      }
      if (mn == "v_max_i32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        regs.writeReg32(B, op.dst(), B.CreateSelect(B.CreateICmpSGT(s0, s1), s0, s1, "vmax"));
        handled = true; break;
      }
      if (mn == "v_min_i32") {
        Value *s0 = op.src(0), *s1 = op.src(1);
        regs.writeReg32(B, op.dst(), B.CreateSelect(B.CreateICmpSLT(s0, s1), s0, s1, "vmin"));
        handled = true; break;
      }
      if (mn == "v_mul_hi_u32") {
        Value *a = B.CreateZExt(op.src(0), i64Ty), *b = B.CreateZExt(op.src(1), i64Ty);
        regs.writeReg32(B, op.dst(), B.CreateTrunc(B.CreateLShr(B.CreateMul(a, b), 32), i32Ty, "vmulhi"));
        handled = true; break;
      }
      if (mn == "v_mul_hi_i32") {
        Value *a = B.CreateSExt(op.src(0), i64Ty), *b = B.CreateSExt(op.src(1), i64Ty);
        regs.writeReg32(B, op.dst(), B.CreateTrunc(B.CreateAShr(B.CreateMul(a, b), 32), i32Ty, "vmulhi"));
        handled = true; break;
      }
      if (mn == "v_mul_u32_u24") {
        Value *a = B.CreateAnd(op.src(0), ConstantInt::get(i32Ty, 0xFFFFFF));
        Value *b = B.CreateAnd(op.src(1), ConstantInt::get(i32Ty, 0xFFFFFF));
        regs.writeReg32(B, op.dst(), B.CreateMul(a, b, "mul24"));
        handled = true; break;
      }
      if (mn == "v_mul_i32_i24") {
        Value *a = B.CreateShl(op.src(0), 8);
        a = B.CreateAShr(a, 8);
        Value *b = B.CreateShl(op.src(1), 8);
        b = B.CreateAShr(b, 8);
        regs.writeReg32(B, op.dst(), B.CreateMul(a, b, "mul24i"));
        handled = true; break;
      }
      if (mn == "v_mad_u32_u24") {
        Value *a = B.CreateAnd(op.src(0), ConstantInt::get(i32Ty, 0xFFFFFF));
        Value *b = B.CreateAnd(op.src(1), ConstantInt::get(i32Ty, 0xFFFFFF));
        regs.writeReg32(B, op.dst(), B.CreateAdd(B.CreateMul(a, b), op.src(2), "mad24"));
        handled = true; break;
      }
      // v_writelane_b32: scalar→vector lane write, in scalar model just a move
      if (mn == "v_writelane_b32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }
      if (mn == "v_readlane_b32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }

      // ---- Reversed-operand shifts ----
      if (mn == "v_lshrrev_b32") {
        regs.writeReg32(B, op.dst(), B.CreateLShr(op.src(1), op.src(0), "vlshr"));
        handled = true; break;
      }
      if (mn == "v_lshlrev_b32") {
        regs.writeReg32(B, op.dst(), B.CreateShl(op.src(1), op.src(0), "vlshl"));
        handled = true; break;
      }
      if (mn == "v_ashrrev_i32") {
        regs.writeReg32(B, op.dst(), B.CreateAShr(op.src(1), op.src(0), "vashr"));
        handled = true; break;
      }

      // ---- FP ALU (srcF applies VOP3 neg/abs modifiers) ----
      if (mn == "v_add_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateFAdd(s0, s1, "fadd"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_mul_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateFMul(s0, s1, "fmul"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_sub_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateFSub(s0, s1, "fsub"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_subrev_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateFSub(s1, s0, "fsubrev"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_max_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        Function *maxFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::maxnum, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(maxFn, {s0, s1}, "fmax"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_min_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        Function *minFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::minnum, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(minFn, {s0, s1}, "fmin"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_fma_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        if (s2->getType() != f32Ty) s2 = B.CreateBitCast(s2, f32Ty);
        Function *fma = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::fma, {f32Ty});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(fma, {s0, s1, s2}, "fma"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_fmac_f32") {
        ParsedReg dstReg = op.dst();
        Value *s0 = op.srcF(0), *s1 = op.srcF(1), *dv = regs.readReg32(B, dstReg);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        if (dv->getType() != f32Ty) dv = B.CreateBitCast(dv, f32Ty);
        Function *fmuladd = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::fmuladd, {f32Ty});
        regs.writeReg32(B, dstReg, B.CreateBitCast(B.CreateCall(fmuladd, {s0, s1, dv}, "fmac"), i32Ty));
        handled = true; break;
      }

      // ---- 3-source integer VOP3 ----
      if (mn == "v_add3_u32") {
        regs.writeReg32(B, op.dst(), B.CreateAdd(B.CreateAdd(op.src(0), op.src(1)), op.src(2), "vadd3"));
        handled = true; break;
      }
      if (mn == "v_lshl_add_u32") {
        regs.writeReg32(B, op.dst(), B.CreateAdd(B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshl_add"));
        handled = true; break;
      }
      if (mn == "v_lshl_or_b32") {
        regs.writeReg32(B, op.dst(), B.CreateOr(B.CreateShl(op.src(0), op.src(1)), op.src(2), "vlshlor"));
        handled = true; break;
      }
      if (mn == "v_max3_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        if (s2->getType() != f32Ty) s2 = B.CreateBitCast(s2, f32Ty);
        Function *maxFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::maxnum, {f32Ty});
        Value *m01 = B.CreateCall(maxFn, {s0, s1});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(maxFn, {m01, s2}, "max3"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_min3_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        if (s2->getType() != f32Ty) s2 = B.CreateBitCast(s2, f32Ty);
        Function *minFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::minnum, {f32Ty});
        Value *m01 = B.CreateCall(minFn, {s0, s1});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(minFn, {m01, s2}, "min3"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_med3_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1), *s2 = op.srcF(2);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        if (s2->getType() != f32Ty) s2 = B.CreateBitCast(s2, f32Ty);
        Function *maxFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::maxnum, {f32Ty});
        Function *minFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::minnum, {f32Ty});
        Value *mn01 = B.CreateCall(minFn, {s0, s1});
        Value *mx01 = B.CreateCall(maxFn, {s0, s1});
        regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateCall(maxFn, {mn01, B.CreateCall(minFn, {mx01, s2})}, "med3"), i32Ty));
        handled = true; break;
      }
      if (mn == "v_cvt_pk_bf16_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        auto *bfTy = Type::getBFloatTy(C);
        Value *bf0 = B.CreateFPTrunc(s0, bfTy, "tobf16_0");
        Value *bf1 = B.CreateFPTrunc(s1, bfTy, "tobf16_1");
        Value *bits0 = B.CreateZExt(B.CreateBitCast(bf0, Type::getInt16Ty(C)), i32Ty);
        Value *bits1 = B.CreateZExt(B.CreateBitCast(bf1, Type::getInt16Ty(C)), i32Ty);
        regs.writeReg32(B, op.dst(), B.CreateOr(bits0, B.CreateShl(bits1, 16), "pk_bf16"));
        handled = true; break;
      }
      if (mn == "v_cvt_pk_fp8_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        // v_cvt_pk_fp8_f32 packs two f32 into two fp8 values in the low 16 bits.
        // The "old" value and word_sel determine where in the dest the result goes.
        // src2 = old value, src3 (imm) = word_sel.
        // Use the LLVM intrinsic which handles this correctly.
        Value *oldVal = (op.nSrcs() >= 3) ? op.src(2) : ConstantInt::get(i32Ty, 0);
        bool wordSel = (op.nSrcs() >= 4 && di.isImm(op.srcIdx(3))) ? (op.srcImm(3) != 0) : false;
        Function *cvtFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_cvt_pk_fp8_f32);
        regs.writeReg32(B, op.dst(), B.CreateCall(cvtFn,
            {s0, s1, oldVal, ConstantInt::get(i1Ty, wordSel)}, "pk_fp8"));
        handled = true; break;
      }
      if (mn == "v_cvt_pk_bf8_f32") {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
        if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        Value *oldVal = (op.nSrcs() >= 3) ? op.src(2) : ConstantInt::get(i32Ty, 0);
        bool wordSel = (op.nSrcs() >= 4 && di.isImm(op.srcIdx(3))) ? (op.srcImm(3) != 0) : false;
        Function *cvtFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_cvt_pk_bf8_f32);
        regs.writeReg32(B, op.dst(), B.CreateCall(cvtFn,
            {s0, s1, oldVal, ConstantInt::get(i1Ty, wordSel)}, "pk_bf8"));
        handled = true; break;
      }
      if (mn == "v_perm_b32") {
        Function *permFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_perm);
        regs.writeReg32(B, op.dst(), B.CreateCall(permFn, {op.src(0), op.src(1), op.src(2)}, "perm"));
        handled = true; break;
      }

      // ---- 64-bit vector ops ----
      if (mn == "v_lshlrev_b64") {
        Value *src = op.src64(1);
        int64_t shamt = op.srcImm(0);
        if (src->getType() != i64Ty) src = B.CreateBitOrPointerCast(src, i64Ty);
        regs.writeReg64(B, op.dst(), B.CreateShl(src, ConstantInt::get(i64Ty, shamt), "shl"));
        handled = true; break;
      }
      if (mn == "v_lshl_add_u64") {
        Value *src0 = op.src64(0);
        int64_t shift = op.srcImm(1);
        Value *src2 = op.src64(2);
        if (src0->getType()->isPointerTy()) src0 = B.CreatePtrToInt(src0, i64Ty);
        if (src0->getType() != i64Ty) src0 = B.CreateBitOrPointerCast(src0, i64Ty);
        if (src2->getType() != i64Ty) src2 = B.CreateBitOrPointerCast(src2, i64Ty);
        Value *shifted = (shift == 0) ? src0 : B.CreateShl(src0, ConstantInt::get(i64Ty, shift));
        regs.writeReg64(B, op.dst(), B.CreateAdd(shifted, src2, "lshl_add"));
        handled = true; break;
      }

      // ---- v_mad_u64_u32 (2 defs: VDST + SDST, firstSrcIdx=2) ----
      if (mn == "v_mad_u64_u32") {
        Value *a = B.CreateZExt(op.src(0), i64Ty), *b = B.CreateZExt(op.src(1), i64Ty);
        Value *res = B.CreateAdd(B.CreateMul(a, b), op.src64(2), "vmad64");
        regs.writeReg64(B, op.dst(0), res);
        regs.writeReg64(B, op.dst(1), ConstantInt::get(i64Ty, 0));
        handled = true; break;
      }

      // ---- Vector compares (VOPC e32 and VOP3 e64) ----
      // ---- Integer vector compares ----
      if (mn.starts_with("v_cmp_") && (mn.contains("_i32") || mn.contains("_u32") || mn.contains("_i64") || mn.contains("_u64"))) {
        Value *s0 = op.src(0), *s1 = op.src(1);
        if (!s0 || !s1) {
          errs() << "ir_proto: " << mn << ": missing operand\n";
          return result;
        }
        Value *cmp = nullptr;
        // Signed integer compares
        if      (mn.contains("_gt_i32") || mn.contains("_gt_i64")) cmp = B.CreateICmpSGT(s0, s1, "vcmp");
        else if (mn.contains("_ge_i32") || mn.contains("_ge_i64")) cmp = B.CreateICmpSGE(s0, s1, "vcmp");
        else if (mn.contains("_lt_i32") || mn.contains("_lt_i64")) cmp = B.CreateICmpSLT(s0, s1, "vcmp");
        else if (mn.contains("_le_i32") || mn.contains("_le_i64")) cmp = B.CreateICmpSLE(s0, s1, "vcmp");
        // Unsigned integer compares (also used for equality)
        else if (mn.contains("_eq_u32") || mn.contains("_eq_i32") || mn.contains("_eq_u64")) cmp = B.CreateICmpEQ(s0, s1, "vcmp");
        else if (mn.contains("_ne_u32") || mn.contains("_ne_i32") || mn.contains("_lg_u32") || mn.contains("_ne_u64")) cmp = B.CreateICmpNE(s0, s1, "vcmp");
        else if (mn.contains("_gt_u32") || mn.contains("_gt_u64")) cmp = B.CreateICmpUGT(s0, s1, "vcmp");
        else if (mn.contains("_ge_u32") || mn.contains("_ge_u64")) cmp = B.CreateICmpUGE(s0, s1, "vcmp");
        else if (mn.contains("_lt_u32") || mn.contains("_lt_u64")) cmp = B.CreateICmpULT(s0, s1, "vcmp");
        else if (mn.contains("_le_u32") || mn.contains("_le_u64")) cmp = B.CreateICmpULE(s0, s1, "vcmp");
        if (cmp) {
          if (di.numDefs >= 1) {
            ParsedReg d = op.dst();
            if (d.kind == ParsedReg::SGPR) regs.storeSGPR64(B, d.baseIdx, B.CreateSExt(cmp, i64Ty));
            else                           regs.storeVCC(B, cmp);
          } else {
            regs.storeVCC(B, cmp);
          }
          handled = true; break;
        }
      }
      // ---- FP vector compares ----
      if (mn.starts_with("v_cmp_") && (mn.contains("_f32") || mn.contains("_f16"))) {
        Value *s0 = op.srcF(0), *s1 = op.srcF(1);
        if (mn.contains("_f32")) {
          if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
          if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
        }
        Value *cmp = nullptr;
        if      (mn.contains("_gt_f"))  cmp = B.CreateFCmpOGT(s0, s1, "vcmpf");
        else if (mn.contains("_ge_f"))  cmp = B.CreateFCmpOGE(s0, s1, "vcmpf");
        else if (mn.contains("_lt_f"))  cmp = B.CreateFCmpOLT(s0, s1, "vcmpf");
        else if (mn.contains("_le_f"))  cmp = B.CreateFCmpOLE(s0, s1, "vcmpf");
        else if (mn.contains("_eq_f"))  cmp = B.CreateFCmpOEQ(s0, s1, "vcmpf");
        else if (mn.contains("_neq_f") || mn.contains("_ne_f") || mn.contains("_lg_f")) cmp = B.CreateFCmpONE(s0, s1, "vcmpf");
        else if (mn.contains("_nlt_f")) cmp = B.CreateFCmpUGE(s0, s1, "vcmpf");
        else if (mn.contains("_nle_f")) cmp = B.CreateFCmpUGT(s0, s1, "vcmpf");
        else if (mn.contains("_ngt_f")) cmp = B.CreateFCmpULE(s0, s1, "vcmpf");
        else if (mn.contains("_nge_f")) cmp = B.CreateFCmpULT(s0, s1, "vcmpf");
        else if (mn.contains("_u_f"))   cmp = B.CreateFCmpUNO(s0, s1, "vcmpf");
        else if (mn.contains("_o_f"))   cmp = B.CreateFCmpORD(s0, s1, "vcmpf");
        if (cmp) {
          if (di.numDefs >= 1) {
            ParsedReg d = op.dst();
            if (d.kind == ParsedReg::SGPR) regs.storeSGPR64(B, d.baseIdx, B.CreateSExt(cmp, i64Ty));
            else                           regs.storeVCC(B, cmp);
          } else {
            regs.storeVCC(B, cmp);
          }
          handled = true; break;
        }
      }

      // ---- VOP3P packed ops (2x fp32 in 2 dwords) ----
      if (mn == "v_pk_mul_f32") {
        auto *v2f32 = FixedVectorType::get(f32Ty, 2);
        if (!op.isSrcReg(0) || !op.isSrcReg(1)) {
          errs() << "ir_proto: " << mn << ": non-register source (immediate in VOP3P not supported)\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "VOP3P"; return result;
        }
        regs.writeRegVec(B, op.dst(), B.CreateFMul(
            regs.readRegVec(B, op.srcReg(0), v2f32), regs.readRegVec(B, op.srcReg(1), v2f32), "pk_mul"));
        handled = true; break;
      }
      if (mn == "v_pk_add_f32") {
        auto *v2f32 = FixedVectorType::get(f32Ty, 2);
        if (!op.isSrcReg(0) || !op.isSrcReg(1)) {
          errs() << "ir_proto: " << mn << ": non-register source (immediate in VOP3P not supported)\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "VOP3P"; return result;
        }
        regs.writeRegVec(B, op.dst(), B.CreateFAdd(
            regs.readRegVec(B, op.srcReg(0), v2f32), regs.readRegVec(B, op.srcReg(1), v2f32), "pk_add"));
        handled = true; break;
      }
      if (mn == "v_pk_fma_f32") {
        auto *v2f32 = FixedVectorType::get(f32Ty, 2);
        if (!op.isSrcReg(0) || !op.isSrcReg(1) || !op.isSrcReg(2)) {
          errs() << "ir_proto: " << mn << ": non-register source (immediate in VOP3P not supported)\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "VOP3P"; return result;
        }
        Function *fma = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::fma, {v2f32});
        regs.writeRegVec(B, op.dst(), B.CreateCall(fma, {
            regs.readRegVec(B, op.srcReg(0), v2f32), regs.readRegVec(B, op.srcReg(1), v2f32),
            regs.readRegVec(B, op.srcReg(2), v2f32)}, "pk_fma"));
        handled = true; break;
      }
      if (mn == "v_pk_max_f32") {
        auto *v2f32 = FixedVectorType::get(f32Ty, 2);
        if (!op.isSrcReg(0) || !op.isSrcReg(1)) {
          errs() << "ir_proto: " << mn << ": non-register source (immediate in VOP3P not supported)\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "VOP3P"; return result;
        }
        Function *maxFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::maxnum, {v2f32});
        regs.writeRegVec(B, op.dst(), B.CreateCall(maxFn, {
            regs.readRegVec(B, op.srcReg(0), v2f32), regs.readRegVec(B, op.srcReg(1), v2f32)}, "pk_max"));
        handled = true; break;
      }
      if (mn == "v_pk_min_f32") {
        auto *v2f32 = FixedVectorType::get(f32Ty, 2);
        if (!op.isSrcReg(0) || !op.isSrcReg(1)) {
          errs() << "ir_proto: " << mn << ": non-register source (immediate in VOP3P not supported)\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "VOP3P"; return result;
        }
        Function *minFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::minnum, {v2f32});
        regs.writeRegVec(B, op.dst(), B.CreateCall(minFn, {
            regs.readRegVec(B, op.srcReg(0), v2f32), regs.readRegVec(B, op.srcReg(1), v2f32)}, "pk_min"));
        handled = true; break;
      }
      if (mn == "v_pk_mov_b32") {
        regs.writeReg64(B, op.dst(), op.src64(0));
        handled = true; break;
      }

      // ---- v_cndmask_b32 (VOP2 or VOP3 — srcMap skips modifiers) ----
      if (mn == "v_cndmask_b32") {
        ParsedReg dest = op.dst();
        Value *src0 = op.src(0);
        Value *src1 = op.src(1);
        Value *cond = nullptr;
        if (op.nSrcs() >= 3 && di.isReg(op.srcIdx(2))) {
          ParsedReg condReg = parseReg(mc, di.getReg(op.srcIdx(2)));
          if (condReg.kind == ParsedReg::SGPR)
            cond = B.CreateICmpNE(regs.loadSGPR64(B, condReg.baseIdx), ConstantInt::get(i64Ty, 0));
          else
            cond = regs.loadVCC(B);
        }
        if (!cond) cond = regs.loadVCC(B);
        regs.writeReg32(B, dest, B.CreateSelect(cond, src1, src0, "cndmask"));
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // FLAT: global_load / global_store
    // ================================================================
    case FormatKind::FLAT: {
      if (mn == "global_load_ushort" || mn == "global_load_short_d16_hi" ||
          mn == "global_load_sshort" || mn == "global_load_ubyte" ||
          mn == "global_load_sbyte") {
        ParsedReg dest = op.dst();
        Value *addr = regs.readReg64(B, op.srcReg(0));
        if (addr->getType() != ptrGlobalTy) addr = B.CreateIntToPtr(addr, ptrGlobalTy);
        int64_t memOffset = 0;
        for (unsigned k = 1; k < op.nSrcs(); k++)
          if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
            memOffset = di.getImm(op.srcIdx(k));
        if (memOffset != 0) addr = B.CreateInBoundsGEP(i8Ty, addr, B.getInt64(memOffset));

        Type *loadTy = mn.contains("short") ? Type::getInt16Ty(C) : i8Ty;
        Value *loaded = B.CreateLoad(loadTy, addr, "gload_sub");
        Value *ext = (mn.contains("ubyte") || mn.contains("ushort"))
                       ? B.CreateZExt(loaded, i32Ty)
                       : B.CreateSExt(loaded, i32Ty);
        if (mn.contains("d16_hi")) {
          Value *prev = regs.readReg32(B, dest);
          ext = B.CreateOr(B.CreateAnd(prev, ConstantInt::get(i32Ty, 0xFFFF)),
                           B.CreateShl(ext, 16), "d16hi");
        }
        regs.writeReg32(B, dest, ext);
        handled = true; break;
      }

      if (mn.starts_with("global_load_dword")) {
        int loadDwords = 1;
        if (mn.contains("dwordx2")) loadDwords = 2;
        else if (mn.contains("dwordx4")) loadDwords = 4;

        ParsedReg dest = op.dst();
        Value *addr = regs.readReg64(B, op.srcReg(0));
        if (addr->getType() != ptrGlobalTy) addr = B.CreateIntToPtr(addr, ptrGlobalTy);
        int64_t memOffset = 0;
        for (unsigned k = 1; k < op.nSrcs(); k++)
          if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
            memOffset = di.getImm(op.srcIdx(k));
        if (memOffset != 0) addr = B.CreateInBoundsGEP(i8Ty, addr, B.getInt64(memOffset));

        if (loadDwords == 1) {
          regs.writeReg32(B, dest, B.CreateBitCast(B.CreateLoad(f32Ty, addr, "gload"), i32Ty));
        } else {
          Type *vecTy = FixedVectorType::get(i32Ty, loadDwords);
          Value *loaded = B.CreateLoad(vecTy, addr, "gload");
          for (int d = 0; d < loadDwords; d++) {
            ParsedReg sub = dest;
            sub.baseIdx = dest.baseIdx + d;
            sub.width = 1;
            regs.writeReg32(B, sub, B.CreateExtractElement(loaded, B.getInt32(d)));
          }
        }
        handled = true; break;
      }

      if (mn.starts_with("global_store")) {
        int storeDwords = 1;
        int storeBits = 32;
        if (mn.contains("dwordx4")) storeDwords = 4;
        else if (mn.contains("dwordx3")) storeDwords = 3;
        else if (mn.contains("dwordx2")) storeDwords = 2;
        else if (mn.contains("dword")) storeDwords = 1;
        else if (mn.contains("short") || mn.contains("_b16")) { storeBits = 16; storeDwords = 0; }
        else if (mn.contains("byte") || mn.contains("_b8")) { storeBits = 8; storeDwords = 0; }

        Value *addr = regs.readReg64(B, op.srcReg(0));
        if (addr->getType() != ptrGlobalTy) addr = B.CreateIntToPtr(addr, ptrGlobalTy);
        int64_t memOffset = 0;
        for (unsigned k = 2; k < op.nSrcs(); k++)
          if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
            memOffset = di.getImm(op.srcIdx(k));
        if (memOffset != 0) addr = B.CreateInBoundsGEP(i8Ty, addr, B.getInt64(memOffset));

        ParsedReg stData = op.srcReg(1);
        if (storeDwords == 0) {
          Type *memTy = Type::getIntNTy(C, storeBits);
          Value *val = B.CreateTrunc(regs.readReg32(B, stData), memTy);
          B.CreateStore(val, addr);
        } else if (storeDwords == 1) {
          B.CreateStore(regs.readReg32(B, stData), addr);
        } else {
          auto *vecTy = FixedVectorType::get(i32Ty, storeDwords);
          B.CreateStore(regs.readRegVec(B, stData, vecTy), addr);
        }
        handled = true; break;
      }

      // ---- Global atomics ----
      if (mn.starts_with("global_atomic_")) {
        ParsedReg addrReg = op.srcReg(0);
        Value *addr = regs.readReg64(B, addrReg);
        if (addr->getType() != ptrGlobalTy) addr = B.CreateIntToPtr(addr, ptrGlobalTy);
        int64_t memOffset = 0;
        unsigned dataIdx = 1;
        for (unsigned k = 1; k < op.nSrcs(); k++) {
          if (di.isImm(op.srcIdx(k)) && di.getImm(op.srcIdx(k)) != 0)
            memOffset = di.getImm(op.srcIdx(k));
          else if (di.isReg(op.srcIdx(k)))
            dataIdx = k;
        }
        if (memOffset != 0) addr = B.CreateInBoundsGEP(i8Ty, addr, B.getInt64(memOffset));
        Value *data = regs.readReg32(B, op.srcReg(dataIdx));

        AtomicRMWInst::BinOp atomicOp;
        Type *atomicTy = i32Ty;
        bool isFP = false;
        if (mn.contains("_add_u32") || mn.contains("_add_i32")) atomicOp = AtomicRMWInst::Add;
        else if (mn.contains("_sub_u32")) atomicOp = AtomicRMWInst::Sub;
        else if (mn.contains("_and_b32")) atomicOp = AtomicRMWInst::And;
        else if (mn.contains("_or_b32")) atomicOp = AtomicRMWInst::Or;
        else if (mn.contains("_xor_b32")) atomicOp = AtomicRMWInst::Xor;
        else if (mn.contains("_smin")) atomicOp = AtomicRMWInst::Min;
        else if (mn.contains("_smax")) atomicOp = AtomicRMWInst::Max;
        else if (mn.contains("_umin")) atomicOp = AtomicRMWInst::UMin;
        else if (mn.contains("_umax")) atomicOp = AtomicRMWInst::UMax;
        else if (mn.contains("_swap")) atomicOp = AtomicRMWInst::Xchg;
        else if (mn.contains("_add_f32")) { atomicOp = AtomicRMWInst::FAdd; atomicTy = f32Ty; isFP = true; }
        else if (mn.contains("_pk_add_bf16")) { atomicOp = AtomicRMWInst::FAdd; atomicTy = FixedVectorType::get(Type::getBFloatTy(C), 2); isFP = true; }
        else if (mn.contains("_pk_add_f16")) { atomicOp = AtomicRMWInst::FAdd; atomicTy = FixedVectorType::get(Type::getHalfTy(C), 2); isFP = true; }
        else {
          errs() << "ir_proto: Unsupported global atomic variant: " << mn << "\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "FLAT"; return result;
        }
        if (isFP) data = B.CreateBitCast(data, atomicTy);
        Value *prev = B.CreateAtomicRMW(atomicOp, addr, data, MaybeAlign(),
                                         AtomicOrdering::Monotonic);
        if (di.numDefs > 0) {
          if (isFP) prev = B.CreateBitCast(prev, i32Ty);
          regs.writeReg32(B, op.dst(), prev);
        }
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // DS: LDS load / store
    // ================================================================
    case FormatKind::DS: {
      bool isDsRead = mn.starts_with("ds_read") || mn.starts_with("ds_load");
      bool isDsWrite = mn.starts_with("ds_write") || mn.starts_with("ds_store");
      if (isDsRead || isDsWrite) {
        int dwords = 1;
        if (mn.contains("_b128") || mn.contains("2_b64")) dwords = 4;
        else if (mn.contains("_b64") || mn.contains("2_b32")) dwords = 2;
        else if (mn.contains("_b32") || mn.contains("_dword")) dwords = 1;
        else if (mn.contains("_b16") || mn.contains("_u16") ||
                 mn.contains("_i16") || mn.contains("_short")) dwords = 0;
        else if (mn.contains("_b8") || mn.contains("_u8") ||
                 mn.contains("_i8") || mn.contains("_byte")) dwords = 0;

        int loadBits = (dwords > 0) ? dwords * 32 : (mn.contains("_b8") || mn.contains("_u8") || mn.contains("_i8") || mn.contains("_byte")) ? 8 : 16;

        // DS addr is a 32-bit VGPR offset into LDS
        Value *addr = B.CreateZExt(op.src(0), i64Ty, "ds_addr");

        // Add immediate offset if present
        for (unsigned k = 1; k < op.nSrcs(); k++) {
          if (di.isImm(op.srcIdx(k))) {
            int64_t imm = di.getImm(op.srcIdx(k));
            if (imm != 0)
              addr = B.CreateAdd(addr, ConstantInt::get(i64Ty, imm), "ds_off");
            break;
          }
        }

        // Use addrspace(3) for LDS
        Value *ptr = B.CreateIntToPtr(addr, PointerType::get(C, 3));

        if (isDsRead) {
          ParsedReg dest = op.dst();
          if (dwords == 0) {
            Type *memTy = Type::getIntNTy(C, loadBits);
            Value *v = B.CreateLoad(memTy, ptr, "ds_ld");
            bool isSigned = mn.contains("_i8") || mn.contains("_i16");
            regs.writeReg32(B, dest, isSigned ? B.CreateSExt(v, i32Ty) : B.CreateZExt(v, i32Ty));
          } else if (dwords == 1) {
            regs.writeReg32(B, dest, B.CreateLoad(i32Ty, ptr, "ds_ld"));
          } else if (dwords == 2) {
            auto *vecTy = FixedVectorType::get(i32Ty, 2);
            regs.writeRegVec(B, dest, B.CreateLoad(vecTy, ptr, "ds_ld"));
          } else {
            auto *vecTy = FixedVectorType::get(i32Ty, dwords);
            regs.writeRegVec(B, dest, B.CreateLoad(vecTy, ptr, "ds_ld"));
          }
          handled = true; break;
        }
        if (isDsWrite) {
          ParsedReg stData = op.srcReg(1);
          if (dwords == 0) {
            Type *memTy = Type::getIntNTy(C, loadBits);
            B.CreateStore(B.CreateTrunc(regs.readReg32(B, stData), memTy), ptr);
          } else if (dwords == 1) {
            B.CreateStore(regs.readReg32(B, stData), ptr);
          } else {
            auto *vecTy = FixedVectorType::get(i32Ty, dwords);
            B.CreateStore(regs.readRegVec(B, stData, vecTy), ptr);
          }
          handled = true; break;
        }
      }
      break;
    }

    // ================================================================
    // MUBUF: buffer_load / buffer_store
    // ================================================================
    case FormatKind::MUBUF: {
      bool isLoad = mn.starts_with("buffer_load");
      bool isStore = mn.starts_with("buffer_store");
      if (isLoad || isStore) {
        int dwords = 1;
        if (mn.contains("dwordx4")) dwords = 4;
        else if (mn.contains("dwordx3")) dwords = 3;
        else if (mn.contains("dwordx2")) dwords = 2;
        else if (mn.contains("dword")) dwords = 1;

        int loadBits = 32 * dwords;
        bool isSubDword = false;
        if (mn.contains("ubyte") || mn.contains("sbyte")) { loadBits = 8; isSubDword = true; }
        else if (mn.contains("ushort") || mn.contains("sshort") ||
                 mn.contains("short_d16")) { loadBits = 16; isSubDword = true; }
        else if (mn.contains("_b8")) { loadBits = 8; isSubDword = true; }
        else if (mn.contains("_b16")) { loadBits = 16; isSubDword = true; }

        // MUBUF buffer resource is a 128-bit descriptor (4 SGPRs):
        //   dword0[47:0]  = base address (low 48 bits)
        //   dword1[29:16] = stride
        //   dword2        = num_records
        //   dword3        = flags (swizzle, format, etc.)
        // We read all 4 dwords. If stride != 0, fail loudly (structured buffers
        // need index*stride address math we don't model).
        ParsedReg vdata = op.dst(0);
        ParsedReg srsrc = op.srcReg(0);
        Value *dw0 = regs.readReg32(B, srsrc);
        ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
        Value *dw1 = regs.readReg32(B, srsrc1);
        if (!dw0 || !dw1) {
          errs() << "ir_proto: MUBUF: cannot read SRSRC for " << mn << "\n";
          return result;
        }
        // Extract 48-bit base address
        Value *lo = B.CreateZExt(dw0, i64Ty);
        Value *hi = B.CreateAnd(B.CreateZExt(dw1, i64Ty), ConstantInt::get(i64Ty, 0xFFFF));
        Value *ptr = B.CreateOr(lo, B.CreateShl(hi, 32), "buf_base");

        // Add vaddr if present and non-zero
        if (op.nSrcs() >= 2 && di.isReg(op.srcIdx(1))) {
          ParsedReg vaddr = op.srcReg(1);
          if (vaddr.kind == ParsedReg::VGPR) {
            Value *voff = B.CreateZExt(regs.readReg32(B, vaddr), i64Ty);
            ptr = B.CreateAdd(ptr, voff, "buf_vaddr");
          }
        }

        // Add soffset
        if (op.nSrcs() >= 3) {
          if (di.isReg(op.srcIdx(2))) {
            ParsedReg soff = op.srcReg(2);
            if (soff.kind == ParsedReg::SGPR) {
              Value *sv = B.CreateZExt(regs.readReg32(B, soff), i64Ty);
              ptr = B.CreateAdd(ptr, sv, "buf_soff");
            }
          }
        }

        // Add immediate offset
        for (unsigned k = 2; k < op.nSrcs(); k++) {
          if (di.isImm(op.srcIdx(k))) {
            int64_t imm = di.getImm(op.srcIdx(k));
            if (imm != 0)
              ptr = B.CreateAdd(ptr, ConstantInt::get(i64Ty, imm), "buf_off");
            break;
          }
        }

        Value *gep = B.CreateIntToPtr(ptr, PointerType::get(C, 0));

        if (isLoad) {
          if (isSubDword) {
            Type *memTy = Type::getIntNTy(C, loadBits);
            Value *loaded = B.CreateLoad(memTy, gep, "buf_ld");
            bool isSigned = mn.contains("sbyte") || mn.contains("sshort");
            Value *ext = isSigned ? B.CreateSExt(loaded, i32Ty)
                                  : B.CreateZExt(loaded, i32Ty);
            regs.writeReg32(B, vdata, ext);
          } else if (dwords == 1) {
            regs.writeReg32(B, vdata, B.CreateLoad(i32Ty, gep, "buf_ld"));
          } else {
            auto *vecTy = FixedVectorType::get(i32Ty, dwords);
            regs.writeRegVec(B, vdata, B.CreateLoad(vecTy, gep, "buf_ld"));
          }
          handled = true; break;
        }
        if (isStore) {
          ParsedReg storeData = op.dst(0);
          if (isSubDword) {
            Type *memTy = Type::getIntNTy(C, loadBits);
            Value *val = B.CreateTrunc(regs.readReg32(B, storeData), memTy);
            B.CreateStore(val, gep);
          } else if (dwords == 1) {
            B.CreateStore(regs.readReg32(B, storeData), gep);
          } else {
            auto *vecTy = FixedVectorType::get(i32Ty, dwords);
            B.CreateStore(regs.readRegVec(B, storeData, vecTy), gep);
          }
          handled = true; break;
        }
      }

      // ---- Buffer atomics ----
      if (mn.starts_with("buffer_atomic_")) {
        ParsedReg srsrc = op.srcReg(0);
        Value *dw0 = regs.readReg32(B, srsrc);
        ParsedReg srsrc1 = srsrc; srsrc1.baseIdx = srsrc.baseIdx + 1;
        Value *dw1 = regs.readReg32(B, srsrc1);
        if (!dw0 || !dw1) {
          errs() << "ir_proto: buffer_atomic: cannot read SRSRC\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "MUBUF"; return result;
        }
        Value *lo = B.CreateZExt(dw0, i64Ty);
        Value *hi = B.CreateAnd(B.CreateZExt(dw1, i64Ty), ConstantInt::get(i64Ty, 0xFFFF));
        Value *ptr = B.CreateOr(lo, B.CreateShl(hi, 32), "buf_base");
        Value *gep = B.CreateIntToPtr(ptr, PointerType::get(C, 0));
        Value *data = regs.readReg32(B, op.dst(0));

        AtomicRMWInst::BinOp atomicOp;
        Type *atomicTy = i32Ty;
        bool isFP = false;
        if (mn.contains("_add_u32") || mn.contains("_add_i32")) atomicOp = AtomicRMWInst::Add;
        else if (mn.contains("_sub_u32")) atomicOp = AtomicRMWInst::Sub;
        else if (mn.contains("_and_b32")) atomicOp = AtomicRMWInst::And;
        else if (mn.contains("_or_b32")) atomicOp = AtomicRMWInst::Or;
        else if (mn.contains("_xor_b32")) atomicOp = AtomicRMWInst::Xor;
        else if (mn.contains("_add_f32")) { atomicOp = AtomicRMWInst::FAdd; atomicTy = f32Ty; isFP = true; }
        else if (mn.contains("_pk_add_bf16")) { atomicOp = AtomicRMWInst::FAdd; atomicTy = FixedVectorType::get(Type::getBFloatTy(C), 2); isFP = true; }
        else if (mn.contains("_pk_add_f16")) { atomicOp = AtomicRMWInst::FAdd; atomicTy = FixedVectorType::get(Type::getHalfTy(C), 2); isFP = true; }
        else {
          errs() << "ir_proto: Unsupported buffer atomic: " << mn << "\n";
          result.failMnemonic = di.mnemonic; result.failFormat = "MUBUF"; return result;
        }
        if (isFP) data = B.CreateBitCast(data, atomicTy);
        B.CreateAtomicRMW(atomicOp, gep, data, MaybeAlign(), AtomicOrdering::Monotonic);
        handled = true; break;
      }
      break;
    }

    // ================================================================
    // MFMA: v_mfma_*
    // ================================================================
    case FormatKind::MFMA: {
      // AGPR move instructions are classified as MFMA format
      if (mn == "v_accvgpr_write_b32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }
      if (mn == "v_accvgpr_read_b32") {
        regs.writeReg32(B, op.dst(), op.src(0));
        handled = true; break;
      }

      struct MfmaInfo {
        Intrinsic::ID id;
        Type *srcTy;
        Type *accumTy;
        unsigned srcDwords;
        unsigned accumDwords;
      };

      auto *v4f16Ty  = FixedVectorType::get(Type::getHalfTy(C), 4);
      auto *v4f32Ty  = FixedVectorType::get(f32Ty, 4);
      auto *v16f32Ty = FixedVectorType::get(f32Ty, 16);
      auto *v32f32Ty = FixedVectorType::get(f32Ty, 32);
      auto *v4i32Ty  = FixedVectorType::get(i32Ty, 4);
      auto *v16i32Ty = FixedVectorType::get(i32Ty, 16);
      auto *v32i32Ty = FixedVectorType::get(i32Ty, 32);
      auto *v2f32Ty  = FixedVectorType::get(f32Ty, 2);
      auto *v4i16Ty  = FixedVectorType::get(Type::getInt16Ty(C), 4);
      auto *v8i32Ty  = FixedVectorType::get(i32Ty, 8);

      // gfx950 scaled MFMA: v_mfma_[scale_]f32_{16x16x128,32x32x64}_f8f6f4
      // These use the llvm.amdgcn.mfma.scale intrinsic with extra scale params.
      // The non-scale version uses scale=0, op_sel=0 (identity scaling).
      if (mn == "v_mfma_f32_16x16x128_f8f6f4" || mn == "v_mfma_scale_f32_16x16x128_f8f6f4" ||
          mn == "v_mfma_f32_32x32x64_f8f6f4" || mn == "v_mfma_scale_f32_32x32x64_f8f6f4") {
        bool is16x16 = mn.contains("16x16x128");
        bool isScale = mn.contains("_scale_");
        Type *accumTy = is16x16 ? (Type*)v4f32Ty : (Type*)v16f32Ty;
        Type *srcTy = v8i32Ty;
        Intrinsic::ID intrId = is16x16
            ? Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4
            : Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4;

        ParsedReg dest = op.dst();
        ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
        ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;
        Value *a = regs.readRegVec(B, srcA, srcTy);
        Value *b = regs.readRegVec(B, srcB, srcTy);
        Value *c = regs.readRegVec(B, srcC, accumTy);

        // Read cbsz and blgp from source immediates
        int cbsz = 0, blgp = 0;
        for (unsigned k = 3; k < op.nSrcs(); k++) {
          if (di.isImm(op.srcIdx(k))) {
            if (cbsz == 0) cbsz = (int)di.getImm(op.srcIdx(k));
            else { blgp = (int)di.getImm(op.srcIdx(k)); break; }
          }
        }

        Value *scaleA = ConstantInt::get(i32Ty, 0);
        Value *scaleB = ConstantInt::get(i32Ty, 0);
        int opSelA = 0, opSelB = 0;
        if (isScale) {
          // Scale versions have additional operands for scale registers and op_sel
          // These come from v_mfma_ld_scale_b32, encoded in extra src operands.
          // For now, read from source registers if available.
          unsigned scaleIdx = 3;
          for (unsigned k = 3; k < op.nSrcs(); k++) {
            if (!di.isImm(op.srcIdx(k)) && op.isSrcReg(k)) {
              if (scaleIdx == 3) { scaleA = regs.readReg32(B, op.srcReg(k)); scaleIdx++; }
              else { scaleB = regs.readReg32(B, op.srcReg(k)); break; }
            }
          }
        }

        Function *fn = Intrinsic::getOrInsertDeclaration(&M, intrId, {srcTy, srcTy});
        Value *result_val = B.CreateCall(fn, {
            a, b, c,
            ConstantInt::get(i32Ty, cbsz), ConstantInt::get(i32Ty, blgp),
            ConstantInt::get(i32Ty, opSelA), scaleA,
            ConstantInt::get(i32Ty, opSelB), scaleB
        }, "mfma_scale");
        regs.writeRegVec(B, dest, result_val);
        handled = true; break;
      }

      std::map<std::string, MfmaInfo> mfmaTable = {
        {"v_mfma_f32_16x16x16_f16",  {Intrinsic::amdgcn_mfma_f32_16x16x16f16,  v4f16Ty, v4f32Ty, 2, 4}},
        {"v_mfma_f32_16x16x16f16",   {Intrinsic::amdgcn_mfma_f32_16x16x16f16,  v4f16Ty, v4f32Ty, 2, 4}},
        {"v_mfma_f32_32x32x8_f16",   {Intrinsic::amdgcn_mfma_f32_32x32x8f16,   v4f16Ty, v16f32Ty, 2, 16}},
        {"v_mfma_f32_32x32x8f16",    {Intrinsic::amdgcn_mfma_f32_32x32x8f16,   v4f16Ty, v16f32Ty, 2, 16}},
        {"v_mfma_f32_16x16x4_f32",   {Intrinsic::amdgcn_mfma_f32_16x16x4f32,   f32Ty,   v16f32Ty, 1, 16}},
        {"v_mfma_f32_16x16x4f32",    {Intrinsic::amdgcn_mfma_f32_16x16x4f32,   f32Ty,   v16f32Ty, 1, 16}},
        {"v_mfma_f32_32x32x1_f32",   {Intrinsic::amdgcn_mfma_f32_32x32x1f32,   f32Ty,   v32f32Ty, 1, 32}},
        {"v_mfma_f32_32x32x2_f32",   {Intrinsic::amdgcn_mfma_f32_32x32x2f32,   f32Ty,   v16f32Ty, 1, 16}},
        {"v_mfma_f32_4x4x1_f32",     {Intrinsic::amdgcn_mfma_f32_4x4x1f32,     f32Ty,   v4f32Ty,  1, 4}},
        {"v_mfma_f32_16x16x1_f32",   {Intrinsic::amdgcn_mfma_f32_16x16x1f32,   f32Ty,   v16f32Ty, 1, 16}},
        {"v_mfma_f32_32x32x4_f16",   {Intrinsic::amdgcn_mfma_f32_32x32x4f16,   v4f16Ty, v32f32Ty, 2, 32}},
        {"v_mfma_f32_16x16x4_f16",   {Intrinsic::amdgcn_mfma_f32_16x16x4f16,   v4f16Ty, v16f32Ty, 2, 16}},
        {"v_mfma_f32_4x4x4_f16",     {Intrinsic::amdgcn_mfma_f32_4x4x4f16,     v4f16Ty, v4f32Ty,  2, 4}},
        {"v_mfma_i32_16x16x32_i8",   {Intrinsic::amdgcn_mfma_i32_16x16x32_i8,  i64Ty,   v4i32Ty,  2, 4}},
        {"v_mfma_i32_32x32x16_i8",   {Intrinsic::amdgcn_mfma_i32_32x32x16_i8,  i64Ty,   v16i32Ty, 2, 16}},
        {"v_mfma_f32_16x16x8_xf32",  {Intrinsic::amdgcn_mfma_f32_16x16x8_xf32, v2f32Ty, v4f32Ty,  2, 4}},
        {"v_mfma_f32_32x32x4_xf32",  {Intrinsic::amdgcn_mfma_f32_32x32x4_xf32, v2f32Ty, v16f32Ty, 2, 16}},
        {"v_mfma_i32_32x32x4_i8",    {Intrinsic::amdgcn_mfma_i32_32x32x4i8,    i32Ty,   v32i32Ty, 1, 32}},
        {"v_mfma_i32_16x16x4_i8",    {Intrinsic::amdgcn_mfma_i32_16x16x4i8,    i32Ty,   v16i32Ty, 1, 16}},
        {"v_mfma_i32_4x4x4_i8",      {Intrinsic::amdgcn_mfma_i32_4x4x4i8,      i32Ty,   v4i32Ty,  1, 4}},
        {"v_mfma_f32_32x32x2_bf16",  {Intrinsic::amdgcn_mfma_f32_32x32x2bf16,  v4i16Ty, v32f32Ty, 2, 32}},
        {"v_mfma_f32_16x16x2_bf16",  {Intrinsic::amdgcn_mfma_f32_16x16x2bf16,  v4i16Ty, v16f32Ty, 2, 16}},
        {"v_mfma_f32_4x4x2_bf16",    {Intrinsic::amdgcn_mfma_f32_4x4x2bf16,    v4i16Ty, v4f32Ty,  2, 4}},
        // gfx942 bf16 "1K" shapes (v4i16 = 2 dwords, but these are v8bf16 equivalent in 1K encoding)
        {"v_mfma_f32_16x16x16_bf16", {Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k, v4i16Ty, v4f32Ty,  2, 4}},
        {"v_mfma_f32_32x32x8_bf16",  {Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k,  v4i16Ty, v16f32Ty, 2, 16}},
        // gfx950 bf16 wider shapes (v8bf16 = 4 dwords)
        {"v_mfma_f32_16x16x32_bf16", {Intrinsic::amdgcn_mfma_f32_16x16x32_bf16, FixedVectorType::get(Type::getBFloatTy(C), 8), v4f32Ty,  4, 4}},
        {"v_mfma_f32_32x32x16_bf16", {Intrinsic::amdgcn_mfma_f32_32x32x16_bf16, FixedVectorType::get(Type::getBFloatTy(C), 8), v16f32Ty, 4, 16}},
        // gfx950 f16 wider shapes (v8f16 = 4 dwords)
        {"v_mfma_f32_16x16x32_f16",  {Intrinsic::amdgcn_mfma_f32_16x16x32_f16,  FixedVectorType::get(Type::getHalfTy(C), 8), v4f32Ty,  4, 4}},
        // gfx942 fp8 variants (i64 = 2 dwords)
        {"v_mfma_f32_16x16x32_fp8_fp8",  {Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8,  i64Ty, v4f32Ty,  2, 4}},
        {"v_mfma_f32_16x16x32_fp8_bf8",  {Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8,  i64Ty, v4f32Ty,  2, 4}},
        {"v_mfma_f32_16x16x32_bf8_fp8",  {Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8,  i64Ty, v4f32Ty,  2, 4}},
        {"v_mfma_f32_16x16x32_bf8_bf8",  {Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8,  i64Ty, v4f32Ty,  2, 4}},
        {"v_mfma_f32_32x32x16_fp8_fp8",  {Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8,  i64Ty, v16f32Ty, 2, 16}},
        {"v_mfma_f32_32x32x16_fp8_bf8",  {Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8,  i64Ty, v16f32Ty, 2, 16}},
        {"v_mfma_f32_32x32x16_bf8_fp8",  {Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8,  i64Ty, v16f32Ty, 2, 16}},
        {"v_mfma_f32_32x32x16_bf8_bf8",  {Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8,  i64Ty, v16f32Ty, 2, 16}},
      };

      auto it = mfmaTable.find(di.mnemonic);
      if (it == mfmaTable.end()) {
        result.failMnemonic = di.mnemonic;
        result.failFormat = "MFMA";
        errs() << "ir_proto: Unknown MFMA: " << di.mnemonic << "\n";
        return result;
      }

      auto &info = it->second;
      ParsedReg dest = op.dst();
      ParsedReg srcA = op.srcReg(0), srcB = op.srcReg(1);
      // The accumulator (src2) may be tied to the destination in some encodings.
      ParsedReg srcC = op.isSrcReg(2) ? op.srcReg(2) : dest;
      if (srcA.kind == ParsedReg::OTHER || srcB.kind == ParsedReg::OTHER) {
        result.failMnemonic = di.mnemonic; result.failFormat = "MFMA";
        errs() << "ir_proto: MFMA " << mn << ": cannot read source registers\n";
        return result;
      }
      Value *a = regs.readRegVec(B, srcA, info.srcTy);
      Value *b = regs.readRegVec(B, srcB, info.srcTy);
      Value *c = regs.readRegVec(B, srcC, info.accumTy);

      int cbsz = 0, abid = 0, blgp = 0;
      unsigned immIdx = 0;
      for (unsigned k = 3; k < op.nSrcs(); k++) {
        if (di.isImm(op.srcIdx(k))) {
          int64_t v = di.getImm(op.srcIdx(k));
          if (immIdx == 0) cbsz = v;
          else if (immIdx == 1) abid = v;
          else if (immIdx == 2) blgp = v;
          immIdx++;
        }
      }

      Function *mfmaFn = Intrinsic::getOrInsertDeclaration(&M, info.id);
      regs.writeRegVec(B, dest, B.CreateCall(mfmaFn, {
          a, b, c,
          ConstantInt::get(i32Ty, cbsz),
          ConstantInt::get(i32Ty, abid),
          ConstantInt::get(i32Ty, blgp)
      }, "mfma"));
      handled = true; break;
    }

    default:
      break;
    } // switch (di.format)

    if (handled) {
      // Auto SCC writeback: if hardware defines SCC and handler didn't
      // write it explicitly, emit SCC = (sccResult != 0).
      if (di.defsSCC && !sccHandled && sccResult) {
        Value *zero = Constant::getNullValue(sccResult->getType());
        regs.storeSCC(B, B.CreateICmpNE(sccResult, zero));
      }
      raisedCount++;
      continue;
    }

    // ---- Unrecognized: fail loudly ----
    result.failMnemonic = di.mnemonic;
    result.failFormat = formatName(di.format);
    errs() << "ir_proto: Unsupported instruction: " << di.mnemonic
           << " (raw: " << di.rawMnemonic << ")"
           << " [format=" << formatName(di.format) << "]"
           << " at offset 0x" << format_hex(di.offset, 1) << "\n";
    return result;
  }

  if (currentBB && !currentBB->getTerminator())
    B.CreateUnreachable();

  result.liftedCount = raisedCount;

  // ==== Phase 6: Promote allocas to SSA ====
  {
    DominatorTree DT(*F);
    AssumptionCache AC(*F);
    SmallVector<AllocaInst *, 512> allocas;
    regs.collectAllocas(allocas);
    PromoteMemToReg(allocas, DT, &AC);
  }

  // ==== Phase 7: Verify IR ====
  std::string verifyErr;
  raw_string_ostream verifyOS(verifyErr);
  if (verifyModule(M, &verifyOS)) {
    errs() << "ir_proto: IR verification failed:\n" << verifyErr << "\n";
    return result;
  }

  {
    raw_string_ostream irOS(result.irText);
    M.print(irOS, nullptr);
  }

  result.success = true;
  return result;
}

} // namespace ir_proto
