#include "raiser.hpp"

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
// Decoded instruction — keeps the raw MCInst for positional operand access
// ============================================================================

struct DecodedInst {
  std::string mnemonic; // encoding suffix stripped
  MCInst inst;
  unsigned numDefs = 0;
  bool isBranch = false;
  bool isConditionalBranch = false;
  uint64_t offset = 0;
  uint64_t size = 0;

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
// Register classification — maps MCPhysReg to logical SGPR/VGPR index + width
// ============================================================================

struct ParsedReg {
  enum Kind { SGPR, VGPR, VCC, EXEC, SCC, MODE, NOREG, OTHER };
  Kind kind = OTHER;
  int baseIdx = -1;
  int width = 1; // in 32-bit dwords
};

ParsedReg parseReg(const MCState &mc, unsigned reg) {
  ParsedReg pr;
  if (reg == 0) {
    pr.kind = ParsedReg::NOREG;
    return pr;
  }
  StringRef name = mc.regInfo->getName(reg);

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
  return pr;
}

// ============================================================================
// Register file — tracks physical register → IR Value mappings
// ============================================================================

class RegFile {
  struct Entry {
    Value *val;
    int width; // 1, 2, or 4 dwords
  };
  std::map<int, Entry> entries_;

public:
  void set(int baseIdx, int width, Value *val) {
    // Remove overlapping entries
    auto it = entries_.begin();
    while (it != entries_.end()) {
      int eStart = it->first;
      int eEnd = eStart + it->second.width;
      int nStart = baseIdx;
      int nEnd = baseIdx + width;
      if (eStart < nEnd && nStart < eEnd)
        it = entries_.erase(it);
      else
        ++it;
    }
    entries_[baseIdx] = {val, width};
  }

  Value *get(int baseIdx, int width, IRBuilder<> &B) const {
    // Exact match
    auto it = entries_.find(baseIdx);
    if (it != entries_.end() && it->second.width == width)
      return it->second.val;

    // Extract a sub-range from a wider entry
    for (auto &[eBase, entry] : entries_) {
      if (eBase <= baseIdx && eBase + entry.width >= baseIdx + width) {
        int subIdx = baseIdx - eBase;
        Value *v = entry.val;
        if (width == 1 && entry.width == 2) {
          Type *i64Ty = B.getInt64Ty();
          Type *i32Ty = B.getInt32Ty();
          if (v->getType()->isPointerTy())
            v = B.CreatePtrToInt(v, i64Ty);
          else if (v->getType() != i64Ty)
            v = B.CreateBitCast(v, i64Ty);
          if (subIdx == 0)
            return B.CreateTrunc(v, i32Ty);
          return B.CreateTrunc(B.CreateLShr(v, 32), i32Ty);
        }
        if (width == 2 && entry.width == 4) {
          Type *i128Ty = Type::getIntNTy(B.getContext(), 128);
          Type *i64Ty = B.getInt64Ty();
          if (v->getType() != i128Ty)
            v = B.CreateBitCast(v, i128Ty);
          if (subIdx == 0)
            return B.CreateTrunc(v, i64Ty);
          return B.CreateTrunc(B.CreateLShr(v, 64), i64Ty);
        }
      }
    }

    // Combine two halves into a pair
    if (width == 2) {
      Value *lo = get(baseIdx, 1, B);
      Value *hi = get(baseIdx + 1, 1, B);
      if (lo && hi) {
        Type *i32Ty = B.getInt32Ty();
        Type *i64Ty = B.getInt64Ty();
        if (lo->getType() != i32Ty)
          lo = B.CreateBitOrPointerCast(lo, i32Ty);
        if (hi->getType() != i32Ty)
          hi = B.CreateBitOrPointerCast(hi, i32Ty);
        Value *loExt = B.CreateZExt(lo, i64Ty);
        Value *hiExt = B.CreateZExt(hi, i64Ty);
        return B.CreateOr(loExt, B.CreateShl(hiExt, 32));
      }
    }
    return nullptr;
  }
};

// ============================================================================
// Kernarg layout — maps byte offsets to function parameters
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

  // Resolve a kernarg load: which function args does it cover?
  void resolveLoad(
      int byteOffset, int loadBytes,
      std::vector<std::pair<int /*regWidth*/, int /*paramIdx*/>> &out) const {
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
                      const std::string &kernelName) {
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
      di.mnemonic = stripEncoding(getMnemonic(mc, inst)).str();
      di.inst = inst;
      di.numDefs = desc.getNumDefs();
      di.isBranch = desc.isBranch();
      di.isConditionalBranch = desc.isConditionalBranch();
      di.offset = off;
      di.size = instSize;

      if (di.isBranch) {
        for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
          if (inst.getOperand(i).isImm()) {
            int64_t brOff = inst.getOperand(i).getImm();
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
  auto *i8Ty = Type::getInt8Ty(C);
  auto *i32Ty = Type::getInt32Ty(C);
  auto *i64Ty = Type::getInt64Ty(C);
  auto *f32Ty = Type::getFloatTy(C);
  auto *ptrGlobalTy = PointerType::get(C, 1);

  // TODO: parse kernel signature from ELF metadata for generality
  auto *funcTy = FunctionType::get(
      voidTy, {ptrGlobalTy, ptrGlobalTy, ptrGlobalTy, i32Ty}, false);
  Function *F =
      Function::Create(funcTy, GlobalValue::ExternalLinkage, kernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1024");
  F->getArg(0)->setName("A");
  F->getArg(1)->setName("B");
  F->getArg(2)->setName("C");
  F->getArg(3)->setName("N");

  KernargLayout kernargs;
  kernargs.params = {{0, 8, 0, true},
                     {8, 8, 1, true},
                     {16, 8, 2, true},
                     {24, 4, 3, false}};
  kernargs.implicitArgsBase = 32;

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

  RegFile sgprs, vgprs;
  Value *vccVal = nullptr;

  // s[0:1] = kernarg segment pointer — sentinel; kernarg loads handle it
  sgprs.set(0, 2, Constant::getNullValue(PointerType::get(C, 4)));
  // s2 = workgroup_id_x
  sgprs.set(2, 1, B.CreateCall(fnWorkgroupIdX, {}, "wg_id_x"));
  // v0 = workitem_id_x
  vgprs.set(0, 1, B.CreateCall(fnWorkitemIdX, {}, "tid"));

  // ==== Phase 5: Raise each instruction ====
  BasicBlock *currentBB = offsetToBB[0];
  int raisedCount = 0;

  // Helper: read a register operand from the appropriate file
  auto readReg = [&](const DecodedInst &di, unsigned opIdx) -> Value * {
    ParsedReg pr = parseReg(mc, di.getReg(opIdx));
    switch (pr.kind) {
    case ParsedReg::SGPR:
      return sgprs.get(pr.baseIdx, pr.width, B);
    case ParsedReg::VGPR:
      return vgprs.get(pr.baseIdx, pr.width, B);
    case ParsedReg::VCC:
      return vccVal;
    default:
      return nullptr;
    }
  };

  // Helper: read an operand that's either register or immediate
  auto readOp32 = [&](const DecodedInst &di, unsigned opIdx) -> Value * {
    if (di.isReg(opIdx))
      return readReg(di, opIdx);
    if (di.isImm(opIdx))
      return ConstantInt::get(i32Ty, di.getImm(opIdx));
    return nullptr;
  };

  auto readOp64 = [&](const DecodedInst &di, unsigned opIdx) -> Value * {
    if (di.isReg(opIdx))
      return readReg(di, opIdx);
    if (di.isImm(opIdx))
      return ConstantInt::get(i64Ty, di.getImm(opIdx));
    return nullptr;
  };

  // Helper: write to the register identified by operand opIdx
  auto writeReg = [&](const DecodedInst &di, unsigned opIdx, Value *v) {
    ParsedReg pr = parseReg(mc, di.getReg(opIdx));
    if (pr.kind == ParsedReg::SGPR)
      sgprs.set(pr.baseIdx, pr.width, v);
    else if (pr.kind == ParsedReg::VGPR)
      vgprs.set(pr.baseIdx, pr.width, v);
  };

  for (size_t instIdx = 0; instIdx < insts.size(); ++instIdx) {
    const DecodedInst &di = insts[instIdx];

    // Switch basic block if this instruction starts a new one
    auto bbIt = offsetToBB.find(di.offset);
    if (bbIt != offsetToBB.end() && bbIt->second != currentBB) {
      if (!currentBB->getTerminator())
        B.CreateBr(bbIt->second);
      currentBB = bbIt->second;
      B.SetInsertPoint(currentBB);
    }

    StringRef mn(di.mnemonic);

    // ---- Non-semantic: skip ----
    if (mn == "s_waitcnt" || mn == "s_nop" || mn == "s_code_end") {
      raisedCount++;
      continue;
    }

    // ---- Scalar loads from kernarg segment ----
    // s_load_dword[xN] dest, s[0:1], offset
    // op0=dest(reg), op1=base(reg), op2=offset(imm), op3=flags(imm)
    if (mn.starts_with("s_load_dword")) {
      int loadDwords = 1;
      if (mn.contains("dwordx2"))
        loadDwords = 2;
      else if (mn.contains("dwordx4"))
        loadDwords = 4;
      else if (mn.contains("dwordx8"))
        loadDwords = 8;
      int loadBytes = loadDwords * 4;

      ParsedReg dest = parseReg(mc, di.getReg(0));
      int64_t byteOffset = di.getImm(2); // op2 is always the offset for SMEM

      if (byteOffset < kernargs.implicitArgsBase) {
        // Explicit kernel arguments
        std::vector<std::pair<int, int>> resolved;
        kernargs.resolveLoad(byteOffset, loadBytes, resolved);

        if (resolved.empty()) {
          errs() << "ir_proto: Cannot resolve kernarg at offset " << byteOffset
                 << "\n";
          return result;
        }

        int regOff = 0;
        for (auto &[regWidth, paramIdx] : resolved) {
          Value *arg = F->getArg(paramIdx);
          if (dest.kind == ParsedReg::SGPR)
            sgprs.set(dest.baseIdx + regOff, regWidth, arg);
          regOff += regWidth;
        }
      } else {
        // Implicit args region
        int implOffset = byteOffset - kernargs.implicitArgsBase;
        Value *implPtr =
            B.CreateCall(fnImplicitArgPtr, {}, "implicitarg_ptr");
        Value *gep = B.CreateInBoundsGEP(i8Ty, implPtr,
                                          B.getInt64(implOffset), "impl_gep");
        Value *loaded = B.CreateLoad(i32Ty, gep, "impl_load");
        sgprs.set(dest.baseIdx, loadDwords, loaded);
      }

      raisedCount++;
      continue;
    }

    // ---- s_and_b32 dest, src0, src1 ----
    // op0=dest(reg), op1=src0(reg), op2=src1(reg or imm)
    if (mn == "s_and_b32") {
      Value *src0 = readOp32(di, 1);
      Value *src1 = readOp32(di, 2);
      if (!src0 || !src1) {
        errs() << "ir_proto: s_and_b32: missing operand\n";
        return result;
      }
      if (src0->getType() != i32Ty)
        src0 = B.CreateBitOrPointerCast(src0, i32Ty);
      writeReg(di, 0, B.CreateAnd(src0, src1, "and"));
      raisedCount++;
      continue;
    }

    // ---- s_mul_i32 dest, src0, src1 ----
    // op0=dest(reg), op1=src0(reg), op2=src1(reg)
    if (mn == "s_mul_i32") {
      Value *src0 = readOp32(di, 1);
      Value *src1 = readOp32(di, 2);
      if (!src0 || !src1) {
        errs() << "ir_proto: s_mul_i32: missing operand\n";
        return result;
      }
      writeReg(di, 0, B.CreateMul(src0, src1, "mul"));
      raisedCount++;
      continue;
    }

    // ---- v_add_u32 dest, src0, src1 ----
    // op0=dest(reg), op1=src0(reg or imm), op2=src1(reg)
    if (mn == "v_add_u32") {
      Value *src0 = readOp32(di, 1);
      Value *src1 = readOp32(di, 2);
      if (!src0 || !src1) {
        errs() << "ir_proto: v_add_u32: missing operand\n";
        return result;
      }
      writeReg(di, 0, B.CreateAdd(src0, src1, "add"));
      raisedCount++;
      continue;
    }

    // ---- v_cmp_gt_i32 src0, src1 (implicit-def VCC) ----
    // op0=src0(reg), op1=src1(reg) — no explicit def, VCC is implicit
    if (mn == "v_cmp_gt_i32") {
      Value *src0 = readOp32(di, 0);
      Value *src1 = readOp32(di, 1);
      if (!src0 || !src1) {
        errs() << "ir_proto: v_cmp_gt_i32: missing operand\n";
        return result;
      }
      // src0 > src1 → in IR: icmp sgt
      vccVal = B.CreateICmpSGT(src0, src1, "cmp");
      raisedCount++;
      continue;
    }

    // ---- s_and_saveexec_b64 dest, src ----
    // Saves old exec to dest, sets exec = exec AND src.
    // For branch translation: just note VCC as the upcoming branch condition.
    // op0=dest(reg), op1=src(VCC reg)
    if (mn == "s_and_saveexec_b64") {
      // Store a placeholder for the old exec mask (not used in simple kernels)
      sgprs.set(parseReg(mc, di.getReg(0)).baseIdx, 2,
                ConstantInt::get(i64Ty, -1));
      // vccVal already holds the condition from v_cmp_*
      raisedCount++;
      continue;
    }

    // ---- s_cbranch_execz offset ----
    // Branch to target if exec is zero (condition was false).
    // op0=offset(imm)
    if (mn == "s_cbranch_execz") {
      int64_t brOff = di.getImm(0);
      uint64_t target = di.offset + 4 + brOff * 4;
      BasicBlock *targetBB = offsetToBB[target];
      BasicBlock *fallthroughBB = offsetToBB[di.offset + di.size];

      if (!vccVal || !targetBB || !fallthroughBB) {
        errs() << "ir_proto: s_cbranch_execz: missing condition or BB\n";
        return result;
      }
      // VCC true → exec active → fall through; VCC false → branch to target
      B.CreateCondBr(vccVal, fallthroughBB, targetBB);
      raisedCount++;
      continue;
    }

    // ---- v_ashrrev_i32 dest, shamt(imm), src(reg) ----
    // dest = src >> shamt (arithmetic)
    // op0=dest(reg), op1=shamt(imm), op2=src(reg)
    if (mn == "v_ashrrev_i32") {
      Value *src = readOp32(di, 2);
      int64_t shamt = di.getImm(1);
      if (!src) {
        errs() << "ir_proto: v_ashrrev_i32: missing operand\n";
        return result;
      }
      writeReg(di, 0,
               B.CreateAShr(src, ConstantInt::get(i32Ty, shamt), "ashr"));
      raisedCount++;
      continue;
    }

    // ---- v_lshlrev_b64 dest, shamt(imm), src(reg pair) ----
    // dest = src << shamt (64-bit)
    // op0=dest(reg pair), op1=shamt(imm), op2=src(reg pair)
    if (mn == "v_lshlrev_b64") {
      Value *src = readOp64(di, 2);
      int64_t shamt = di.getImm(1);
      if (!src) {
        errs() << "ir_proto: v_lshlrev_b64: missing operand\n";
        return result;
      }
      if (src->getType() != i64Ty)
        src = B.CreateBitOrPointerCast(src, i64Ty);
      writeReg(di, 0,
               B.CreateShl(src, ConstantInt::get(i64Ty, shamt), "shl"));
      raisedCount++;
      continue;
    }

    // ---- v_lshl_add_u64 dest, src0, shift(imm), src2 ----
    // dest = (src0 << shift) + src2
    // op0=dest(reg pair), op1=src0(reg pair), op2=shift(imm), op3=src2(reg pair)
    if (mn == "v_lshl_add_u64") {
      Value *src0 = readOp64(di, 1);
      int64_t shift = di.getImm(2);
      Value *src2 = readOp64(di, 3);

      if (!src0 || !src2) {
        errs() << "ir_proto: v_lshl_add_u64: missing operand\n";
        return result;
      }

      // Special case: if src0 is a pointer and shift is 0, use GEP
      if (shift == 0 && src0->getType()->isPointerTy()) {
        if (src2->getType() != i64Ty)
          src2 = B.CreateBitOrPointerCast(src2, i64Ty);
        writeReg(di, 0,
                 B.CreateInBoundsGEP(i8Ty, src0, src2, "gep_addr"));
      } else {
        if (src0->getType()->isPointerTy())
          src0 = B.CreatePtrToInt(src0, i64Ty);
        if (src0->getType() != i64Ty)
          src0 = B.CreateBitOrPointerCast(src0, i64Ty);
        if (src2->getType() != i64Ty)
          src2 = B.CreateBitOrPointerCast(src2, i64Ty);
        Value *shifted = (shift == 0) ? src0
                                      : B.CreateShl(src0,
                                                    ConstantInt::get(i64Ty, shift));
        writeReg(di, 0, B.CreateAdd(shifted, src2, "lshl_add"));
      }
      raisedCount++;
      continue;
    }

    // ---- global_load_dword dest, addr(reg pair), off ----
    // op0=dest(reg), op1=addr(reg pair), op2=offset(imm), ...
    if (mn.starts_with("global_load_dword") && !mn.contains("x")) {
      Value *addr = readReg(di, 1);
      if (!addr) {
        errs() << "ir_proto: global_load_dword: missing address\n";
        return result;
      }
      // Convert integer address to pointer if needed
      if (addr->getType() != ptrGlobalTy) {
        if (addr->getType() != i64Ty)
          addr = B.CreateBitOrPointerCast(addr, i64Ty);
        addr = B.CreateIntToPtr(addr, ptrGlobalTy);
      }
      Value *loaded = B.CreateLoad(f32Ty, addr, "gload");
      writeReg(di, 0, loaded);
      raisedCount++;
      continue;
    }

    // ---- global_store_dword addr(reg pair), data(reg), off ----
    // Note: store has NO def — op0=addr(use), op1=data(use), op2=offset(imm)
    if (mn.starts_with("global_store_dword") && !mn.contains("x")) {
      Value *addr = readReg(di, 0);
      Value *data = readReg(di, 1);
      if (!addr || !data) {
        errs() << "ir_proto: global_store_dword: missing operand\n";
        return result;
      }
      if (addr->getType() != ptrGlobalTy) {
        if (addr->getType() != i64Ty)
          addr = B.CreateBitOrPointerCast(addr, i64Ty);
        addr = B.CreateIntToPtr(addr, ptrGlobalTy);
      }
      if (data->getType() != f32Ty)
        data = B.CreateBitCast(data, f32Ty);
      B.CreateStore(data, addr);
      raisedCount++;
      continue;
    }

    // ---- v_add_f32 dest, src0, src1 ----
    // op0=dest(reg), op1=src0(reg), op2=src1(reg)
    if (mn == "v_add_f32") {
      Value *src0 = readReg(di, 1);
      Value *src1 = readReg(di, 2);
      if (!src0 || !src1) {
        errs() << "ir_proto: v_add_f32: missing operand\n";
        return result;
      }
      if (src0->getType() != f32Ty)
        src0 = B.CreateBitCast(src0, f32Ty);
      if (src1->getType() != f32Ty)
        src1 = B.CreateBitCast(src1, f32Ty);
      writeReg(di, 0, B.CreateFAdd(src0, src1, "fadd"));
      raisedCount++;
      continue;
    }

    // ---- s_endpgm ----
    if (mn == "s_endpgm") {
      B.CreateRetVoid();
      raisedCount++;
      continue;
    }

    // ---- Unrecognized instruction: fail loudly ----
    errs() << "ir_proto: Unsupported instruction: " << di.mnemonic
           << " at offset 0x" << format_hex(di.offset, 1) << "\n";
    return result;
  }

  // Close any open block that lacks a terminator (shouldn't happen for valid code)
  if (currentBB && !currentBB->getTerminator())
    B.CreateUnreachable();

  result.liftedCount = raisedCount;

  // ==== Phase 6: Verify IR ====
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
