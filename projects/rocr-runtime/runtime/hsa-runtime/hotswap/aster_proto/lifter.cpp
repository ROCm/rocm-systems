#include "lifter.hpp"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Target/ASM/TranslateModule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"

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

#include <map>
#include <set>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace aster_proto {

namespace {

struct MCState {
  const llvm::Target *target = nullptr;
  std::unique_ptr<llvm::MCInstrInfo> instrInfo;
  std::unique_ptr<llvm::MCRegisterInfo> regInfo;
  std::unique_ptr<llvm::MCSubtargetInfo> subtargetInfo;
  std::unique_ptr<const llvm::MCAsmInfo> asmInfo;
  std::unique_ptr<llvm::MCContext> ctx;
  std::unique_ptr<llvm::MCDisassembler> disasm;
  std::unique_ptr<llvm::MCInstPrinter> printer;
};

bool initMCState(MCState &state, const std::string &targetISA) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  llvm::Triple triple("amdgcn-amd-amdhsa");
  std::string error;
  state.target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!state.target) {
    llvm::errs() << "aster_proto: Failed to find AMDGPU target: " << error
                 << "\n";
    return false;
  }

  state.instrInfo.reset(state.target->createMCInstrInfo());
  state.regInfo.reset(state.target->createMCRegInfo(triple));
  state.subtargetInfo.reset(
      state.target->createMCSubtargetInfo(triple, targetISA, ""));
  state.asmInfo.reset(
      state.target->createMCAsmInfo(*state.regInfo, triple,
                                    llvm::MCTargetOptions()));
  state.ctx = std::make_unique<llvm::MCContext>(
      triple, state.asmInfo.get(), state.regInfo.get(),
      state.subtargetInfo.get());
  state.disasm.reset(
      state.target->createMCDisassembler(*state.subtargetInfo, *state.ctx));
  state.printer.reset(state.target->createMCInstPrinter(
      triple, 0, *state.asmInfo, *state.instrInfo, *state.regInfo));
  state.printer->setPrintImmHex(true);

  return state.disasm != nullptr;
}

std::string getMnemonic(const MCState &mc, const llvm::MCInst &inst) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mc.printer->printInst(&inst, 0, "", *mc.subtargetInfo, os);
  llvm::StringRef ref(s);
  ref = ref.ltrim();
  auto spacePos = ref.find_first_of(" \t");
  if (spacePos != llvm::StringRef::npos)
    ref = ref.substr(0, spacePos);
  return ref.str();
}

std::string getFullText(const MCState &mc, const llvm::MCInst &inst) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mc.printer->printInst(&inst, 0, "", *mc.subtargetInfo, os);
  return llvm::StringRef(s).trim().str();
}

enum class RegClass { SGPR, VGPR, AGPR, VCC, EXEC, SCC, Other };

struct RegInfo {
  RegClass cls;
  int index;
  int rangeSize;
};

RegInfo classifyReg(const MCState &mc, unsigned mcReg) {
  llvm::StringRef name = mc.regInfo->getName(mcReg);

  if (name == "VCC" || name == "VCC_LO" || name == "VCC_HI")
    return {RegClass::VCC, 0, name == "VCC" ? 2 : 1};
  if (name.starts_with("EXEC"))
    return {RegClass::EXEC, 0, name == "EXEC" ? 2 : 1};
  if (name == "SCC")
    return {RegClass::SCC, 0, 1};

  auto tryRange = [&](llvm::StringRef prefix, RegClass cls) -> RegInfo {
    auto subRegs = mc.regInfo->subregs(mcReg);
    int first = 9999, count = 0;
    for (auto sub : subRegs) {
      llvm::StringRef sn = mc.regInfo->getName(sub);
      // Only count direct scalar sub-registers (e.g., SGPR0, VGPR1),
      // not nested tuple registers (e.g., SGPR0_SGPR1)
      if (sn.starts_with(prefix)) {
        auto suffix = sn.substr(prefix.size());
        int idx = 0;
        if (!suffix.getAsInteger(10, idx) && suffix.size() > 0) {
          if (idx < first) first = idx;
          count++;
        }
      }
    }
    if (count > 0 && first < 9999)
      return {cls, first, count};
    return {RegClass::Other, 0, 0};
  };

  auto isSingleReg = [](llvm::StringRef name, llvm::StringRef prefix) -> bool {
    if (!name.starts_with(prefix)) return false;
    auto suffix = name.substr(prefix.size());
    int idx;
    return !suffix.getAsInteger(10, idx) && suffix.size() > 0;
  };

  // Try ranges first (e.g., SGPR0_SGPR1, VGPR_32_...)
  if (name.starts_with("SGPR") && !isSingleReg(name, "SGPR")) {
    auto r = tryRange("SGPR", RegClass::SGPR);
    if (r.rangeSize > 0) return r;
  }
  if (name.starts_with("VGPR") && !isSingleReg(name, "VGPR")) {
    auto r = tryRange("VGPR", RegClass::VGPR);
    if (r.rangeSize > 0) return r;
  }
  if (name.starts_with("AGPR") && !isSingleReg(name, "AGPR")) {
    auto r = tryRange("AGPR", RegClass::AGPR);
    if (r.rangeSize > 0) return r;
  }

  // Single registers
  if (name.starts_with("VGPR")) {
    int idx = 0;
    name.substr(4).getAsInteger(10, idx);
    return {RegClass::VGPR, idx, 1};
  }
  if (name.starts_with("SGPR")) {
    int idx = 0;
    name.substr(4).getAsInteger(10, idx);
    return {RegClass::SGPR, idx, 1};
  }
  if (name.starts_with("AGPR")) {
    int idx = 0;
    name.substr(4).getAsInteger(10, idx);
    return {RegClass::AGPR, idx, 1};
  }
  return {RegClass::Other, 0, 1};
}

Value getOrCreateSingleReg(OpBuilder &builder, Location loc,
                           std::map<std::pair<RegClass, int>, Value> &cache,
                           MLIRContext *ctx, RegClass cls, int index) {
  auto key = std::make_pair(cls, index);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;

  Type regTy;
  if (cls == RegClass::VGPR)
    regTy = VGPRType::get(ctx, Register(index));
  else if (cls == RegClass::SGPR)
    regTy = SGPRType::get(ctx, Register(index));
  else if (cls == RegClass::AGPR)
    regTy = AGPRType::get(ctx, Register(index));
  else if (cls == RegClass::VCC)
    regTy = VCCType::get(ctx, Register(0));
  else if (cls == RegClass::EXEC)
    regTy = EXECType::get(ctx, Register(0));
  else if (cls == RegClass::SCC)
    regTy = SCCType::get(ctx, Register(0));
  else
    return {};

  auto alloca = AllocaOp::create(builder, loc, regTy);
  cache[key] = alloca;
  return alloca;
}

Value getOrCreateReg(OpBuilder &builder, Location loc,
                     std::map<std::pair<RegClass, int>, Value> &singleCache,
                     std::map<std::pair<RegClass, std::pair<int, int>>, Value> &rangeCache,
                     MLIRContext *ctx, RegClass cls, int index, int size) {
  if (size <= 1)
    return getOrCreateSingleReg(builder, loc, singleCache, ctx, cls, index);

  auto rangeKey = std::make_pair(cls, std::make_pair(index, size));
  auto rit = rangeCache.find(rangeKey);
  if (rit != rangeCache.end())
    return rit->second;

  SmallVector<Value> regs;
  for (int i = 0; i < size; i++)
    regs.push_back(
        getOrCreateSingleReg(builder, loc, singleCache, ctx, cls, index + i));

  auto rangeOp = MakeRegisterRangeOp::create(builder, loc, regs);
  rangeCache[rangeKey] = rangeOp;
  return rangeOp;
}

} // anonymous namespace

LiftResult liftToAster(MLIRContext &ctx,
                       const std::vector<uint8_t> &textBytes,
                       const std::string &targetISA,
                       const std::string &kernelName) {
  LiftResult result;

  MCState mc;
  if (!initMCState(mc, targetISA)) {
    llvm::errs() << "aster_proto: Failed to initialize LLVM MC for "
                 << targetISA << "\n";
    return result;
  }

  ctx.loadDialect<AMDGCNDialect>();
  ctx.loadDialect<arith::ArithDialect>();

  OpBuilder builder(&ctx);
  auto loc = UnknownLoc::get(&ctx);

  // Disambiguate: use mlir::ModuleOp for the builtin module
  auto builtinMod = ::mlir::ModuleOp::create(builder, loc);
  builder.setInsertionPointToEnd(builtinMod.getBody());

  auto amdgcnMod = amdgcn::ModuleOp::create(builder, loc,
      Target::GFX942, ISAVersion::CDNA3, llvm::StringRef("mod"),
      /*sym_visibility=*/mlir::StringAttr(),
      /*normal_forms=*/mlir::ArrayAttr());
  // ModuleOp starts with empty region — we must create an entry block
  amdgcnMod.getBodyRegion().emplaceBlock();
  builder.setInsertionPointToStart(&amdgcnMod.getBodyRegion().front());

  SmallVector<KernelArgAttrInterface> kernelArgs;
  auto globalAS = AddressSpaceKind::Global;
  auto noFlags = KernelArgumentFlags::None;
  auto nullType = mlir::Type();

  kernelArgs.push_back(BufferArgAttr::get(&ctx, globalAS, AccessKind::ReadOnly,
                                          noFlags, "", nullType));
  kernelArgs.push_back(BufferArgAttr::get(&ctx, globalAS, AccessKind::ReadOnly,
                                          noFlags, "", nullType));
  kernelArgs.push_back(BufferArgAttr::get(&ctx, globalAS, AccessKind::WriteOnly,
                                          noFlags, "", nullType));
  kernelArgs.push_back(ByValueArgAttr::get(&ctx, 4, 4, "",
                                           builder.getI32Type()));

  auto kernelOp = amdgcn::KernelOp::create(builder, loc, kernelName,
      kernelArgs, /*shared_memory_size=*/0, /*private_memory_size=*/0,
      /*enable_private_segment_buffer=*/false,
      /*enable_dispatch_ptr=*/false,
      /*enable_kernarg_segment_ptr=*/true);

  // KernelOp starts with empty region — we must create an entry block
  kernelOp.getBodyRegion().emplaceBlock();
  builder.setInsertionPointToStart(&kernelOp.getBodyRegion().front());

  std::map<std::pair<RegClass, int>, Value> singleRegCache;
  std::map<std::pair<RegClass, std::pair<int, int>>, Value> rangeCache;
  std::set<std::string> unsupported;

  auto getRegVal = [&](const llvm::MCOperand &op) -> Value {
    if (!op.isReg()) return {};
    auto ri = classifyReg(mc, op.getReg());
    return getOrCreateReg(builder, loc, singleRegCache, rangeCache,
                          &ctx, ri.cls, ri.index, ri.rangeSize);
  };

  auto getImmVal = [&](int64_t imm) -> Value {
    return arith::ConstantIntOp::create(builder, loc, imm, 32);
  };

  auto getOperandVal = [&](const llvm::MCOperand &op) -> Value {
    if (op.isReg()) return getRegVal(op);
    if (op.isImm()) return getImmVal(op.getImm());
    return {};
  };

  // Track VCC from most recent v_cmp for exec-mask → VCC-branch conversion
  Value lastVCC;
  // For multi-block control flow: exit block for the if-guarded body
  Block *exitBlock = nullptr;

  uint64_t offset = 0;
  uint64_t size = textBytes.size();
  llvm::ArrayRef<uint8_t> bytes(textBytes);
  bool endpgmSeen = false;

  while (offset < size) {
    llvm::MCInst inst;
    uint64_t instSize = 0;
    auto status = mc.disasm->getInstruction(
        inst, instSize, bytes.slice(offset), offset, llvm::nulls());

    if (status != llvm::MCDisassembler::Success) {
      offset += 4;
      continue;
    }

    std::string mnemonic = getMnemonic(mc, inst);
    std::string fullText = getFullText(mc, inst);
    bool handled = false;

    // --- VOP2: v_add_f32_e32, v_add_u32_e32, v_lshlrev_b32_e32, v_ashrrev_i32_e32 ---
    if (mnemonic == "v_add_f32_e32" || mnemonic == "v_add_u32_e32" ||
        mnemonic == "v_lshlrev_b32_e32" || mnemonic == "v_ashrrev_i32_e32") {
      if (inst.getNumOperands() >= 3) {
        Value dst = getRegVal(inst.getOperand(0));
        Value src0 = getOperandVal(inst.getOperand(1));
        Value src1 = getRegVal(inst.getOperand(2));
        if (dst && src0 && src1) {
          if (mnemonic == "v_add_f32_e32")
            V_ADD_F32::create(builder, loc, dst, src0, src1);
          else if (mnemonic == "v_add_u32_e32")
            V_ADD_U32::create(builder, loc, dst, src0, src1);
          else if (mnemonic == "v_ashrrev_i32_e32")
            V_ASHRREV_I32::create(builder, loc, dst, src0, src1);
          else if (mnemonic == "v_lshlrev_b32_e32")
            V_LSHLREV_B32_E32::create(builder, loc, dst, src0, src1);
          handled = true;
        }
      }
    }

    // --- VOP1: v_mov_b32_e32 ---
    if (!handled && mnemonic == "v_mov_b32_e32") {
      if (inst.getNumOperands() >= 2) {
        Value dst = getRegVal(inst.getOperand(0));
        Value src = getOperandVal(inst.getOperand(1));
        if (dst && src) {
          V_MOV_B32_E32::create(builder, loc, dst, src);
          handled = true;
        }
      }
    }

    // --- SOP2: s_and_b32, s_or_b32, s_mul_i32, s_lshl_b32 ---
    if (!handled && (mnemonic == "s_and_b32" || mnemonic == "s_mul_i32" ||
                     mnemonic == "s_or_b32" || mnemonic == "s_lshl_b32")) {
      if (inst.getNumOperands() >= 3) {
        Value dst = getRegVal(inst.getOperand(0));
        Value src0 = getOperandVal(inst.getOperand(1));
        Value src1 = getOperandVal(inst.getOperand(2));
        if (dst && src0 && src1) {
          if (mnemonic == "s_and_b32")
            S_AND_B32::create(builder, loc, dst, src0, src1);
          else if (mnemonic == "s_mul_i32")
            S_MUL_I32::create(builder, loc, dst, src0, src1);
          else if (mnemonic == "s_or_b32")
            S_OR_B32::create(builder, loc, dst, src0, src1);
          else if (mnemonic == "s_lshl_b32")
            S_LSHL_B32::create(builder, loc, dst, src0, src1);
          handled = true;
        }
      }
    }

    // --- SOP1: s_mov_b32 ---
    if (!handled && mnemonic == "s_mov_b32") {
      if (inst.getNumOperands() >= 2) {
        Value dst = getRegVal(inst.getOperand(0));
        Value src = getOperandVal(inst.getOperand(1));
        if (dst && src) {
          S_MOV_B32::create(builder, loc, dst, src);
          handled = true;
        }
      }
    }

    // --- VOPC: v_cmp_gt_i32_e32 (implicit VCC dest) ---
    // In _e32 VOPC encoding, VCC is the implicit destination.
    // MCInst only has 2 explicit operands: src0, src1.
    if (!handled && llvm::StringRef(mnemonic).starts_with("v_cmp_gt_i32")) {
      unsigned numOps = inst.getNumOperands();
      if (numOps >= 2) {
        Value vcc = getOrCreateSingleReg(builder, loc, singleRegCache,
                                         &ctx, RegClass::VCC, 0);
        Value src0 = getOperandVal(inst.getOperand(numOps == 2 ? 0 : 1));
        Value src1 = getOperandVal(inst.getOperand(numOps == 2 ? 1 : 2));
        if (vcc && src0 && src1) {
          V_CMP_GT_I32::create(builder, loc, vcc, src0, src1);
          lastVCC = vcc;
          handled = true;
        }
      }
    }

    // --- Exec-mask pattern: s_and_saveexec_b64 + s_cbranch_execz ---
    // Convert to Aster's native VCC-based branching (multi-block CFG).
    // Semantics: the original binary uses EXEC-mask predication to guard
    // memory ops per-lane. We convert this to wavefront-level VCC branching
    // which is correct when the grid dispatch is aligned to wavefront size.
    if (!handled && mnemonic == "s_and_saveexec_b64") {
      // The v_cmp that precedes this already wrote VCC. We skip saveexec
      // and convert the following s_cbranch_execz into s_cbranch_vccz.
      if (lastVCC) {
        handled = true;
        result.liftedCount++;
        result.notes.push_back(
            "s_and_saveexec_b64 → elided (exec-mask converted to VCC branch)");
      }
    }

    if (!handled && mnemonic == "s_cbranch_execz") {
      if (lastVCC) {
        // Create body and exit blocks within the kernel region
        auto &region = kernelOp.getBodyRegion();
        Block *bodyBlock = new Block();
        exitBlock = new Block();
        region.push_back(bodyBlock);
        region.push_back(exitBlock);

        // Terminate current block with VCC-based conditional branch:
        // If VCC==0 (all lanes failed compare), branch to exit; else body.
        CBranchOp::create(builder, loc, OpCode::S_CBRANCH_VCCZ,
                          lastVCC, exitBlock, bodyBlock);

        // Switch builder to body block for remaining instructions
        builder.setInsertionPointToStart(bodyBlock);
        handled = true;
        result.notes.push_back(
            "s_cbranch_execz → s_cbranch_vccz (multi-block CFG)");
      }
    }

    // --- SMEM loads ---
    if (!handled && llvm::StringRef(mnemonic).starts_with("s_load_dword")) {
      if (inst.getNumOperands() >= 3) {
        Value dest = getRegVal(inst.getOperand(0));
        Value addr = getRegVal(inst.getOperand(1));
        int64_t constOff = inst.getOperand(2).isImm()
                               ? inst.getOperand(2).getImm() : 0;
        Value constOffVal = getImmVal(constOff);
        Value noDynOff;
        if (dest && addr) {
          if (mnemonic == "s_load_dword")
            S_LOAD_DWORD::create(builder, loc, dest, addr, noDynOff, constOffVal);
          else if (mnemonic == "s_load_dwordx2")
            S_LOAD_DWORDX2::create(builder, loc, dest, addr, noDynOff, constOffVal);
          else if (mnemonic == "s_load_dwordx4")
            S_LOAD_DWORDX4::create(builder, loc, dest, addr, noDynOff, constOffVal);
          else
            goto track_unsupported;
          handled = true;
        }
      }
    }

    // --- Global load/store ---
    if (!handled && mnemonic == "global_load_dword") {
      if (inst.getNumOperands() >= 2) {
        Value dest = getRegVal(inst.getOperand(0));
        Value addr = getRegVal(inst.getOperand(1));
        Value noDynOff;
        Value zeroConst = getImmVal(0);
        if (dest && addr) {
          GLOBAL_LOAD_DWORD::create(builder, loc, dest, addr, noDynOff, zeroConst);
          handled = true;
        }
      }
    }
    if (!handled && mnemonic == "global_store_dword") {
      if (inst.getNumOperands() >= 2) {
        Value addr = getRegVal(inst.getOperand(0));
        Value data = getRegVal(inst.getOperand(1));
        Value noDynOff;
        Value zeroConst = getImmVal(0);
        if (addr && data) {
          GLOBAL_STORE_DWORD::create(builder, loc, data, addr, noDynOff, zeroConst);
          handled = true;
        }
      }
    }

    // --- VOP3: v_lshlrev_b64, v_lshl_add_u64 ---
    if (!handled && (mnemonic == "v_lshlrev_b64" ||
                     mnemonic == "v_lshl_add_u64")) {
      if (inst.getNumOperands() >= 3) {
        Value dst = getRegVal(inst.getOperand(0));
        Value src0 = getOperandVal(inst.getOperand(1));
        Value src1 = getOperandVal(inst.getOperand(2));
        if (dst && src0 && src1) {
          if (mnemonic == "v_lshlrev_b64") {
            V_LSHLREV_B64::create(builder, loc, dst, src0, src1);
          } else {
            Value src2 = (inst.getNumOperands() > 3)
                             ? getOperandVal(inst.getOperand(3))
                             : getImmVal(0);
            if (src2) {
              V_LSHL_ADD_U64::create(builder, loc, dst, src0, src1, src2);
            }
          }
          handled = true;
        }
      }
    }

    // --- s_waitcnt ---
    if (!handled && mnemonic == "s_waitcnt") {
      int64_t packed = (inst.getNumOperands() > 0 && inst.getOperand(0).isImm())
                           ? inst.getOperand(0).getImm() : 0;
      int vmcnt = packed & 0xF;
      int expcnt = (packed >> 4) & 0x7;
      int lgkmcnt = (packed >> 8) & 0x3F;
      S_WAITCNT::create(builder, loc, vmcnt, expcnt, lgkmcnt);
      handled = true;
    }

    // --- s_nop ---
    if (!handled && mnemonic == "s_nop") {
      int imm = (inst.getNumOperands() > 0 && inst.getOperand(0).isImm())
                    ? inst.getOperand(0).getImm() : 0;
      S_NOP::create(builder, loc, imm);
      handled = true;
    }

    // --- s_endpgm ---
    if (!handled && mnemonic == "s_endpgm") {
      if (exitBlock) {
        // Multi-block: terminate body with branch to exit, emit endpgm there
        BranchOp::create(builder, loc, OpCode::S_BRANCH, exitBlock);
        builder.setInsertionPointToStart(exitBlock);
      }
      EndKernelOp::create(builder, loc);
      handled = true;
      endpgmSeen = true;
    }

    if (endpgmSeen) {
      offset += instSize;
      break;
    }

track_unsupported:
    if (handled) {
      result.liftedCount++;
    } else {
      result.unsupportedCount++;
      if (unsupported.insert(mnemonic).second)
        result.unsupportedMnemonics.push_back(mnemonic);
      result.unsupportedInstructions.push_back({offset, fullText});
    }

    offset += instSize;
  }

  result.module = builtinMod.getOperation();
  result.success = true;
  return result;
}

} // namespace aster_proto
