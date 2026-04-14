////////////////////////////////////////////////////////////////////////////////
//
// Binary Lifter: AMDGPU machine code (MCInst) -> waveasm MLIR ops
//
////////////////////////////////////////////////////////////////////////////////

#include "lifter.hpp"

#include "waveasm/Dialect/WaveASMAttrs.h"

#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/IR/Verifier.h>

#include <mutex>

using namespace hotswap;

//===----------------------------------------------------------------------===//
// LLVM target initialization (one-time)
//===----------------------------------------------------------------------===//

static std::once_flag g_init_flag;
static void ensureAMDGPUTargets() {
  std::call_once(g_init_flag, [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUDisassembler();
  });
}

MCState hotswap::initMCState(llvm::StringRef isaName) {
  ensureAMDGPUTargets();

  MCState state;
  state.cpu = isaName.str();
  llvm::Triple triple("amdgcn-amd-amdhsa");
  std::string error;

  state.target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!state.target)
    return state;

  state.MRI.reset(state.target->createMCRegInfo(triple));
  llvm::MCTargetOptions mcOpts;
  state.MAI.reset(state.target->createMCAsmInfo(*state.MRI, triple, mcOpts));
  state.MCII.reset(state.target->createMCInstrInfo());
  state.STI.reset(
      state.target->createMCSubtargetInfo(triple, state.cpu, ""));

  if (!state.MRI || !state.MAI || !state.MCII || !state.STI)
    return state;

  state.Ctx = std::make_unique<llvm::MCContext>(
      triple, state.MAI.get(), state.MRI.get(),
      state.STI.get());
  state.disasm.reset(
      state.target->createMCDisassembler(*state.STI, *state.Ctx));
  state.printer.reset(state.target->createMCInstPrinter(
      triple, 0, *state.MAI, *state.MCII, *state.MRI));

  if (!state.disasm || !state.printer)
    return state;

  state.valid = true;
  return state;
}

//===----------------------------------------------------------------------===//
// Register classification
//===----------------------------------------------------------------------===//

enum class RegKind { VGPR, SGPR, AGPR, VCC, EXEC, SCC, M0, TTMP, Other };

static RegKind classifyMCReg(unsigned reg, const MCState &mc) {
  if (reg == 0)
    return RegKind::Other;
  const char *name = mc.MRI->getName(reg);
  if (!name)
    return RegKind::Other;
  if (strncmp(name, "VGPR", 4) == 0)
    return RegKind::VGPR;
  if (strncmp(name, "SGPR", 4) == 0)
    return RegKind::SGPR;
  if (strncmp(name, "AGPR", 4) == 0)
    return RegKind::AGPR;
  if (strcmp(name, "SCC") == 0)
    return RegKind::SCC;
  if (strncmp(name, "VCC", 3) == 0)
    return RegKind::VCC;
  if (strncmp(name, "EXEC", 4) == 0)
    return RegKind::EXEC;
  if (strncmp(name, "M0", 2) == 0)
    return RegKind::M0;
  if (strncmp(name, "TTMP", 4) == 0)
    return RegKind::TTMP;
  return RegKind::Other;
}

static int64_t extractRegIndex(unsigned reg, const MCState &mc) {
  const char *name = mc.MRI->getName(reg);
  if (!name)
    return 0;
  llvm::StringRef sref(name);
  llvm::StringRef num;
  if (sref.starts_with("VGPR") || sref.starts_with("SGPR") ||
      sref.starts_with("AGPR"))
    num = sref.drop_front(4);
  else if (sref.starts_with("TTMP"))
    num = sref.drop_front(4);
  else
    return 0;

  auto underscorePos = num.find('_');
  if (underscorePos != llvm::StringRef::npos)
    num = num.take_front(underscorePos);

  int64_t idx = 0;
  if (num.getAsInteger(10, idx))
    return 0;
  return idx;
}

static int64_t getMCRegSize(unsigned reg, const MCState &mc) {
  const char *name = mc.MRI->getName(reg);
  if (!name)
    return 1;
  llvm::StringRef sref(name);

  int64_t count = 1;
  for (char c : sref)
    if (c == '_')
      ++count;

  if (sref.starts_with("VCC") || sref.starts_with("EXEC") ||
      sref.starts_with("FLAT_SCRATCH"))
    return 2;
  if (sref == "SCC" || sref == "M0")
    return 1;

  return count;
}

//===----------------------------------------------------------------------===//
// Lifter implementation
//===----------------------------------------------------------------------===//

Lifter::Lifter(mlir::MLIRContext &ctx) : ctx(ctx) {
  ctx.loadDialect<waveasm::WaveASMDialect>();
  ctx.allowUnregisteredDialects();
}

mlir::Type Lifter::classifyRegister(unsigned reg, const MCState &mc) const {
  auto kind = classifyMCReg(reg, mc);
  int64_t size = getMCRegSize(reg, mc);
  int64_t idx = extractRegIndex(reg, mc);

  switch (kind) {
  case RegKind::VGPR:
    return waveasm::PVRegType::get(&ctx, idx, size);
  case RegKind::SGPR:
    return waveasm::PSRegType::get(&ctx, idx, size);
  case RegKind::AGPR:
    return waveasm::PARegType::get(&ctx, idx, size);
  case RegKind::SCC:
    return waveasm::SCCType::get(&ctx);
  case RegKind::VCC:
    return waveasm::PSRegType::get(&ctx, 106, 2);
  case RegKind::EXEC:
    return waveasm::PSRegType::get(&ctx, 126, 2);
  case RegKind::M0:
    return waveasm::PSRegType::get(&ctx, 124, 1);
  case RegKind::TTMP:
    return waveasm::PSRegType::get(&ctx, idx + 1000, size);
  case RegKind::Other:
    return waveasm::ImmType::get(&ctx, 0);
  }
  llvm_unreachable("unhandled RegKind");
}

mlir::Type Lifter::getOperandType(const llvm::MCOperand &op,
                                  const MCState &mc) const {
  if (op.isReg())
    return classifyRegister(op.getReg(), mc);
  if (op.isImm())
    return waveasm::ImmType::get(&ctx, op.getImm());
  return waveasm::ImmType::get(&ctx, 0);
}

std::string Lifter::getMnemonic(const llvm::MCInst &inst,
                                const MCState &mc) const {
  std::string text;
  llvm::raw_string_ostream rso(text);
  mc.printer->printInst(&inst, 0, "", *mc.STI, rso);
  rso.flush();
  auto start = text.find_first_not_of(" \t");
  if (start == std::string::npos)
    return "";
  text = text.substr(start);
  auto space = text.find_first_of(" \t");
  if (space != std::string::npos)
    text = text.substr(0, space);
  return text;
}

mlir::Value Lifter::getOrCreatePhysReg(unsigned reg, const MCState &mc,
                                       mlir::OpBuilder &builder,
                                       mlir::Location loc) {
  auto it = physRegCache.find(reg);
  if (it != physRegCache.end())
    return it->second;

  auto kind = classifyMCReg(reg, mc);
  int64_t idx = extractRegIndex(reg, mc);
  int64_t size = getMCRegSize(reg, mc);
  mlir::Value val;

  auto createPrecolored = [&](llvm::StringRef opName, mlir::Type ty,
                              int64_t index, int64_t sz) -> mlir::Value {
    mlir::OperationState state(loc, ("waveasm." + opName).str());
    state.addTypes({ty});
    state.addAttribute("index",
                       builder.getIntegerAttr(builder.getI64Type(), index));
    state.addAttribute("size",
                       builder.getIntegerAttr(builder.getI64Type(), sz));
    auto *op = builder.create(state);
    return op->getResult(0);
  };

  switch (kind) {
  case RegKind::VGPR:
    val = createPrecolored("precolored.vreg",
                           waveasm::PVRegType::get(&ctx, idx, size), idx, size);
    break;
  case RegKind::SGPR:
    val = createPrecolored("precolored.sreg",
                           waveasm::PSRegType::get(&ctx, idx, size), idx, size);
    break;
  case RegKind::AGPR:
    val = createPrecolored("precolored.areg",
                           waveasm::PARegType::get(&ctx, idx, size), idx, size);
    break;
  case RegKind::VCC:
    val = createPrecolored("precolored.sreg",
                           waveasm::PSRegType::get(&ctx, 106, 2), 106, 2);
    break;
  case RegKind::EXEC:
    val = createPrecolored("precolored.sreg",
                           waveasm::PSRegType::get(&ctx, 126, 2), 126, 2);
    break;
  case RegKind::M0:
    val = createPrecolored("precolored.sreg",
                           waveasm::PSRegType::get(&ctx, 124, 1), 124, 1);
    break;
  default:
    val = createPrecolored("precolored.sreg",
                           waveasm::PSRegType::get(&ctx, 0, 1), 0, 1);
    break;
  }

  physRegCache[reg] = val;
  return val;
}

mlir::Value Lifter::createImmediate(int64_t value, mlir::OpBuilder &builder,
                                    mlir::Location loc) {
  auto ty = waveasm::ImmType::get(&ctx, value);
  mlir::OperationState state(loc, "waveasm.constant");
  state.addTypes({ty});
  state.addAttribute("value", builder.getI64IntegerAttr(value));
  auto *op = builder.create(state);
  return op->getResult(0);
}

mlir::Operation *
Lifter::createWaveasmOp(llvm::StringRef opName, mlir::TypeRange resultTypes,
                        mlir::ValueRange operands, mlir::OpBuilder &builder,
                        mlir::Location loc,
                        llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  std::string fullName = ("waveasm." + opName).str();
  mlir::OperationState state(loc, fullName);
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addAttributes(attrs);
  return builder.create(state);
}

//===----------------------------------------------------------------------===//
// Instruction dispatch
//===----------------------------------------------------------------------===//

bool Lifter::dispatchToWaveasmOp(llvm::StringRef mnemonic,
                                 const llvm::MCInst &inst, uint64_t pc,
                                 const MCState &mc, mlir::OpBuilder &builder,
                                 mlir::Location loc) {
  // Handle SOPP instructions first (before materializing operands),
  // since they encode operands as attributes, not SSA values.
  if (mnemonic == "s_endpgm" || mnemonic == "s_barrier") {
    createWaveasmOp(mnemonic, {}, {}, builder, loc);
    return true;
  }
  if (mnemonic == "s_nop") {
    int64_t count = 0;
    if (inst.getNumOperands() > 0 && inst.getOperand(0).isImm())
      count = inst.getOperand(0).getImm();
    createWaveasmOp(mnemonic, {}, {}, builder, loc,
                    {builder.getNamedAttr(
                        "simm16", builder.getI64IntegerAttr(count))});
    return true;
  }
  if (mnemonic == "s_waitcnt") {
    createWaveasmOp(mnemonic, {}, {}, builder, loc);
    return true;
  }

  // Handle branch instructions: resolve PC-relative offsets to label refs.
  if (mnemonic.starts_with("s_branch") || mnemonic.starts_with("s_cbranch_")) {
    for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
      if (inst.getOperand(i).isImm()) {
        int64_t offset = inst.getOperand(i).getImm();
        if (offset > 32767)
          offset -= 65536;
        int64_t targetPC = static_cast<int64_t>(pc) + 4 + offset * 4;
        if (targetPC >= 0) {
          auto it = branchLabels.find(static_cast<uint64_t>(targetPC));
          if (it != branchLabels.end()) {
            createWaveasmOp(
                mnemonic, {}, {}, builder, loc,
                {builder.getNamedAttr(
                    "target",
                    mlir::FlatSymbolRefAttr::get(&ctx, it->second))});
            return true;
          }
        }
        break;
      }
    }
  }

  // For register-based instructions, materialize operands.
  const llvm::MCInstrDesc &desc = mc.MCII->get(inst.getOpcode());
  unsigned numDefs = desc.getNumDefs();

  llvm::SmallVector<mlir::Value> operandValues;
  llvm::SmallVector<mlir::Type> resultTypes;

  for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
    const auto &op = inst.getOperand(i);
    if (i < numDefs) {
      if (op.isReg())
        resultTypes.push_back(classifyRegister(op.getReg(), mc));
      else
        resultTypes.push_back(waveasm::ImmType::get(&ctx, 0));
    } else {
      if (op.isReg() && op.getReg() != 0)
        operandValues.push_back(
            getOrCreatePhysReg(op.getReg(), mc, builder, loc));
      else if (op.isImm())
        operandValues.push_back(createImmediate(op.getImm(), builder, loc));
    }
  }

  // Try known instruction registry first.
  auto &registry = waveasm::InstructionRegistry::get();
  if (registry.hasInstruction(mnemonic)) {
    createWaveasmOp(mnemonic, resultTypes, operandValues, builder, loc);
    return true;
  }

  // Try dynamic creation for ops registered via TableGen.
  auto *result =
      createWaveasmOp(mnemonic, resultTypes, operandValues, builder, loc);
  if (result)
    return true;

  return false;
}

bool Lifter::liftInstruction(const llvm::MCInst &inst, uint64_t pc,
                             const MCState &mc, mlir::OpBuilder &builder,
                             mlir::Location loc) {
  std::string mnemonic = getMnemonic(inst, mc);
  if (mnemonic.empty()) {
    std::string hex;
    llvm::raw_string_ostream rso(hex);
    rso << llvm::format_hex(inst.getOpcode(), 10);
    createWaveasmOp("raw", {}, {}, builder, loc,
                    {builder.getNamedAttr("text", builder.getStringAttr(hex))});
    return false;
  }

  if (dispatchToWaveasmOp(mnemonic, inst, pc, mc, builder, loc))
    return true;

  // Fallback: emit full assembly text as waveasm.raw
  std::string fullText;
  llvm::raw_string_ostream rso(fullText);
  mc.printer->printInst(&inst, 0, "", *mc.STI, rso);
  rso.flush();
  auto start = fullText.find_first_not_of(" \t");
  if (start != std::string::npos)
    fullText = fullText.substr(start);

  createWaveasmOp("raw", {}, {}, builder, loc,
                  {builder.getNamedAttr("text",
                       builder.getStringAttr(fullText))});
  return false;
}

//===----------------------------------------------------------------------===//
// Main lift entry point
//===----------------------------------------------------------------------===//

mlir::OwningOpRef<mlir::ModuleOp>
Lifter::lift(llvm::ArrayRef<uint8_t> bytes, llvm::StringRef isaName,
             llvm::StringRef kernelName) {
  stats = LiftStats{};
  physRegCache.clear();

  auto mc = initMCState(isaName);
  if (!mc.valid)
    return nullptr;

  mlir::OpBuilder builder(&ctx);
  auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto loc = builder.getUnknownLoc();

  // Build the target attribute from the ISA name.
  waveasm::TargetAttrInterface targetKindAttr;
  if (isaName == "gfx942")
    targetKindAttr = waveasm::GFX942TargetAttr::get(&ctx);
  else if (isaName == "gfx950")
    targetKindAttr = waveasm::GFX950TargetAttr::get(&ctx);
  else if (isaName == "gfx1250")
    targetKindAttr = waveasm::GFX1250TargetAttr::get(&ctx);
  else
    targetKindAttr = waveasm::GFX942TargetAttr::get(&ctx);
  auto targetAttr = waveasm::TargetAttr::get(&ctx, targetKindAttr, 5);

  auto abiAttr = waveasm::KernelABIAttr::get(
      &ctx, /*tid=*/0, /*kernarg=*/0,
      /*wg_id_x=*/std::optional<int64_t>(4),
      /*wg_id_y=*/std::nullopt,
      /*wg_id_z=*/std::nullopt);

  auto programOp = builder.create<waveasm::ProgramOp>(
      loc, kernelName, targetAttr, abiAttr);

  mlir::Block &body = programOp.getBody().front();
  builder.setInsertionPointToEnd(&body);

  // Pre-scan: identify branch targets and assign label names.
  branchLabels.clear();
  {
    uint64_t scanPos = 0;
    int labelCounter = 0;
    while (scanPos < bytes.size()) {
      llvm::MCInst scanInst;
      uint64_t scanSize = 0;
      auto scanBytes = llvm::ArrayRef<uint8_t>(bytes.data() + scanPos,
                                               bytes.size() - scanPos);
      auto status = mc.disasm->getInstruction(scanInst, scanSize, scanBytes,
                                              scanPos, llvm::nulls());
      if (status == llvm::MCDisassembler::Fail) {
        scanPos += 4;
        continue;
      }
      std::string mnem = getMnemonic(scanInst, mc);
      llvm::StringRef mnemRef(mnem);
      if (mnemRef.starts_with("s_branch") || mnemRef.starts_with("s_cbranch_")) {
        for (unsigned i = 0; i < scanInst.getNumOperands(); ++i) {
          if (scanInst.getOperand(i).isImm()) {
            int64_t offset = scanInst.getOperand(i).getImm();
            // Sign-extend from 16-bit (SOPP simm16 is signed)
            if (offset > 32767)
              offset -= 65536;
            int64_t targetPC =
                static_cast<int64_t>(scanPos) + 4 + offset * 4;
            if (targetPC >= 0 &&
                static_cast<uint64_t>(targetPC) <= bytes.size()) {
              auto key = static_cast<uint64_t>(targetPC);
              if (branchLabels.find(key) == branchLabels.end())
                branchLabels[key] =
                    "L_br" + std::to_string(labelCounter++);
            }
            break;
          }
        }
      }
      scanPos += scanSize;
    }
  }

  uint64_t pos = 0;
  while (pos < bytes.size()) {
    // Insert label if this PC is a branch target.
    auto labelIt = branchLabels.find(pos);
    if (labelIt != branchLabels.end()) {
      auto labelLoc = mlir::FileLineColLoc::get(
          builder.getStringAttr(kernelName), 0, pos);
      createWaveasmOp("label", {}, {}, builder, labelLoc,
                      {builder.getNamedAttr(
                          "sym_name",
                          builder.getStringAttr(labelIt->second))});
    }

    llvm::MCInst inst;
    uint64_t instSize = 0;
    llvm::ArrayRef<uint8_t> remaining(bytes.data() + pos,
                                      bytes.size() - pos);

    auto status = mc.disasm->getInstruction(inst, instSize, remaining, pos,
                                            llvm::nulls());
    stats.totalInstructions++;

    if (status == llvm::MCDisassembler::Fail) {
      uint32_t dword = 0;
      if (pos + 4 <= bytes.size())
        memcpy(&dword, bytes.data() + pos, 4);
      std::string hex;
      llvm::raw_string_ostream rso(hex);
      rso << ".long " << llvm::format_hex(dword, 10);
      auto rawLoc = mlir::FileLineColLoc::get(
          builder.getStringAttr(kernelName), 0, pos);
      createWaveasmOp("raw", {}, {}, builder, rawLoc,
                      {builder.getNamedAttr("text",
                           builder.getStringAttr(hex))});
      stats.failedDisassembly++;
      pos += 4;
      continue;
    }

    auto instLoc = mlir::FileLineColLoc::get(
        builder.getStringAttr(kernelName), 0, pos);
    if (liftInstruction(inst, pos, mc, builder, instLoc))
      stats.liftedInstructions++;
    else
      stats.rawFallbacks++;

    pos += instSize;
  }

  return moduleOp;
}
