#include "lifter.hpp"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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

namespace mir_proto {

// ---------------------------------------------------------------------------
// LiveMF — PIMPL body holding the live MachineFunction and MC layer objects
// ---------------------------------------------------------------------------
struct LiftResult::LiveMF {
  // LLVM IR + CodeGen
  std::unique_ptr<LLVMContext> llvmCtx;
  std::unique_ptr<Module> mod;
  std::unique_ptr<TargetMachine> tm;
  std::unique_ptr<MachineModuleInfo> mmi;
  MachineFunction *mf = nullptr;

  // MC layer (kept alive for assembly printing in generateAssembly)
  const Target *target = nullptr;
  std::unique_ptr<MCInstrInfo> instrInfo;
  std::unique_ptr<MCRegisterInfo> regInfo;
  std::unique_ptr<MCSubtargetInfo> subtargetInfo;
  std::unique_ptr<const MCAsmInfo> asmInfo;
  std::unique_ptr<MCContext> mcCtx;
  std::unique_ptr<MCInstPrinter> printer;
};

LiftResult::LiftResult() = default;
LiftResult::~LiftResult() = default;
LiftResult::LiftResult(LiftResult &&) noexcept = default;
LiftResult &LiftResult::operator=(LiftResult &&) noexcept = default;

// ---------------------------------------------------------------------------
// MC component initialisation (stored directly in LiveMF)
// ---------------------------------------------------------------------------
namespace {

bool initMCComponents(LiftResult::LiveMF &live, const std::string &targetISA) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  Triple triple("amdgcn-amd-amdhsa");
  std::string error;
  live.target = TargetRegistry::lookupTarget(triple, error);
  if (!live.target) {
    errs() << "mir_proto: Failed to find AMDGPU target: " << error << "\n";
    return false;
  }

  live.instrInfo.reset(live.target->createMCInstrInfo());
  live.regInfo.reset(live.target->createMCRegInfo(triple));
  live.subtargetInfo.reset(
      live.target->createMCSubtargetInfo(triple, targetISA, ""));
  live.asmInfo.reset(live.target->createMCAsmInfo(*live.regInfo, triple,
                                                  MCTargetOptions()));
  live.mcCtx = std::make_unique<MCContext>(triple, live.asmInfo.get(),
                                           live.regInfo.get(),
                                           live.subtargetInfo.get());
  live.printer.reset(live.target->createMCInstPrinter(
      triple, 0, *live.asmInfo, *live.instrInfo, *live.regInfo));
  live.printer->setPrintImmHex(true);

  return true;
}

std::string getMnemonic(const LiftResult::LiveMF &live, const MCInst &inst) {
  std::string s;
  raw_string_ostream os(s);
  live.printer->printInst(&inst, 0, "", *live.subtargetInfo, os);
  StringRef sr(s);
  sr = sr.ltrim();
  return sr.split('\t').first.split(' ').first.str();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// liftToMIR — disassemble .text bytes into a live MachineFunction
// ---------------------------------------------------------------------------
LiftResult liftToMIR(const std::vector<uint8_t> &textBytes,
                     const std::string &targetISA,
                     const std::string &kernelName) {
  LiftResult result;
  result.liveMF = std::make_unique<LiftResult::LiveMF>();
  auto &live = *result.liveMF;

  if (!initMCComponents(live, targetISA)) {
    errs() << "mir_proto: Failed to initialize MC state\n";
    return result;
  }

  // Disassembler is only needed during lifting; destroyed at function exit.
  std::unique_ptr<MCDisassembler> disasm(
      live.target->createMCDisassembler(*live.subtargetInfo, *live.mcCtx));

  // --- LLVM IR scaffolding (stub function for the MachineFunction) ---------
  live.llvmCtx = std::make_unique<LLVMContext>();
  live.mod = std::make_unique<Module>("mir_proto_module", *live.llvmCtx);
  live.mod->setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  TargetOptions opts;
  live.tm.reset(live.target->createTargetMachine(
      Triple("amdgcn-amd-amdhsa"), targetISA, "", opts, Reloc::PIC_));
  if (!live.tm) {
    errs() << "mir_proto: Failed to create TargetMachine\n";
    return result;
  }
  live.mod->setDataLayout(live.tm->createDataLayout());

  auto *i32Ty = Type::getInt32Ty(*live.llvmCtx);
  auto *ptrTy = PointerType::get(*live.llvmCtx, 1);
  auto *funcTy = FunctionType::get(Type::getVoidTy(*live.llvmCtx),
                                   {ptrTy, ptrTy, ptrTy, i32Ty}, false);
  auto *func = Function::Create(funcTy, GlobalValue::ExternalLinkage,
                                kernelName, live.mod.get());
  func->setCallingConv(CallingConv::AMDGPU_KERNEL);

  // --- MachineFunction creation --------------------------------------------
  live.mmi = std::make_unique<MachineModuleInfo>(live.tm.get());
  MachineFunction &MF = live.mmi->getOrCreateMachineFunction(*func);
  live.mf = &MF;

  const auto *TII = MF.getSubtarget().getInstrInfo();
  if (!TII) {
    errs() << "mir_proto: TargetInstrInfo not available\n";
    return result;
  }

  MF.getRegInfo().invalidateLiveness();

  // --- First pass: identify basic-block boundaries -------------------------
  ArrayRef<uint8_t> bytes(textBytes.data(), textBytes.size());
  uint64_t size = textBytes.size();

  std::set<uint64_t> blockStarts;
  blockStarts.insert(0);
  {
    uint64_t scanOffset = 0;
    while (scanOffset < size) {
      MCInst inst;
      uint64_t instSize = 0;
      auto status = disasm->getInstruction(inst, instSize,
                                           bytes.slice(scanOffset),
                                           scanOffset, nulls());
      if (status != MCDisassembler::Success) {
        scanOffset += 4;
        continue;
      }

      const MCInstrDesc &desc = live.instrInfo->get(inst.getOpcode());
      std::string mnem = getMnemonic(live, inst);

      if (desc.isBranch() || desc.isConditionalBranch()) {
        for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
          if (inst.getOperand(i).isImm()) {
            int64_t brOffset = inst.getOperand(i).getImm();
            uint64_t target = scanOffset + 4 + (brOffset * 4);
            blockStarts.insert(target);
          }
        }
        if (desc.isConditionalBranch())
          blockStarts.insert(scanOffset + instSize);
      }

      if (mnem == "s_endpgm")
        break;
      scanOffset += instSize;
    }
  }

  // Create MBBs for every block start
  std::map<uint64_t, MachineBasicBlock *> offsetToMBB;
  for (uint64_t addr : blockStarts) {
    auto *newMBB = MF.CreateMachineBasicBlock();
    MF.push_back(newMBB);
    offsetToMBB[addr] = newMBB;
  }

  // --- Second pass: lift MCInst → MachineInstr -----------------------------
  MachineBasicBlock *MBB = offsetToMBB[0];
  uint64_t offset = 0;
  int liftedCount = 0;
  int totalCount = 0;

  while (offset < size) {
    auto it = offsetToMBB.find(offset);
    if (it != offsetToMBB.end())
      MBB = it->second;

    MCInst inst;
    uint64_t instSize = 0;
    auto status = disasm->getInstruction(inst, instSize, bytes.slice(offset),
                                         offset, nulls());
    if (status != MCDisassembler::Success) {
      errs() << "mir_proto: Disassembly failed at offset 0x"
             << format_hex(offset, 1) << ", skipping 4 bytes\n";
      offset += 4;
      continue;
    }

    totalCount++;
    const MCInstrDesc &desc = live.instrInfo->get(inst.getOpcode());
    std::string mnem = getMnemonic(live, inst);

    if (mnem == "s_nop" || mnem == "s_code_end") {
      offset += instSize;
      continue;
    }

    if (mnem == "s_endpgm") {
      auto MIB2 =
          BuildMI(*MBB, MBB->end(), DebugLoc(), TII->get(inst.getOpcode()));
      if (inst.getNumOperands() > 0 && inst.getOperand(0).isImm())
        MIB2.addImm(inst.getOperand(0).getImm());
      else
        MIB2.addImm(0);
      liftedCount++;
      offset += instSize;
      break;
    }

    unsigned numExplicitDefs = desc.getNumDefs();
    unsigned mcNumOps = inst.getNumOperands();

    // Resolve branch targets to MBBs
    MachineBasicBlock *branchTargetMBB = nullptr;
    if (desc.isBranch()) {
      for (unsigned i = 0; i < mcNumOps; ++i) {
        if (inst.getOperand(i).isImm()) {
          int64_t brOffset = inst.getOperand(i).getImm();
          uint64_t target = offset + 4 + (brOffset * 4);
          auto tgtIt = offsetToMBB.find(target);
          if (tgtIt != offsetToMBB.end())
            branchTargetMBB = tgtIt->second;
        }
      }
    }

    auto MIB = BuildMI(*MBB, MBB->end(), DebugLoc(), desc);

    for (unsigned i = 0; i < mcNumOps; ++i) {
      const MCOperand &mcOp = inst.getOperand(i);

      if (mcOp.isReg()) {
        MCRegister reg(mcOp.getReg());
        RegState flags =
            i < numExplicitDefs ? RegState::Define : RegState{};
        MIB.addReg(reg, flags);
      } else if (mcOp.isImm()) {
        if (desc.isBranch() && branchTargetMBB) {
          MIB.addMBB(branchTargetMBB);
          branchTargetMBB = nullptr;
        } else {
          MIB.addImm(mcOp.getImm());
        }
      }
    }

    // CFG edges for conditional branches
    if (desc.isConditionalBranch()) {
      uint64_t fallthrough = offset + instSize;
      auto ftIt = offsetToMBB.find(fallthrough);
      if (ftIt != offsetToMBB.end())
        MBB->addSuccessor(ftIt->second);
      for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
        if (inst.getOperand(i).isImm()) {
          int64_t brOffset = inst.getOperand(i).getImm();
          uint64_t target = offset + 4 + (brOffset * 4);
          auto tgtIt = offsetToMBB.find(target);
          if (tgtIt != offsetToMBB.end())
            MBB->addSuccessor(tgtIt->second);
        }
      }
    }

    liftedCount++;
    offset += instSize;
  }

  result.liftedCount = liftedCount;
  result.totalCount = totalCount;

  // --- Serialize MIR text for inspection ---
  {
    raw_string_ostream mirOS(result.mirText);
    printMIR(mirOS, *live.mod);
    printMIR(mirOS, *live.mmi, MF);
  }

  result.notes.push_back(
      "Implicit operands (VCC, SCC, EXEC) auto-populated from MCInstrDesc");
  result.notes.push_back("Opcode-based dispatch — no mnemonic string parsing "
                         "for instruction creation");
  result.notes.push_back("Assembly generated from live MachineFunction, "
                         "not from re-encoding original bytes");
  result.success = true;
  return result;
}

// ---------------------------------------------------------------------------
// generateAssembly — emit assembly by walking the live MachineFunction
// ---------------------------------------------------------------------------
std::string LiftResult::generateAssembly(const std::string &targetISA,
                                         const std::string &kernelName) const {
  if (!liveMF || !liveMF->mf) {
    errs() << "mir_proto: generateAssembly called with no live MachineFunction\n";
    return {};
  }

  const auto &live = *liveMF;
  MachineFunction &MF = *live.mf;

  // Use decimal immediates for assembly output (llvm-mc accepts both).
  live.printer->setPrintImmHex(false);

  std::string asmText;
  raw_string_ostream os(asmText);

  // --- Prologue ---
  os << "\t.amdgcn_target \"amdgcn-amd-amdhsa--" << targetISA << "\"\n";
  os << "\t.amdhsa_code_object_version 6\n";
  os << "\t.text\n";
  os << "\t.globl\t" << kernelName << "\n";
  os << "\t.p2align\t8\n";
  os << "\t.type\t" << kernelName << ",@function\n";
  os << kernelName << ":\n";

  // --- Assign labels to MBBs ---
  std::map<const MachineBasicBlock *, std::string> mbbLabels;
  int labelIdx = 0;
  for (const auto &MBB : MF)
    mbbLabels[&MBB] = ".LBB0_" + std::to_string(labelIdx++);

  // --- Walk MachineFunction, lower each MI to MCInst, print ----------------
  bool isFirst = true;
  for (const auto &MBB : MF) {
    if (!isFirst)
      os << mbbLabels[&MBB] << ":\n";
    isFirst = false;

    for (const auto &MI : MBB) {
      // Check whether any explicit operand is an MBB reference (branch).
      const MachineOperand *mbbOp = nullptr;
      for (unsigned i = 0; i < MI.getNumExplicitOperands(); ++i) {
        if (MI.getOperand(i).isMBB()) {
          mbbOp = &MI.getOperand(i);
          break;
        }
      }

      if (!mbbOp) {
        // Regular instruction: build MCInst from explicit operands and print.
        MCInst mcInst;
        mcInst.setOpcode(MI.getOpcode());
        for (unsigned i = 0; i < MI.getNumExplicitOperands(); ++i) {
          const MachineOperand &MO = MI.getOperand(i);
          if (MO.isReg())
            mcInst.addOperand(MCOperand::createReg(MO.getReg()));
          else if (MO.isImm())
            mcInst.addOperand(MCOperand::createImm(MO.getImm()));
        }

        std::string instStr;
        raw_string_ostream ios(instStr);
        live.printer->printInst(&mcInst, 0, "", *live.subtargetInfo, ios);
        os << instStr << "\n";
      } else {
        // Branch with MBB target: extract mnemonic via a dummy MCInst,
        // then emit  mnemonic <label>.
        MCInst dummy;
        dummy.setOpcode(MI.getOpcode());
        for (unsigned i = 0; i < MI.getNumExplicitOperands(); ++i) {
          const MachineOperand &MO = MI.getOperand(i);
          if (MO.isReg())
            dummy.addOperand(MCOperand::createReg(MO.getReg()));
          else if (MO.isImm())
            dummy.addOperand(MCOperand::createImm(MO.getImm()));
          else if (MO.isMBB())
            dummy.addOperand(MCOperand::createImm(0));
        }
        std::string s;
        raw_string_ostream tos(s);
        live.printer->printInst(&dummy, 0, "", *live.subtargetInfo, tos);
        StringRef sr(s);
        sr = sr.ltrim();
        std::string mnem = sr.split('\t').first.split(' ').first.str();

        os << "\t" << mnem << " " << mbbLabels[mbbOp->getMBB()] << "\n";
      }
    }
  }

  // --- Kernel descriptor (hardcoded for vecadd) ---
  os << "\t.section\t.rodata,\"a\",@progbits\n";
  os << "\t.p2align\t6, 0x0\n";
  os << "\t.amdhsa_kernel " << kernelName << "\n";
  os << "\t\t.amdhsa_group_segment_fixed_size 0\n";
  os << "\t\t.amdhsa_private_segment_fixed_size 0\n";
  os << "\t\t.amdhsa_kernarg_size 288\n";
  os << "\t\t.amdhsa_user_sgpr_count 2\n";
  os << "\t\t.amdhsa_user_sgpr_dispatch_ptr 0\n";
  os << "\t\t.amdhsa_user_sgpr_queue_ptr 0\n";
  os << "\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1\n";
  os << "\t\t.amdhsa_user_sgpr_dispatch_id 0\n";
  os << "\t\t.amdhsa_user_sgpr_kernarg_preload_length 0\n";
  os << "\t\t.amdhsa_user_sgpr_kernarg_preload_offset 0\n";
  os << "\t\t.amdhsa_user_sgpr_private_segment_size 0\n";
  os << "\t\t.amdhsa_uses_dynamic_stack 0\n";
  os << "\t\t.amdhsa_enable_private_segment 0\n";
  os << "\t\t.amdhsa_system_sgpr_workgroup_id_x 1\n";
  os << "\t\t.amdhsa_system_sgpr_workgroup_id_y 0\n";
  os << "\t\t.amdhsa_system_sgpr_workgroup_id_z 0\n";
  os << "\t\t.amdhsa_system_sgpr_workgroup_info 0\n";
  os << "\t\t.amdhsa_system_vgpr_workitem_id 0\n";
  os << "\t\t.amdhsa_next_free_vgpr 8\n";
  os << "\t\t.amdhsa_next_free_sgpr 8\n";
  os << "\t\t.amdhsa_accum_offset 8\n";
  os << "\t\t.amdhsa_reserve_vcc 1\n";
  os << "\t\t.amdhsa_float_round_mode_32 0\n";
  os << "\t\t.amdhsa_float_round_mode_16_64 0\n";
  os << "\t\t.amdhsa_float_denorm_mode_32 3\n";
  os << "\t\t.amdhsa_float_denorm_mode_16_64 3\n";
  os << "\t\t.amdhsa_dx10_clamp 1\n";
  os << "\t\t.amdhsa_ieee_mode 1\n";
  os << "\t\t.amdhsa_fp16_overflow 0\n";
  os << "\t\t.amdhsa_tg_split 0\n";
  os << "\t\t.amdhsa_exception_fp_ieee_invalid_op 0\n";
  os << "\t\t.amdhsa_exception_fp_denorm_src 0\n";
  os << "\t\t.amdhsa_exception_fp_ieee_div_zero 0\n";
  os << "\t\t.amdhsa_exception_fp_ieee_overflow 0\n";
  os << "\t\t.amdhsa_exception_fp_ieee_underflow 0\n";
  os << "\t\t.amdhsa_exception_fp_ieee_inexact 0\n";
  os << "\t\t.amdhsa_exception_int_div_zero 0\n";
  os << "\t.end_amdhsa_kernel\n";
  os << "\t.text\n";

  // --- Metadata YAML (hardcoded for vecadd) ---
  os << "\n\t.amdgpu_metadata\n";
  os << "---\n";
  os << "amdhsa.kernels:\n";
  os << "  - .args:\n";
  os << "      - .address_space: global\n";
  os << "        .offset:         0\n";
  os << "        .size:           8\n";
  os << "        .value_kind:     global_buffer\n";
  os << "      - .address_space: global\n";
  os << "        .offset:         8\n";
  os << "        .size:           8\n";
  os << "        .value_kind:     global_buffer\n";
  os << "      - .address_space: global\n";
  os << "        .offset:         16\n";
  os << "        .size:           8\n";
  os << "        .value_kind:     global_buffer\n";
  os << "      - .offset:         24\n";
  os << "        .size:           4\n";
  os << "        .value_kind:     by_value\n";
  os << "      - .offset:         32\n";
  os << "        .size:           4\n";
  os << "        .value_kind:     hidden_block_count_x\n";
  os << "      - .offset:         36\n";
  os << "        .size:           4\n";
  os << "        .value_kind:     hidden_block_count_y\n";
  os << "      - .offset:         40\n";
  os << "        .size:           4\n";
  os << "        .value_kind:     hidden_block_count_z\n";
  os << "      - .offset:         44\n";
  os << "        .size:           2\n";
  os << "        .value_kind:     hidden_group_size_x\n";
  os << "      - .offset:         46\n";
  os << "        .size:           2\n";
  os << "        .value_kind:     hidden_group_size_y\n";
  os << "      - .offset:         48\n";
  os << "        .size:           2\n";
  os << "        .value_kind:     hidden_group_size_z\n";
  os << "      - .offset:         50\n";
  os << "        .size:           2\n";
  os << "        .value_kind:     hidden_remainder_x\n";
  os << "      - .offset:         52\n";
  os << "        .size:           2\n";
  os << "        .value_kind:     hidden_remainder_y\n";
  os << "      - .offset:         54\n";
  os << "        .size:           2\n";
  os << "        .value_kind:     hidden_remainder_z\n";
  os << "      - .offset:         72\n";
  os << "        .size:           8\n";
  os << "        .value_kind:     hidden_global_offset_x\n";
  os << "      - .offset:         80\n";
  os << "        .size:           8\n";
  os << "        .value_kind:     hidden_global_offset_y\n";
  os << "      - .offset:         88\n";
  os << "        .size:           8\n";
  os << "        .value_kind:     hidden_global_offset_z\n";
  os << "      - .offset:         96\n";
  os << "        .size:           2\n";
  os << "        .value_kind:     hidden_grid_dims\n";
  os << "    .group_segment_fixed_size: 0\n";
  os << "    .kernarg_segment_align: 8\n";
  os << "    .kernarg_segment_size: 288\n";
  os << "    .language:       OpenCL C\n";
  os << "    .language_version:\n";
  os << "      - 2\n";
  os << "      - 0\n";
  os << "    .max_flat_workgroup_size: 1024\n";
  os << "    .name:           " << kernelName << "\n";
  os << "    .private_segment_fixed_size: 0\n";
  os << "    .sgpr_count:     8\n";
  os << "    .sgpr_spill_count: 0\n";
  os << "    .symbol:         " << kernelName << ".kd\n";
  os << "    .vgpr_count:     8\n";
  os << "    .vgpr_spill_count: 0\n";
  os << "    .wavefront_size: 64\n";
  os << "amdhsa.target:   amdgcn-amd-amdhsa--" << targetISA << "\n";
  os << "amdhsa.version:\n";
  os << "  - 1\n";
  os << "  - 2\n";
  os << "...\n\n";
  os << "\t.end_amdgpu_metadata\n";

  // Restore hex mode for any subsequent MIR printing.
  live.printer->setPrintImmHex(true);

  return asmText;
}

} // namespace mir_proto
