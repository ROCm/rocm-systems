#include "raiser.hpp"
#include "amdgpu_formats.hpp"
#include "code_object_utils.hpp"
#include "semop.hpp"
#include "isa_profile.hpp"
#include "decoded_inst.hpp"
#include "parsed_reg.hpp"

#include "mc_state.hpp"
#include "opcode_map.hpp"
#include "canonicalize.hpp"
#include "reg_file.hpp"
#include "kernarg_layout.hpp"
#include "raise_context.hpp"
#include "handlers.hpp"

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
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
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

namespace transpiler {

// parseReg, readOp32/64/ExecWidth, and OpResolver are now in raise_context.hpp/cpp

// ============================================================================
// Main raising function
// ============================================================================

RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &sourceISA,
                      const std::string &kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset,
                      const std::string &compilationTargetISA) {
  RaiseResult result;

  MCState mc;
  if (!initMCState(mc, sourceISA))
    return result;

  ISAProfile isa = ISAProfile::fromTarget(StringRef(sourceISA));
  ISAProfile targetIsa = compilationTargetISA.empty()
      ? isa
      : ISAProfile::fromTarget(StringRef(compilationTargetISA));

  // Build opcode → SemOp map from MCInstrInfo
  OpcodeMap opcMap;
  opcMap.build(*mc.instrInfo);

  // ==== Phase 1: Disassemble + identify block boundaries ====
  ArrayRef<uint8_t> bytes(textBytes.data(), textBytes.size());
  uint64_t totalSize = textBytes.size();
  std::vector<DecodedInst> insts;
  std::set<uint64_t> blockStarts;
  blockStarts.insert(kernelOffset);

  if (kernelOffset > 0)
    errs() << "transpiler: Starting disassembly at kernel offset 0x"
           << utohexstr(kernelOffset) << "\n";

  {
    uint64_t off = kernelOffset;
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
      {
        std::string s;
        raw_string_ostream os(s);
        mc.printer->printInst(&inst, 0, "", *mc.subtargetInfo, os);
        di.fullText = StringRef(s).ltrim().str();
      }
      std::string stripped = stripEncoding(StringRef(di.rawMnemonic)).str();
      di.mnemonic = canonicalizeMnemonic(StringRef(stripped));
      di.inst = inst;
      di.semOp = opcMap.lookup(inst.getOpcode());
      di.numDefs = desc.getNumDefs();
      di.isBranch = desc.isBranch();
      di.isConditionalBranch = desc.isConditionalBranch();
      di.offset = off;
      di.size = instSize;

      di.tsFlags = desc.TSFlags;
      di.format = classifyFormat(desc.TSFlags);
      // VOPD detection: mnemonic prefix is reliable across LLVM versions
      if (StringRef(di.mnemonic).starts_with("v_dual_"))
        di.format = FormatKind::VOPD;
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
      if (isEnd) {
        // s_endpgm may appear mid-binary (early-return path); if there are
        // known block starts at later offsets, keep disassembling.
        uint64_t nextOff = off + instSize;
        auto it = blockStarts.upper_bound(off);
        if (it != blockStarts.end() && *it < textBytes.size()) {
          off = nextOff;
          continue;
        }
        break;
      }
      off += instSize;
    }
  }

  result.totalCount = (int)insts.size();

  // ==== Phase 2: Build LLVM IR module + function ====
  result.ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *result.ctx;
  result.module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *result.module;
  M.setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  TargetOptions opts;
  std::unique_ptr<TargetMachine> tm(mc.target->createTargetMachine(
      Triple("amdgcn-amd-amdhsa"),
      compilationTargetISA.empty() ? sourceISA : compilationTargetISA,
      "", opts, Reloc::PIC_));
  if (!tm) {
    errs() << "transpiler: Failed to create TargetMachine\n";
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
  {
    int maxWg = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 1024;
    int waveSz = targetIsa.isWave32() ? 32 : 64;
    int minWaves = (maxWg + waveSz - 1) / waveSz;
    F->addFnAttr("amdgpu-flat-work-group-size",
                  std::to_string(maxWg) + "," + std::to_string(maxWg));
    if (minWaves > 1)
      F->addFnAttr("amdgpu-waves-per-eu",
                    std::to_string(minWaves) + "," + std::to_string(minWaves));
  }

  for (int i = 0; i < paramIdx; i++)
    F->getArg(i)->setName("arg" + std::to_string(i));

  errs() << "transpiler: Kernel '" << kernelName << "' has " << paramIdx
         << " args (kernarg_segment_size=" << meta.kernargSegmentSize << ")\n";

  Function *fnWorkgroupIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_x);
  Function *fnWorkitemIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workitem_id_x);
  // ==== Phase 3: Create basic blocks ====
  std::map<uint64_t, BasicBlock *> offsetToBB;
  for (uint64_t addr : blockStarts)
    offsetToBB[addr] = BasicBlock::Create(C, "bb_0x" + utohexstr(addr - kernelOffset), F);

  // ==== Phase 4: Init entry registers ====
  IRBuilder<> B(offsetToBB[kernelOffset]);

  AllocaRegFile regs;
  regs.init(B, i32Ty, i1Ty, isa);

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

  // On RDNA3+ (gfx12xx), the hardware command processor uses TTMP registers
  // for workgroup scheduling.
  //   ttmp9 = workgroup_id_x (accelerated launch)
  //   ttmp8[29:25] = wave_id within workgroup (subgroup ID)
  if (isa.target.find("gfx12") != std::string::npos) {
    B.CreateStore(B.CreateCall(fnWorkgroupIdX, {}, "ttmp9_wg_id"), regs.ttmp[9]);

    // wave_id = workitem_id_x / wavefront_size (32 for gfx12)
    Value *tidForTtmp = B.CreateCall(fnWorkitemIdX, {}, "ttmp8_tid");
    Value *waveId = B.CreateLShr(tidForTtmp, B.getInt32(5), "wave_id_in_wg");
    Value *ttmp8Val = B.CreateShl(waveId, B.getInt32(25), "ttmp8_val");
    B.CreateStore(ttmp8Val, regs.ttmp[8]);
  }

  // ==== Phase 5: Raise each instruction ====

  auto *f16Ty = Type::getHalfTy(C);
  RaiseContext ctx{C, M, B, regs, mc, isa, targetIsa, kernargs, F,
                   i1Ty, i8Ty, i32Ty, i64Ty, f32Ty, f16Ty,
                   ptrGlobalTy, offsetToBB};

  BasicBlock *currentBB = offsetToBB[kernelOffset];
  int raisedCount = 0;

  for (size_t instIdx = 0; instIdx < insts.size(); ++instIdx) {
    const DecodedInst &di = insts[instIdx];

    auto bbIt = offsetToBB.find(di.offset);
    if (bbIt != offsetToBB.end() && bbIt->second != currentBB) {
      if (currentBB->empty() || !currentBB->getTerminator())
        B.CreateBr(bbIt->second);
      currentBB = bbIt->second;
      B.SetInsertPoint(currentBB);
    }

    ctx.computeVGPRAdjust(di);
    OpResolver op{ctx, di};

    HandlerResult hr;
    switch (di.format) {
    case FormatKind::SOPP:  hr = handleSOPP(ctx, di, op, result); break;
    case FormatKind::SMEM:  hr = handleSMEM(ctx, di, op, result); break;
    case FormatKind::SOPC:  hr = handleSOPC(ctx, di, op, result); break;
    case FormatKind::SOP1:  hr = handleSOP1(ctx, di, op, result); break;
    case FormatKind::SOPK:  hr = handleSOPK(ctx, di, op, result); break;
    case FormatKind::SOP2:  hr = handleSOP2(ctx, di, op, result); break;
    case FormatKind::DPP:
    case FormatKind::SDWA:
    case FormatKind::VOP1:
    case FormatKind::VOP2:
    case FormatKind::VOP3:
    case FormatKind::VOPC:
    case FormatKind::VOP3P:  hr = handleVALU(ctx, di, op, result); break;
    case FormatKind::FLAT:   hr = handleFLAT(ctx, di, op, result); break;
    case FormatKind::DS:     hr = handleDS(ctx, di, op, result); break;
    case FormatKind::MUBUF:  hr = handleMUBUF(ctx, di, op, result); break;
    case FormatKind::MFMA:   hr = handleMFMA(ctx, di, op, result); break;
    case FormatKind::VOPD:   hr = handleVOPD(ctx, di, op, result); break;
    default: break;
    }

    // Handler may have set failure on result directly (e.g. SMEM kernarg fail)
    if (!hr.handled && !result.failMnemonic.empty())
      return result;

    if (hr.handled) {
      if (di.defsSCC && !hr.sccHandled && hr.sccResult) {
        Value *zero = Constant::getNullValue(hr.sccResult->getType());
        ctx.regs.storeSCC(ctx.B, ctx.B.CreateICmpNE(hr.sccResult, zero));
      }
      if (di.defsEXEC)
        result.hasDivergentExec = true;
      raisedCount++;
      continue;
    }

    result.failMnemonic = di.mnemonic;
    result.failFormat = formatName(di.format);
    errs() << "transpiler: Unsupported instruction: " << di.mnemonic
           << " (raw: " << di.rawMnemonic << ")"
           << " [format=" << formatName(di.format) << "]"
           << " at offset 0x" << format_hex(di.offset, 1) << "\n";
    return result;
  }

  // Ensure all BBs have terminators
  for (auto &BB : *F) {
    if (BB.empty() || !BB.getTerminator()) {
      B.SetInsertPoint(&BB);
      B.CreateUnreachable();
    }
  }

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
    errs() << "transpiler: IR verification failed:\n" << verifyErr << "\n";
    return result;
  }

  {
    raw_string_ostream irOS(result.irText);
    M.print(irOS, nullptr);
  }

  result.success = true;
  return result;
}

} // namespace transpiler
