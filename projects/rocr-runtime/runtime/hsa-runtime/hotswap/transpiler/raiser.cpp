#include "raiser.hpp"
#include "amdgpu_formats.hpp"
#include "code_object_utils.hpp"
#include "semop.hpp"
#include "isa_profile.hpp"
#include "decoded_inst.hpp"
#include "parsed_reg.hpp"

#include "mc_state.hpp"
#include "opcode_map.hpp"
#include "Utils/AMDGPUBaseInfo.h"
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
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
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
  initMCState(mc, sourceISA);

  ISAProfile isa = ISAProfile::fromSubtarget(*mc.subtargetInfo);
  // When the caller does not specify a distinct compilation target we raise
  // in place and reuse the source profile; otherwise we spin up a throwaway
  // MCSubtargetInfo just to snapshot the target's feature bits.
  ISAProfile targetIsa = isa;
  std::unique_ptr<MCSubtargetInfo> targetSTI;
  if (!compilationTargetISA.empty()) {
    targetSTI = buildSubtargetInfo(*mc.target, compilationTargetISA);
    targetIsa = ISAProfile::fromSubtarget(*targetSTI);
  }

  // Build opcode → SemOp map from MCInstrInfo
  OpcodeMap opcMap;
  opcMap.build(*mc.instrInfo);

  // Fail loudly if any MFMA-format SemOp is missing a handler row. Cheap
  // startup walk that catches table drift before any kernel is lifted.
  verifyMFMACoverage(*mc.instrInfo, opcMap);

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
      di.mnemonic = stripEncoding(StringRef(di.rawMnemonic)).str();
      di.inst = inst;
      di.semOp = opcMap.lookup(inst.getOpcode());
      if (di.semOp == SemOp::V_CMP || di.semOp == SemOp::V_CMPX)
        di.vcmp = opcMap.lookupVCmp(inst.getOpcode());
      di.numDefs = desc.getNumDefs();
      di.isBranch = desc.isBranch();
      di.isConditionalBranch = desc.isConditionalBranch();
      di.offset = off;
      di.size = instSize;

      di.tsFlags = desc.TSFlags;
      di.firstSrcIdx = desc.getNumDefs();

      // Build the logical-source view of the MCInst. We walk `desc.operands()`
      // and classify each operand using TableGen-generated metadata only:
      //
      //   * Operand types carrying the AMDGPU-specific `OPERAND_INPUT_MODS`
      //     tag are VOP3 source modifiers (neg/abs/opsel packed as an imm).
      //     They attach to the next logical source via `modMap`.
      //   * DPP/SDWA encodings carry a tied "old" input (fallback value for
      //     inactive lanes, named `$old` or `$vdst_in` in TableGen). In our
      //     all-lanes-active scalar model that slot is never read, so we
      //     skip it. Not every tied-to-def operand is a fallback — VOP2 MAC
      //     forms (v_fmac_f32, v_mac_f32, v_dot2c_*) tie `$src2` to the dst
      //     and atomics tie `$vdata_in`/`$sdst_in`/`$addr_in`; in those
      //     cases the tied operand is a real accumulator/read-modify input
      //     and must stay in srcMap. We therefore select on the named-
      //     operand id rather than the TIED_TO bit alone.
      //   * Everything else is a logical source recorded in MCInst order.
      //
      // A reportErr helper factors out the fatal-error text builder used by
      // the validation checks below (item 3: drift detection).
      auto reportErr = [&](const Twine &prefix, int index, int ours,
                           int expected) -> void {
        std::string msg;
        raw_string_ostream os(msg);
        os << prefix << " for " << di.rawMnemonic
           << " (opcode=" << inst.getOpcode() << "): index=" << index
           << ", srcMap/modMap=" << ours << ", named=" << expected
           << ", numSrcs=" << di.numSrcs
           << ", numDefs=" << desc.getNumDefs()
           << ", numOps=" << inst.getNumOperands();
        report_fatal_error(StringRef(msg));
      };

      unsigned opc = inst.getOpcode();
      int oldIdx = AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::old);
      int vdstInIdx =
          AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::vdst_in);
      auto opInfos = desc.operands();
      unsigned pendingModIdx = UINT_MAX;
      for (unsigned i = di.firstSrcIdx; i < inst.getNumOperands(); ++i) {
        if (i < opInfos.size() &&
            opInfos[i].OperandType == OPERAND_INPUT_MODS) {
          pendingModIdx = i;
          continue;
        }
        if ((int)i == oldIdx || (int)i == vdstInIdx) {
          pendingModIdx = UINT_MAX;
          continue;
        }
        if (di.numSrcs >= DecodedInst::kMaxSrcs)
          report_fatal_error("transpiler: DecodedInst::kMaxSrcs exceeded; "
                             "bump kMaxSrcs to match the widest LLVM operand "
                             "list");
        di.srcMap[di.numSrcs] = i;
        di.modMap[di.numSrcs] = pendingModIdx;
        di.numSrcs++;
        pendingModIdx = UINT_MAX;
      }

      // Drift check A: every tied-to-def operand on this instruction must
      // have an OpName we've explicitly classified. If LLVM introduces a new
      // tied-input OpName we haven't audited (so we don't know whether to
      // skip or keep it), stop and make a human decide. `kKnownTiedIn` is
      // the exhaustive audit as of this commit. Two semantic categories:
      //
      //   skipped-as-fallback (DPP/SDWA inactive-lane value; never read
      //                       in the all-lanes-active scalar model):
      //     `old`, `vdst_in`.
      //
      //   kept-as-real-input (read-modify accumulator, atomic compare, or
      //                      MAC-style third source; the instruction
      //                      semantically reads the prior def value):
      //     `sdst_in`, `vdata_in`, `addr_in`, `srcTiedDef`,
      //     `src0`, `src1`, `src2`,
      //     `src0X`, `src0Y`, `src2X`, `src2Y`,
      //     `vsrc2X`, `vsrc2Y`.
      //
      // srcN and VOPD variants all appear here because SOPK `S_ADDK_I32`
      // ties `$src0`, SOP2 `sdst,sdst_in` variants may also surface `$src0`,
      // VALU MAC forms tie `$src2`, and VOPD3 FMAC halves tie `$src2X` /
      // `$src2Y` (plus potentially the separate VOPD3 third source).
      static constexpr AMDGPU::OpName kKnownTiedIn[] = {
          AMDGPU::OpName::old,        AMDGPU::OpName::vdst_in,
          AMDGPU::OpName::sdst_in,    AMDGPU::OpName::vdata_in,
          AMDGPU::OpName::addr_in,    AMDGPU::OpName::srcTiedDef,
          AMDGPU::OpName::src0,       AMDGPU::OpName::src1,
          AMDGPU::OpName::src2,       AMDGPU::OpName::src0X,
          AMDGPU::OpName::src0Y,      AMDGPU::OpName::src2X,
          AMDGPU::OpName::src2Y,      AMDGPU::OpName::vsrc2X,
          AMDGPU::OpName::vsrc2Y,
      };
      for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
        int tied = desc.getOperandConstraint(i, MCOI::TIED_TO);
        if (tied < 0)
          continue;
        // Only flag operands tied to a def. Use-to-use ties exist in LLVM's
        // constraint system but are not relevant to the fallback/accumulator
        // distinction this check protects.
        if ((unsigned)tied >= desc.getNumDefs())
          continue;
        bool known = false;
        for (AMDGPU::OpName n : kKnownTiedIn) {
          if ((int)i == AMDGPU::getNamedOperandIdx(opc, n)) {
            known = true;
            break;
          }
        }
        if (!known)
          reportErr("transpiler: tied-to-def operand has an OpName not in "
                    "the audited set — classify explicitly (fallback to skip "
                    "vs. real input to keep) before proceeding",
                    (int)i, tied, -1);
      }

      // Drift check B: for every opcode that exposes `srcN` / `srcN_modifiers`
      // naming (VALU, VOPC, SOP1/SOP2, a handful of scalar forms), the first
      // N entries of srcMap / modMap must agree with LLVM's named-operand
      // table. Catches operand-layout drift for the large majority of
      // opcodes — but notably NOT for DS / MUBUF / FLAT / SMEM / image
      // encodings, which don't use srcN naming; those formats are only
      // protected by the walk's correctness and drift check A.
      {
        static constexpr AMDGPU::OpName kSrcNames[] = {
            AMDGPU::OpName::src0, AMDGPU::OpName::src1,
            AMDGPU::OpName::src2};
        static constexpr AMDGPU::OpName kModNames[] = {
            AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
            AMDGPU::OpName::src2_modifiers};
        for (unsigned k = 0; k < 3; ++k) {
          int namedSrc = AMDGPU::getNamedOperandIdx(opc, kSrcNames[k]);
          if (namedSrc < 0)
            break;
          int ourSrc =
              (k < di.numSrcs) ? (int)di.srcMap[k] : -1;
          if (ourSrc != namedSrc)
            reportErr("transpiler: srcMap disagrees with OpName::srcN table",
                      (int)k, ourSrc, namedSrc);
          int namedMod = AMDGPU::getNamedOperandIdx(opc, kModNames[k]);
          int ourMod = (di.modMap[k] == UINT_MAX) ? -1 : (int)di.modMap[k];
          int expectedMod = (namedMod < 0) ? -1 : namedMod;
          if (ourMod != expectedMod) {
            // Scaled MFMA instructions (ScaledMAIInst in TableGen) append
            // src0_modifiers / src1_modifiers AFTER all source operands,
            // not interleaved as in VOP3. Our walk can't discover them
            // because it only looks for OPERAND_INPUT_MODS *before* each
            // source. Repair the modMap from LLVM's authoritative named-
            // operand table, but ONLY for MAI-format instructions so we
            // don't silently mask future layout drift in other formats.
            bool isMAI = di.tsFlags & SIInstrFlags::IsMAI;
            if (isMAI && namedMod >= 0 && ourMod == -1) {
              di.modMap[k] = (unsigned)namedMod;
            } else {
              reportErr(
                  "transpiler: modMap disagrees with OpName::srcN_modifiers "
                  "table",
                  (int)k, ourMod, expectedMod);
            }
          }
        }
      }

      // Identify implicit defs of wave-mask / condition-flag registers via
      // identity constants rather than register-name string matches. We
      // normalise through `mc2PseudoReg` first, which strips subtarget
      // suffixes (``_gfxNplus``) and converts aliases to their canonical
      // pseudo-register id — same pattern used by `parseReg`.
      for (MCPhysReg r : desc.implicit_defs()) {
        llvm::MCRegister reg = AMDGPU::mc2PseudoReg(r);
        switch (reg) {
        case AMDGPU::SCC:
          di.defsSCC = true;
          break;
        case AMDGPU::VCC:
        case AMDGPU::VCC_LO:
        case AMDGPU::VCC_HI:
          di.defsVCC = true;
          break;
        case AMDGPU::EXEC:
        case AMDGPU::EXEC_LO:
        case AMDGPU::EXEC_HI:
          di.defsEXEC = true;
          break;
        default:
          break;
        }
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

      bool isEnd = (di.semOp == SemOp::S_ENDPGM);
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

  {
    raw_string_ostream disOS(result.disasmText);
    for (const auto &di : insts) {
      disOS << format_hex_no_prefix(di.offset, 8) << ":  " << di.fullText
            << "\n";
    }
  }

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
    // Pin the workgroup size to exactly what the source kernel declared, so
    // the backend lays out LDS / workitem IDs the same way the original
    // gfx1250 binary did.
    int maxWg = meta.maxFlatWorkgroupSize > 0 ? meta.maxFlatWorkgroupSize : 1024;
    F->addFnAttr("amdgpu-flat-work-group-size",
                  std::to_string(maxWg) + "," + std::to_string(maxWg));

    // Deliberately do NOT set "amdgpu-waves-per-eu".  Pinning occupancy
    // constrains register allocation and caused spurious VGPR spills for
    // wide kernels (e.g. the Triton 128x128 matmul on gfx942), which then
    // triggered memory faults because our raised IR is register-pressure
    // heavy compared to a from-source compile.  Letting the backend choose
    // occupancy freely keeps register pressure safe.
    // TODO(gfx1250→gfx942): revisit once the raiser emits tighter IR; we may
    // want to propagate the source kernel's waves-per-eu for parity.
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

  // On gfx12+ the hardware command processor uses TTMP registers for
  // workgroup scheduling (RDNA4+ / CDNA-next layout):
  //   ttmp9        = workgroup_id_x  (accelerated launch)
  //   ttmp8[29:25] = wave_id within workgroup (subgroup ID)
  // gfx11 (RDNA3) passes these via SGPRs set up by the CP instead.
  if (AMDGPU::isGFX12Plus(*mc.subtargetInfo)) {
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
      // LLVM's AMDGPULowerVGPREncoding pass resets VGPR MSB mode at every
      // basic-block boundary (both before terminators and at BB fall-through
      // exits).  Mirror that behaviour so we do not inherit stale MSB state
      // from a previous linear instruction that does not control-flow into
      // this BB.
      ctx.vgprMSBs = 0;
    }

    ctx.computeVGPRAdjust(di);
    OpResolver op{ctx, di};

    // Dispatch to the format-specific handler by querying TSFlags (and
    // `AMDGPU::isVOPD` for the one encoding without a dedicated flag bit)
    // directly, rather than going through a hand-rolled FormatKind enum.
    // Check precedence mirrors LLVM's decoder:
    //   * VOPD first — it has no TSFlags bit; detect by named-operand id.
    //   * IsMAI before VOP3 — MFMA is a VOP3 subclass with its own handler.
    //   * DPP / SDWA / VOPC / VOP3P / VOP3 / VOP2 / VOP1 all route to
    //     handleVALU, so they're collapsed into one mask test; ordering
    //     within the VOP family is therefore irrelevant here.
    //   * Scalar / memory family bits are mutually exclusive.
    // `default: break;` semantics are preserved: anything without a matching
    // bit falls through with `hr.handled == false` and hits the unsupported-
    // instruction error path below.
    const uint64_t kVALU =
        SIInstrFlags::DPP | SIInstrFlags::SDWA | SIInstrFlags::VOP1 |
        SIInstrFlags::VOP2 | SIInstrFlags::VOP3 | SIInstrFlags::VOPC |
        SIInstrFlags::VOP3P;
    const uint64_t flags = di.tsFlags;
    const unsigned opc = di.inst.getOpcode();
    HandlerResult hr;
    if (AMDGPU::isVOPD(opc))
      hr = handleVOPD(ctx, di, op, result);
    else if (flags & SIInstrFlags::IsMAI)
      hr = handleMFMA(ctx, di, op, result);
    else if (flags & kVALU)
      hr = handleVALU(ctx, di, op, result);
    else if (flags & SIInstrFlags::SOPP)
      hr = handleSOPP(ctx, di, op, result);
    else if (flags & SIInstrFlags::SOPC)
      hr = handleSOPC(ctx, di, op, result);
    else if (flags & SIInstrFlags::SOP1)
      hr = handleSOP1(ctx, di, op, result);
    else if (flags & SIInstrFlags::SOP2)
      hr = handleSOP2(ctx, di, op, result);
    else if (flags & SIInstrFlags::SOPK)
      hr = handleSOPK(ctx, di, op, result);
    else if (flags & SIInstrFlags::SMRD)
      hr = handleSMEM(ctx, di, op, result);
    else if (flags & SIInstrFlags::FLAT)
      hr = handleFLAT(ctx, di, op, result);
    else if (flags & SIInstrFlags::MUBUF)
      hr = handleMUBUF(ctx, di, op, result);
    else if (flags & SIInstrFlags::DS)
      hr = handleDS(ctx, di, op, result);

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
    result.failFormat = formatName(di.tsFlags, di.inst.getOpcode());
    errs() << "transpiler: Unsupported instruction: " << di.mnemonic
           << " (raw: " << di.rawMnemonic << ")"
           << " [format=" << result.failFormat << "]"
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
