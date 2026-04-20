#include "raiser.hpp"
#include "amdgpu_formats.hpp"
#include "code_object_utils.hpp"
#include "decode.hpp"
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
#include "sem_op_attrs.hpp"
#include "setpc_analysis.hpp"
#include "user_sgpr_layout.hpp"
#include "wave_projection.hpp"
#include "wave_size_obstruction.hpp"
#include "handlers.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
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
#include "llvm/Support/Debug.h"

#include <map>

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace transpiler {

// parseReg, readOp32/64/ExecWidth, and OpResolver are in raise_context.hpp/cpp
// instructionWritesEXEC and the cross-wave gate live in wave_projection.hpp/cpp
// RaiseFailure + reasonString are in raise_failure.hpp/cpp

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

  // LLVMContext + common IR types are created here (earlier than they used
  // to be) so the WaveProjection has access to i32/i64 before the cross-
  // wave gate runs. The module is still created lazily in Phase 2 so
  // early-return paths (pre-translation aborts) don't leave behind a
  // half-built module.
  result.ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *result.ctx;
  auto *i32Ty = Type::getInt32Ty(C);
  auto *i64Ty = Type::getInt64Ty(C);

  ModuloReplicationProjection projection(isa, targetIsa, i32Ty, i64Ty);

  // Build opcode → SemOp map from MCInstrInfo
  OpcodeMap opcMap;
  opcMap.build(*mc.instrInfo);

  // Fail loudly if any MFMA-format SemOp is missing a handler row. Cheap
  // startup walk that catches table drift before any kernel is lifted.
  verifyMFMACoverage(*mc.instrInfo, opcMap);

  // Startup invariant: every MC opcode that implicitly defines EXEC must
  // map to a SemOp that has `routesExecThroughStoreExec` set. Explicit-
  // operand EXEC writers (where EXEC is an operand value rather than a
  // TableGen def) stay the per-kernel Phase 1.5 gate's responsibility
  // since they depend on runtime operand values.
  verifyExecAttrCoverage(*mc.instrInfo, opcMap);

  // ==== Phase 1: Disassemble + identify block boundaries ====
  //
  // The decode loop (and its two LLVM-drift guards) lives in decode.cpp so
  // this function stays focused on IR emission. decodeKernel returns a
  // linearised instruction stream + the set of CFG block-start offsets.
  DecodeResult decoded =
      decodeKernel(mc, opcMap,
                   ArrayRef<uint8_t>(textBytes.data(), textBytes.size()),
                   kernelOffset);
  auto &insts = decoded.insts;
  auto &blockStarts = decoded.blockStarts;

  // ==== Phase 1.1: s_set_pc_i64 analysis ====
  //
  // Classify every s_set_pc_i64 site (Pattern A direct branch /
  // Pattern B subroutine return / Unresolvable) and discover the
  // extra basic-block leaders the indirect control-flow implies
  // (Pattern A targets + Pattern B return targets + the offset
  // immediately following each set-PC, which is otherwise unreachable
  // by linear fall-through). Merging the extra leaders into
  // `blockStarts` here is mandatory: Phase 3 only creates LLVM
  // BasicBlocks for offsets in this set, and the handler / call-site
  // rewrite both look up those BBs via `ctx.lookupBB`.
  // See setpc_analysis.hpp + semop.hpp's `S_SET_PC_I64` doc for the
  // analysis contract.
  SetPcAnalysis setpcAnalysis = analyseSetPC(insts, blockStarts, mc);
  for (uint64_t addr : setpcAnalysis.extraBlockStarts)
    blockStarts.insert(addr);

  result.totalCount = (int)insts.size();

  {
    raw_string_ostream disOS(result.disasmText);
    for (const auto &di : insts) {
      disOS << format_hex_no_prefix(di.offset, 8) << ":  " << di.fullText
            << "\n";
    }
  }

  // ==== Phase 1.4: Cross-wave legacy diagnostic (LLVM_DEBUG) ====
  //
  // Kept as a fallback diagnostic under `-debug-only=wave-projection`;
  // the structured classifier in Phase 1.4.5 below is the primary
  // decision surface. See wave_projection.cpp for the text of the
  // legacy diagnostic.
  emitCrossWaveWarning(projection, mc, insts, sourceISA,
                       compilationTargetISA);

  // ==== Phase 1.4.5: Wave-size obstruction classifier
  // (hotswap/docs/wave-size-translation.md §7) ====
  //
  // The classifier walks the decoded instruction stream and tags every
  // site that violates the wave-size-obliviousness theorem (see
  // wave-size-translation.md §6 for the precise definition). The
  // decider then applies the 3-outcome procedure:
  //   (a) no obstructions, or every obstruction is covered by an
  //       implemented rewrite → emit modulo-replication.
  //   (b) at least one obstruction has a rewrite structurally
  //       recognised but not yet implemented (the "Pending rewrite"
  //       table in wave-size-translation.md §7) → refuse with a
  //       `CrossWaveShuffleRewritePending` diagnostic naming the P-item.
  //   (c) at least one obstruction has no rewrite in the decision
  //       procedure's unrewritable table → refuse with the kind-
  //       specific CrossWave* diagnostic (`CrossWaveLaneIdLeak`,
  //       `CrossWaveUnrewritableShuffle`, `CrossWaveReplicaRace`,
  //       `CrossWaveLanePredicatedExec`).
  //
  // Refusal diagnostics are written to `errs()` (user-visible) AND the
  // full per-site trace is routed through LLVM_DEBUG so operators can
  // inspect the oblivious/pass path under `-debug-only=wave-projection`
  // without recompiling.
  {
    ObstructionReport report =
        buildObstructionReport(insts, mc, isa, targetIsa);
    std::string trace = renderObstructionTrace(
        report, kernelName, sourceISA,
        compilationTargetISA.empty() ? sourceISA : compilationTargetISA,
        isa.waveSize, targetIsa.waveSize);
    LLVM_DEBUG(dbgs() << trace);
    if (report.hasUnrewritable() || report.hasPendingRewrite()) {
      RaiseFailure f = selectFailureFromReport(report);
      // The factory names the class in `format`; surface the full
      // trace in `detail` so raise_cli / batch_raise_test can carry
      // the per-site context forward without re-invoking the
      // classifier.
      if (!f.detail.empty())
        f.detail += "\n";
      f.detail += trace;
      // `format_hex(value, width)` prepends "0x" itself; do NOT add a
      // literal "0x" here or the output will read "0x0x...". Use
      // `format_hex_no_prefix` if a manual prefix is desired (the
      // trace-renderer below uses that variant).
      errs() << "transpiler: pre-translation abort: " << f.format
             << " on '" << f.mnemonic << "' at offset "
             << format_hex(f.offset, 1) << " \u2014 "
             << (report.firstUnrewritable()
                     ? "no rewrite in wave-size-translation.md "
                       "\u00a77's unrewritable table"
                     : "rewrite pending (wave-size-translation.md "
                       "\u00a77's pending-rewrite table)")
             << "\n"
             << trace;
      result.failure = std::move(f);
      return result;
    }
  }

  // ==== Phase 1.5: SPE A-level gate (EXEC-writer attribute check) ====
  //
  // SPE (SIMT Predicated Execution) is correct only when every runtime
  // change to EXEC either (a) propagates through the EXEC alloca via a
  // handler we have audited, or (b) follows the standard dataflow form
  // `exec = f(old_exec, sgprs, ...)` where `f` is a bitwise / shift /
  // move / compare-based scalar op — the IR's live EXEC value then
  // matches the hardware EXEC that the backend re-materialises when it
  // lowers our predicated-store diamonds back to v_cmpx / s_and_saveexec
  // pairs. Anything outside this set risks silently generating IR that
  // looks well-typed but diverges from hardware semantics.
  //
  // The allow-list lives as per-SemOp attributes in `sem_op_attrs.{hpp,
  // cpp}`; `verifyExecAttrCoverage` above already enforces it for
  // implicit-def EXEC writers at startup. This per-kernel scan covers
  // the remaining case: explicit-operand EXEC writers (e.g.
  // `s_mov_b32 exec_lo, s2`) where "writes EXEC" depends on the
  // runtime operand value rather than the MCInstrDesc alone.
  for (const DecodedInst &di : insts) {
    if (!instructionWritesEXEC(di, mc))
      continue;
    if (getSemOpAttrs(di.semOp).routesExecThroughStoreExec)
      continue;
    result.failure = RaiseFailure::speUnsafeExecWriter(di);
    errs() << "transpiler: pre-translation abort: '" << di.rawMnemonic
           << "' writes EXEC but its SemOp (" << semOpName(di.semOp)
           << ") is not marked routesExecThroughStoreExec. Auditing "
              "the handler path against SPE (lane-active predication "
              "assumption) is required before declaring the SemOp in "
              "the handler's get*Attrs() registration.\n";
    return result;
  }

  // ==== Phase 2: Build LLVM IR module + function ====
  // LLVMContext + i32/i64 were created earlier for the WaveProjection.
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
    result.failure = RaiseFailure::targetMachineCreationFailed();
    return result;
  }
  M.setDataLayout(tm->createDataLayout());

  auto *voidTy = Type::getVoidTy(C);
  auto *i1Ty = Type::getInt1Ty(C);
  auto *i8Ty = Type::getInt8Ty(C);
  auto *f32Ty = Type::getFloatTy(C);
  auto *ptrGlobalTy = PointerType::get(C, 1);

  // Build function signature dynamically from kernel metadata.
  //
  // The IR-level argument list must reproduce the source binary's
  // kernarg byte layout exactly: every byte the source reads from the
  // kernarg buffer at offset O must be reachable through some IR
  // argument anchored at that offset. The AMDGPU backend places kernel
  // arguments in the kernarg buffer by their natural alignment + size,
  // so as long as we emit the right type at the right cumulative
  // offset, the buffer layout matches the runtime's packing.
  //
  // Three slot shapes are emitted:
  //   * `global_buffer` (size==8) → ptr addrspace(1).
  //   * non-pointer `by_value` size==4 → i32.
  //   * non-pointer `by_value` size==8 → i64.
  //   * non-pointer `by_value` size > 8 (and divisible by 4, i.e. an
  //     aggregate kernarg like Triton's tensor-descriptor struct) is
  //     DECOMPOSED into one i32 slot per dword. Without this split the
  //     IR would carry a single i32 placeholder for the whole struct
  //     and codegen would only allocate 4 bytes for it — silently
  //     shifting every downstream arg's runtime byte offset and turning
  //     all kernarg loads past the struct into reads of garbage. The
  //     per-dword split also makes SMEM kernarg loads against the
  //     interior of the struct addressable through `extractKernargDword`
  //     in handle_smem.cpp without needing any aggregate-aware extract
  //     logic. Sizes that are not 4, 8, or a multiple of 4 are refused
  //     loudly: they would require partial-dword extraction that no
  //     current handler supports, and the no-fallback rule applies.
  //
  // Test back-reference: lit_tests/s_load_b96_kernarg/ pins the i32
  // slot signature this branch produces for a 16-byte by_value
  // aggregate; any change to the dword-decomposition logic must keep
  // that fixture's `(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, ptr
  // addrspace(1) %arg4)` signature green.
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
    if (isPtr) {
      if (arg.size != 8)
        report_fatal_error(
            Twine("transpiler: kernel '") + kernelName + "' arg '" +
            arg.name + "' is global_buffer but size=" +
            Twine(arg.size) + " (expected 8)");
      paramTypes.push_back(ptrGlobalTy);
      kernargs.params.push_back({arg.offset, 8, paramIdx, true});
      paramIdx++;
      continue;
    }
    if (arg.size == 4) {
      paramTypes.push_back(i32Ty);
      kernargs.params.push_back({arg.offset, 4, paramIdx, false});
      paramIdx++;
      continue;
    }
    if (arg.size == 8) {
      paramTypes.push_back(i64Ty);
      kernargs.params.push_back({arg.offset, 8, paramIdx, false});
      paramIdx++;
      continue;
    }
    if (arg.size > 0 && arg.size % 4 == 0) {
      int nDwords = arg.size / 4;
      for (int d = 0; d < nDwords; ++d) {
        paramTypes.push_back(i32Ty);
        kernargs.params.push_back(
            {arg.offset + d * 4, 4, paramIdx, false});
        paramIdx++;
      }
      continue;
    }
    report_fatal_error(
        Twine("transpiler: kernel '") + kernelName + "' arg '" +
        arg.name + "' has unsupported by_value size=" + Twine(arg.size) +
        " (expected 4, 8, or a positive multiple of 4); partial-dword "
        "kernarg extraction is not modelled and silent rounding is "
        "rejected by the no-fallback rule.");
  }
  kernargs.implicitArgsBase = meta.implicitArgsBase();
  kernargs.kernargSegmentSize = meta.kernargSegmentSize;

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
  regs.init(B, i32Ty, i1Ty, isa, *mc.regInfo, projection);

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
  // Build the user-SGPR ABI for the source ISA. handle_smem and any other
  // handler that needs to identify a specific source-ABI SGPR (e.g. the
  // kernarg pointer) reads this through `ctx.userSgprLayout`. The layout
  // is owned by this stack frame; the `RaiseContext` borrows a pointer
  // valid for the duration of `raiseToIR`. `fromKernelMeta` aborts loudly
  // if the kernel descriptor is missing — there is no fallback layout.
  UserSgprLayout userSgprLayout = UserSgprLayout::fromKernelMeta(meta);
  RaiseContext ctx{C, M, B, regs, projection, mc, isa, targetIsa, kernargs,
                   &userSgprLayout, F,
                   i1Ty, i8Ty, i32Ty, i64Ty, f32Ty, f16Ty,
                   ptrGlobalTy, offsetToBB};
  ctx.setpcAnalysis = &setpcAnalysis;

  // Wire the reg-file's EXEC-write invalidation hook to ctx's lane_active
  // memo. This catches every EXEC mutation — ctx.storeExec, the various
  // ctx.writeReg*(EXEC, …) wrappers, *and* the handful of handlers that
  // still call ctx.regs.storeExec / ctx.regs.writeRegExecWidth directly
  // (SAVEEXEC family in handle_sop1, V_CMPX in handle_valu). Without
  // this hook those direct paths would leave the memo pointing at a
  // pre-write `lane_active`, silently mispredicating subsequent
  // emitUnderExec diamonds.
  regs.onExecWritten = [&ctx] { ctx.resetLaneActiveCache(); };

  int raisedCount = 0;

  for (size_t instIdx = 0; instIdx < insts.size(); ++instIdx) {
    const DecodedInst &di = insts[instIdx];

    // Source-BB boundary handling uses `B.GetInsertBlock()` rather than a
    // tracked `currentBB` so that intra-handler CFG splits (emitUnderExec
    // diamonds under SPE) propagate correctly: fall-through must leave
    // from whatever block the builder is currently at — which is the
    // `spe_skip` tail when the last emission was wrapped — not from the
    // block that started the source instruction.
    auto bbIt = offsetToBB.find(di.offset);
    if (bbIt != offsetToBB.end() && bbIt->second != B.GetInsertBlock()) {
      BasicBlock *insertBB = B.GetInsertBlock();
      if (insertBB->empty() || !insertBB->getTerminator())
        B.CreateBr(bbIt->second);
      B.SetInsertPoint(bbIt->second);
      // LLVM's AMDGPULowerVGPREncoding pass resets VGPR MSB mode at every
      // basic-block boundary (both before terminators and at BB fall-through
      // exits).  Mirror that behaviour so we do not inherit stale MSB state
      // from a previous linear instruction that does not control-flow into
      // this BB.
      ctx.vgprMSBs = 0;
    }

    ctx.computeVGPRAdjust(di);
    // Invalidate the SPE lane_active memoisation at every instruction
    // boundary. Any instruction is a potential EXEC writer (either through
    // our modeled SemOp allow-list, or through a path we haven't yet
    // covered), and emitLaneActiveBit is load-bearing for per-lane
    // predication correctness: reusing a stale lane_active from before an
    // EXEC write would silently mispredicate side effects. See
    // RaiseContext::resetLaneActiveCache in raise_context.hpp for the full
    // invalidation contract.
    ctx.resetLaneActiveCache();
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
      hr = handleVOPD(ctx, di, op);
    else if (flags & SIInstrFlags::IsMAI)
      hr = handleMFMA(ctx, di, op);
    else if (flags & kVALU)
      hr = handleVALU(ctx, di, op);
    else if (flags & SIInstrFlags::SOPP)
      hr = handleSOPP(ctx, di, op);
    else if (flags & SIInstrFlags::SOPC)
      hr = handleSOPC(ctx, di, op);
    else if (flags & SIInstrFlags::SOP1)
      hr = handleSOP1(ctx, di, op);
    else if (flags & SIInstrFlags::SOP2)
      hr = handleSOP2(ctx, di, op);
    else if (flags & SIInstrFlags::SOPK)
      hr = handleSOPK(ctx, di, op);
    else if (flags & SIInstrFlags::SMRD)
      hr = handleSMEM(ctx, di, op);
    else if (flags & SIInstrFlags::FLAT)
      hr = handleFLAT(ctx, di, op);
    else if (flags & SIInstrFlags::MUBUF)
      hr = handleMUBUF(ctx, di, op);
    else if (flags & SIInstrFlags::DS)
      hr = handleDS(ctx, di, op);
    // VIMAGE TENSOR pseudo-instructions (`tensor_load_to_lds_d{2,4}`,
    // `tensor_store_from_lds_d{2,4}`, MIMGInstructions.td:2049-2113).
    // The pseudo extends `InstSI` directly and only sets `let VALU =
    // 1` and `let TENSOR_CNT = 1` (NOT `let VIMAGE = 1`), so the
    // `SIInstrFlags::VIMAGE` bit stays 0 on these. Dispatch on
    // `TENSOR_CNT` instead — the only other carrier of that bit is
    // `s_wait_tensorcnt` (SOPP), which is already claimed by the
    // SOPP arm above and never reaches this fallthrough. Routed
    // late because TENSOR ops are exclusive to the gfx1250
    // (`isGFX125xOnly`) generation and the handler's only contract
    // today is a cross-target loud refusal; the same gating applies
    // when the same-target intrinsic-emit path lands.
    else if (flags & SIInstrFlags::TENSOR_CNT)
      hr = handleVIMAGE(ctx, di, op);

    // Operand-read paths (`readOp32` / `readOp64`) cannot bail mid-
    // handler, so they record any unsupported-register failures into
    // `ctx.pendingFailure`. Promote that to the structured failure
    // *before* the `hr.handled` check — a handler that "succeeded"
    // by returning undef from a read is still an unraised kernel.
    if (ctx.pendingFailure.hasFailed()) {
      result.failure = std::move(ctx.pendingFailure);
      ctx.pendingFailure = RaiseFailure{};
      return result;
    }

    if (hr.handled) {
      if (di.defsSCC && !hr.sccHandled && hr.sccResult) {
        Value *zero = Constant::getNullValue(hr.sccResult->getType());
        ctx.regs.storeSCC(ctx.B, ctx.B.CreateICmpNE(hr.sccResult, zero));
      }
      if (di.defsEXEC)
        result.hasDivergentExec = true;
      // Pattern B call-site post-processing: if this s_add_co_ci_u32
      // is the high-half terminator of a getpc+add chain that feeds
      // a Pattern B `s_set_pc_i64` enumerated-dispatch cascade (i.e.
      // some downstream s_set_pc_i64 reads the same ret-pair this
      // chain populated), overwrite the ret-pair SGPR with the plain
      // i64 marker `resolvedReturnAddr` — i.e. the source-MC byte
      // offset of the BB this chain meant to return to. The
      // downstream cascade compares against the same offsets via
      // `icmp eq i64 %marker, <offset_k>` for each enumerated
      // target; when this predecessor's marker matches one of the
      // enumerated offsets, mem2reg + SCCP + InstCombine fold the
      // compare to `i1 true` across the phi join and SimplifyCFG
      // collapses the cmp+br cascade into a direct
      // `br label %BB_<offset>`. The SOP2 handler has already done
      // its (binary-PC-producing) arithmetic above; this commit
      // happens *after* and clobbers that result on purpose — that
      // value was an opaque runtime PC we never want to see
      // downstream.
      //
      // An earlier revision of this hook wrote
      // `ptrtoint(blockaddress(@kernel, %BB_returnAddr)) to i64`
      // here so the cascade could compare against a `blockaddress`
      // constant. That form survived mem2reg + SCCP unfolded in
      // irreducible tensilelite-shaped CFGs (the `storeSGPR64`
      // hi/lo split prevented the cross-phi fold), leaving a
      // `BlockAddress` SDNode alive into AMDGPU ISel, which has no
      // codegen pattern for it and aborts llc with
      //   `Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
      // Using a plain integer marker keeps `BlockAddress` solely
      // as a direct-branch `label` operand (which DOES have a
      // codegen pattern), sidestepping the ISel crash entirely.
      // See setpc_analysis.hpp + semop.hpp's S_SET_PC_I64 doc +
      // `emitEnumeratedDispatch` in handle_sop1.cpp.
      if (di.semOp == SemOp::S_ADDC_U32) {
        auto it = setpcAnalysis.chainTerminators.find(di.offset);
        if (it != setpcAnalysis.chainTerminators.end()) {
          // Force the BB to exist so the downstream cascade's
          // direct branch has a destination; we don't use the
          // pointer here.
          (void)ctx.lookupBB(it->second.resolvedReturnAddr);
          Value *retMarker =
              ConstantInt::get(ctx.i64Ty, it->second.resolvedReturnAddr);
          ctx.regs.storeSGPR64(ctx.B,
                                static_cast<int>(it->second.retPairLowReg),
                                retMarker);
        }
      }
      raisedCount++;
      continue;
    }

    // The handler either recognised the instruction but refused the
    // specific shape (hr.failure.reason != None), or no handler claimed
    // it at all — promote to `UnsupportedOpcode` and bucket by format.
    if (hr.failure.hasFailed()) {
      result.failure = std::move(hr.failure);
    } else {
      result.failure = RaiseFailure::unsupportedOpcode(
          di, formatName(di.tsFlags, di.inst.getOpcode()));
      errs() << "transpiler: Unsupported instruction: " << di.mnemonic
             << " (raw: " << di.rawMnemonic << ")"
             << " [format=" << result.failure.format << "]"
             << " at offset 0x" << format_hex(di.offset, 1) << "\n";
    }
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
    result.failure = RaiseFailure::irVerificationFailed(verifyErr);
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
