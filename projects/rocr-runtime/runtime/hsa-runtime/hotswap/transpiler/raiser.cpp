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
#include "rewrite_cross_lane_divergent.hpp"
#include "c5_predicate_chain_classifier.hpp"

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

#include "rewrite_permlane16_xor3_partner.hpp"
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
                      const std::string &compilationTargetISA,
                      bool enableWritelaneRewrite,
                      bool enableWaveNative) {
  RaiseResult result;

  // NOTE. The `HSA_SALMON_WAVE_NATIVE=1` process-environment override
  // that lived here through the empirical graduation sweep (pre-
  // 2026-04-21) has been removed now that `enableWaveNative`
  // defaults to `true`. The override served one purpose — flipping
  // every call-site's projection without editing each caller —
  // which is no longer needed. Keeping it around would subtly
  // break the opt-OUT path: `--disable-wave-native` on
  // `raise_cli` (and `enableWaveNative=false` on programmatic
  // callers) are how lit fixtures and operators pin MODREP for
  // projection-specific debugging, and a silent env-var that
  // unconditionally flips to WaveNative would defeat that. If
  // future evidence needs a global toggle, add a proper
  // `PipelineConfig` field rather than re-introducing the env var.

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

  // Projection choice.
  //
  // `ModuloReplicationProjection` is the long-standing default: it fans
  // each target lane onto `lane_id mod W_src` of the source EXEC mask
  // and truncates cross-wave ballots to source width. Correct under
  // the wave-size-obliviousness theorem (hotswap/docs/wave-size-
  // translation.md §6); insufficient for kernels whose WMMA → MFMA
  // redistribute / collect pipeline needs hardware EXEC = -1 on the
  // upper half of the Wave64 target (lanes 32..63 would otherwise
  // never update their MFMA destination VGPRs — see the file-header
  // comment in `wmma_lowering.cpp`).
  //
  // `WaveNativeProjection` is the opt-in alternative for wave32
  // source → wave64 target. Its `emitInitialExec` calls
  // `@llvm.amdgcn.init_whole_wave` at kernel entry to force hardware
  // EXEC = -1 for the whole kernel body while saving the original
  // per-lane active mask into the (widened) EXEC alloca; every VGPR
  // write / memory store / LDS op already routes through
  // `emitUnderExec`, which rematerialises the per-lane predicate at
  // each side-effect site. The direction gate inside the
  // `WaveNativeProjection` constructor enforces that this projection
  // is only instantiated when `isa.isWave32() && !targetIsa.isWave32()`
  // — other directions fatal-error loudly to prevent a decider bug
  // from silently picking an unsupported shape.
  //
  // Phantom-lane fallback to MODREP.  WaveNative's `init_whole_wave`
  // sets hardware EXEC = -1 and relies on SPE `emitUnderExec`
  // diamonds (gated by `saved_exec`) to keep inactive source lanes
  // from committing side effects.  That model is correct when every
  // target-wavefront lane has a source-kernel workitem — i.e. when
  // the HSACO's `max_flat_workgroup_size` is at least
  // `targetWaveSize` so every launch fills the target wave.  When
  // `max_flat_workgroup_size < targetWaveSize` (the phantom-lane
  // regime, e.g. Triton's `num_warps=1` kernels whose source WG is
  // 32 on wave32 compiled for a wave64 target), the "extra" target
  // lanes have no source workitem: their `workitem.id.x()` is their
  // hardware lane index (e.g. 32..63 for a 32-thread block on
  // wave64), their VGPRs hold undef / dispatcher state, and their
  // cross-lane ops (`ds_bpermute`, `ds_swizzle`, `permlane*`) read
  // from / contribute to actively-masked source lanes with
  // undef-derived values — producing addresses that fault on
  // subsequent SPE-gated loads (the active lane's pointer
  // arithmetic picks up undef data through a cross-lane op, then
  // the gated load fires with that poisoned address).  Empirically
  // surfaced by `compare_correctness`'s `matmul_fp16` /
  // `matmul_fp16_16x16` Triton recipes (HIP error 700 on every
  // shape under WaveNative; bumping `num_warps` to 2 fills the
  // target wavefront and eliminates the fault, confirming the
  // phantom-lane attribution).
  //
  // `ModuloReplicationProjection` leaves hardware EXEC at the
  // dispatcher's boot state (the source-wave-sized active mask,
  // with the target wave's upper lanes inactive) and uses
  // `lane_id mod W_src` to project the target mask onto the source
  // EXEC alloca.  Under MODREP, phantom lanes are hardware-inactive
  // for the entire kernel body — every ISA instruction (VALU,
  // cross-lane, memory, control flow) is HW-EXEC-masked — so
  // undef-VGPR contamination can't escape into active lanes.  The
  // trade-off is that MODREP cannot express WMMA → MFMA layout
  // transposes that need all 64 target lanes active (see
  // `wmma_lowering.cpp`); those kernels will refuse at lift time
  // rather than silently running wrong.  That's the principled
  // outcome for the phantom-lane regime.
  const bool phantomLaneRegime =
      meta.maxFlatWorkgroupSize > 0 &&
      static_cast<unsigned>(meta.maxFlatWorkgroupSize) < targetIsa.waveSize;
  const bool useWaveNative = enableWaveNative && isa.isWave32() &&
                              !targetIsa.isWave32() && !phantomLaneRegime;
  std::unique_ptr<WaveProjection> projectionPtr;
  if (useWaveNative)
    projectionPtr = std::make_unique<WaveNativeProjection>(isa, targetIsa,
                                                             i32Ty, i64Ty);
  else
    projectionPtr = std::make_unique<ModuloReplicationProjection>(
        isa, targetIsa, i32Ty, i64Ty);
  WaveProjection &projection = *projectionPtr;

  if (enableWaveNative && phantomLaneRegime && isa.isWave32() &&
      !targetIsa.isWave32()) {
    // Log the fallback so operators can trace which kernels moved to
    // MODREP and why.  A regression that silently flips WaveNative's
    // selection on a phantom-lane kernel would then (re-)produce the
    // HIP-700 miscompile this fallback guards against.
    errs() << "transpiler: kernel '" << kernelName
           << "' is in phantom-lane regime (max_flat_workgroup_size="
           << meta.maxFlatWorkgroupSize << " < target wavefront width="
           << targetIsa.waveSize
           << "); falling back to ModuloReplicationProjection even "
              "though enableWaveNative=true, so phantom target lanes "
              "stay hardware-inactive and their undef-VGPR state "
              "cannot contaminate active-lane pointer arithmetic via "
              "cross-lane ops. See the block comment above in "
              "`raiser.cpp` for the full rationale.\n";
  }

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

  result.totalCount = static_cast<int>(insts.size());

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
  // Number of `WaveIdLiftScalarized` sites the classifier matched.
  // Needed after Phase 6.5 for the rewrite-pass safety net (see
  // below): when this is > 0, the rewrite pass is *expected* to have
  // rewritten at least one divergent writelane/readlane site; if it
  // rewrote zero, the oracle disagrees with the syntactic
  // classifier and we refuse post-raise rather than emit silently
  // unchanged IR that scalarises the divergent wave_id lift.
  unsigned classifierWaveIdLiftScalarizedSites = 0;
  {
    ObstructionReport report =
        buildObstructionReport(insts, mc, isa, targetIsa,
                               enableWritelaneRewrite);
    for (const auto &s : report.sites)
      if (s.kind == ObstructionKind::WaveIdLiftScalarized)
        ++classifierWaveIdLiftScalarizedSites;
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
  //   ttmp7[15:0]  = workgroup_id_y  (low 16 bits)
  //   ttmp7[31:16] = workgroup_id_z  (high 16 bits; 0 when grid has no Z)
  //   ttmp8[29:25] = wave_id within workgroup (subgroup ID)
  //   ttmp9        = workgroup_id_x  (accelerated launch)
  // The packed-Y-and-Z layout in ttmp7 is from the AMDGPU backend's
  // `loadInputValue` path (see LLVM's `AMDGPULegalizerInfo.cpp` —
  // `WorkGroupIDY = ArgDescriptor::createRegister(TTMP7, 0xFFFFu)`,
  // `WorkGroupIDZ = ArgDescriptor::createRegister(TTMP7, 0xFFFF0000u)`).
  // Triton-generated gfx1250 kernels read the Y component via
  // `s_and_b32 sN, ttmp7, 0xffff` (e.g. matmul_fp16_16x16's `pid_n =
  // tl.program_id(1)` lowering), so a kernel raised without ttmp7
  // initialised always sees `workgroup_id_y == 0` — only the
  // leftmost column of workgroups in a 2D-grid kernel writes its
  // tile, and the right-side tiles stay at whatever the destination
  // memory held at dispatch (verified empirically: matmul_fp16_16x16
  // M=32 with an all-1s input shows cols 0..15 = correct 32.0,
  // cols 16..31 = poison-fill from the host's pre-launch memset).
  // gfx11 (RDNA3) passes these via SGPRs set up by the CP instead.
  if (AMDGPU::isGFX12Plus(*mc.subtargetInfo)) {
    B.CreateStore(B.CreateCall(fnWorkgroupIdX, {}, "ttmp9_wg_id"), regs.ttmp[9]);

    // ttmp7 = (workgroup_id_z << 16) | (workgroup_id_y & 0xFFFF).
    // Mask Y to 16 bits before shifting Z so a stray-high-bit Y
    // doesn't bleed into the Z field; the AMDGPU backend's mask is
    // `~0u` on no-Z kernels (the upper bits are simply ignored by
    // the consumer's `s_and ttmp7, 0xffff` pattern), so masking
    // unconditionally is safe and matches the principled all-cases
    // shape.
    Value *wgIdY = B.CreateCall(fnWorkgroupIdY, {}, "ttmp7_wg_id_y");
    Function *fnWorkgroupIdZ =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_z);
    Value *wgIdZ = B.CreateCall(fnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
    Value *wgIdYLo = B.CreateAnd(wgIdY, B.getInt32(0xFFFF), "wg_id_y_lo16");
    Value *wgIdZHi = B.CreateShl(wgIdZ, B.getInt32(16), "wg_id_z_hi16");
    Value *ttmp7Val = B.CreateOr(wgIdYLo, wgIdZHi, "ttmp7_val");
    B.CreateStore(ttmp7Val, regs.ttmp[7]);

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

  // Wire the reg-file's per-SGPR write invalidation hook to ctx's
  // V_CMP -> V_CNDMASK per-lane-i1 shadow map
  // (`lastSgprWaveMaskI1`). Fires on every `storeSGPR32 / storeSGPR64`
  // and therefore on every path that mutates an SGPR — including
  // handlers that bypass `writeReg32 / writeReg64` to call the
  // low-level stores directly (handle_smem's multi-dword load
  // splitting, handle_valu's SCC-flag SGPR writes, etc.). The V_CMP
  // wave-mask write path also fires this hook; the V_CMP handler
  // immediately re-populates the shadow with the per-lane `i1`
  // afterwards via `ctx.recordSgprWaveMaskI1`. See hotswap/docs/sgpr-
  // wave-mask-translation.md section 3.1 for the full contract.
  regs.onSgprWritten = [&ctx](int idx) { ctx.invalidateSgprWaveMaskI1(idx); };

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
      if (!insertBB->hasTerminator())
        B.CreateBr(bbIt->second);
      B.SetInsertPoint(bbIt->second);
      // LLVM's AMDGPULowerVGPREncoding pass resets VGPR MSB mode at every
      // basic-block boundary (both before terminators and at BB fall-through
      // exits).  Mirror that behaviour so we do not inherit stale MSB state
      // from a previous linear instruction that does not control-flow into
      // this BB.
      ctx.vgprMSBs = 0;
      // Drop the V_CMP -> V_CNDMASK per-lane-i1 shadow at every BB
      // transition. The cached `i1` SSA values dominate only the BB
      // they were emitted in; carrying them into a successor would
      // read an SSA value out of its dominance scope. A future
      // reaching-definitions pass on the raised IR could upgrade this
      // to a proper per-BB merge (see sgpr-wave-mask-translation.md
      // section 7 evolution path).
      ctx.clearSgprWaveMaskShadow();
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
    if (!BB.hasTerminator()) {
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

  // ==== Phase 6.04: Triton cross-16 bitonic-merge xor3-partner rewrite ====
  //
  // Detects the gfx1250-only `permlane16_swap + v_xor3_b32` idiom
  // Triton emits at the cross-16 stage of `tl.sort` / `tl.topk`'s
  // bitonic merge.  See `rewrite_permlane16_xor3_partner.hpp` for
  // the full pattern, the (a)/(b) hypothesis split that motivates
  // the rewrite, and the GTest pinning evidence.
  //
  // Runs AFTER `PromoteMemToReg` (Phase 6) so the
  // alloca-backed VGPR round-trips are folded out and the bpermute
  // results flow directly into the xor inputs.  Runs BEFORE Phase
  // 6.5 (writelane/readlane rewrite) and Phase 6.6 (predicate-chain
  // classifier) so the rewrite's substitutions don't perturb the
  // SSA shape those downstream passes inspect — both downstream
  // passes only walk specific intrinsic call sites
  // (`amdgcn.{writelane,readlane,workitem.id.x}`) that this
  // rewrite never touches.
  {
    Permlane16Xor3PartnerRewriteReport report =
        rewritePermLane16Xor3Partner(*F);
    if (report.matchedSites > 0) {
      LLVM_DEBUG({
        dbgs() << "permlane16-xor3-partner: rewrote "
               << report.matchedSites << " site(s) in '" << kernelName
               << "'\n";
      });
    }
  }

  // ==== Phase 6.5: Cross-widen writelane/readlane rewrite ====
  //
  // Opt-in symmetric rewrite of `v_writelane_b32` / `v_readlane_b32`
  // sites under cross-widening. Disabled by default; the caller
  // (raise_cli's `--enable-writelane-rewrite`, PipelineConfig's
  // `enableWritelaneRewrite`) must ask for it explicitly. See
  // `rewrite_cross_lane_divergent.{hpp,cpp}` and
  // wave-size-translation.md §5.6.3 for the principled derivation,
  // and hotswap/docs/learnings.md for the asymmetric-rewrite bug
  // that motivated the symmetry-plus-use-chain design.
  //
  // Runs AFTER `PromoteMemToReg` by construction: the rewrite pass's
  // forward use-chain classifier needs post-mem2reg SSA so a
  // scratch-addrspace round-trip (load / store through an alloca) does
  // not obscure the fact that a writelane / readlane result eventually
  // reaches an SGPR-constrained consumer. No behavioural change on
  // same-wave / narrowing directions — the rewrite pass short-
  // circuits internally on `targetWaveSize <= sourceWaveSize`.
  //
  // Refusal path. If any writelane / readlane site's forward use chain
  // reaches an SGPR-forced consumer that the classifier cannot prove
  // safe (`s_buffer_load` rsrc, `s_sendmsg` message, `readfirstlane`,
  // addrspace(4) load, inline asm with `"s"` constraint, or any
  // unaudited intrinsic / instruction), the rewrite pass performs
  // zero rewrites and populates `report.sgprForcedDetail`. The raiser
  // surfaces that detail as a `crossWaveRewriteOracleDisagreement`
  // refusal — principled per the no-silent-miscompile contract:
  // rewriting the ds_bpermute output into an SGPR-forced consumer
  // would re-introduce `v_readfirstlane_b32` at the SGPR boundary and
  // recreate the source-wave collapse the rewrite exists to avoid.
  if (enableWritelaneRewrite) {
    CrossLaneDivergentRewriteReport rewriteReport =
        rewriteCrossLaneDivergent(*F, isa.waveSize, targetIsa.waveSize);

    if (rewriteReport.refusedSgprForced()) {
      RaiseFailure f = RaiseFailure::crossWaveRewriteOracleDisagreement(
          kernelName, rewriteReport.sgprForcedDetail);
      errs() << "transpiler: post-raise abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      result.failure = std::move(f);
      return result;
    }

    // Unsupported `dpp_ctrl` on an i32 update.dpp site — the rewrite
    // family covers quad_perm / row_shl / row_shr today (all stay
    // within a single 16-lane row).  Any ctrl outside that set is
    // either wave-size-dependent (wave_* shifts / rotations) or
    // hasn't been audited yet (row_mirror / row_half_mirror /
    // row_share / row_xmask — expressible but no corpus demand
    // yet).  Refusing loudly surfaces the demand so the next
    // extension has a concrete test pointer.  See
    // `buildDppLaneMap` in rewrite_cross_lane_divergent.cpp for
    // the per-ctrl widening protocol.
    if (rewriteReport.refusedUnsupportedDpp()) {
      RaiseFailure f = RaiseFailure::crossWaveRewriteOracleDisagreement(
          kernelName, rewriteReport.unsupportedDppDetail);
      errs() << "transpiler: post-raise abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      result.failure = std::move(f);
      return result;
    }

    // Second-order invariant: the syntactic Phase 1.4.5 classifier
    // matched `WaveIdLiftScalarized` iff the decoded instruction
    // stream contains at least one `v_writelane_b32` /
    // `v_readlane_b32`. Under the symmetry rule every such intrinsic
    // is rewritten (or the whole function refuses above), so a non-
    // zero classifier count MUST coincide with a non-zero count of
    // writelane + readlane rewrites specifically. Checking that
    // specific sum (not the grand total including `dppRewritten`)
    // matters: a kernel that emits DPP sites alongside missing
    // writelane / readlane would otherwise silently satisfy the
    // invariant via the DPP count, masking the handler-emission
    // regression this gate exists to catch.
    if (classifierWaveIdLiftScalarizedSites > 0 &&
        (rewriteReport.writelaneRewritten +
         rewriteReport.readlaneRewritten) == 0) {
      std::string msg;
      raw_string_ostream os(msg);
      os << "classifier matched WaveIdLiftScalarized on "
         << classifierWaveIdLiftScalarizedSites
         << " site(s) but rewriteCrossLaneDivergent rewrote 0 \u2014 the "
            "raised IR is missing the writelane/readlane intrinsic(s) "
            "that the decoded instruction stream contained. This is a "
            "handler-emission regression, not a classifier/rewrite "
            "disagreement. Refusing rather than risk a silent "
            "miscompile (see wave-size-translation.md \u00a75.6.3).";
      RaiseFailure f = RaiseFailure::crossWaveRewriteOracleDisagreement(
          kernelName, os.str());
      errs() << "transpiler: post-raise abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      result.failure = std::move(f);
      return result;
    }
  }

  // ==== Phase 6.6: Cross-widen predicate-chain classifier (C5) ====
  //
  // Post-mem2reg classifier for the Class-5 predicate-chain class
  // documented in hotswap/docs/modrep-predicate-chain.md §5 (narrow-O1).
  // Walks every `@llvm.amdgcn.workitem.id.x()` call in the function and
  // refuses the lift if any call's forward use chain reaches an `icmp`
  // against a compile-time constant K in `(0, W_s - 1]` without being
  // AND-masked by `(W_s - 1)` first — i.e. a lane-position-scoped
  // predicate (`tid < 2^s`, `tid < W_s/2`, quad-level masks) that would
  // evaluate differently on target replica-1 lanes than source wave 0
  // under modulo-replication despite sharing the source EXEC bit.
  //
  // Intentionally narrow: Phase-2 IR inspection (modrep-predicate-chain.md
  // §5 O1) established that the broader "any unmasked tid → icmp →
  // side-effect refuses" rule would also refuse baselines
  // `vecadd_f16` / `rope_fp32` / `canary_dpp_compound_add_fp32` (their
  // IR has structurally identical shapes but with a dynamic kernarg as
  // the icmp constant, not a compile-time K). The compile-time-K-only
  // rule catches `canary_bpermute_scan_fp32`'s Kogge-Stone scan-stage
  // predicates (K ∈ {1, 3, 7, 15}) while leaving the baselines green.
  //
  // Runs AFTER Phase 6 `PromoteMemToReg` so scratch-addrspace round-trips
  // are gone and the forward use-chain classifier operates on clean SSA.
  // Runs AFTER the Phase 6.5 writelane/readlane rewrite so the chain sees
  // the post-rewrite shapes (relevant when a future iteration widens the
  // classifier to audit additional users). Direction gate inside
  // `classifyPredicateChain` short-circuits when
  // `targetWaveSize <= sourceWaveSize`.
  //
  // No companion rewrite today. The design doc's §5 O2 "tid AND (W_s-1)"
  // rewrite is deferred (§6.2 documents the semantic-incorrectness of
  // the norm-family failing recipes and are a no-op for sub-case-2
  // scan-shaped recipes). If a future design iteration adds a principled
  // rewrite, pair it with a `RewriteId` alongside
  // `ObstructionKind::WorkitemIdPredicateChain`.
  {
    // Pass `enableWaveNative` + the kernel's
    // `max_flat_workgroup_size` to the classifier. The
    // `waveNative` arm suppresses refusal for the common case
    // (target-wave-sized WG, no phantom lanes) and enables
    // refusal for the phantom-lane sub-case
    // (`max_flat_workgroup_size < targetWaveSize`) — see
    // `c5_predicate_chain_classifier.hpp`'s file-header docstring
    // for the full two-arm rationale. The walk runs in both
    // arms so `observedSites` is populated for the attribution
    // breadcrumb below.
    PredicateChainClassifierReport predReport =
        classifyPredicateChain(*F, isa.waveSize, targetIsa.waveSize,
                                enableWaveNative,
                                /*maxFlatWorkgroupSize=*/
                                meta.maxFlatWorkgroupSize > 0
                                    ? static_cast<unsigned>(
                                          meta.maxFlatWorkgroupSize)
                                    : 0u);

    if (enableWaveNative && !predReport.refused &&
        !predReport.observedSites.empty()) {
      LLVM_DEBUG({
        dbgs() << "c5-predicate-chain: observed "
               << predReport.observedSites.size()
               << " C5-shape site(s) in '" << kernelName
               << "' under WaveNativeProjection (refusal "
                  "suppressed per c5_predicate_chain_classifier.hpp "
                  "`waveNative` contract; no phantom-lane "
                  "configuration detected):\n";
        for (const std::string &site : predReport.observedSites)
          dbgs() << "  - " << site << "\n";
      });
    }

    if (predReport.refused) {
      RaiseFailure f = RaiseFailure::crossWavePredicateChain(
          kernelName, predReport.refusalDetail);
      errs() << "transpiler: pre-translation abort: " << f.format << " on '"
             << f.mnemonic << "' \u2014 " << f.detail << "\n";
      errs() << "  outcome: (c) refuse \u2014 "
                "WorkitemIdPredicateChain (\u00a73 Class 5"
             << (predReport.phantomLaneRefusal
                     ? " phantom-lane sub-case"
                     : "")
             << ")\n";
      result.failure = std::move(f);
      return result;
    }
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
