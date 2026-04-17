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
// EXEC-writer detection (shared by Phase 1.4 cross-wave gate and Phase 1.5
// SPE allow-list gate).
//
// Derivation strategy (principled, no string matching, no per-opcode
// allowlists for detection):
//
//   (a) Implicit defs — canonical LLVM TableGen-derived source of truth.
//       `MCInstrDesc::implicit_defs()` is iterated during decoding
//       (see `raiser.cpp` Phase 1) and each result is normalised through
//       `AMDGPU::mc2PseudoReg` (strips subtarget-variant suffixes) before
//       classification. Results are cached in `DecodedInst::defsEXEC`,
//       `defsVCC`, `defsSCC`. This is strictly equivalent to
//       `desc.hasImplicitDefOfPhysReg(EXEC)` extended across the
//       EXEC/EXEC_LO/EXEC_HI canonical-alias family. Instructions like
//       `v_cmpx_*` and `s_*_saveexec_*` surface here.
//
//   (b) Explicit defs — required because LLVM models some EXEC writers
//       with EXEC as an *explicit* destination operand (e.g.
//       `s_mov_b64 exec, sN`, `s_and_b32 exec_lo, exec_lo, s2`).
//       `hasImplicitDefOfPhysReg` does NOT cover these by design.
//       Walk the first `desc.getNumDefs()` operands (TableGen convention:
//       defs always come first in the MCInst operand list) and classify
//       each through `AMDGPU::mc2PseudoReg`, same as (a).
//
// (a) ∪ (b) is exhaustive for AMDGPU: an MCInst either defines a register
// implicitly (via TableGen `let Defs = [...]`) or explicitly (as an
// `outs` operand). There is no third path. Both halves ground in
// MCInstrDesc; no mnemonic parsing and no per-opcode lists.
// ============================================================================
static bool instructionWritesEXEC(const DecodedInst &di, const MCState &mc) {
  if (di.defsEXEC)
    return true;
  const MCInstrDesc &desc = mc.instrInfo->get(di.inst.getOpcode());
  for (unsigned i = 0; i < desc.getNumDefs() &&
                       i < di.inst.getNumOperands();
       ++i) {
    const MCOperand &mop = di.inst.getOperand(i);
    if (!mop.isReg() || !mop.getReg())
      continue;
    MCRegister reg = AMDGPU::mc2PseudoReg(mop.getReg());
    if (reg == AMDGPU::EXEC || reg == AMDGPU::EXEC_LO ||
        reg == AMDGPU::EXEC_HI)
      return true;
  }
  return false;
}

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

  // ==== Phase 1.4: Cross-wave-size safety check (warning-only) ====
  //
  // There is no hardware-defined semantic for "running a binary compiled
  // for wave size W_src on hardware with wave size W_tgt != W_src". The
  // source author never specified what lanes outside [0, W_src) should
  // do, so any translation policy here is *our* choice, not a derivable
  // fact from the source ISA or the binary itself.
  //
  // The raiser implements modulo-replication in
  // `RaiseContext::emitLaneActiveBit`: target lane L reads bit
  // (L mod W_src) of the source EXEC mask. That policy is *provably
  // equivalent to launching the kernel (W_tgt / W_src) times in parallel
  // on independent sub-waves* iff the kernel never writes EXEC — i.e.
  // the kernel is embarrassingly per-lane parallel and has no intra-BB
  // divergence. The moment the kernel manipulates EXEC via
  // `v_cmpx` / `s_*_saveexec` / `s_and_b64 exec, ...`, replication
  // doubles (or N-folds) the author's divergence decisions onto
  // sub-waves the author never authored. That is a silent miscompile
  // category in the general case, but for many simple kernels
  // (e.g. vecadd-style bounds checks against a uniform larger than the
  // target wave) replication happens to be observationally identical to
  // native execution because the decision is lane-position-independent.
  //
  // KNOWN LIMITATION: The raiser currently accepts cross-wave
  // translation of EXEC-manipulating kernels with a warning rather than
  // an abort. This is a deliberate, time-boxed retention of the
  // pre-existing (unsound-in-general) behaviour — without it, every
  // current cross-wave GPU regression test (Gfx1250Gpu.{Vecadd, Softmax,
  // Matmul*}, CrossArchGpu.VecaddCrossArch) would fail despite
  // empirically producing correct output. Tightening this to an abort
  // requires either (a) a same-wave test corpus, or (b) a divergence-
  // analysis pass that can statically prove a kernel's EXEC writers are
  // replication-safe (e.g. v_cmpx compares tid against a uniform known
  // to be ≥ target_wave_bits, no tid-indexed stores inside narrowed-
  // EXEC regions). See SPE_DESIGN.md (`Cross-wave translation policy`)
  // for the full trade-off analysis and the alternatives considered.
  //
  // Today we still perform the static scan — both so operators see the
  // risk on their logs and so downstream tooling (pipeline runners, CI
  // gates) can escalate the warning to an error if they want a strict
  // policy. `result.failFormat` is intentionally NOT set: success path
  // continues so existing kernels still raise.
  if (isa.waveSize != targetIsa.waveSize) {
    const DecodedInst *firstEXECWriter = nullptr;
    for (const DecodedInst &di : insts) {
      if (instructionWritesEXEC(di, mc)) {
        firstEXECWriter = &di;
        break;
      }
    }
    if (firstEXECWriter) {
      errs()
          << "transpiler: WARNING: cross-wave translation of an "
             "EXEC-manipulating kernel relies on modulo-replication, "
             "which is not provably correct in general.\n"
          << "  source ISA wave size: " << isa.waveSize << " ("
          << sourceISA << ")\n"
          << "  target ISA wave size: " << targetIsa.waveSize << " ("
          << (compilationTargetISA.empty() ? sourceISA : compilationTargetISA)
          << ")\n"
          << "  first EXEC-writer: " << firstEXECWriter->rawMnemonic
          << " at offset 0x"
          << format_hex_no_prefix(firstEXECWriter->offset, 4) << "\n"
          << "  rationale: the kernel manipulates EXEC; replicating it "
             "across wave halves will double per-lane side effects in a "
             "way the source author did not specify. Empirically this is "
             "correct for kernels whose EXEC writers are lane-position-"
             "independent (pointwise ops with bounds checks against a "
             "uniform ≥ target_wave_bits). Same-wave translation is the "
             "principled path for any other EXEC-divergent kernel.\n";
    }
  }

  // ==== Phase 1.5: SPE A-level gate (EXEC-writer allow-list) ====
  //
  // SPE (SIMT Predicated Execution) is correct only when every runtime
  // change to EXEC either (a) propagates through the EXEC alloca via a
  // handler we have audited, or (b) follows the standard dataflow form
  // `exec = f(old_exec, sgprs, ...)` where `f` is a bitwise / shift /
  // move / compare-based scalar op — the IR's live EXEC value then
  // matches the hardware EXEC that the backend re-materialises when it
  // lowers our predicated-store diamonds back to v_cmpx / s_and_saveexec
  // pairs. Anything outside this set (e.g. `s_setreg_b32` aimed at a
  // hypothetical EXEC HWREG, cross-lane ops that broadcast one lane's
  // value into EXEC, future opcodes we have not yet modelled) risks
  // silently generating IR that looks well-typed but diverges from
  // hardware semantics.
  //
  // We scan the fully decoded instruction stream once, identify every
  // instruction that defines EXEC (implicit_def or an explicit dst
  // register that resolves to EXEC / EXEC_LO / EXEC_HI), and abort if
  // its SemOp is not in the audited set below. The gate is permanent:
  // every new EXEC-writer must be added here only after its handler has
  // been verified against SPE, making accidental silent regressions
  // impossible.
  {
    // Allow-list of SemOps whose handlers have been audited to route EXEC
    // writes through regs.storeExec (either directly or through
    // writeReg{32,64,ExecWidth} which dispatch EXEC→storeExec). Each entry
    // has been verified by following the corresponding handler and
    // confirming a write to ParsedReg::EXEC lands in the exec alloca — so
    // mem-to-reg later produces SSA that correctly carries the per-lane
    // EXEC bit through the SPE diamonds. See AGENTS.md / SPE_DESIGN.md
    // for the SPE invariant and the audit protocol; do not add to this
    // list without repeating the audit.
    //
    // Audit trail (handler:line → mechanism):
    //   V_CMPX                        → handle_valu.cpp: explicit storeExec
    //   S_{AND,OR,XOR,ANDN2,ORN2}_SAVEEXEC_B32
    //                                 → handle_sop1.cpp: explicit storeExec
    //   S_MOV_B{32,64}, S_NOT_B{32,64}
    //                                 → handle_sop1.cpp: writeReg32/64 → storeExec
    //   S_{AND,OR,XOR,ANDN2,ORN2}_B{32,64}, S_LSHL_B{32,64},
    //   S_LSHR_B32, S_BFM_B{32,64}, S_CSELECT_B{32,64}
    //                                 → handle_sop2.cpp: writeReg32/64 → storeExec
    auto isSPEModeledExecWriter = [](SemOp sop) -> bool {
      switch (sop) {
      case SemOp::V_CMPX:
      case SemOp::S_AND_SAVEEXEC_B32:
      case SemOp::S_OR_SAVEEXEC_B32:
      case SemOp::S_XOR_SAVEEXEC_B32:
      case SemOp::S_ANDN2_SAVEEXEC_B32:
      case SemOp::S_ORN2_SAVEEXEC_B32:
      case SemOp::S_MOV_B32:    case SemOp::S_MOV_B64:
      case SemOp::S_NOT_B32:    case SemOp::S_NOT_B64:
      case SemOp::S_AND_B32:    case SemOp::S_AND_B64:
      case SemOp::S_OR_B32:     case SemOp::S_OR_B64:
      case SemOp::S_XOR_B32:    case SemOp::S_XOR_B64:
      case SemOp::S_ANDN2_B32:  case SemOp::S_ANDN2_B64:
      case SemOp::S_ORN2_B32:   case SemOp::S_ORN2_B64:
      case SemOp::S_LSHL_B32:   case SemOp::S_LSHL_B64:
      case SemOp::S_LSHR_B32:
      case SemOp::S_BFM_B32:    case SemOp::S_BFM_B64:
      case SemOp::S_CSELECT_B32:case SemOp::S_CSELECT_B64:
        return true;
      default:
        return false;
      }
    };

    for (const DecodedInst &di : insts) {
      if (!instructionWritesEXEC(di, mc))
        continue;
      if (isSPEModeledExecWriter(di.semOp))
        continue;
      result.failMnemonic = di.mnemonic;
      result.failFormat = "SPE-unmodeled-EXEC-writer";
      errs() << "transpiler: pre-translation abort: '" << di.rawMnemonic
             << "' writes EXEC but its SemOp (" << (int)di.semOp
             << ") is not in the SPE-modelled allow-list. Adding it "
                "requires auditing the handler path against SPE "
                "(lane-active predication assumption).\n";
      return result;
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
  regs.init(B, i32Ty, i1Ty, isa, targetIsa);

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
