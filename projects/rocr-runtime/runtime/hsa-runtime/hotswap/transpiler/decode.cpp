#include "decode.hpp"

#include "amdgpu_formats.hpp"
#include "decoded_inst.hpp"
#include "mc_state.hpp"
#include "opcode_map.hpp"
#include "semop.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, VCC, SCC, ...
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace transpiler {

namespace {

// Build the logical-source view of an MCInst. Walks `desc.operands()` and
// classifies each operand using TableGen-generated metadata only:
//
//   * Operand types carrying the AMDGPU-specific `OPERAND_INPUT_MODS`
//     tag are VOP3 source modifiers (neg/abs/opsel packed as an imm).
//     They attach to the next logical source via `modMap`.
//   * DPP/SDWA encodings carry a tied "old" input (fallback value for
//     inactive lanes, named `$old` or `$vdst_in` in TableGen). In our
//     all-lanes-active scalar model that slot is never read, so we skip
//     it. Not every tied-to-def operand is a fallback — VOP2 MAC forms
//     (v_fmac_f32, v_mac_f32, v_dot2c_*) tie `$src2` to the dst and
//     atomics tie `$vdata_in`/`$sdst_in`/`$addr_in`; in those cases the
//     tied operand is a real accumulator/read-modify input and must stay
//     in srcMap. We therefore select on the named-operand id rather than
//     the TIED_TO bit alone.
//   * Everything else is a logical source recorded in MCInst order.
void buildSrcMap(DecodedInst &di, const MCInstrDesc &desc) {
  const MCInst &inst = di.inst;
  unsigned opc = inst.getOpcode();
  int oldIdx = AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::old);
  int vdstInIdx = AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::vdst_in);
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
}

// Drift check A: every tied-to-def operand on this instruction must have
// an OpName we've explicitly classified. If LLVM introduces a new
// tied-input OpName we haven't audited (so we don't know whether to skip
// or keep it), stop and make a human decide. `kKnownTiedIn` is the
// exhaustive audit as of this commit. Two semantic categories:
//
//   skipped-as-fallback (DPP/SDWA inactive-lane value; never read in the
//                       all-lanes-active scalar model):
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
void driftCheckTiedIn(const DecodedInst &di, const MCInstrDesc &desc) {
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
  const MCInst &inst = di.inst;
  unsigned opc = inst.getOpcode();
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
    if (!known) {
      std::string msg;
      raw_string_ostream os(msg);
      os << "transpiler: tied-to-def operand has an OpName not in the "
            "audited set — classify explicitly (fallback to skip vs. real "
            "input to keep) before proceeding for " << di.rawMnemonic
         << " (opcode=" << opc << "): index=" << i
         << ", tiedTo=" << tied
         << ", numDefs=" << desc.getNumDefs()
         << ", numOps=" << inst.getNumOperands();
      report_fatal_error(StringRef(msg));
    }
  }
}

// Drift check B: for every opcode that exposes `srcN` / `srcN_modifiers`
// naming (VALU, VOPC, SOP1/SOP2, a handful of scalar forms), the first
// N entries of srcMap / modMap must agree with LLVM's named-operand
// table. Catches operand-layout drift for the large majority of opcodes
// — but notably NOT for DS / MUBUF / FLAT / SMEM / image encodings,
// which don't use srcN naming; those formats are only protected by the
// walk's correctness and drift check A.
//
// Scaled MFMA instructions (ScaledMAIInst in TableGen) append
// src0_modifiers / src1_modifiers AFTER all source operands, not
// interleaved as in VOP3. The walk can't discover them because it only
// looks for OPERAND_INPUT_MODS *before* each source. We repair the
// modMap from LLVM's authoritative named-operand table here, but ONLY
// for MAI-format instructions so we don't silently mask future layout
// drift in other formats.
void driftCheckSrcN(DecodedInst &di, const MCInstrDesc &desc) {
  static constexpr AMDGPU::OpName kSrcNames[] = {
      AMDGPU::OpName::src0, AMDGPU::OpName::src1, AMDGPU::OpName::src2};
  static constexpr AMDGPU::OpName kModNames[] = {
      AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
      AMDGPU::OpName::src2_modifiers};

  auto reportErr = [&](const Twine &prefix, int index, int ours,
                       int expected) -> void {
    std::string msg;
    raw_string_ostream os(msg);
    os << prefix << " for " << di.rawMnemonic
       << " (opcode=" << di.inst.getOpcode() << "): index=" << index
       << ", srcMap/modMap=" << ours << ", named=" << expected
       << ", numSrcs=" << di.numSrcs
       << ", numDefs=" << desc.getNumDefs()
       << ", numOps=" << di.inst.getNumOperands();
    report_fatal_error(StringRef(msg));
  };

  unsigned opc = di.inst.getOpcode();
  for (unsigned k = 0; k < 3; ++k) {
    int namedSrc = AMDGPU::getNamedOperandIdx(opc, kSrcNames[k]);
    if (namedSrc < 0)
      break;
    int ourSrc = (k < di.numSrcs) ? (int)di.srcMap[k] : -1;
    if (ourSrc != namedSrc)
      reportErr("transpiler: srcMap disagrees with OpName::srcN table",
                (int)k, ourSrc, namedSrc);
    int namedMod = AMDGPU::getNamedOperandIdx(opc, kModNames[k]);
    int ourMod = (di.modMap[k] == UINT_MAX) ? -1 : (int)di.modMap[k];
    int expectedMod = (namedMod < 0) ? -1 : namedMod;
    if (ourMod != expectedMod) {
      bool isMAI = di.tsFlags & SIInstrFlags::IsMAI;
      if (isMAI && namedMod >= 0 && ourMod == -1) {
        di.modMap[k] = (unsigned)namedMod;
      } else {
        reportErr(
            "transpiler: modMap disagrees with OpName::srcN_modifiers table",
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
void classifyImplicitDefs(DecodedInst &di, const MCInstrDesc &desc) {
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
}

// Decode the `scale_offset` bit out of the CPol operand once, so handlers
// can consume a typed boolean instead of string-searching the disassembled
// `fullText`. gfx12+ FLAT/GLOBAL forms carry the bit in `cpol`; earlier
// ISAs have no `cpol` operand and the flag is inherently absent
// (`hasScaleOffset` stays false).
void decodeScaleOffset(DecodedInst &di) {
  const MCInst &inst = di.inst;
  int cpolIdx =
      AMDGPU::getNamedOperandIdx(inst.getOpcode(), AMDGPU::OpName::cpol);
  if (cpolIdx < 0 || (unsigned)cpolIdx >= inst.getNumOperands())
    return;
  const MCOperand &mop = inst.getOperand((unsigned)cpolIdx);
  if (!mop.isImm())
    return;
  int64_t cpol = mop.getImm();
  di.hasScaleOffset = (cpol & AMDGPU::CPol::SCAL) != 0;
}

// Decode the DPP modifier operands (dpp_ctrl / row_mask / bank_mask /
// bound_ctrl) once, so the raiser can lift DPP-modified VALU ops
// through `llvm.amdgcn.update.dpp` without each handler having to
// re-inspect the MCInst operand list. Triggered for every instruction
// whose original (pre-canonicalisation) MCInstrDesc carries
// `TSFlags & SIInstrFlags::DPP`. For non-DPP instructions `hasDpp`
// stays false and the modifier fields retain their default values.
//
// `fi` is intentionally NOT surfaced: the supported LLVM intrinsic
// `llvm.amdgcn.update.dpp` has no fi argument (it encodes DPP16;
// DPP8's fi control is a separate intrinsic family not covered here).
// If a future corpus kernel uses DPP8 we will extend this to emit
// `llvm.amdgcn.mov.dpp8` + an additional field.
//
// Precondition: `di.tsFlags` is already populated from the ORIGINAL
// MCInstrDesc (i.e. before any SemOp-level canonicalisation), so
// testing the DPP bit here is the authoritative source of truth.
void decodeDppModifiers(DecodedInst &di) {
  if (!(di.tsFlags & SIInstrFlags::DPP))
    return;
  const MCInst &inst = di.inst;
  const unsigned opc = inst.getOpcode();
  auto immOpt = [&](AMDGPU::OpName name) -> std::optional<int64_t> {
    int idx = AMDGPU::getNamedOperandIdx(opc, name);
    if (idx < 0 || static_cast<unsigned>(idx) >= inst.getNumOperands())
      return std::nullopt;
    const MCOperand &mop = inst.getOperand(static_cast<unsigned>(idx));
    if (!mop.isImm())
      return std::nullopt;
    return mop.getImm();
  };
  auto ctrl = immOpt(AMDGPU::OpName::dpp_ctrl);
  auto rowMask = immOpt(AMDGPU::OpName::row_mask);
  auto bankMask = immOpt(AMDGPU::OpName::bank_mask);
  auto boundCtrl = immOpt(AMDGPU::OpName::bound_ctrl);
  if (!ctrl || !rowMask || !bankMask || !boundCtrl) {
    // MCInstrDesc declared DPP but the MCInst operand list is missing
    // one of the four DPP16 modifier fields. This is a decoder-vs-
    // tblgen drift situation — fail loudly rather than emit IR with
    // default (possibly wrong) values.
    std::string msg;
    raw_string_ostream os(msg);
    os << "decodeDppModifiers: TSFlags::DPP is set for '" << di.rawMnemonic
       << "' (opcode=" << opc
       << ") but at least one of {dpp_ctrl, row_mask, bank_mask, "
          "bound_ctrl} is missing or not an immediate. LLVM likely "
          "added a new DPP variant whose operand layout this decoder "
          "does not yet recognise; extend decodeDppModifiers.";
    report_fatal_error(os.str().c_str());
  }
  di.hasDpp = true;
  di.dppCtrl = static_cast<uint16_t>(*ctrl & 0xFFFF);
  di.dppRowMask = static_cast<uint8_t>(*rowMask & 0xF);
  di.dppBankMask = static_cast<uint8_t>(*bankMask & 0xF);
  di.dppBoundCtrl = (*boundCtrl) != 0;
}

// Pull every branch-target offset out of a branch instruction's
// immediates and insert the resulting byte offsets into `blockStarts`.
// Signed 16-bit PC-relative offset * 4 bytes, relative to the
// instruction's successor (off + 4, not off + instSize — matches the
// hardware encoding definition).
void collectBranchTargets(const DecodedInst &di, uint64_t off,
                          uint64_t instSize,
                          std::set<uint64_t> &blockStarts) {
  const MCInst &inst = di.inst;
  for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
    if (!inst.getOperand(i).isImm())
      continue;
    int64_t raw = inst.getOperand(i).getImm();
    int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
    blockStarts.insert(off + 4 + brOff * 4);
  }
  if (di.isConditionalBranch)
    blockStarts.insert(off + instSize);
}

} // namespace

DecodeResult decodeKernel(const MCState &mc,
                          const OpcodeMap &opcMap,
                          ArrayRef<uint8_t> textBytes,
                          uint64_t kernelOffset) {
  DecodeResult out;
  out.blockStarts.insert(kernelOffset);

  if (kernelOffset > 0)
    errs() << "transpiler: Starting disassembly at kernel offset 0x"
           << utohexstr(kernelOffset) << "\n";

  const uint64_t totalSize = textBytes.size();
  uint64_t off = kernelOffset;
  while (off < totalSize) {
    MCInst inst;
    uint64_t instSize = 0;
    auto status = mc.disasm->getInstruction(inst, instSize,
                                            textBytes.slice(off), off,
                                            nulls());
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

    decodeScaleOffset(di);
    decodeDppModifiers(di);
    buildSrcMap(di, desc);
    driftCheckTiedIn(di, desc);
    driftCheckSrcN(di, desc);
    classifyImplicitDefs(di, desc);

    if (di.isBranch)
      collectBranchTargets(di, off, instSize, out.blockStarts);

    bool isEnd = (di.semOp == SemOp::S_ENDPGM);
    out.insts.push_back(std::move(di));
    if (isEnd) {
      // `s_endpgm` may appear mid-binary (early-return path); if there are
      // known block starts at later offsets, keep disassembling.
      uint64_t nextOff = off + instSize;
      auto it = out.blockStarts.upper_bound(off);
      if (it != out.blockStarts.end() && *it < totalSize) {
        off = nextOff;
        continue;
      }
      break;
    }
    off += instSize;
  }

  return out;
}

} // namespace transpiler
