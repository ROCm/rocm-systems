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
// CAVEAT: this list reflects whether the *handler* should treat the
// tied operand as a real read (yes for `sdst_in`/`vdata_in`/etc.; no
// for `old`/`vdst_in`). It does NOT promise that the AMDGPU
// disassembler will materialise an MCOperand for that slot — for
// SOP1 `sdst_in` (S_BITSET0/1_B{32,64}) and SOP1 `S_CMOV_B{32,64}`
// the disassembler collapses the tied slot and produces only
// `(sdst, src0)`, so `srcMap` won't contain an entry for the prior-
// dst read. Handlers in those cases must fetch the prior value
// directly via `ctx.regs.readReg{32,64}(op.dst())`. For
// `vdata_in` / `addr_in` / `srcTiedDef` (atomics, MAC accumulators)
// the disassembler does emit a full MCOperand and the handler reads
// it through the normal `op.src(N)` path.
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

  // VOP2 MADMK exception: `v_fmamk_f32` and friends use VOP_MADMK
  // (VOP2Instructions.td), whose Ins32 is `(src0, K-imm, src1)` —
  // i.e., the 32-bit literal sits at MCInst index 2 BETWEEN src0
  // (index 1) and src1 (index 3). The natural positional walk in
  // `buildSrcMap` produces srcMap = [src0, K-imm, src1], which
  // matches what the V_FMAMK_F32 handler in handle_valu.cpp
  // expects (`srcF(0)=src0, srcF(1)=K, srcF(2)=src2` per its own
  // documentation block).
  //
  // The strict `srcMap[k] == OpName::srcN` invariant breaks for
  // this layout because OpName::src1 lives at MCInst index 3, not
  // srcMap[1] = 2. The drift is INTENTIONAL: handlers index by
  // MCInst order, not OpName order, and MADMK is a known stable
  // form. Skip the strict srcN-position check at the AFFECTED
  // index (k=1, the src1 slot) for this signature; k=0 (src0)
  // still passes naturally and k=2 returns -1 (no src2) so the
  // outer loop breaks before reaching it.
  //
  // Detection: the opcode exposes both `OpName::imm` and
  // `OpName::src0` / `OpName::src1`, with the imm operand index
  // strictly between src0 and src1. (Compare to MADAK forms like
  // `v_fmaak_f32`, whose Ins32 is `(src0, src1, K-imm)` — K
  // trailing — where the positional walk happens to coincide with
  // OpName order and the drift check passes naturally.)
  //
  // The modifier-map check below remains in force; MADMK has no
  // src{0,1,2}_modifiers operands, so the loop's modMap branch
  // simply finds expected=-1 and our=-1, agreement.
  int immIdx = AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::imm);
  int src0Idx = AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::src0);
  int src1Idx = AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::src1);
  bool isMADMK = immIdx >= 0 && src0Idx >= 0 && src1Idx >= 0 &&
                 src0Idx < immIdx && immIdx < src1Idx;

  for (unsigned k = 0; k < 3; ++k) {
    int namedSrc = AMDGPU::getNamedOperandIdx(opc, kSrcNames[k]);
    if (namedSrc < 0)
      break;
    int ourSrc = (k < di.numSrcs) ? (int)di.srcMap[k] : -1;
    // Skip ONLY the genuinely-affected index (k=1) for MADMK. k=0
    // (src0) still receives the strict check, so a hypothetical
    // future drift in src0's MCInst position is still caught even
    // for MADMK opcodes.
    bool skipThis = isMADMK && k == 1;
    if (!skipThis && ourSrc != namedSrc)
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

// Decode DPP16 modifier operands (dpp_ctrl / row_mask / bank_mask /
// bound_ctrl) so the raiser can lift DPP-modified VALU ops through
// `llvm.amdgcn.update.dpp`. Sets `di.hasDpp = true` only when every
// DPP16 operand is present and immediate-typed.
//
// Preconditions:
//   - `di.tsFlags` is populated from the ORIGINAL (pre-canonicalisation)
//     MCInstrDesc — the DPP bit here is the authoritative signal that
//     SOME DPP form is in play, but it does NOT distinguish DPP16 from
//     DPP8 (both `VOP_DPP8_Base` and VOP_DPP set `let DPP = 1`, see
//     VOPInstructions.td).
//
// DPP8 handling (present corpus: 0 instances, but architecturally
// possible): DPP8 encodes an 8-lane permutation as a single `OpName::
// dpp8` operand and has NO `dpp_ctrl` / `row_mask` / `bank_mask` /
// `bound_ctrl`. When we detect DPP8 (named operand `dpp8` exists) we
// leave `di.hasDpp` false; the classifier's DppCrossLane site will
// then mark the kernel as `rewriteImplemented = false` (pending P5
// extension to `llvm.amdgcn.mov.dpp8`), so the raiser refuses loudly
// rather than crashing on a partially-populated DPP modifier set.
//
// `fi` (fetch-invalid) is not surfaced for DPP16 — `llvm.amdgcn.
// update.dpp` does not take it. A future DPP8 lift would route
// through `llvm.amdgcn.mov.dpp8` which also does not take `fi`.
void decodeDppModifiers(DecodedInst &di) {
  if (!(di.tsFlags & SIInstrFlags::DPP))
    return;
  const MCInst &inst = di.inst;
  const unsigned opc = inst.getOpcode();
  // Detect DPP8 form by presence of the `dpp8` named operand. If this
  // is a DPP8 instruction, leave `hasDpp` false — see the header
  // comment for the classifier-refusal contract.
  if (AMDGPU::getNamedOperandIdx(opc, AMDGPU::OpName::dpp8) >= 0)
    return;
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
    // MCInstrDesc declared DPP and it is not a DPP8 variant, yet the
    // MCInst operand list is missing one of the four DPP16 modifier
    // fields. This is a decoder-vs-tblgen drift situation — fail
    // loudly rather than emit IR with default (possibly wrong)
    // values. DPP8 was already filtered above, so we only reach here
    // on a genuinely unrecognised DPP form.
    std::string msg;
    raw_string_ostream os(msg);
    os << "decodeDppModifiers: TSFlags::DPP is set for '" << di.rawMnemonic
       << "' (opcode=" << opc
       << ") with no OpName::dpp8 operand, yet at least one of "
          "{dpp_ctrl, row_mask, bank_mask, bound_ctrl} is missing or "
          "not an immediate. LLVM likely added a new DPP variant "
          "whose operand layout this decoder does not yet recognise; "
          "extend decodeDppModifiers.";
    report_fatal_error(os.str().c_str());
  }
  di.hasDpp = true;
  di.dppCtrl = static_cast<uint16_t>(*ctrl & 0xFFFF);
  di.dppRowMask = static_cast<uint8_t>(*rowMask & 0xF);
  di.dppBankMask = static_cast<uint8_t>(*bankMask & 0xF);
  di.dppBoundCtrl = (*boundCtrl) != 0;
}

// Decode the 16-bit `OpName::offset` immediate of `ds_swizzle_b32`
// into `di.dsSwizzleImm` so the obstruction classifier and the DS
// handler share a single canonical extraction point. Mirrors the
// `decodeDppModifiers` pattern: decode-time field population, no
// per-call MCInst probing in downstream consumers.
//
// Only fires for `SemOp::DS_SWIZZLE_B32`. For every other instruction
// `hasDsSwizzleImm` stays false and `dsSwizzleImm` is meaningless;
// consumers MUST gate on `hasDsSwizzleImm`.
//
// Soundness: refuses to populate the field if the operand is missing,
// non-immediate, or outside the unsigned 16-bit range. The classifier
// treats `!hasDsSwizzleImm` as "rewriteImplemented = false" so the
// kernel refuses loudly with a malformed-disassembly diagnostic
// rather than silently truncating a wider value to uint16_t (which
// could land in either the QUAD_PERM or BITMASK_PERM safe envelope
// and cause a silent miscompile).
void decodeDsSwizzleImm(DecodedInst &di) {
  if (di.semOp != SemOp::DS_SWIZZLE_B32)
    return;
  const MCInst &inst = di.inst;
  int idx = AMDGPU::getNamedOperandIdx(inst.getOpcode(),
                                        AMDGPU::OpName::offset);
  if (idx < 0 || static_cast<unsigned>(idx) >= inst.getNumOperands())
    return;
  const MCOperand &mop = inst.getOperand(static_cast<unsigned>(idx));
  if (!mop.isImm())
    return;
  int64_t raw = mop.getImm();
  if (raw < 0 || raw > 0xFFFF)
    return;
  di.dsSwizzleImm = static_cast<uint16_t>(raw);
  di.hasDsSwizzleImm = true;
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
    decodeDsSwizzleImm(di);
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
