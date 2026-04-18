// Static analysis pass that classifies every `s_set_pc_i64` site in a
// decoded kernel into Pattern A (statically resolvable intra-kernel
// branch — lowers to `br label %BB_target`) or Pattern B (subroutine
// return via an SGPR pair stashed at the call site — lowers to
// `indirectbr ptr %ret_pc, [list of resolved return targets]`). See
// semop.hpp's `S_SET_PC_I64` doc for the full lowering contract.
//
// Key invariants this pass enforces:
//   * Per-block reset: the symbolic-PC table is wiped at every
//     decoder-known basic-block leader. PC chains do not survive
//     control flow, so a chain interrupted by a branch is dropped
//     rather than misclassified.
//   * Conservative cleanup: any SGPR write the pass cannot model
//     (e.g. an s_mov, an unrelated s_add) clears the destination's
//     entry. Pattern A is only claimed when the entire chain
//     (s_get_pc_i64 → s_add_co_u32 → s_add_co_ci_u32) is observable.
//   * Loud refusal: unresolvable sites are recorded with a
//     human-readable reason; the handler turns this into an
//     unsupportedShape failure rather than a silent stub branch.
//
// Pattern B handling enumerates call sites whose getpc+add chain
// targets the same ret-pair SGPR low index the indirect set-PC reads
// from. Call sites are identified as: (a) a complete getpc+add chain
// terminating in s_add_co_ci_u32 to a ret-pair SGPR, immediately
// followed by (b) an unconditional s_branch into the subroutine
// region. The post-branch instruction's offset is the return target
// the call site committed to; we record it as a chain-terminator hook
// so the raiser can rewrite the chain's effect to materialise a
// blockaddress(@kernel, %BB_returnAddr) into the ret-pair (rather
// than the binary PC the chain would otherwise yield).

#include "setpc_analysis.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "mc_state.hpp"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <unordered_map>

using namespace llvm;

namespace transpiler {

namespace {

// Classify a register operand as an SGPR and return its hardware
// index. Returns nullopt for non-SGPR regs (VCC/EXEC/SCC/M0/...).
// Mirrors the SGPR branch of RaiseContext::parseReg but stays scoped
// to this analysis (no shared state, so it can run pre-IR).
std::optional<unsigned> sgprIdx(const MCRegisterInfo &MRI, MCRegister reg) {
  if (!reg)
    return std::nullopt;
  MCRegister lane = MRI.getSubReg(reg, AMDGPU::sub0);
  if (!lane)
    lane = reg;
  lane = AMDGPU::mc2PseudoReg(lane);
  // Filter the special-purpose registers parseReg handles before
  // falling through to SGPR_32. None of them participate in PC chains.
  switch (lane) {
  case AMDGPU::VCC_LO:
  case AMDGPU::VCC_HI:
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
  case AMDGPU::SCC:
  case AMDGPU::MODE:
  case AMDGPU::M0:
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
  case AMDGPU::LDS_DIRECT:
    return std::nullopt;
  default:
    break;
  }
  unsigned enc = MRI.getEncodingValue(reg);
  if (enc & (AMDGPU::HWEncoding::IS_VGPR | AMDGPU::HWEncoding::IS_AGPR))
    return std::nullopt;
  if (!MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(lane))
    return std::nullopt;
  return enc & AMDGPU::HWEncoding::REG_IDX_MASK;
}

// Width of the as-decoded register in 32-bit lanes. A pair (s[X:X+1])
// reports width 2; a scalar reports width 1.
unsigned regWidth32(const MCRegisterInfo &MRI, MCRegister reg) {
  const unsigned maxSubIdx = MRI.getNumSubRegIndices();
  unsigned w = 0;
  for (unsigned subIdx = AMDGPU::sub0; subIdx < maxSubIdx; ++subIdx) {
    if (!MRI.getSubReg(reg, subIdx))
      break;
    ++w;
  }
  return w ? w : 1;
}

// Convenience: extract a 32-bit unsigned immediate from an MCOperand.
// Returns nullopt if the operand is not an integer immediate.
std::optional<uint32_t> imm32(const MCInst &inst, unsigned opIdx) {
  if (opIdx >= inst.getNumOperands())
    return std::nullopt;
  const MCOperand &op = inst.getOperand(opIdx);
  if (!op.isImm())
    return std::nullopt;
  return static_cast<uint32_t>(op.getImm() & 0xFFFFFFFFu);
}

// Per-pair PC-chain state. We track the symbolic absolute kernel
// offset stored in an SGPR pair sX:X+1, plus the offset of the chain
// terminator (the s_add_co_ci_u32 high-half add) so the raiser knows
// which instruction to attach the blockaddress-materialisation hook
// to. `lowAddDone` records whether the low-half s_add_co_u32 has
// already fired (a Pattern A chain follows the strict order
// getpc → low-add → high-add).
struct PcChain {
  uint64_t value = 0;       // symbolic absolute kernel offset
  uint64_t terminator = 0;  // offset of the high-half s_add_co_ci_u32
  bool lowAddDone = false;
};

// Per-scalar immediate state: tracks SGPR_32 values that are known
// constants (from `s_add_co_i32 sZ, IMM, IMM`-style folding). Used to
// resolve the offset operand of the low-half PC add.
struct ScalarImm {
  uint32_t value = 0;
};

class State {
public:
  State(const MCRegisterInfo &MRI) : MRI_(MRI) {}

  // Wipe everything at a basic-block leader. Chain values do not
  // survive a control-flow boundary (any inbound edge could clobber
  // the ret-pair through some other path).
  void resetBlock() {
    pcChains_.clear();
    scalars_.clear();
  }

  // Record a known absolute PC in the SGPR pair starting at `lowIdx`.
  void recordPc(unsigned lowIdx, uint64_t value) {
    PcChain c;
    c.value = value;
    pcChains_[lowIdx] = c;
    // The pair's high half can no longer be a tracked scalar imm.
    scalars_.erase(lowIdx);
    scalars_.erase(lowIdx + 1);
  }

  // Look up a tracked PC chain by SGPR low index.
  PcChain *findPc(unsigned lowIdx) {
    auto it = pcChains_.find(lowIdx);
    return it == pcChains_.end() ? nullptr : &it->second;
  }

  void recordScalar(unsigned idx, uint32_t value) {
    scalars_[idx].value = value;
    // A scalar write to one half of a tracked PC pair invalidates it.
    invalidatePcAt(idx);
  }

  std::optional<uint32_t> findScalar(unsigned idx) const {
    auto it = scalars_.find(idx);
    if (it == scalars_.end())
      return std::nullopt;
    return it->second.value;
  }

  // Drop any tracked PC pair whose low or high half is `idx`.
  void invalidatePcAt(unsigned idx) {
    auto it = pcChains_.find(idx);
    if (it != pcChains_.end()) {
      pcChains_.erase(it);
      return;
    }
    if (idx == 0)
      return;
    it = pcChains_.find(idx - 1);
    if (it != pcChains_.end())
      pcChains_.erase(it);
  }

  void invalidateScalarAt(unsigned idx) { scalars_.erase(idx); }

  // Promote an in-progress chain to "low add done": the next
  // s_add_co_ci_u32 to the matching high-half register completes the
  // chain.
  void markLowAddDone(unsigned lowIdx, uint64_t newValue) {
    auto it = pcChains_.find(lowIdx);
    if (it == pcChains_.end())
      return;
    it->second.value = newValue;
    it->second.lowAddDone = true;
  }

  // Finalise the chain at `lowIdx`, returning the resolved value and
  // the terminator offset. Caller must have already verified the
  // chain reached lowAddDone.
  void finishHighAdd(unsigned lowIdx, uint64_t terminatorOff,
                     uint64_t addedHi) {
    auto it = pcChains_.find(lowIdx);
    if (it == pcChains_.end())
      return;
    it->second.value += addedHi << 32;
    it->second.terminator = terminatorOff;
    // Stays in the table so the next instruction (s_set_pc_i64 or
    // s_branch) can read it.
  }

  // Drop everything that is not a finalised PC pair (i.e. invalidate
  // any chain whose lowAddDone is false). Called when the analysis
  // observes any SGPR write that breaks the strict chain order.
  void dropInProgressChains() {
    for (auto it = pcChains_.begin(); it != pcChains_.end();) {
      if (!it->second.lowAddDone)
        it = pcChains_.erase(it);
      else
        ++it;
    }
  }

private:
  const MCRegisterInfo &MRI_;
  std::unordered_map<unsigned, PcChain> pcChains_;
  std::unordered_map<unsigned, ScalarImm> scalars_;
};

// Encapsulates "what does this instruction write to SGPR space" for
// the cases we model directly. Anything not covered by the explicit
// SemOp dispatch falls into a generic write-detection pass that
// invalidates any tracked SGPR the instruction defines.
void invalidateGeneralSgprDefs(const DecodedInst &di,
                               const MCRegisterInfo &MRI, State &state) {
  for (unsigned i = 0; i < di.numDefs && i < di.numOps(); ++i) {
    if (!di.isReg(i))
      continue;
    auto idx = sgprIdx(MRI, di.getReg(i));
    if (!idx)
      continue;
    unsigned w = regWidth32(MRI, di.getReg(i));
    for (unsigned k = 0; k < w; ++k) {
      state.invalidatePcAt(*idx + k);
      state.invalidateScalarAt(*idx + k);
    }
  }
}

} // namespace

SetPcAnalysis analyseSetPC(ArrayRef<DecodedInst> insts,
                           const std::set<uint64_t> &blockStarts,
                           const MCState &mc) {
  SetPcAnalysis result;
  if (insts.empty())
    return result;

  const MCRegisterInfo &MRI = *mc.regInfo;
  State state(MRI);

  // Pass 1: walk linearly, building per-instruction symbolic-PC state
  // and classifying each s_set_pc_i64 site. We also record candidate
  // call sites: a finalised PC chain immediately followed by an
  // s_branch is the canonical call-site shape (Pattern B).
  //
  // Indices into `insts` of every Pattern B site we encountered,
  // together with the SGPR low index of the ret-pair the indirect
  // set-PC reads. Pass 2 enumerates the matching return targets.
  struct PendingB {
    uint64_t setpcOffset = 0;
    unsigned retPairLowReg = 0;
  };
  SmallVector<PendingB, 16> pendingB;

  // The first instruction is always a block leader; reset there.
  state.resetBlock();
  uint64_t prevOff = insts.front().offset;

  for (size_t i = 0; i < insts.size(); ++i) {
    const DecodedInst &di = insts[i];

    // Reset on every basic-block boundary.  We treat the very first
    // instruction's offset as a leader implicitly via resetBlock()
    // above; subsequent leaders come from `blockStarts`.
    if (di.offset != prevOff && blockStarts.count(di.offset))
      state.resetBlock();
    prevOff = di.offset;

    switch (di.semOp) {
    case SemOp::S_GETPC_B64: {
      // dst = absolute address of the next instruction (di.offset + 4).
      if (di.numDefs >= 1 && di.isReg(0)) {
        auto idx = sgprIdx(MRI, di.getReg(0));
        if (idx) {
          state.recordPc(*idx, di.offset + di.size);
          continue;
        }
      }
      // No usable destination — fall through to default invalidation.
      break;
    }

    case SemOp::S_ADD_U32: {
      // We model two cases:
      //   (a) `s_add_co_i32 sZ, IMM, IMM`  → record sZ as a scalar imm
      //   (b) `s_add_co_u32 sX, sX, <imm-or-tracked-scalar>` where sX
      //        is the low half of a tracked PC pair → extend the chain
      //        and mark lowAddDone.
      if (di.numDefs < 1 || !di.isReg(0))
        break;
      auto dstIdx = sgprIdx(MRI, di.getReg(0));
      if (!dstIdx)
        break;
      // Source operand indices: the SOP2 layout puts dst at 0, src0
      // at 1, src1 at 2 (ignoring the implicit SCC def at numDefs-1).
      // We use the firstSrcIdx from the decoder for safety.
      unsigned s0 = di.firstSrcIdx;
      unsigned s1 = s0 + 1;
      auto src0Imm = imm32(di.inst, s0);
      auto src1Imm = imm32(di.inst, s1);
      if (src0Imm && src1Imm) {
        // Case (a): const+const fold.
        state.recordScalar(*dstIdx, *src0Imm + *src1Imm);
        continue;
      }
      // Case (b): low-half PC add. dst must equal src0 must equal a
      // tracked PC low half; src1 must be a known constant (imm or
      // tracked scalar).
      std::optional<unsigned> src0Idx;
      if (di.isReg(s0))
        src0Idx = sgprIdx(MRI, di.getReg(s0));
      if (!src0Idx || *src0Idx != *dstIdx)
        break;
      PcChain *chain = state.findPc(*dstIdx);
      if (!chain || chain->lowAddDone)
        break;
      uint32_t addend = 0;
      if (src1Imm) {
        addend = *src1Imm;
      } else if (di.isReg(s1)) {
        auto src1Reg = sgprIdx(MRI, di.getReg(s1));
        if (!src1Reg)
          break;
        auto sc = state.findScalar(*src1Reg);
        if (!sc)
          break;
        addend = *sc;
      } else {
        break;
      }
      uint64_t newVal = chain->value + static_cast<uint64_t>(addend);
      state.markLowAddDone(*dstIdx, newVal);
      continue;
    }

    case SemOp::S_ADDC_U32: {
      // High-half PC add: `s_add_co_ci_u32 sY, sY, IMM` where sY is
      // the high half of a tracked PC pair whose low half has already
      // been added to (lowAddDone). The chain terminates here.
      if (di.numDefs < 1 || !di.isReg(0))
        break;
      auto dstIdx = sgprIdx(MRI, di.getReg(0));
      if (!dstIdx || *dstIdx == 0)
        break;
      unsigned lowIdx = *dstIdx - 1;
      PcChain *chain = state.findPc(lowIdx);
      if (!chain || !chain->lowAddDone)
        break;
      unsigned s0 = di.firstSrcIdx;
      unsigned s1 = s0 + 1;
      // src0 must be the same high half (sY).
      std::optional<unsigned> src0Idx;
      if (di.isReg(s0))
        src0Idx = sgprIdx(MRI, di.getReg(s0));
      if (!src0Idx || *src0Idx != *dstIdx)
        break;
      auto src1Imm = imm32(di.inst, s1);
      if (!src1Imm) {
        // We accept only a constant high-half addend (the canonical
        // pattern is +0 with carry from the low add). Anything else
        // we cannot fold.
        break;
      }
      state.finishHighAdd(lowIdx, di.offset, static_cast<uint64_t>(*src1Imm));
      // Tentatively register a chain-terminator hook. Pass 2 will
      // confirm whether this chain feeds Pattern A (consumed by the
      // very next s_set_pc_i64) or Pattern B (consumed by an s_branch
      // that crosses into a subroutine — i.e. a true call site). For
      // Pattern A we drop the hook; for Pattern B we keep it and
      // record the resolved return target. We hold the entry in a
      // side map keyed by chain low-index for now.
      result.chainTerminators[di.offset] =
          SetPcCallSiteInfo{state.findPc(lowIdx)->value, lowIdx};
      continue;
    }

    case SemOp::S_SET_PC_I64: {
      // The source SGPR pair is the indirect target.
      unsigned srcOpIdx = di.firstSrcIdx;
      if (!di.isReg(srcOpIdx)) {
        SetPcSiteInfo info;
        info.kind = SetPcSiteInfo::Kind::Unresolvable;
        info.refusalReason =
            "s_set_pc_i64 source operand is not a register";
        result.setpcSites[di.offset] = std::move(info);
        continue;
      }
      auto srcIdx = sgprIdx(MRI, di.getReg(srcOpIdx));
      if (!srcIdx) {
        SetPcSiteInfo info;
        info.kind = SetPcSiteInfo::Kind::Unresolvable;
        info.refusalReason =
            "s_set_pc_i64 source register is not an SGPR pair";
        result.setpcSites[di.offset] = std::move(info);
        continue;
      }
      PcChain *chain = state.findPc(*srcIdx);
      // s_set_pc_i64 is a control-flow terminator; the next linear
      // instruction is reachable only via some other branch (or is
      // dead). The raiser BB-layout phase will append IR after our
      // emitted br/indirectbr unless the next instruction is a
      // labelled BB leader, so we always promote off+size to a leader
      // here. If no instruction lives at that offset (s_set_pc_i64
      // was the kernel's last instruction) the entry is harmless.
      result.extraBlockStarts.insert(di.offset + di.size);
      if (chain && chain->lowAddDone) {
        // Pattern A: chain resolves the absolute target.
        SetPcSiteInfo info;
        info.kind = SetPcSiteInfo::Kind::DirectA;
        info.directTarget = chain->value;
        result.setpcSites[di.offset] = std::move(info);
        result.extraBlockStarts.insert(chain->value);
        // The chain that fed this Pattern A site is consumed inline;
        // no call-site rewrite is needed for its terminator.
        if (chain->terminator)
          result.chainTerminators.erase(chain->terminator);
        // Consume the chain (the SGPR pair is now logically dead for
        // PC-tracking purposes; further reads would indicate stale
        // state we do not want to honour).
        state.invalidatePcAt(*srcIdx);
        continue;
      }
      // Pattern B candidate: defer until Pass 2 has the full set of
      // call sites enumerated.
      pendingB.push_back(PendingB{di.offset, *srcIdx});
      continue;
    }

    default:
      break;
    }

    // Generic fallthrough: invalidate every SGPR (and overlapping PC
    // pair) the instruction writes to. This prevents stale chain
    // values from leaking past instructions whose semantics we did
    // not model. Also drop in-progress chains so a stray write
    // between getpc and the low add cannot be silently absorbed into
    // the chain by a later matching add.
    invalidateGeneralSgprDefs(di, MRI, state);
    state.dropInProgressChains();
  }

  // Pass 2: classify pending Pattern B sites and prune the chain
  // terminator side-table.
  //
  // Invariant established by Pass 1: every entry left in
  // `chainTerminators` is the high-add of a fully-resolved getpc+add
  // chain that was NOT consumed by an immediately-following Pattern A
  // `s_set_pc_i64` (Pattern A erases its own terminator entry on the
  // way out). The remaining entries are call-site candidates whose
  // ret-pair value crosses the next s_branch.
  //
  // We further restrict: a chain-terminator hook only matters if its
  // ret-pair low index is actually read by some Pattern B site we
  // observed. Otherwise the chain is dead code (or a pattern this
  // analysis does not recognise) and we drop the hook so the raiser
  // does not gratuitously rewrite that chain into a blockaddress
  // materialisation.

  // Collect the set of ret-pair low indices read by pending Pattern
  // B sites.
  std::set<unsigned> retPairsConsumed;
  for (const PendingB &pb : pendingB)
    retPairsConsumed.insert(pb.retPairLowReg);

  // Drop chain terminator entries that do not feed any consumer.
  for (auto it = result.chainTerminators.begin();
       it != result.chainTerminators.end();) {
    if (!retPairsConsumed.count(it->second.retPairLowReg))
      it = result.chainTerminators.erase(it);
    else
      ++it;
  }

  // Build per-pair return-target lists from the surviving terminators
  // and seed extraBlockStarts with each return target.
  std::unordered_map<unsigned, SmallVector<uint64_t, 4>> targetsByPair;
  for (const auto &kv : result.chainTerminators) {
    targetsByPair[kv.second.retPairLowReg].push_back(
        kv.second.resolvedReturnAddr);
    result.extraBlockStarts.insert(kv.second.resolvedReturnAddr);
  }
  // Deduplicate + sort each target list so lit fixtures see a stable
  // shape.
  for (auto &kv : targetsByPair) {
    auto &v = kv.second;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }

  for (const PendingB &pb : pendingB) {
    auto it = targetsByPair.find(pb.retPairLowReg);
    if (it == targetsByPair.end() || it->second.empty()) {
      SetPcSiteInfo info;
      info.kind = SetPcSiteInfo::Kind::Unresolvable;
      info.refusalReason =
          ("s_set_pc_i64 reads SGPR pair s[" +
           std::to_string(pb.retPairLowReg) + ":" +
           std::to_string(pb.retPairLowReg + 1) +
           "] but no statically resolvable call-site getpc+add chain "
           "targets that pair");
      result.setpcSites[pb.setpcOffset] = std::move(info);
      continue;
    }
    SetPcSiteInfo info;
    info.kind = SetPcSiteInfo::Kind::IndirectB;
    info.indirectTargets.assign(it->second.begin(), it->second.end());
    info.indirectRetPairLowReg = pb.retPairLowReg;
    result.setpcSites[pb.setpcOffset] = std::move(info);
  }

  return result;
}

} // namespace transpiler
