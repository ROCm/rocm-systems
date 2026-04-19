// Static analysis pass that classifies every `s_set_pc_i64` and
// `s_swap_pc_i64` site in a decoded kernel into one of three principled
// shapes:
//
//   * DirectA: source SGPR pair is produced by a complete
//     `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32` chain reachable
//     in the same basic block. Lowers to `br label %BB_target`.
//
//   * IndirectB: subroutine-return shape — the source SGPR pair is the
//     ret-pair populated by a caller's chain (Pattern B). Lowers to
//     `indirectbr ptr %ret_pc, [list of resolved return targets]` where
//     each call site in the kernel that wrote that ret-pair contributes
//     one target via the `chainTerminators` rewrite hook.
//
//   * DispatchSet: multi-target dispatch — the source SGPR pair holds
//     one of N statically-known absolute targets reaching the use site
//     through distinct CFG paths (e.g. a tensilelite "activation
//     function dispatcher" — each predecessor block writes a different
//     chain target into the same pair, then a join block consumes it).
//     Lowers to `indirectbr ptr %target, [list]`. For `s_swap_pc_i64`
//     it ALSO writes the return-PC `blockaddress` into sdst before the
//     indirectbr (mirroring the DirectA dst-write).
//
// See semop.hpp's `S_SET_PC_I64` and `S_SWAP_PC_I64` doc for the full
// lowering contracts. The handler in `handle_sop1.cpp` consumes the
// classification.
//
// The pass runs in five phases. The last three together implement the
// inter-block PC-chain dataflow that distinguishes DispatchSet from
// "still Unresolvable". Without that dataflow, dispatcher-shaped
// kernels (the common tensilelite case where the call target is
// computed in a predecessor of the swap_pc's block) fall into
// Unresolvable and the handler refuses, blocking large slices of the
// corpus.
//
//   Phase 1 — pre-pass. Walk insts once to enumerate every swap/set_pc
//             instruction's fallthrough offset and pre-add it to the
//             working block-leader set. This guarantees the per-block
//             walk in Phase 2 sees every swap/set_pc as the LAST inst
//             of its block (so its fallthrough is in a separate block,
//             and so the block-exit transfer correctly summarises the
//             pair state up to and including the swap/set_pc).
//
//   Phase 2 — per-block intra-block walk. Mirrors the original
//             single-pass analysis: build per-instruction symbolic-PC
//             state (chains, scalar imms), classify swap/set_pc sites
//             that resolve in-block as DirectA (or, if the source pair
//             was dirtied without a complete chain, as Unresolvable
//             with the intra-block-specific reason). Sites whose
//             source pair is pristine through the block are deferred
//             to Phase 4 with a `PendingDataflow` record. Also collect
//             every chain terminator (Phase 5 prunes), every PendingB
//             site (Phase 5 classifies as IndirectB), and per-block
//             transfer summaries: per-pair {SET(value, terminator),
//             KILL, PASS}.
//
//   Phase 3 — CFG + forward dataflow. Build per-block successor lists
//             from the last instruction (S_BRANCH / S_CBRANCH_* /
//             S_SWAP_PC fallthrough / fallthrough-to-leader). Run a
//             worklist forward dataflow on a finite lattice
//             (`(SGPR pair) -> (sorted set<uint64_t>, incomplete)`)
//             where the join is set-union with an incomplete bit OR'd
//             across paths and a hard cap of `kMaxDispatchTargets`
//             values per pair (over the cap → mark incomplete and
//             refuse the use site). Convergence is guaranteed by the
//             bounded-height lattice.
//
//   Phase 4 — re-classify deferred sites. For each PendingDataflow
//             site, look up the entry facts at its block for the
//             source pair. Empty / incomplete → Unresolvable with the
//             dataflow-specific reason. One value → DirectA. Two or
//             more values (within the cap) → DispatchSet with that
//             enumerated set as `indirectTargets`. Add every target
//             to `extraBlockStarts` so the BB-layout phase promotes
//             it to a leader.
//
//   Phase 5 — chain terminator retention. Keep every chain terminator
//             that feeds either (a) an IndirectB ret-pair (existing
//             logic, identifies it by `retPairLowReg` membership) OR
//             (b) a DispatchSet site, where retention is conditional
//             on BOTH `retPairLowReg == DispatchSet.indirectRetPairLowReg`
//             AND `resolvedReturnAddr ∈ DispatchSet.indirectTargets`
//             (so dead chain terminators that don't match a target
//             are still pruned). Build per-pair return-target lists
//             for IndirectB. Drop unused terminators so the raiser's
//             S_ADDC_U32 hook does not gratuitously rewrite chains
//             that don't feed any classified site.
//
// Key invariants this pass enforces:
//   * Per-block reset: the symbolic-PC table is wiped at every
//     decoder-known basic-block leader. PC chains do not survive
//     control flow (within a block). Inter-block survival is encoded
//     entirely through the dataflow lattice; the per-block intra walk
//     only sees its own writes.
//   * Conservative cleanup: any SGPR write the pass cannot model
//     (e.g. an s_mov, an unrelated s_add) clears the destination's
//     entry. DirectA is only claimed when the entire chain
//     (s_get_pc_i64 → s_add_co_u32 → s_add_co_ci_u32) is observable
//     intra-block; DispatchSet is only claimed when the dataflow
//     joins exclusively over complete chains (any path that kills the
//     pair sets `incomplete` and refuses).
//   * Loud refusal: unresolvable sites are recorded with a
//     human-readable reason; the handler turns this into an
//     unsupportedShape failure rather than a silent stub branch.
//
// IndirectB call-site handling (Phase 5) is unchanged from the original
// design: a complete getpc+add chain terminating in s_add_co_ci_u32 to
// a ret-pair SGPR, immediately followed by an unconditional s_branch
// into the subroutine region. The post-branch instruction's offset is
// the return target the call site committed to; we record it as a
// chain-terminator hook so the raiser rewrites the chain's effect to
// materialise a `blockaddress(@kernel, %BB_returnAddr)` into the
// ret-pair (rather than the binary PC the chain would otherwise yield).
// The same rewrite hook fires for DispatchSet retained terminators —
// the value materialised is the dispatch target, not a return address,
// but the mechanism is identical (the field's name remains
// `resolvedReturnAddr` for backward compatibility with the existing
// SetPcCallSiteInfo struct).

#include "setpc_analysis.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "mc_state.hpp"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <deque>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm;

namespace transpiler {

namespace {

// Cap on the number of distinct PC-chain values we will enumerate per
// SGPR pair across CFG edges. Beyond this cap we mark the lattice
// element `incomplete` and the use site falls into Unresolvable. A
// practical dispatcher (e.g. an N-way activation-function switch) tops
// out far below this; any larger fan-out is much more likely to be a
// runtime-derived chain we cannot enumerate, and refusing loudly is
// the principled response.
constexpr size_t kMaxDispatchTargets = 16;

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
// already fired (a complete chain follows the strict order
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

// Intra-block analysis state. Tracks per-pair PC chains and per-half
// scalar immediates. `intraDirtyHalf_` records every SGPR-half index
// the block has WRITTEN (chain ops + generic SGPR writes). It is the
// signal Phase 2 uses to decide whether a swap/set_pc site whose
// chain didn't resolve intra-block is allowed to fall back on dataflow
// entry facts (pristine half → defer; dirtied half → refuse loudly).
class State {
public:
  State(const MCRegisterInfo &MRI) : MRI_(MRI) {}

  // Wipe everything at a basic-block leader. Chain values do not
  // survive a control-flow boundary at the intra-block layer; inter-
  // block survival is encoded entirely through the dataflow lattice.
  void resetBlock() {
    pcChains_.clear();
    scalars_.clear();
    intraDirtyHalf_.clear();
  }

  // Record a known absolute PC in the SGPR pair starting at `lowIdx`.
  void recordPc(unsigned lowIdx, uint64_t value) {
    PcChain c;
    c.value = value;
    pcChains_[lowIdx] = c;
    // The pair's high half can no longer be a tracked scalar imm.
    scalars_.erase(lowIdx);
    scalars_.erase(lowIdx + 1);
    intraDirtyHalf_.insert(lowIdx);
    intraDirtyHalf_.insert(lowIdx + 1);
  }

  // Look up a tracked PC chain by SGPR low index.
  PcChain *findPc(unsigned lowIdx) {
    auto it = pcChains_.find(lowIdx);
    return it == pcChains_.end() ? nullptr : &it->second;
  }

  void recordScalar(unsigned idx, uint32_t value) {
    scalars_[idx].value = value;
    intraDirtyHalf_.insert(idx);
    invalidatePcAt(idx);
  }

  std::optional<uint32_t> findScalar(unsigned idx) const {
    auto it = scalars_.find(idx);
    if (it == scalars_.end())
      return std::nullopt;
    return it->second.value;
  }

  // Drop any tracked PC pair whose low or high half is `idx`. Marks
  // `idx` (and the affected pair's other half) as dirty so the
  // dataflow transfer correctly KILLS pass-through entry facts.
  void invalidatePcAt(unsigned idx) {
    intraDirtyHalf_.insert(idx);
    auto it = pcChains_.find(idx);
    if (it != pcChains_.end()) {
      intraDirtyHalf_.insert(it->first + 1);
      pcChains_.erase(it);
      return;
    }
    if (idx == 0)
      return;
    it = pcChains_.find(idx - 1);
    if (it != pcChains_.end()) {
      intraDirtyHalf_.insert(it->first);
      pcChains_.erase(it);
    }
  }

  void invalidateScalarAt(unsigned idx) {
    intraDirtyHalf_.insert(idx);
    scalars_.erase(idx);
  }

  // Promote an in-progress chain to "low add done": the next
  // s_add_co_ci_u32 to the matching high-half register completes the
  // chain.
  void markLowAddDone(unsigned lowIdx, uint64_t newValue) {
    auto it = pcChains_.find(lowIdx);
    if (it == pcChains_.end())
      return;
    it->second.value = newValue;
    it->second.lowAddDone = true;
    intraDirtyHalf_.insert(lowIdx);
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
    intraDirtyHalf_.insert(lowIdx + 1);
  }

  // Drop everything that is not a finalised PC pair (i.e. invalidate
  // any chain whose lowAddDone is false). Called when the analysis
  // observes any SGPR write that breaks the strict chain order.
  void dropInProgressChains() {
    for (auto it = pcChains_.begin(); it != pcChains_.end();) {
      if (!it->second.lowAddDone) {
        intraDirtyHalf_.insert(it->first);
        intraDirtyHalf_.insert(it->first + 1);
        it = pcChains_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Whether the block has performed any SGPR write to either half of
  // pair `lowIdx`. Used by Phase 2 to decide whether a swap/set_pc
  // site may fall back on dataflow entry facts.
  bool isPairDirty(unsigned lowIdx) const {
    return intraDirtyHalf_.count(lowIdx) ||
           intraDirtyHalf_.count(lowIdx + 1);
  }

  // Accessors used by Phase 2 to construct the per-block transfer
  // summary at end-of-block.
  const std::unordered_map<unsigned, PcChain> &pcChains() const {
    return pcChains_;
  }
  const std::unordered_set<unsigned> &dirtyHalves() const {
    return intraDirtyHalf_;
  }

private:
  const MCRegisterInfo &MRI_;
  std::unordered_map<unsigned, PcChain> pcChains_;
  std::unordered_map<unsigned, ScalarImm> scalars_;
  std::unordered_set<unsigned> intraDirtyHalf_;
};

// Mark every SGPR-half written by `di` as dirty in `state` and drop
// any tracked PC pair whose halves overlap. This is the generic
// fallthrough for instructions whose semantics we did not model.
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

// Per-pair transfer summary computed at end of each block in Phase 2,
// consumed by the Phase 3 dataflow.
struct PairTransfer {
  enum class Kind {
    // Block did not write either half of this pair. Entry facts pass
    // through unchanged.
    Pass,
    // Block ended with this pair holding a complete PC chain whose
    // value is `value` and whose chain terminator is at offset
    // `terminator`. Overrides any incoming entry fact.
    Set,
    // Block wrote one or both halves of this pair but did not end
    // with a complete chain (or wrote a non-chain value). Pass-
    // through entry facts are killed; downstream sees a constraint-
    // less pair.
    Kill,
  };
  Kind kind = Kind::Pass;
  uint64_t value = 0;       // when kind == Set
  uint64_t terminator = 0;  // when kind == Set
};

// Per-block descriptor used by Phase 2 / 3 / 4. `lastIdx` is inclusive.
struct BlockData {
  uint64_t offset = 0;
  size_t firstIdx = 0;
  size_t lastIdx = 0;
  std::unordered_map<unsigned, PairTransfer> transfers;
  SmallVector<uint64_t, 2> successors;
};

// A swap/set_pc site whose source pair was pristine through its block
// and so was deferred to the Phase 4 dataflow re-classification.
struct PendingDataflowSite {
  uint64_t blockOffset = 0;
  uint64_t siteOffset = 0;
  unsigned srcPair = 0;
  bool isSwap = false;
};

// Pattern B set_pc site whose source ret-pair could not be resolved
// to a single chain in Phase 2; Phase 5 enumerates matching chain
// terminators and either classifies as IndirectB or refuses with the
// pair-no-call-site reason.
struct PendingB {
  uint64_t setpcOffset = 0;
  unsigned retPairLowReg = 0;
};

// Inter-block lattice value for one SGPR pair.
//   - `values` is the enumerated set of statically-known absolute
//     kernel offsets the pair can hold at the start of a block, sorted
//     ascending and capped at kMaxDispatchTargets.
//   - `incomplete` records that at least one CFG predecessor leaves
//     the pair in an unmodeled state (overwritten by a non-chain
//     operation, never constrained at all — pristine kernel entry —
//     or omitted from a predecessor's exit facts entirely).
//
// Use sites consume the lattice: incomplete OR cap-exceeded ⇒ refuse
// loudly; otherwise the value count picks DirectA (1) or DispatchSet
// (>1) classification.
//
// Lattice element absent from a block's entry-fact map is interpreted
// by use sites as the unconstrained default (incomplete=true, no
// values). The JOIN-over-predecessors formulation in Phase 3 only
// inserts an entry into the map when at least one predecessor's exit
// facts mention the pair; it then OR's in incomplete from any
// predecessor that DIDN'T mention the pair.
struct PcLatticeValue {
  llvm::SmallVector<uint64_t, 8> values;  // sorted, deduped
  bool incomplete = false;
};

bool operator==(const PcLatticeValue &a, const PcLatticeValue &b) {
  return a.incomplete == b.incomplete && a.values == b.values;
}
bool operator!=(const PcLatticeValue &a, const PcLatticeValue &b) {
  return !(a == b);
}

// Merge `src` into `dst` (set-union of values, OR of incomplete bits,
// hard cap at kMaxDispatchTargets values — over-cap promotes to
// incomplete and stops growing the value list).
void joinValue(PcLatticeValue &dst, const PcLatticeValue &src) {
  if (src.incomplete)
    dst.incomplete = true;
  for (uint64_t v : src.values) {
    auto it = std::lower_bound(dst.values.begin(), dst.values.end(), v);
    if (it != dst.values.end() && *it == v)
      continue;
    if (dst.values.size() >= kMaxDispatchTargets) {
      dst.incomplete = true;
      break;
    }
    dst.values.insert(it, v);
  }
}

// Compute the set of CFG successor block-offsets for the block whose
// last instruction is `lastInst`. `nextBlockOffset` is the offset of
// the next block in linear layout (used for fallthrough); pass
// `nextBlockExists = false` if `lastInst` is the kernel's final inst.
//
// The successor model is intentionally conservative for analysis
// safety:
//   * S_BRANCH: 1 successor (decoded target).
//   * S_CBRANCH_*: 2 successors (target + linear fallthrough).
//   * S_ENDPGM: 0 successors.
//   * S_SET_PC_I64: 0 successors. The classification table tells the
//     raiser where this jumps; for dataflow purposes any survivors
//     would have to flow through the destination's other predecessors
//     anyway.
//   * S_SWAP_PC_I64: 1 successor (linear fallthrough). The swap is a
//     branch-and-link; control eventually returns into the
//     fallthrough, which Phase 1 added to extraBlockStarts so it is a
//     known leader. Source-pair facts survive across the swap (the
//     swap reads but does not write the source pair); the dst pair
//     is killed by the swap's transfer.
//   * Anything else (block ended only because the next inst was an
//     external BB leader): linear fallthrough.
SmallVector<uint64_t, 2>
computeSuccessors(const DecodedInst &lastInst, uint64_t nextBlockOffset,
                  bool nextBlockExists) {
  SmallVector<uint64_t, 2> result;
  auto branchTargetFromImm =
      [&](unsigned opIdx) -> std::optional<uint64_t> {
    if (opIdx >= lastInst.inst.getNumOperands())
      return std::nullopt;
    const MCOperand &op = lastInst.inst.getOperand(opIdx);
    if (!op.isImm())
      return std::nullopt;
    int64_t raw = op.getImm();
    int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
    return lastInst.offset + 4 + brOff * 4;
  };
  switch (lastInst.semOp) {
  case SemOp::S_BRANCH: {
    auto t = branchTargetFromImm(0);
    if (t)
      result.push_back(*t);
    break;
  }
  case SemOp::S_CBRANCH_SCC0:
  case SemOp::S_CBRANCH_SCC1:
  case SemOp::S_CBRANCH_VCCZ:
  case SemOp::S_CBRANCH_VCCNZ:
  case SemOp::S_CBRANCH_EXECZ:
  case SemOp::S_CBRANCH_EXECNZ: {
    auto t = branchTargetFromImm(0);
    if (t)
      result.push_back(*t);
    if (nextBlockExists)
      result.push_back(nextBlockOffset);
    break;
  }
  case SemOp::S_ENDPGM:
  case SemOp::S_SET_PC_I64:
    break;
  case SemOp::S_SWAP_PC_I64:
  default:
    if (nextBlockExists)
      result.push_back(nextBlockOffset);
    break;
  }
  return result;
}

} // namespace

SetPcAnalysis analyseSetPC(ArrayRef<DecodedInst> insts,
                           const std::set<uint64_t> &blockStarts,
                           const MCState &mc) {
  SetPcAnalysis result;
  if (insts.empty())
    return result;

  const MCRegisterInfo &MRI = *mc.regInfo;

  // ---------------------------------------------------------------
  // Phase 1 — pre-pass: enumerate every swap/set_pc fallthrough as a
  // block leader. This guarantees the per-block walk sees each
  // swap/set_pc as the LAST inst of its block (so its block-exit
  // transfer correctly summarises the pair state up to the
  // swap/set_pc, not past it).
  // ---------------------------------------------------------------
  std::set<uint64_t> mergedBlockStarts(blockStarts.begin(), blockStarts.end());
  for (const DecodedInst &di : insts) {
    if (di.semOp == SemOp::S_SWAP_PC_I64 ||
        di.semOp == SemOp::S_SET_PC_I64) {
      uint64_t fallthrough = di.offset + di.size;
      if (mergedBlockStarts.insert(fallthrough).second)
        result.extraBlockStarts.insert(fallthrough);
    }
  }

  // ---------------------------------------------------------------
  // Build BlockData skeletons. A "block" is identified by the offset
  // of an instruction that appears in `insts` AND is in
  // `mergedBlockStarts`. We skip leader offsets that have no inst
  // (e.g. fallthroughs past the last decoded inst, or branch targets
  // outside the decoded range).
  // ---------------------------------------------------------------
  std::vector<BlockData> blocks;
  std::unordered_map<uint64_t, size_t> offsetToBlockIdx;
  blocks.reserve(mergedBlockStarts.size());
  for (size_t i = 0; i < insts.size(); ++i) {
    if (mergedBlockStarts.count(insts[i].offset)) {
      BlockData bd;
      bd.offset = insts[i].offset;
      bd.firstIdx = i;
      bd.lastIdx = i; // fixed up below
      offsetToBlockIdx[bd.offset] = blocks.size();
      blocks.push_back(bd);
    }
  }
  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    size_t end = (bi + 1 < blocks.size()) ? blocks[bi + 1].firstIdx
                                          : insts.size();
    blocks[bi].lastIdx = end - 1;
  }

  // Always treat the first instruction's block as kernel entry. Phase
  // 3 seeds that block's entry facts as "unconstrained" (the default
  // PcLatticeValue), which correctly models pristine register state.

  // ---------------------------------------------------------------
  // Phase 2 — per-block intra-block walk. For each block, run the
  // existing chain analysis and collect:
  //   * setpcSites entries for sites that resolve in-block (DirectA
  //     direct-branch, or Unresolvable-with-intra-block-reason when
  //     the source pair was dirtied without a complete chain).
  //   * pendingDataflow entries for sites whose source pair was
  //     pristine through the block — Phase 4 reclassifies these.
  //   * pendingB entries for s_set_pc sites whose source pair could
  //     not be resolved by a chain (intra OR dataflow) — Phase 5
  //     classifies these as IndirectB by matching against
  //     chainTerminators.
  //   * chainTerminators[high_add_offset] = {chain_value, lowReg} for
  //     every completed chain. Phase 5 prunes those that don't feed
  //     any classified site.
  //   * extraBlockStarts entries for DirectA chain targets and for
  //     swap_pc fallthroughs (the fallthrough leader was added in
  //     Phase 1 for analysis correctness; we re-record it here for
  //     the caller's BB-layout merge).
  //   * the per-block transfer summary (transfers map).
  //   * the per-block successor list (computeSuccessors on lastInst).
  // ---------------------------------------------------------------
  std::vector<PendingDataflowSite> pendingDataflow;
  std::vector<PendingB> pendingB;

  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    BlockData &bd = blocks[bi];
    State state(MRI);
    state.resetBlock();

    for (size_t i = bd.firstIdx; i <= bd.lastIdx; ++i) {
      const DecodedInst &di = insts[i];

      switch (di.semOp) {
      case SemOp::S_GETPC_B64: {
        if (di.numDefs >= 1 && di.isReg(0)) {
          auto idx = sgprIdx(MRI, di.getReg(0));
          if (idx) {
            state.recordPc(*idx, di.offset + di.size);
            continue;
          }
        }
        break;
      }

      case SemOp::S_ADD_U32: {
        if (di.numDefs < 1 || !di.isReg(0))
          break;
        auto dstIdx = sgprIdx(MRI, di.getReg(0));
        if (!dstIdx)
          break;
        unsigned s0 = di.firstSrcIdx;
        unsigned s1 = s0 + 1;
        auto src0Imm = imm32(di.inst, s0);
        auto src1Imm = imm32(di.inst, s1);
        if (src0Imm && src1Imm) {
          state.recordScalar(*dstIdx, *src0Imm + *src1Imm);
          continue;
        }
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
        std::optional<unsigned> src0Idx;
        if (di.isReg(s0))
          src0Idx = sgprIdx(MRI, di.getReg(s0));
        if (!src0Idx || *src0Idx != *dstIdx)
          break;
        auto src1Imm = imm32(di.inst, s1);
        if (!src1Imm)
          break;
        state.finishHighAdd(lowIdx, di.offset,
                            static_cast<uint64_t>(*src1Imm));
        result.chainTerminators[di.offset] =
            SetPcCallSiteInfo{state.findPc(lowIdx)->value, lowIdx};
        continue;
      }

      case SemOp::S_SWAP_PC_I64: {
        // Phase 1 already added the fallthrough to mergedBlockStarts;
        // re-record for the caller's BB-layout merge.
        result.extraBlockStarts.insert(di.offset + di.size);

        std::optional<unsigned> dstLow;
        if (di.numDefs >= 1 && di.isReg(0))
          dstLow = sgprIdx(MRI, di.getReg(0));

        // Synthetic chain-terminator registration. We always record
        // it when dstLow is known; Phase 5 drops it if no downstream
        // s_set_pc_i64 reads `sdst`. The key (di.offset) is unique
        // because LLVM never lays two instructions at the same
        // offset; the value carries (retPair=sdstLow, returnAddr=
        // swap.end). The raiser's S_ADDC_U32 post-hook does not fire
        // on S_SWAP_PC_I64 (gated by SemOp), so the equivalent
        // blockaddress materialisation happens inline in handleSOP1.
        if (dstLow)
          result.chainTerminators[di.offset] =
              SetPcCallSiteInfo{di.offset + di.size, *dstLow};

        // Classify the call-target (source pair).
        unsigned srcOpIdx = di.firstSrcIdx;
        if (!di.isReg(srcOpIdx)) {
          SetPcSiteInfo info;
          info.kind = SetPcSiteInfo::Kind::Unresolvable;
          info.refusalReason =
              "s_swap_pc_i64 source operand is not a register";
          result.setpcSites[di.offset] = std::move(info);
          if (dstLow) {
            state.invalidatePcAt(*dstLow);
            state.invalidatePcAt(*dstLow + 1);
          }
          continue;
        }
        auto srcLow = sgprIdx(MRI, di.getReg(srcOpIdx));
        if (!srcLow) {
          SetPcSiteInfo info;
          info.kind = SetPcSiteInfo::Kind::Unresolvable;
          info.refusalReason =
              "s_swap_pc_i64 source register is not an SGPR pair";
          result.setpcSites[di.offset] = std::move(info);
          if (dstLow) {
            state.invalidatePcAt(*dstLow);
            state.invalidatePcAt(*dstLow + 1);
          }
          continue;
        }
        PcChain *chain = state.findPc(*srcLow);
        if (chain && chain->lowAddDone) {
          // DirectA: chain resolves the absolute callee target intra-block.
          SetPcSiteInfo info;
          info.kind = SetPcSiteInfo::Kind::DirectA;
          info.directTarget = chain->value;
          result.setpcSites[di.offset] = std::move(info);
          result.extraBlockStarts.insert(chain->value);
          if (chain->terminator)
            result.chainTerminators.erase(chain->terminator);
          // The src pair is consumed inline. Mark it dirty so the
          // block transfer KILLs entry facts (a downstream block on
          // a back edge would otherwise see the chain value, but the
          // intra-block consumption invalidates further reasoning).
          state.invalidatePcAt(*srcLow);
        } else if (state.isPairDirty(*srcLow)) {
          // The block wrote to srcPair (chain-or-otherwise) but it
          // didn't end with a complete chain. Dataflow entry facts
          // are dead. Refuse loudly.
          SetPcSiteInfo info;
          info.kind = SetPcSiteInfo::Kind::Unresolvable;
          info.refusalReason =
              ("s_swap_pc_i64 source SGPR pair s[" +
               std::to_string(*srcLow) + ":" +
               std::to_string(*srcLow + 1) +
               "] was modified intra-block without producing a "
               "statically resolvable getpc+add chain (the block "
               "either started a chain that did not complete or "
               "overwrote the pair with a non-chain value); inter-"
               "block dataflow facts cannot recover this");
          result.setpcSites[di.offset] = std::move(info);
        } else {
          // SrcPair is pristine through the block. Defer to Phase 4
          // dataflow re-classification.
          PendingDataflowSite pds;
          pds.blockOffset = bd.offset;
          pds.siteOffset = di.offset;
          pds.srcPair = *srcLow;
          pds.isSwap = true;
          pendingDataflow.push_back(pds);
        }
        // Dst pair now holds an opaque (return-PC) value; remove from
        // PC tracking so a downstream s_set_pc_i64 reading dst falls
        // into Pattern B (indirectbr) rather than DirectA.
        if (dstLow) {
          state.invalidatePcAt(*dstLow);
          state.invalidatePcAt(*dstLow + 1);
        }
        continue;
      }

      case SemOp::S_SET_PC_I64: {
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
        result.extraBlockStarts.insert(di.offset + di.size);
        PcChain *chain = state.findPc(*srcIdx);
        if (chain && chain->lowAddDone) {
          // DirectA intra-block.
          SetPcSiteInfo info;
          info.kind = SetPcSiteInfo::Kind::DirectA;
          info.directTarget = chain->value;
          result.setpcSites[di.offset] = std::move(info);
          result.extraBlockStarts.insert(chain->value);
          if (chain->terminator)
            result.chainTerminators.erase(chain->terminator);
          state.invalidatePcAt(*srcIdx);
          continue;
        }
        if (state.isPairDirty(*srcIdx)) {
          // Pair was dirtied intra-block without a complete chain;
          // dataflow facts are dead. Defer to Phase 5 PendingB
          // classification (matches against chainTerminators) which
          // correctly handles the subroutine-return shape regardless
          // of intra-block dirtiness — the IndirectB pattern relies
          // on the source pair being populated by a CALLER's chain
          // terminator, not by anything in this block. If no chain
          // terminator matches this source pair, Phase 5 emits the
          // pair-no-call-site Unresolvable diagnostic.
          PendingB pb;
          pb.setpcOffset = di.offset;
          pb.retPairLowReg = *srcIdx;
          pendingB.push_back(pb);
          continue;
        }
        // SrcPair pristine through the block. Defer to Phase 4
        // dataflow re-classification. If dataflow leaves it
        // unconstrained, Phase 4 falls through to a PendingB-style
        // resolution attempt (so subroutine-return shapes still get
        // classified as IndirectB, not refused).
        PendingDataflowSite pds;
        pds.blockOffset = bd.offset;
        pds.siteOffset = di.offset;
        pds.srcPair = *srcIdx;
        pds.isSwap = false;
        pendingDataflow.push_back(pds);
        continue;
      }

      default:
        break;
      }

      // Generic fallthrough: invalidate every SGPR (and overlapping
      // PC pair) the instruction writes to. This prevents stale
      // chain values from leaking past instructions whose semantics
      // we did not model. Also drop in-progress chains so a stray
      // write between getpc and the low add cannot be silently
      // absorbed into the chain by a later matching add.
      invalidateGeneralSgprDefs(di, MRI, state);
      state.dropInProgressChains();
    }

    // Compute block-exit transfers from the final state.
    //   * Every pair in `state.pcChains()` with lowAddDone=true →
    //     SET(value, terminator). Pass-through is overridden.
    //   * Every dirty half whose pair is not in pcChains-with-
    //     lowAddDone → KILL of that pair. Cover BOTH the "low" view
    //     (half index treated as the pair's low) and the "high" view
    //     (half index - 1 treated as the pair's low). A dirty half
    //     can mean two distinct pairs were partially touched; we
    //     conservatively KILL both. (In the common case the second
    //     pair is unused and the KILL is harmless.)
    //   * Pairs with no entry in transfers default to PASS.
    auto setKill = [&](unsigned lowIdx) {
      auto &t = bd.transfers[lowIdx];
      if (t.kind != PairTransfer::Kind::Set)
        t.kind = PairTransfer::Kind::Kill;
    };
    for (const auto &kv : state.pcChains()) {
      if (kv.second.lowAddDone) {
        PairTransfer &t = bd.transfers[kv.first];
        t.kind = PairTransfer::Kind::Set;
        t.value = kv.second.value;
        t.terminator = kv.second.terminator;
      } else {
        setKill(kv.first);
        if (kv.first > 0)
          setKill(kv.first - 1);
      }
    }
    for (unsigned half : state.dirtyHalves()) {
      // Don't downgrade a SET pair to KILL.
      auto it = bd.transfers.find(half);
      if (it == bd.transfers.end() ||
          it->second.kind != PairTransfer::Kind::Set)
        setKill(half);
      if (half > 0) {
        auto it2 = bd.transfers.find(half - 1);
        if (it2 == bd.transfers.end() ||
            it2->second.kind != PairTransfer::Kind::Set)
          setKill(half - 1);
      }
    }

    // Compute CFG successors.
    bool hasNext = (bi + 1) < blocks.size();
    uint64_t nextOff = hasNext ? blocks[bi + 1].offset : 0;
    bd.successors = computeSuccessors(insts[bd.lastIdx], nextOff, hasNext);
  }

  // ---------------------------------------------------------------
  // Phase 3 — forward dataflow to fixpoint.
  //
  //   entryFacts[blockIdx][pairLow] = PcLatticeValue
  //
  // Formulation: at each block B,
  //   entryFacts[B] = JOIN over P ∈ preds(B) of exitFacts(P)
  //
  // where exitFacts(P) = transfer(entryFacts[P], P.transfers):
  //   - SET overrides any incoming entry with {value, !incomplete}
  //   - KILL overrides with {∅, incomplete}
  //   - PASS leaves the entry unchanged
  //
  // and JOIN is set-union of `values` + OR of `incomplete` bits, with
  // an additional rule: any pair P that appears in SOME predecessor
  // P_i's exit but is MISSING from P_j's exit gets incomplete=true
  // (the missing predecessor leaves P at its kernel-entry-pristine
  // default = unconstrained).
  //
  // Pairs absent from entryFacts[B] are interpreted by use sites as
  // the unconstrained default, so we only insert entries into the map
  // when at least one predecessor's exit mentions the pair.
  //
  // Convergence is guaranteed by the bounded-height lattice: per
  // pair, at most kMaxDispatchTargets values + 1 incomplete bit.
  // The lattice is also monotone (values only grow, incomplete only
  // 0→1), so JOIN-over-preds-from-scratch with re-entry-on-change is
  // a sound fixpoint algorithm.
  // ---------------------------------------------------------------

  // Build predecessor map.
  std::vector<llvm::SmallVector<size_t, 4>> predecessors(blocks.size());
  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    for (uint64_t succOff : blocks[bi].successors) {
      auto sit = offsetToBlockIdx.find(succOff);
      if (sit != offsetToBlockIdx.end())
        predecessors[sit->second].push_back(bi);
    }
  }

  // computeExit applies the per-block transfer to a given entry-fact
  // map and returns the exit-fact map. PASS pairs flow through; SET
  // and KILL pairs override.
  auto computeExit =
      [&](const std::unordered_map<unsigned, PcLatticeValue> &entry,
          const BlockData &bd) {
        std::unordered_map<unsigned, PcLatticeValue> exit = entry;
        for (const auto &kv : bd.transfers) {
          if (kv.second.kind == PairTransfer::Kind::Set) {
            PcLatticeValue v;
            v.values.push_back(kv.second.value);
            v.incomplete = false;
            exit[kv.first] = std::move(v);
          } else if (kv.second.kind == PairTransfer::Kind::Kill) {
            PcLatticeValue v;
            v.incomplete = true;
            exit[kv.first] = std::move(v);
          }
        }
        return exit;
      };

  std::vector<std::unordered_map<unsigned, PcLatticeValue>> entryFacts(
      blocks.size());

  std::deque<size_t> worklist;
  std::vector<bool> onWorklist(blocks.size(), false);
  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    worklist.push_back(bi);
    onWorklist[bi] = true;
  }

  while (!worklist.empty()) {
    size_t bi = worklist.front();
    worklist.pop_front();
    onWorklist[bi] = false;

    // Recompute entry from JOIN over predecessors.
    std::unordered_map<unsigned, PcLatticeValue> newEntry;
    if (!predecessors[bi].empty()) {
      // Collect predecessor exits.
      std::vector<std::unordered_map<unsigned, PcLatticeValue>> predExits;
      predExits.reserve(predecessors[bi].size());
      for (size_t pi : predecessors[bi])
        predExits.push_back(computeExit(entryFacts[pi], blocks[pi]));

      // Determine the union of pairs mentioned by any predecessor
      // exit. These are the only pairs whose entry fact differs from
      // the unconstrained default at this block.
      std::unordered_set<unsigned> mentioned;
      for (const auto &pe : predExits)
        for (const auto &kv : pe)
          mentioned.insert(kv.first);

      // For each mentioned pair, JOIN every predecessor's
      // contribution. Predecessors whose exit doesn't mention the
      // pair contribute incomplete=true (the kernel-entry-pristine
      // default).
      for (unsigned pair : mentioned) {
        PcLatticeValue acc;
        for (const auto &pe : predExits) {
          auto it = pe.find(pair);
          if (it == pe.end()) {
            acc.incomplete = true;
          } else {
            joinValue(acc, it->second);
          }
        }
        newEntry[pair] = std::move(acc);
      }
    }
    // Block 0 (kernel entry) has no predecessors → newEntry is empty,
    // which correctly represents "every pair is unconstrained".

    if (newEntry != entryFacts[bi]) {
      entryFacts[bi] = std::move(newEntry);
      for (uint64_t succOff : blocks[bi].successors) {
        auto sit = offsetToBlockIdx.find(succOff);
        if (sit != offsetToBlockIdx.end() &&
            !onWorklist[sit->second]) {
          worklist.push_back(sit->second);
          onWorklist[sit->second] = true;
        }
      }
    }
  }

  // ---------------------------------------------------------------
  // Phase 4 — re-classify deferred sites using dataflow facts.
  // ---------------------------------------------------------------
  for (const PendingDataflowSite &pds : pendingDataflow) {
    auto bit = offsetToBlockIdx.find(pds.blockOffset);
    if (bit == offsetToBlockIdx.end())
      continue;
    const auto &facts = entryFacts[bit->second];
    auto it = facts.find(pds.srcPair);
    bool resolved =
        (it != facts.end()) && !it->second.incomplete &&
        !it->second.values.empty() &&
        it->second.values.size() <= kMaxDispatchTargets;

    if (!resolved) {
      if (pds.isSwap) {
        SetPcSiteInfo info;
        info.kind = SetPcSiteInfo::Kind::Unresolvable;
        info.refusalReason =
            ("s_swap_pc_i64 source SGPR pair s[" +
             std::to_string(pds.srcPair) + ":" +
             std::to_string(pds.srcPair + 1) +
             "] does not have a statically resolvable getpc+add "
             "chain reaching this site (intra-block analysis found "
             "no chain; inter-block dataflow could not enumerate a "
             "bounded set of targets — the value comes from a "
             "kernarg/runtime source, an unbounded fan-in, or a "
             "control-flow path that overwrites the pair with an "
             "unmodelled value)");
        result.setpcSites[pds.siteOffset] = std::move(info);
      } else {
        // For s_set_pc_i64, fall through to PendingB — a subroutine-
        // return shape relies on caller-side chain terminators, not
        // on the source pair being constrained by intra-kernel
        // dataflow.
        PendingB pb;
        pb.setpcOffset = pds.siteOffset;
        pb.retPairLowReg = pds.srcPair;
        pendingB.push_back(pb);
      }
      continue;
    }

    SmallVector<uint64_t, 4> targets(it->second.values.begin(),
                                     it->second.values.end());
    SetPcSiteInfo info;
    if (targets.size() == 1) {
      info.kind = SetPcSiteInfo::Kind::DirectA;
      info.directTarget = targets[0];
    } else {
      info.kind = SetPcSiteInfo::Kind::DispatchSet;
      info.indirectTargets = targets;
      info.indirectRetPairLowReg = pds.srcPair;
    }
    for (uint64_t t : targets)
      result.extraBlockStarts.insert(t);
    result.setpcSites[pds.siteOffset] = std::move(info);
  }

  // ---------------------------------------------------------------
  // Phase 5 — chain terminator retention + IndirectB classification.
  //
  // Retention rule: keep a chain terminator iff it feeds either
  //   (a) an IndirectB ret-pair (pair-low-index match), OR
  //   (b) a DispatchSet site (pair-low-index AND value match).
  // Drop unused terminators so the raiser's S_ADDC_U32 hook does not
  // gratuitously rewrite chains that don't feed any classified site.
  // ---------------------------------------------------------------

  // Collect Pattern B consumers.
  std::set<unsigned> retPairsConsumedByB;
  for (const PendingB &pb : pendingB)
    retPairsConsumedByB.insert(pb.retPairLowReg);

  // Collect DispatchSet consumers (pair → set of allowed values).
  std::map<unsigned, std::set<uint64_t>> dispatchSetTargets;
  for (const auto &kv : result.setpcSites) {
    if (kv.second.kind == SetPcSiteInfo::Kind::DispatchSet) {
      auto &set = dispatchSetTargets[kv.second.indirectRetPairLowReg];
      for (uint64_t t : kv.second.indirectTargets)
        set.insert(t);
    }
  }

  // Prune chain terminators.
  for (auto it = result.chainTerminators.begin();
       it != result.chainTerminators.end();) {
    bool keepForB = retPairsConsumedByB.count(it->second.retPairLowReg);
    bool keepForDispatch = false;
    auto dt = dispatchSetTargets.find(it->second.retPairLowReg);
    if (dt != dispatchSetTargets.end() &&
        dt->second.count(it->second.resolvedReturnAddr))
      keepForDispatch = true;
    if (!keepForB && !keepForDispatch)
      it = result.chainTerminators.erase(it);
    else
      ++it;
  }

  // Build per-pair return-target lists for IndirectB from the
  // surviving terminators.
  std::unordered_map<unsigned, SmallVector<uint64_t, 4>> targetsByPair;
  for (const auto &kv : result.chainTerminators) {
    if (!retPairsConsumedByB.count(kv.second.retPairLowReg))
      continue;
    targetsByPair[kv.second.retPairLowReg].push_back(
        kv.second.resolvedReturnAddr);
    result.extraBlockStarts.insert(kv.second.resolvedReturnAddr);
  }
  for (auto &kv : targetsByPair) {
    auto &v = kv.second;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }

  // Classify PendingB sites.
  for (const PendingB &pb : pendingB) {
    if (result.setpcSites.count(pb.setpcOffset))
      continue;  // already classified by Phase 4 (e.g. DispatchSet)
    auto it = targetsByPair.find(pb.retPairLowReg);
    if (it == targetsByPair.end() || it->second.empty()) {
      SetPcSiteInfo info;
      info.kind = SetPcSiteInfo::Kind::Unresolvable;
      info.refusalReason =
          ("s_set_pc_i64 reads SGPR pair s[" +
           std::to_string(pb.retPairLowReg) + ":" +
           std::to_string(pb.retPairLowReg + 1) +
           "] but no statically resolvable call-site getpc+add chain "
           "targets that pair (and inter-block dataflow could not "
           "enumerate a bounded target set either)");
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
