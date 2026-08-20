// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/indirect_branch_discovery.h"

#include "rocjitsu/analysis/control_flow.h"
#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

// Static indirect branch recovery has to answer a narrow CFG question:
// for each s_setpc_b64 or s_swappc_b64, can we prove the source SGPR pair
// contains a concrete text offset built from s_getpc_b64? If yes, BasicBlock
// can model that target as either an ordinary successor or a context-sensitive
// call edge. If no, the safest answer is silence: do not add a guessed edge,
// and let later translation diagnostics deal with any still-unhandled indirect
// branch.
//
// The pass is split into four phases.
//
// Phase 1 - Decode cheap instruction facts and build an analysis block graph.
// The graph contains only direct CFG edges: direct branches, direct calls, and
// ordinary fallthrough. Recovered indirect edges are intentionally absent at
// this point, otherwise the analysis would be using the result it is trying to
// prove. The caller-provided extra leaders are included because kernel entry
// boundaries are real CFG cut points for later BasicBlock construction.
//
// Phase 2 - Scan each analysis block once. The local state tracks only SGPR
// pairs that currently hold a PC builder. At the end of the block, the scan
// produces a per-pair transfer summary:
//   * SET(value): this block leaves the pair holding a complete concrete PC.
//   * KILL: this block writes the pair in a way the analysis does not model.
//   * PASS: this block did not touch the pair, so incoming facts flow through.
// Consumers that resolve inside their own block emit fixups immediately.
// Consumers whose source pair is pristine in the block are deferred to Phase 4,
// because their value, if any, must come from predecessors.
//
// Phase 3 - Run bounded forward dataflow over block summaries. The lattice is:
//
//   map<sgpr_pair_low, {set<PcValue>, incomplete, killed}>
//
// `incomplete` means at least one predecessor path is unconstrained, killed, or
// over the target cap. `killed` records the specific case where a predecessor
// reached this point after an unmodeled write to the pair. Concrete values are
// still useful when incompleteness came from path-insensitive CFG joins:
// generated kernels often build a small return-address set in a dispatcher and
// then jump into a shared body, but the syntactic CFG can also contain infeasible
// paths into that body. We therefore emit bounded concrete values when only
// incomplete=true, but fail closed when killed=true or when the value set is
// saturated because either case may hide the real branch target.
//
// Phase 4 - Revisit deferred consumers and emit fixups when their block entry
// fact contains a bounded concrete target set. Multiple concrete values are
// allowed up to the cap. BasicBlock will decide whether each recovered target
// is a CFG successor or a call edge.
//
// The four phases run to a small fixed point. The first round uses only direct
// CFG edges. Later rounds add already-proven recovered edges to the temporary
// graph, then rerun the same transfer/dataflow formulation. This is needed for
// nested helper code such as "build return PC A; branch to helper; helper
// setpc A; later setpc B", where discovering the first setpc edge exposes the
// path that proves the second one. If a round has no pending inter-block
// consumers, dataflow is skipped entirely because local block scanning already
// found everything that can be found in that graph.
//
// Important invariants:
//   * PC-builder facts do not cross an analysis block by carrying local state.
//     Cross-block propagation exists only through PairTransfer and the lattice.
//   * Any unrecognized write to either half of a relevant SGPR pair is a KILL.
//   * Direct s_call_b64 is a call boundary for this analysis. The temporary CFG
//     keeps both the callee edge and the fallthrough continuation edge so
//     reachability is not lost, but register effects from the callee are not
//     modeled interprocedurally. Therefore every carried PC-builder fact is
//     killed at a direct call instead of being allowed to flow straight into the
//     continuation.
//   * s_setpc_b64/s_swappc_b64 read their source pair; they do not destroy it.
//     A real kernel may build one callee address once and call it multiple
//     times. Only the swappc destination pair is killed because it receives a
//     return PC, not an editable target builder.
//   * Return PCs from s_call/s_swappc are not modeled as branch targets. They
//     are hardware return addresses, and treating them as normal getpc builders
//     would create edges that can jump across unrelated kernel regions.

/// @brief Maximum number of concrete targets we will enumerate for one consumer.
///
/// @details The analysis is intentionally a finite, bounded dataflow. Once a
/// single SGPR pair can hold more than this many distinct static PC values at a
/// consumer, the value is no longer a small compiler-emitted dispatch set from
/// the DBT's point of view. We mark the fact incomplete and refuse to emit that
/// saturated partial set rather than creating an over-approximate edge set that
/// may connect unrelated regions.
constexpr size_t kMaxIndirectTargetsPerConsumer = 16;

/// @brief AMDGPU source-operand selector for the inline integer value 0.
///
/// @details SOP2 scalar source fields use the shared AMDGPU inline-constant
/// encoding where selector 128 represents integer 0 and each following selector
/// increments the integer value by one. The CDNA/RDNA manuals checked for this
/// pass all use the same selector table, so these are target-independent operand
/// selector values for the AMDGPU ISA families analyzed here.
constexpr uint16_t kInlineInt0 = 128;

/// @brief AMDGPU source-operand selector for the inline integer value 4.
///
/// @details This is `kInlineInt0 + 4`. The PC-delta recovery patterns use it to
/// recognize compiler-emitted `literal + 4` address builders without adding a
/// general scalar constant-propagation pass.
constexpr uint16_t kInlineInt4 = kInlineInt0 + 4;

/// @brief Maximum fixed-point rounds for nested recovered branches.
///
/// @details Most generated code needs one round: a dispatcher builds a concrete
/// PC and immediately reaches a setpc/swappc consumer through direct CFG edges.
/// Some kernels nest that pattern by returning from one recovered setpc into a
/// second region that later returns through another saved PC pair. Each round
/// adds the newly recovered edges to the temporary analysis graph and can expose
/// the next nesting level. The cap keeps malformed or extremely cyclic inputs
/// from turning CFG discovery into an unbounded compile-time search; returning
/// the edges already proven is still conservative.
constexpr size_t kMaxIndirectDiscoveryIterations = 8;
constexpr uint16_t kMaxTrackedSgprPair = static_cast<uint16_t>(REGISTER_SET_MAX_SGPRS - 1);

enum class ScalarPcOp {
  GetPc64,
  SetPc64,
  SwapPc64,
};

/// A scalar register pair accepted by a canonical scalar PC operation.
/// selector preserves the encoded SOP1 operand for relocation, while ref is
/// the decoded architectural identity used for proofs.
struct ScalarPcCarrier {
  uint16_t selector = 0;
  RegisterRef ref{};
};

[[nodiscard]] std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word);

/// @brief SOP2 arithmetic opcodes this pass knows how to interpret.
///
/// @details We do not need full scalar ALU semantics. These are only the
/// arithmetic forms observed in PC materialization chains:
///   s_getpc_b64 pair
///   s_add/sub/add_i32 pair_lo, pair_lo, literal
///   s_addc/subb pair_hi, pair_hi, 0-or-sign-carry
/// If the pair is edited by any other instruction, the generic SGPR-write path
/// kills the fact.
enum class ScalarSop2Op {
  AddU32,
  SubU32,
  AddI32,
  AddcU32,
  SubbU32,
  AddNcU64,
};

/// @brief Concrete PC-builder value carried by one SGPR pair.
///
/// @details The value is a byte offset inside the current text section. The
/// source fields are not needed for CFG construction itself; they are preserved
/// so the translator can later relocate the original getpc/add instruction
/// range that materialized this address.
struct PcValue {
  int64_t offset = 0;
  uint64_t source_getpc_offset = 0;
  uint64_t source_recovery_begin_offset = 0;
  uint64_t source_recovery_end_offset = 0;
  /// @brief False once a non-chain instruction was observed inside the recovery
  /// range. patch_recovered_builder_fixups NOPs the whole
  /// [begin, end) interval as one contiguous run, so a gap instruction between
  /// two builder steps would be erased. A value that stops being contiguous can
  /// never regain the property, so any later delta step keeps it false.
  bool contiguous = true;
  /// @brief True between a split low `s_add_u32` and the `s_addc_u32` that closes it.
  ///
  /// @details The high-half step only advances the recovery range; it never changes @ref offset,
  /// so a half-built chain is numerically indistinguishable from a finished one. Publishing the
  /// half-built state would let the patcher regenerate `[begin, end)` -- which stops before the
  /// carry -- and leave that `s_addc_u32` to apply a stale SCC to the freshly written high half,
  /// because the gfx1250 replacement is the SCC-neutral `s_add_nc_u64`. The single-instruction
  /// `s_add_nc_u64` form and the temp-delta patterns that already absorb their own carry never set
  /// this.
  bool pending_high_carry = false;

  friend bool operator==(const PcValue &, const PcValue &) = default;
};

struct VgprLane {
  uint16_t vgpr = 0;
  uint8_t lane = 0;

  friend bool operator==(const VgprLane &, const VgprLane &) = default;
};

struct FixedLaneTransfer {
  enum class Kind { Write, Read };

  Kind kind = Kind::Write;
  VgprLane lane;
  uint16_t sgpr = 0;
};

struct TempDeltaPattern {
  int64_t delta = 0;
  uint64_t end_offset = 0;
  size_t instruction_count = 0;
};

/// @brief Compact SGPR-only write mask for the indirect-PC recovery pass.
///
/// @details This analysis only needs to know which scalar general-purpose
/// registers an unmodeled instruction writes. Storing that local fact in a full
/// RegisterSet causes recovery performance regressions because every
/// invalidation scans SGPR, VGPR, and AccVGPR bitsets even though vector classes
/// are irrelevant here. Keep two words instead and iterate only set SGPR bits.
struct SgprWriteMask {
  uint64_t lo = 0;
  uint64_t hi = 0;

  void set(uint16_t sgpr) {
    if (sgpr < 64) {
      lo |= uint64_t{1} << sgpr;
    } else if (sgpr < REGISTER_SET_MAX_SGPRS) {
      hi |= uint64_t{1} << (sgpr - 64);
    }
  }

  void expand(RegisterRef ref) {
    if (ref.cls != RegClass::SGPR)
      return;
    const uint16_t width = std::max<uint16_t>(1, ref.width);
    for (uint16_t i = 0; i < width; ++i)
      set(static_cast<uint16_t>(ref.index + i));
  }

  [[nodiscard]] bool contains(uint16_t sgpr) const {
    if (sgpr < 64)
      return (lo & (uint64_t{1} << sgpr)) != 0;
    if (sgpr < REGISTER_SET_MAX_SGPRS)
      return (hi & (uint64_t{1} << (sgpr - 64))) != 0;
    return false;
  }

  [[nodiscard]] bool test(uint16_t sgpr) const { return contains(sgpr); }

  SgprWriteMask &operator|=(const SgprWriteMask &other) {
    lo |= other.lo;
    hi |= other.hi;
    return *this;
  }

  template <typename F> void for_each(F &&f) const {
    uint64_t bits = lo;
    while (bits != 0) {
      const auto sgpr = static_cast<uint16_t>(std::countr_zero(bits));
      f(sgpr);
      bits &= bits - 1;
    }

    bits = hi;
    while (bits != 0) {
      const auto sgpr = static_cast<uint16_t>(64 + std::countr_zero(bits));
      f(sgpr);
      bits &= bits - 1;
    }
  }
};

struct CalleeSummary {
  SgprWriteMask sgprs;
  std::bitset<REGISTER_SET_MAX_VGPRS> vgprs;
  std::optional<uint8_t> return_mode;
  std::optional<bool> return_gpr_idx_enabled;

  CalleeSummary &operator|=(const CalleeSummary &other) {
    sgprs |= other.sgprs;
    vgprs |= other.vgprs;
    if (return_mode != other.return_mode)
      return_mode = std::nullopt;
    if (return_gpr_idx_enabled != other.return_gpr_idx_enabled)
      return_gpr_idx_enabled = std::nullopt;
    return *this;
  }
};

struct CalleeSummaryCacheKey {
  uint64_t target = 0;
  uint16_t return_pair = 0;
  std::optional<uint8_t> mode;
  std::optional<bool> gpr_idx_enabled;

  friend bool operator==(const CalleeSummaryCacheKey &, const CalleeSummaryCacheKey &) = default;
};

struct CalleeSummaryCacheKeyHash {
  [[nodiscard]] size_t operator()(const CalleeSummaryCacheKey &key) const {
    size_t hash = std::hash<uint64_t>{}(key.target);
    hash ^= std::hash<uint16_t>{}(key.return_pair) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    hash ^= std::hash<std::optional<uint8_t>>{}(key.mode) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    hash ^= std::hash<std::optional<bool>>{}(key.gpr_idx_enabled) + 0x9e3779b9u + (hash << 6) +
            (hash >> 2);
    return hash;
  }
};

struct CalleeSummaryGroupKey {
  uint64_t target = 0;
  uint16_t return_pair = 0;

  friend bool operator==(const CalleeSummaryGroupKey &, const CalleeSummaryGroupKey &) = default;
};

struct CalleeSummaryGroupKeyHash {
  [[nodiscard]] size_t operator()(const CalleeSummaryGroupKey &key) const {
    size_t hash = std::hash<uint64_t>{}(key.target);
    hash ^= std::hash<uint16_t>{}(key.return_pair) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    return hash;
  }
};

/// @brief Cached per-instruction facts used by the analysis.
///
/// @details Decoding instruction operands can be expensive on large code
/// objects, so the pass separates cheap raw-word recognition from lazy
/// destination-register extraction. The SOP1/SOP2 fields are identified from
/// encoded words because their layouts are stable for the forms we care about.
/// Generic SGPR writes are computed only when a local block scan reaches an
/// instruction whose unmodeled writes could kill an active or dirty pair.
struct InstructionFacts {
  uint32_t word = 0;
  std::optional<uint16_t> getpc_sdst;
  std::optional<uint16_t> setpc_ssrc;
  std::optional<uint16_t> swappc_ssrc;
  std::optional<uint16_t> swappc_sdst;
  std::optional<uint16_t> call_sdst;
  SgprWriteMask written_sgprs;
  bool written_sgprs_computed = false;
};

[[nodiscard]] bool is_lane_fixup_consumer(const InstructionFacts &facts) {
  return facts.swappc_ssrc.has_value();
}

struct AnalysisContext {
  std::span<const Instruction *const> insts;
  std::span<const uint8_t> text;
  rj_code_arch_t arch;
  uint32_t wavefront_size = 0;
  std::vector<InstructionFacts> facts;

  // Whole-text superset of every SGPR pair that a deferred cross-block
  // consumer can name. Every PendingConsumer producer must originate from one
  // of the setpc/swappc operands recorded here.
  std::bitset<REGISTER_SET_MAX_SGPRS> consumer_pairs;
};

/// @brief Per-pair summary for one analysis block.
///
/// @details Blocks are scanned only once. The dataflow phase does not re-run
/// instruction semantics; it applies this compact summary to incoming facts:
/// Pass leaves an incoming pair unchanged, Set overwrites it with a known
/// builder value, and Kill turns it into an incomplete fact. Kill is used for
/// every write to either half of the pair that we did not model as a PC-builder
/// update.
struct PairTransfer {
  enum class Kind {
    Pass,
    Set,
    Kill,
  };

  Kind kind = Kind::Pass;
  PcValue value;
};

/// @brief One PC-relative address producer while a discovery round is running.
///
/// @details `poisoned` is sticky. Once a producer is observed in a way that no
/// single delta rewrite can repair, no later observation may resurrect it.
struct PcAddressBuilderEntry {
  PcAddressBuilder record;
  bool poisoned = false;
};

/// @brief Accumulator for every PC-relative address producer seen in one round.
///
/// @details Keyed by the producer's `s_getpc_b64` source offset so a getpc that
/// is observed several times (at its consumer, at a call that clobbers it, and
/// again at block exit) collapses to one record. Two observations that disagree
/// cannot both be satisfied by one delta rewrite, so a disagreement poisons the
/// record instead of picking one.
using PcAddressBuilderMap = std::unordered_map<uint64_t, PcAddressBuilderEntry>;

void seed_pc_builder(PcAddressBuilderMap &builders, uint64_t getpc_offset, uint16_t pair_lo) {
  // Every s_getpc_b64 is recorded even when nothing can be proven about it. A
  // whole-scope "no stale PC values" claim must account for the producers the
  // pass failed to follow, not silently omit them.
  builders.try_emplace(getpc_offset,
                       PcAddressBuilderEntry{.record = {.source_getpc_offset = getpc_offset,
                                                        .source_sreg = pair_lo,
                                                        .resolved = false}});
}

void poison_pc_builder(PcAddressBuilderMap &builders, uint64_t getpc_offset) {
  auto it = builders.find(getpc_offset);
  if (it == builders.end())
    return;
  it->second.poisoned = true;
  it->second.record.resolved = false;
}

/// @brief Record the value a builder leaves in its pair at a stable program point.
///
/// @details A stable point is one where the pair stops being tracked: the block
/// exit, a call that clobbers it, or the consumer that reads it. The recorded
/// value is exactly what the original builder range produces there, which is the
/// precondition for rewriting that range to produce the relocated address.
void note_pc_builder(PcAddressBuilderMap &builders, uint16_t pair_lo, const PcValue &value) {
  const PcAddressBuilder record{
      .source_getpc_offset = value.source_getpc_offset,
      .source_recovery_begin_offset = value.source_recovery_begin_offset,
      .source_recovery_end_offset = value.source_recovery_end_offset,
      .source_target_offset = value.offset,
      .source_sreg = pair_lo,
      .resolved = true,
      .contiguous = value.contiguous,
  };

  auto it = builders.find(value.source_getpc_offset);
  if (it == builders.end()) {
    builders.emplace(value.source_getpc_offset, PcAddressBuilderEntry{.record = record});
    return;
  }
  if (it->second.poisoned)
    return;
  if (it->second.record.resolved && it->second.record != record) {
    it->second.poisoned = true;
    it->second.record.resolved = false;
    return;
  }
  it->second.record = record;
}

struct AnalysisBlock {
  /// Byte offset of the first instruction in this temporary analysis block.
  uint64_t offset = 0;

  /// Inclusive instruction-index range in AnalysisContext::insts.
  size_t first_index = 0;
  size_t last_index = 0;

  /// Sparse transfer summary. Pairs not present here are implicit PASS.
  std::unordered_map<uint16_t, PairTransfer> transfers;

  /// Direct-CFG successors by AnalysisBlock index. Recovered indirect edges are
  /// not added here; they are outputs of this analysis, not inputs.
  std::vector<size_t> successors;

  /// Section start or caller-declared entry; recovered leaders are not roots.
  bool external_entry = false;
};

/// @brief A setpc/swappc consumer that must be resolved from block-entry facts.
///
/// @details During the intra-block scan, if the source pair is pristine in the
/// current block, the block cannot prove or disprove the value locally. The
/// consumer is recorded here and classified after Phase 3 dataflow has computed
/// the facts that reach the block entry.
struct PendingConsumer {
  size_t block_index = 0;
  size_t inst_index = 0;
  uint16_t pair_lo = 0;
};

/// @brief Lattice value at a block entry for one SGPR pair.
///
/// @details `values` is the bounded set of concrete PC-builder values the pair
/// may hold. `incomplete` means at least one path reaches this point with an
/// untracked value: the pair came from kernel-entry state, the target set
/// exceeded the cap, or the pair was killed. `killed` distinguishes that last
/// case because a concrete value from another predecessor does not prove the
/// consumer is safe when a real unmodeled write also reaches it.
struct LatticeValue {
  std::vector<PcValue> values;
  bool incomplete = false;
  bool killed = false;

  friend bool operator==(const LatticeValue &, const LatticeValue &) = default;
};

/// @brief Compact sorted block-entry facts keyed by the low SGPR of a pair.
///
/// @details The key domain is bounded by the architectural SGPR count and the
/// dataflow constructs entries in ascending key order. Keeping the sparse facts
/// contiguous avoids one allocation per key plus one bucket array per block,
/// which is especially expensive for generated objects with millions of
/// analysis blocks.
class LatticeFacts {
  using Entry = std::pair<uint16_t, LatticeValue>;

public:
  void reserve(size_t size) { entries_.reserve(size); }

  void append(uint16_t pair_lo, LatticeValue value) {
    assert(entries_.empty() || entries_.back().first < pair_lo);
    entries_.emplace_back(pair_lo, std::move(value));
  }

  [[nodiscard]] const LatticeValue *find(uint16_t pair_lo) const {
    const auto it =
        std::lower_bound(entries_.begin(), entries_.end(), pair_lo,
                         [](const Entry &entry, uint16_t key) { return entry.first < key; });
    return it != entries_.end() && it->first == pair_lo ? &it->second : nullptr;
  }

  friend bool operator==(const LatticeFacts &, const LatticeFacts &) = default;

private:
  std::vector<Entry> entries_;
};

/// @brief Mutable symbolic state for one straight-line analysis block.
///
/// @details This state is deliberately reset at every analysis block boundary.
/// Local instruction semantics are handled here; cross-block propagation is
/// handled only by the finite lattice above. This separation is what prevents
/// the analysis from re-walking large regions once for every s_getpc seed.
class BlockState {
public:
  void set_builder(uint16_t pair_lo, PcValue value) {
    if (pair_lo >= kMaxTrackedSgprPair)
      return;
    invalidate_half(pair_lo, pair_lo);
    invalidate_half(static_cast<uint16_t>(pair_lo + 1), pair_lo);
    mark_dirty(pair_lo);
    mark_dirty(static_cast<uint16_t>(pair_lo + 1));
    if (!builders_[pair_lo])
      active_pairs_.push_back(pair_lo);
    builders_[pair_lo] = value;
  }

  [[nodiscard]] PcValue *builder(uint16_t pair_lo) {
    if (pair_lo >= builders_.size() || !builders_[pair_lo])
      return nullptr;
    return &*builders_[pair_lo];
  }

  [[nodiscard]] const PcValue *builder(uint16_t pair_lo) const {
    if (pair_lo >= builders_.size() || !builders_[pair_lo])
      return nullptr;
    return &*builders_[pair_lo];
  }

  [[nodiscard]] const std::vector<uint16_t> &active_pairs() const { return active_pairs_; }

  [[nodiscard]] bool pair_dirty(uint16_t pair_lo) const {
    return dirty(pair_lo) || dirty(static_cast<uint16_t>(pair_lo + 1));
  }

  [[nodiscard]] bool dirty(uint16_t sgpr) const {
    if (sgpr < 64)
      return (dirty_lo_ & (uint64_t{1} << sgpr)) != 0;
    if (sgpr < REGISTER_SET_MAX_SGPRS)
      return (dirty_hi_ & (uint64_t{1} << (sgpr - 64))) != 0;
    return false;
  }

  /// @brief Visit dirty SGPR halves in ascending register order.
  template <typename F> void for_each_dirty(F &&f) const {
    uint64_t bits = dirty_lo_;
    while (bits != 0) {
      f(static_cast<uint16_t>(std::countr_zero(bits)));
      bits &= bits - 1;
    }

    bits = dirty_hi_;
    while (bits != 0) {
      f(static_cast<uint16_t>(64 + std::countr_zero(bits)));
      bits &= bits - 1;
    }
  }

  /// @brief Invalidate every builder overlapping @p sgpr.
  ///
  /// @details A write to sN can corrupt the pair s[N:N+1] when sN is the low
  /// half, or s[N-1:N] when sN is the high half. We kill both interpretations
  /// because the consumer operand only tells us a pair low register later.
  void invalidate_half(uint16_t sgpr, std::optional<uint16_t> protected_pair = std::nullopt) {
    mark_dirty(sgpr);
    if (sgpr >= builders_.size()) {
      return;
    } else if (protected_pair && *protected_pair == sgpr) {
      // This write is the modeled low-half update for the protected builder.
    } else {
      builders_[sgpr].reset();
    }

    if (sgpr == 0)
      return;
    const uint16_t previous_pair = static_cast<uint16_t>(sgpr - 1);
    if (!protected_pair || *protected_pair != previous_pair)
      builders_[previous_pair].reset();
  }

  void invalidate_pair(uint16_t pair_lo) {
    invalidate_half(pair_lo);
    invalidate_half(static_cast<uint16_t>(pair_lo + 1));
  }

private:
  void mark_dirty(uint16_t sgpr) {
    if (sgpr < 64) {
      dirty_lo_ |= uint64_t{1} << sgpr;
    } else if (sgpr < REGISTER_SET_MAX_SGPRS) {
      dirty_hi_ |= uint64_t{1} << (sgpr - 64);
    }
  }

  std::array<std::optional<PcValue>, REGISTER_SET_MAX_SGPRS> builders_;
  std::vector<uint16_t> active_pairs_;
  static_assert(REGISTER_SET_MAX_SGPRS <= 128, "dirty set uses two 64-bit words");
  uint64_t dirty_lo_ = 0;
  uint64_t dirty_hi_ = 0;
};

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  // The decoder has already produced Instruction objects, but several scalar
  // PC idioms are easier and cheaper to recognize from the encoded word. Return
  // zero for out-of-range literal reads; the surrounding matcher will then fail
  // naturally instead of needing a separate bounds status.
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] std::optional<FixedLaneTransfer> fixed_lane_transfer(const Instruction &inst,
                                                                   uint32_t wavefront_size) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic != "v_writelane_b32" && mnemonic != "v_readlane_b32")
    return std::nullopt;
  if (inst.num_dst_operands() < 1 || inst.num_src_operands() < 2)
    return std::nullopt;

  // Use the encoded selector so a literal whose value resembles an inline
  // lane never gains fixed-lane provenance.
  if (inst.raw_encoding() == nullptr || static_cast<size_t>(inst.size()) < 2u * sizeof(uint32_t))
    return std::nullopt;
  constexpr uint32_t kVop3Src1Shift = 9u;
  constexpr uint32_t kVop3SourceMask = 0x1ffu;
  const uint32_t selector = (inst.raw_encoding()[1] >> kVop3Src1Shift) & kVop3SourceMask;
  if (selector < kInlineInt0 || selector >= kInlineInt0 + 64u)
    return std::nullopt;
  const uint32_t lane = selector - kInlineInt0;
  if (wavefront_size != 0 && lane >= wavefront_size)
    return std::nullopt;

  const auto register_ref = [](const Operand *operand, RegClass cls) {
    if (operand == nullptr)
      return std::optional<RegisterRef>{};
    auto ref = operand->to_register_ref();
    if (!ref || ref->cls != cls || ref->width != 1)
      return std::optional<RegisterRef>{};
    return ref;
  };

  if (mnemonic == "v_writelane_b32") {
    const auto vgpr = register_ref(inst.dst_operand(0), RegClass::VGPR);
    std::optional<RegisterRef> sgpr;
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      const auto candidate = register_ref(inst.src_operand(i), RegClass::SGPR);
      if (!candidate)
        continue;
      if (sgpr)
        return std::nullopt;
      sgpr = candidate;
    }
    if (!vgpr || !sgpr)
      return std::nullopt;
    return FixedLaneTransfer{
        .kind = FixedLaneTransfer::Kind::Write,
        .lane = VgprLane{.vgpr = vgpr->index, .lane = static_cast<uint8_t>(lane)},
        .sgpr = sgpr->index,
    };
  }

  const auto sgpr = register_ref(inst.dst_operand(0), RegClass::SGPR);
  const auto vgpr = register_ref(inst.src_operand(0), RegClass::VGPR);
  if (!vgpr || !sgpr)
    return std::nullopt;
  return FixedLaneTransfer{
      .kind = FixedLaneTransfer::Kind::Read,
      .lane = VgprLane{.vgpr = vgpr->index, .lane = static_cast<uint8_t>(lane)},
      .sgpr = sgpr->index,
  };
}

[[nodiscard]] std::optional<uint8_t> scalar_pc_opcode(rj_code_arch_t arch, ScalarPcOp op) {
  // s_getpc/setpc/swappc are adjacent SOP1 opcodes within each AMDGPU ISA
  // family currently supported by rocjitsu. Keep this mapping local to the
  // analysis because it is an instruction-recognition detail, not semantic
  // lowering logic.
  auto add_base = [&](uint8_t base) -> uint8_t {
    switch (op) {
    case ScalarPcOp::GetPc64:
      return base;
    case ScalarPcOp::SetPc64:
      return static_cast<uint8_t>(base + 1);
    case ScalarPcOp::SwapPc64:
      return static_cast<uint8_t>(base + 2);
    }
    return base;
  };

  // \NPI new ISA family: classify its scalar PC instruction encodings here.
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return add_base(0x1c);
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return add_base(0x1f);
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return add_base(0x47);
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint8_t> scalar_sop2_opcode(rj_code_arch_t arch, ScalarSop2Op op) {
  // \NPI new ISA family: classify its scalar SOP2 opcode mapping here.
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    switch (op) {
    case ScalarSop2Op::AddU32:
      return 0;
    case ScalarSop2Op::SubU32:
      return 1;
    case ScalarSop2Op::AddI32:
      return 2;
    case ScalarSop2Op::AddcU32:
      return 4;
    case ScalarSop2Op::SubbU32:
      return 5;
    case ScalarSop2Op::AddNcU64:
      return std::nullopt;
    }
    return std::nullopt;
  case ROCJITSU_CODE_ARCH_GFX1250:
    switch (op) {
    case ScalarSop2Op::AddU32:
      return 0;
    case ScalarSop2Op::SubU32:
      return 1;
    case ScalarSop2Op::AddI32:
      return 2;
    case ScalarSop2Op::AddcU32:
      return 4;
    case ScalarSop2Op::SubbU32:
      return 5;
    case ScalarSop2Op::AddNcU64:
      return 83;
    }
    return std::nullopt;
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<RegisterRef> scalar_register_pair_ref(const Operand &operand) {
  if (operand.size_bits() != 64 || !operand.is_register())
    return std::nullopt;
  auto ref = operand.to_register_ref();
  return ref && ref->width == 2 ? ref : std::nullopt;
}

[[nodiscard]] bool is_supported_special_pc_carrier(RegisterRef ref) {
  if (ref.cls == RegClass::VCC)
    return ref.index == 0 && ref.width == 2;
  if (ref.cls == RegClass::TTMP)
    return ref.width == 2 && ref.index % 2 == 0 && ref.index + ref.width <= 16;
  return false;
}

[[nodiscard]] std::optional<ScalarPcCarrier>
scalar_pc_carrier(rj_code_arch_t arch, const Instruction &inst, uint32_t word, ScalarPcOp op) {
  // This function intentionally recognizes only the canonical 32-bit SOP1
  // encoding. If the instruction is not exactly the scalar PC form, returning
  // nullopt is safer than trying to recover from the generic Instruction API:
  // false positives here would create real CFG edges.
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((word >> 23) != kSop1EncodingPrefix)
    return std::nullopt;
  auto opcode = scalar_pc_opcode(arch, op);
  if (!opcode || ((word >> 8) & 0xffu) != *opcode)
    return std::nullopt;
  const uint16_t selector = op == ScalarPcOp::GetPc64 ? static_cast<uint16_t>((word >> 16) & 0x7fu)
                                                      : static_cast<uint16_t>(word & 0xffu);
  const Operand *operand = op == ScalarPcOp::GetPc64 ? inst.dst_operand(0) : inst.src_operand(0);
  if (operand == nullptr)
    return std::nullopt;
  auto ref = scalar_register_pair_ref(*operand);
  if (!ref)
    return std::nullopt;
  return ScalarPcCarrier{.selector = selector, .ref = *ref};
}

[[nodiscard]] std::optional<uint16_t> scalar_pc_sreg(rj_code_arch_t arch, const Instruction &inst,
                                                     uint32_t word, ScalarPcOp op) {
  auto carrier = scalar_pc_carrier(arch, inst, word, op);
  if (!carrier || carrier->ref.cls != RegClass::SGPR || carrier->ref.index >= kMaxTrackedSgprPair ||
      carrier->selector != carrier->ref.index)
    return std::nullopt;
  return carrier->selector;
}

[[nodiscard]] bool sop2_literal_to_sreg(const Instruction &inst, uint32_t word,
                                        uint32_t literal_word, uint32_t opcode, uint16_t sdst,
                                        uint16_t ssrc0, uint32_t &literal) {
  if (inst.size() != 2 * sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != 255u)
    return false;
  if ((word & 0xffu) != ssrc0)
    return false;
  literal = literal_word;
  return true;
}

[[nodiscard]] bool sop2_literal_inline_to_sreg(const Instruction &inst, uint32_t word,
                                               uint32_t literal_word, uint32_t opcode,
                                               uint16_t sdst, uint16_t inline_src1,
                                               uint32_t &literal) {
  if (inst.size() != 2 * sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != inline_src1)
    return false;
  if ((word & 0xffu) != 255u)
    return false;
  literal = literal_word;
  return true;
}

[[nodiscard]] bool sop2_sreg_inline_to_sreg(const Instruction &inst, uint32_t word, uint32_t opcode,
                                            uint16_t sdst, uint16_t ssrc0, uint16_t inline_src1) {
  if (inst.size() != sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != inline_src1)
    return false;
  return (word & 0xffu) == ssrc0;
}

[[nodiscard]] bool sop2_sreg_literal_to_sreg(const Instruction &inst, uint32_t word,
                                             uint32_t literal_word, uint32_t opcode, uint16_t sdst,
                                             uint16_t ssrc0, uint32_t &literal) {
  return sop2_literal_to_sreg(inst, word, literal_word, opcode, sdst, ssrc0, literal);
}

[[nodiscard]] bool sop2_sreg_inline_zero_to_sreg(const Instruction &inst, uint32_t word,
                                                 uint32_t opcode, uint16_t sdst, uint16_t ssrc0) {
  return sop2_sreg_inline_to_sreg(inst, word, opcode, sdst, ssrc0, kInlineInt0);
}

void record_written_sgpr_ref(InstructionFacts &facts, RegisterRef ref) {
  facts.written_sgprs.expand(ref);
}

void record_written_sgprs(const Instruction &inst, InstructionFacts &facts,
                          uint32_t wavefront_size) {
  // This analysis only needs SGPR defs. Avoid the heavier def-use helper here:
  // computing use sets and vector metadata for every instruction was a major
  // cost on large generated kernels, and none of that information participates
  // in this lattice.
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = wave_mode_destination_ref(inst, *op, i, wavefront_size))
      record_written_sgpr_ref(facts, *ref);
    else if (auto ref = op->to_register_ref())
      record_written_sgpr_ref(facts, *ref);
  }

  RegisterSet implicit_defs;
  inst.implicit_defs(implicit_defs);
  if (!implicit_defs.none()) {
    implicit_defs.for_each([&](RegisterRef ref) { record_written_sgpr_ref(facts, ref); });
  }
  facts.written_sgprs_computed = true;
}

void ensure_written_sgprs(AnalysisContext &ctx, size_t index) {
  InstructionFacts &facts = ctx.facts[index];
  if (!facts.written_sgprs_computed)
    record_written_sgprs(*ctx.insts[index], facts, ctx.wavefront_size);
}

void invalidate_written_sgprs(AnalysisContext &ctx, size_t index, BlockState &state,
                              std::optional<uint16_t> protected_pair = std::nullopt) {
  // This is the conservative cleanup path for instructions whose semantics are
  // not modeled by the PC-builder transfer functions. It marks every SGPR def
  // as dirty and removes any tracked builder pair that overlaps the def.
  //
  // protected_pair is used when a recognized transfer writes the tracked pair
  // itself. For example, s_add_u32 pair_lo, pair_lo, literal is not a kill; it
  // edits the known value. Other defs in the same instruction, if any, are still
  // processed normally.
  ensure_written_sgprs(ctx, index);
  const InstructionFacts &facts = ctx.facts[index];
  facts.written_sgprs.for_each([&](uint16_t sgpr) { state.invalidate_half(sgpr, protected_pair); });
}

[[nodiscard]] bool is_unconditional_branch(const Instruction &inst) {
  return (inst.flags() & BRANCH) && !(inst.flags() & COND_BRANCH);
}

[[nodiscard]] bool is_indirect_branch(const Instruction &inst) {
  return (inst.flags() & INDIRECT_BRANCH) != 0;
}

[[nodiscard]] bool is_block_terminator(const Instruction &inst) {
  return is_program_path_terminator(inst) ||
         (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL));
}

[[nodiscard]] bool is_direct_call(const Instruction &inst) {
  return (inst.flags() & INDIRECT_CALL) != 0 && inst.branch_offset_bytes().has_value();
}

[[nodiscard]] bool is_recoverable_indirect_consumer(const Instruction &inst) {
  // Every fixup producer in this pass targets one of these consumer kinds.
  // Extend this predicate when adding recovery for another terminator.
  return is_indirect_branch(inst) || ((inst.flags() & INDIRECT_CALL) != 0 && !is_direct_call(inst));
}

[[nodiscard]] bool has_no_direct_successor(const Instruction &inst) {
  // Indirect branches have no known target until this analysis recovers one.
  // Indirect calls still expose their ordinary fallthrough/return continuation
  // in the direct CFG, which is required for liveness and for callers that do
  // not care about the callee body.
  return is_program_path_terminator(inst) || is_indirect_branch(inst);
}

[[nodiscard]] std::optional<size_t>
instruction_index_for_offset(std::span<const Instruction *const> insts, uint64_t offset) {
  // Decoded instructions are in ascending src_loc order. Using binary search
  // keeps leader construction linear apart from the small number of branch
  // targets and extra leaders that require lookups.
  const auto it = std::ranges::lower_bound(insts, offset, {},
                                           [](const Instruction *inst) { return inst->src_loc(); });
  if (it == insts.end() || (*it)->src_loc() != offset)
    return std::nullopt;
  return static_cast<size_t>(std::distance(insts.begin(), it));
}

/// @brief Mark every basic-block leader in the decoded instruction stream.
///
/// @details A leader is the first instruction of a basic block: index 0, any
/// direct branch target, the fallthrough after a terminator, an instruction
/// following an address discontinuity, and any caller-supplied extra leader.
/// Returns a per-instruction bitmap (1 == leader). This is the single source of
/// truth for both the temporary CFG skeleton and the block-local lane-stash
/// scan, so the two agree on where a block begins.
[[nodiscard]] std::vector<uint8_t> compute_block_leaders(std::span<const Instruction *const> insts,
                                                         std::span<const uint64_t> extra_leaders) {
  std::vector<uint8_t> leaders(insts.size(), 0);
  if (insts.empty())
    return leaders;
  leaders.front() = 1;

  const uint64_t section_end =
      insts.back()->src_loc() + static_cast<uint64_t>(insts.back()->size());
  for (uint64_t leader : extra_leaders) {
    if (leader >= section_end)
      continue;
    if (auto index = instruction_index_for_offset(insts, leader))
      leaders[*index] = 1;
  }

  for (size_t i = 0; i < insts.size(); ++i) {
    const Instruction &inst = *insts[i];
    const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
    // An address discontinuity or the instruction after any terminator begins a
    // new block.
    if (i + 1 < insts.size() && insts[i + 1]->src_loc() != next_offset)
      leaders[i + 1] = 1;
    if (is_block_terminator(inst) && next_offset < section_end && i + 1 < insts.size() &&
        insts[i + 1]->src_loc() == next_offset)
      leaders[i + 1] = 1;

    if (auto delta = inst.branch_offset_bytes()) {
      const int64_t target = static_cast<int64_t>(next_offset) + static_cast<int64_t>(*delta);
      if (target >= 0 && static_cast<uint64_t>(target) < section_end) {
        if (auto index = instruction_index_for_offset(insts, static_cast<uint64_t>(target)))
          leaders[*index] = 1;
      }
    }
  }
  return leaders;
}

[[nodiscard]] bool sop1_same_sreg(const Instruction &inst, uint32_t word, std::string_view mnemonic,
                                  uint16_t sreg);

[[nodiscard]] std::optional<TempDeltaPattern> match_temp_add_pattern(const AnalysisContext &ctx,
                                                                     size_t index,
                                                                     size_t last_index,
                                                                     uint16_t pair_lo) {
  // Match:
  //   s_add_i32 tmp, literal, 4
  //   [s_delay_alu]
  //   s_add_u32 pair_lo, pair_lo, tmp
  //   s_addc_u32 pair_hi, pair_hi, 0
  //
  // A one-instruction transfer cannot model this because the low-half add reads
  // a temporary whose value is not part of the lattice. Recognizing the compact
  // idiom as a single transfer lets the block scan keep tracking the pair
  // without adding arbitrary scalar-value analysis.
  if (index + 2 > last_index)
    return std::nullopt;

  size_t low_index = index + 1;
  if (ctx.insts[low_index]->mnemonic() == "s_delay_alu")
    ++low_index;
  const size_t high_index = low_index + 1;
  if (high_index > last_index)
    return std::nullopt;

  const auto add_i32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddI32);
  const auto add_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddU32);
  const auto addc_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddcU32);
  if (!add_i32_opcode || !add_u32_opcode || !addc_u32_opcode)
    return std::nullopt;

  const Instruction &temp_inst = *ctx.insts[index];
  const Instruction &low_inst = *ctx.insts[low_index];
  const Instruction &high_inst = *ctx.insts[high_index];
  const uint32_t temp_word = ctx.facts[index].word;
  const uint32_t low_word = ctx.facts[low_index].word;
  const uint32_t high_word = ctx.facts[high_index].word;
  const auto temp_sdst = static_cast<uint16_t>((temp_word >> 16) & 0x7fu);
  const Operand *temp_dst = temp_inst.dst_operand(0);
  const auto temp_ref =
      temp_dst == nullptr ? std::optional<RegisterRef>{} : temp_dst->to_register_ref();
  if (!temp_ref || temp_ref->cls != RegClass::SGPR || temp_ref->index != temp_sdst ||
      temp_ref->width != 1 || temp_sdst == pair_lo || temp_sdst == pair_lo + 1)
    return std::nullopt;

  uint32_t literal = 0;
  if (!sop2_literal_inline_to_sreg(temp_inst, temp_word,
                                   text_word_at(ctx.text, temp_inst.src_loc() + sizeof(uint32_t)),
                                   *add_i32_opcode, temp_sdst, kInlineInt4, literal))
    return std::nullopt;
  if (!sop2_sreg_inline_to_sreg(low_inst, low_word, *add_u32_opcode, pair_lo, pair_lo, temp_sdst))
    return std::nullopt;
  if (!sop2_sreg_inline_zero_to_sreg(high_inst, high_word, *addc_u32_opcode,
                                     static_cast<uint16_t>(pair_lo + 1),
                                     static_cast<uint16_t>(pair_lo + 1)))
    return std::nullopt;

  return TempDeltaPattern{
      .delta = static_cast<int64_t>(static_cast<int32_t>(literal)) + 4,
      .end_offset = high_inst.src_loc() + static_cast<uint64_t>(high_inst.size()),
      .instruction_count = high_index - index + 1,
  };
}

[[nodiscard]] std::optional<TempDeltaPattern> match_temp_sub_pattern(const AnalysisContext &ctx,
                                                                     size_t index,
                                                                     size_t last_index,
                                                                     uint16_t pair_lo) {
  // Match:
  //   s_add_i32  tmp, literal, 4
  //   s_abs_i32  tmp, tmp
  //   s_sub_u32  pair_lo, pair_lo, tmp
  //   s_subb_u32 pair_hi, pair_hi, 0
  //
  // This is the straight-line negative half of the signed PC-delta template.
  // The actual delta is still the signed `literal + 4`; the abs/sub pair is
  // just the encoding sequence used when that delta is negative.
  if (index + 3 > last_index)
    return std::nullopt;

  const auto add_i32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddI32);
  const auto sub_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubU32);
  const auto subb_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubbU32);
  if (!add_i32_opcode || !sub_u32_opcode || !subb_u32_opcode)
    return std::nullopt;

  const Instruction &temp_inst = *ctx.insts[index];
  const Instruction &abs_inst = *ctx.insts[index + 1];
  const Instruction &low_inst = *ctx.insts[index + 2];
  const Instruction &high_inst = *ctx.insts[index + 3];
  const uint32_t temp_word = ctx.facts[index].word;
  const auto temp_sdst = static_cast<uint16_t>((temp_word >> 16) & 0x7fu);
  const Operand *temp_dst = temp_inst.dst_operand(0);
  const auto temp_ref =
      temp_dst == nullptr ? std::optional<RegisterRef>{} : temp_dst->to_register_ref();
  if (!temp_ref || temp_ref->cls != RegClass::SGPR || temp_ref->index != temp_sdst ||
      temp_ref->width != 1 || temp_sdst == pair_lo || temp_sdst == pair_lo + 1)
    return std::nullopt;

  uint32_t literal = 0;
  if (!sop2_literal_inline_to_sreg(temp_inst, temp_word,
                                   text_word_at(ctx.text, temp_inst.src_loc() + sizeof(uint32_t)),
                                   *add_i32_opcode, temp_sdst, kInlineInt4, literal))
    return std::nullopt;
  if (!sop1_same_sreg(abs_inst, ctx.facts[index + 1].word, "s_abs_i32", temp_sdst))
    return std::nullopt;
  if (!sop2_sreg_inline_to_sreg(low_inst, ctx.facts[index + 2].word, *sub_u32_opcode, pair_lo,
                                pair_lo, temp_sdst))
    return std::nullopt;
  if (!sop2_sreg_inline_zero_to_sreg(high_inst, ctx.facts[index + 3].word, *subb_u32_opcode,
                                     static_cast<uint16_t>(pair_lo + 1),
                                     static_cast<uint16_t>(pair_lo + 1)))
    return std::nullopt;

  return TempDeltaPattern{
      .delta = static_cast<int64_t>(static_cast<int32_t>(literal)) + 4,
      .end_offset = high_inst.src_loc() + static_cast<uint64_t>(high_inst.size()),
      .instruction_count = 4,
  };
}

[[nodiscard]] bool sop1_same_sreg(const Instruction &inst, uint32_t word, std::string_view mnemonic,
                                  uint16_t sreg) {
  if (inst.size() != sizeof(uint32_t))
    return false;
  if (inst.mnemonic() != mnemonic)
    return false;
  if ((word >> 23) != kSop1EncodingPrefix)
    return false;
  return ((word >> 16) & 0x7fu) == sreg && (word & 0xffu) == sreg;
}

[[nodiscard]] bool apply_high_pc_canonicalization(const Instruction &inst, uint32_t word,
                                                  uint16_t pair_lo, PcValue &value) {
  const uint16_t pair_hi = static_cast<uint16_t>(pair_lo + 1);
  if (!sop1_same_sreg(inst, word, "s_sext_i32_i16", pair_hi))
    return false;
  value.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
  return true;
}

[[nodiscard]] bool apply_low_literal_update(const Instruction &inst, uint32_t word,
                                            std::span<const uint8_t> text, rj_code_arch_t arch,
                                            uint16_t pair_lo, PcValue &value) {
  // The low-half update is where the byte target usually changes. We interpret
  // literal add/sub forms only when the destination and first source are the
  // tracked low half. Any non-literal source falls through to the generic write
  // invalidation path and kills the pair.
  const auto add_u32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddU32);
  const auto sub_u32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::SubU32);
  const auto add_i32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddI32);
  if (!add_u32_opcode || !sub_u32_opcode || !add_i32_opcode)
    return false;

  uint32_t literal = 0;
  const uint32_t literal_word = text_word_at(text, inst.src_loc() + sizeof(uint32_t));
  int64_t delta = 0;
  if (sop2_sreg_literal_to_sreg(inst, word, literal_word, *add_u32_opcode, pair_lo, pair_lo,
                                literal)) {
    delta = static_cast<int64_t>(static_cast<int32_t>(literal));
  } else if (sop2_sreg_literal_to_sreg(inst, word, literal_word, *add_i32_opcode, pair_lo, pair_lo,
                                       literal)) {
    delta = static_cast<int64_t>(static_cast<int32_t>(literal));
  } else if (sop2_sreg_literal_to_sreg(inst, word, literal_word, *sub_u32_opcode, pair_lo, pair_lo,
                                       literal)) {
    delta = -static_cast<int64_t>(static_cast<int32_t>(literal));
  } else {
    return false;
  }

  value.offset += delta;
  value.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
  value.pending_high_carry = true;
  return true;
}

[[nodiscard]] bool apply_high_carry_update(const Instruction &inst, uint32_t word,
                                           std::span<const uint8_t> text, rj_code_arch_t arch,
                                           uint16_t pair_lo, PcValue &value) {
  // The high-half carry instruction completes the 64-bit edit. The common
  // getpc-relative chains use 0 or -1 as the second operand so the high half
  // only absorbs carry/borrow from the low half. Other high-half edits are not
  // modeled because they can change the absolute target in ways this pass does
  // not prove.
  const auto addc_u32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::AddcU32);
  const auto subb_u32_opcode = scalar_sop2_opcode(arch, ScalarSop2Op::SubbU32);
  if (!addc_u32_opcode || !subb_u32_opcode)
    return false;

  const uint16_t pair_hi = static_cast<uint16_t>(pair_lo + 1);
  if (sop2_sreg_inline_zero_to_sreg(inst, word, *addc_u32_opcode, pair_hi, pair_hi) ||
      sop2_sreg_inline_zero_to_sreg(inst, word, *subb_u32_opcode, pair_hi, pair_hi)) {
    value.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
    value.pending_high_carry = false;
    return true;
  }

  uint32_t literal = 0;
  const uint32_t literal_word = text_word_at(text, inst.src_loc() + sizeof(uint32_t));
  if (sop2_sreg_literal_to_sreg(inst, word, literal_word, *addc_u32_opcode, pair_hi, pair_hi,
                                literal) ||
      sop2_sreg_literal_to_sreg(inst, word, literal_word, *subb_u32_opcode, pair_hi, pair_hi,
                                literal)) {
    if (literal == 0 || literal == 0xffffffffu) {
      value.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
      value.pending_high_carry = false;
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool apply_gfx1250_add_nc_u64_update(const Instruction &inst, uint32_t word,
                                                   std::span<const uint8_t> text,
                                                   rj_code_arch_t arch, RegisterRef pair_ref,
                                                   PcValue &value) {
  // gfx1250 compilers use one SCC-neutral 64-bit add instead of the legacy
  // low-add/high-carry pair:
  //
  //   s_get_pc_i64  s[lo:lo+1]
  //   s_add_nc_u64  s[lo:lo+1], s[lo:lo+1], literal
  //   s_set_pc_i64  s[lo:lo+1]
  //
  // Match only the self-update literal forms. Register addends would require a
  // separate constant-propagation proof and must continue to fail closed.
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || inst.mnemonic() != "s_add_nc_u64" ||
      inst.num_dst_operands() != 1 || inst.num_src_operands() != 2)
    return false;

  const Operand *dst = inst.dst_operand(0);
  const Operand *src0 = inst.src_operand(0);
  const Operand *src1 = inst.src_operand(1);
  if (dst == nullptr || src0 == nullptr || src1 == nullptr)
    return false;
  const auto dst_ref = scalar_register_pair_ref(*dst);
  const auto src0_ref = scalar_register_pair_ref(*src0);
  if (!dst_ref || !src0_ref || *dst_ref != pair_ref || *src0_ref != pair_ref)
    return false;

  uint64_t literal = 0;
  if (auto literal64 = src1->literal64_value()) {
    literal = *literal64;
  } else {
    constexpr uint16_t kLiteralOperand = 255;
    if (inst.size() != 2 * sizeof(uint32_t) || ((word >> 8) & 0xffu) != kLiteralOperand)
      return false;
    literal = text_word_at(text, inst.src_loc() + sizeof(uint32_t));
  }

  // s_add_nc_u64 performs modulo-2^64 arithmetic. A valid local text target is
  // representable as a non-negative int64 offset after the modulo addition.
  const uint64_t updated = static_cast<uint64_t>(value.offset) + literal;
  if (updated > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  value.offset = static_cast<int64_t>(updated);
  value.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
  return true;
}

[[nodiscard]] bool apply_pair_literal64_update(const Instruction &inst, RegisterRef pair_ref,
                                               PcValue &value) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic != "s_add_nc_u64" && mnemonic != "s_sub_nc_u64")
    return false;

  const Operand *dst = inst.dst_operand(0);
  const Operand *src0 = inst.src_operand(0);
  const Operand *src1 = inst.src_operand(1);
  if (dst == nullptr || src0 == nullptr || src1 == nullptr)
    return false;

  const auto is_tracked_pair = [pair_ref](const Operand &operand) {
    const auto ref = scalar_register_pair_ref(operand);
    return ref && *ref == pair_ref;
  };
  if (!is_tracked_pair(*dst))
    return false;

  std::optional<uint64_t> literal;
  bool subtract = false;
  if (is_tracked_pair(*src0)) {
    literal = src1->literal64_value();
    subtract = mnemonic == "s_sub_nc_u64";
  } else if (mnemonic == "s_add_nc_u64" && is_tracked_pair(*src1)) {
    literal = src0->literal64_value();
  }
  if (!literal)
    return false;

  uint64_t offset_bits = 0;
  static_assert(sizeof(offset_bits) == sizeof(value.offset));
  std::memcpy(&offset_bits, &value.offset, sizeof(offset_bits));
  offset_bits = subtract ? offset_bits - *literal : offset_bits + *literal;
  std::memcpy(&value.offset, &offset_bits, sizeof(value.offset));
  value.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
  return true;
}

[[nodiscard]] std::optional<IndirectCallFixup> fixup_for_value(const AnalysisContext &ctx,
                                                               size_t inst_index, uint16_t selector,
                                                               RegisterRef carrier_ref,
                                                               const PcValue &value) {
  // A recovered target outside the current text section cannot become a local
  // BasicBlock successor. Drop it here rather than forcing the caller to filter
  // impossible leaders.
  if (value.offset < 0 || static_cast<uint64_t>(value.offset) >= ctx.text.size())
    return std::nullopt;

  return IndirectCallFixup{
      .source_getpc_offset = value.source_getpc_offset,
      .source_recovery_begin_offset = value.source_recovery_begin_offset,
      .source_recovery_end_offset = value.source_recovery_end_offset,
      .source_call_offset = ctx.insts[inst_index]->src_loc(),
      .source_target_offset = static_cast<uint64_t>(value.offset),
      .source_call_sreg = carrier_ref.cls == RegClass::SGPR ? carrier_ref.index : uint16_t{0},
      .source_call_selector = selector,
      .source_call_carrier = carrier_ref,
      .source_is_call = ctx.facts[inst_index].swappc_sdst.has_value(),
      .source_return_sreg = ctx.facts[inst_index].swappc_sdst.value_or(0),
      .source_return_selector = ctx.facts[inst_index].swappc_sdst.value_or(0),
  };
}

[[nodiscard]] std::optional<IndirectCallFixup> fixup_for_value(const AnalysisContext &ctx,
                                                               size_t inst_index, uint16_t pair_lo,
                                                               const PcValue &value) {
  return fixup_for_value(ctx, inst_index, pair_lo, RegisterRef{RegClass::SGPR, pair_lo, 2}, value);
}

/// @brief Identity of a fixup for deduplication: exactly the fields append_unique() compares.
struct FixupIdentity {
  uint64_t call;
  uint64_t target;
  uint64_t getpc;
  uint64_t begin;
  uint64_t end;
  uint16_t sreg;
  uint16_t selector;
  RegClass carrier_class;
  uint16_t carrier_index;
  uint8_t carrier_width;
  friend bool operator==(const FixupIdentity &, const FixupIdentity &) = default;
};

struct FixupIdentityHash {
  size_t operator()(const FixupIdentity &id) const noexcept {
    size_t h = std::hash<uint64_t>{}(id.call);
    const auto mix = [&h](uint64_t v) {
      h ^= std::hash<uint64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(id.target);
    mix(id.getpc);
    mix(id.begin);
    mix(id.end);
    mix(id.sreg);
    mix(id.selector);
    mix(static_cast<uint8_t>(id.carrier_class));
    mix(id.carrier_index);
    mix(id.carrier_width);
    return h;
  }
};

[[nodiscard]] FixupIdentity fixup_identity(const IndirectCallFixup &fixup) {
  return {fixup.source_call_offset,         fixup.source_target_offset,
          fixup.source_getpc_offset,        fixup.source_recovery_begin_offset,
          fixup.source_recovery_end_offset, fixup.source_call_sreg,
          fixup.source_call_selector,       fixup.source_call_carrier.cls,
          fixup.source_call_carrier.index,  fixup.source_call_carrier.width};
}

using FixupIndex = std::unordered_map<FixupIdentity, size_t, FixupIdentityHash>;

bool append_unique(std::vector<IndirectCallFixup> &out, IndirectCallFixup fixup) {
  // Deduplicate only FULLY identical fixups. A consumer with several distinct
  // s_getpc builders reaching the same target keeps the translator rewriting each
  // builder to its relocated address, so the builder identity
  // (source_getpc_offset and the recovery range) must participate in the compare.
  // Collapsing on {call, target, sreg} alone would drop a distinct builder, which
  // then keeps its stale pre-relocation address and can branch into unrelated
  // translated bytes. This mirrors the lattice value identity, which likewise
  // includes source_getpc_offset.
  const auto duplicate = std::ranges::find_if(out, [&](const IndirectCallFixup &existing) {
    return existing.source_call_offset == fixup.source_call_offset &&
           existing.source_target_offset == fixup.source_target_offset &&
           existing.source_call_selector == fixup.source_call_selector &&
           existing.source_call_carrier == fixup.source_call_carrier &&
           existing.source_getpc_offset == fixup.source_getpc_offset &&
           existing.source_recovery_begin_offset == fixup.source_recovery_begin_offset &&
           existing.source_recovery_end_offset == fixup.source_recovery_end_offset;
  });
  if (duplicate != out.end()) {
    // Both flags below are monotonic and must survive the merge, because only
    // the first record of a duplicate group is kept. Incompleteness: if any
    // iteration observes this fixup as incomplete, the merged record must stay
    // incomplete, or a later fixed-point pass that rediscovers an
    // earlier-complete fact as incomplete would be dropped here and leave the
    // consumer wrongly eligible for a direct window. A drain requirement is the
    // same shape -- it is a demand on the replacement, so a producer that needs
    // it cannot be outvoted by one that does not. Any future requirement that
    // changes the bytes relocation emits belongs here too.
    duplicate->source_incomplete = duplicate->source_incomplete || fixup.source_incomplete;
    duplicate->source_targets_exhaustive =
        duplicate->source_targets_exhaustive && fixup.source_targets_exhaustive;
    duplicate->source_requires_xcnt_drain =
        duplicate->source_requires_xcnt_drain || fixup.source_requires_xcnt_drain;
    return false;
  }
  out.push_back(fixup);
  return true;
}

/// @brief append_unique() with an O(1) duplicate lookup.
///
/// @details Same semantics as the scanning form -- same identity, same monotonic flag merge -- but
/// the caller keeps an index so accumulating N fixups costs O(N) rather than O(N^2). A large
/// device library recovers tens of thousands of them, and the scan was quadratic in that count on
/// every round.
bool append_unique_indexed(std::vector<IndirectCallFixup> &out, FixupIndex &index,
                           IndirectCallFixup fixup) {
  const auto [it, inserted] = index.try_emplace(fixup_identity(fixup), out.size());
  if (!inserted) {
    IndirectCallFixup &duplicate = out[it->second];
    duplicate.source_incomplete = duplicate.source_incomplete || fixup.source_incomplete;
    duplicate.source_targets_exhaustive =
        duplicate.source_targets_exhaustive && fixup.source_targets_exhaustive;
    duplicate.source_requires_xcnt_drain =
        duplicate.source_requires_xcnt_drain || fixup.source_requires_xcnt_drain;
    return false;
  }
  out.push_back(fixup);
  return true;
}

bool append_lattice_value(LatticeValue &dst, PcValue value) {
  // Keep values sorted and deduplicated so equality checks in the worklist
  // algorithm are deterministic. The key includes source_getpc_offset because
  // two builders can target the same byte offset but require different
  // relocation metadata later.
  const std::array<uint64_t, 2> key{static_cast<uint64_t>(value.offset), value.source_getpc_offset};
  auto it = std::ranges::lower_bound(dst.values, key, {}, [](const PcValue &pc_value) {
    return std::array<uint64_t, 2>{static_cast<uint64_t>(pc_value.offset),
                                   pc_value.source_getpc_offset};
  });
  if (it != dst.values.end() && *it == value)
    return false;
  if (dst.values.size() >= kMaxIndirectTargetsPerConsumer) {
    dst.incomplete = true;
    return false;
  }
  dst.values.insert(it, value);
  return true;
}

void join_lattice_value(LatticeValue &dst, const LatticeValue &src) {
  // JOIN is monotone: concrete values only accumulate, and incomplete/killed
  // only change from false to true. The finite target cap bounds the height of
  // the lattice and guarantees worklist convergence.
  if (src.incomplete)
    dst.incomplete = true;
  if (src.killed)
    dst.killed = true;
  for (const PcValue &value : src.values)
    append_lattice_value(dst, value);
}

[[nodiscard]] AnalysisContext build_context(std::span<const Instruction *const> insts,
                                            std::span<const uint8_t> text, rj_code_arch_t arch,
                                            uint32_t wavefront_size) {
  // Phase 1a: collect cheap facts that are independent of CFG. We do not build
  // full def-use information here. Generic writes are intentionally lazy because
  // many instructions never interact with a PC-builder pair, and decoding all
  // their operands dominated runtime on large code objects.
  AnalysisContext ctx;
  ctx.insts = insts;
  ctx.text = text;
  ctx.arch = arch;
  ctx.wavefront_size = wavefront_size;

  ctx.facts.resize(insts.size());
  for (size_t i = 0; i < insts.size(); ++i) {
    const Instruction &inst = *insts[i];
    InstructionFacts &facts = ctx.facts[i];
    facts.word = text_word_at(text, inst.src_loc());
    facts.getpc_sdst = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::GetPc64);
    facts.setpc_ssrc = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::SetPc64);
    facts.swappc_ssrc = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::SwapPc64);
    if (facts.swappc_ssrc) {
      const uint16_t return_pair = static_cast<uint16_t>((facts.word >> 16) & 0x7fu);
      const Operand *return_operand = inst.dst_operand(0);
      const auto return_ref = return_operand == nullptr ? std::optional<RegisterRef>{}
                                                        : scalar_register_pair_ref(*return_operand);
      if (return_ref && return_ref->cls == RegClass::SGPR && return_ref->index == return_pair &&
          return_pair < kMaxTrackedSgprPair) {
        facts.swappc_sdst = return_pair;
      } else {
        // A special or malformed destination cannot be modeled as a returning
        // call without inventing preservation semantics for that carrier.
        facts.swappc_ssrc.reset();
      }
    }
    if (facts.setpc_ssrc && *facts.setpc_ssrc < kMaxTrackedSgprPair)
      ctx.consumer_pairs.set(*facts.setpc_ssrc);
    if (facts.swappc_ssrc && *facts.swappc_ssrc < kMaxTrackedSgprPair)
      ctx.consumer_pairs.set(*facts.swappc_ssrc);
    facts.call_sdst = s_call_sdst(inst, facts.word);
  }

  return ctx;
}

[[nodiscard]] std::vector<AnalysisBlock>
build_analysis_blocks(const AnalysisContext &ctx, std::span<const uint64_t> extra_leaders,
                      std::span<const uint64_t> external_entries) {
  // Phase 1b: build the direct-CFG block skeleton used by dataflow. This
  // duplicates part of BasicBlock::build on purpose: recovered indirect targets
  // are not known yet, but we need a temporary block graph to prove them.
  //
  // The leader set is represented as an instruction-index bitmap instead of an
  // ordered set of offsets. The decoded instruction stream is already sorted,
  // and index marking avoids an O(number_of_instructions * log leaders)
  // membership check on very large kernels. Splitting after terminators makes
  // each setpc/swappc the last instruction in its analysis block, so the block
  // transfer summarizes the state at the control-transfer boundary rather than
  // after unrelated fallthrough instructions.
  const std::vector<uint8_t> leaders = compute_block_leaders(ctx.insts, extra_leaders);

  std::vector<AnalysisBlock> blocks;
  blocks.reserve(std::ranges::count(leaders, uint8_t{1}));
  for (size_t i = 0; i < ctx.insts.size(); ++i) {
    if (leaders[i] == 0)
      continue;
    AnalysisBlock block;
    block.offset = ctx.insts[i]->src_loc();
    block.first_index = i;
    block.last_index = i;
    block.external_entry =
        i == 0 || std::ranges::find(external_entries, block.offset) != external_entries.end();
    blocks.push_back(std::move(block));
  }

  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const size_t end_index =
        block_index + 1 < blocks.size() ? blocks[block_index + 1].first_index : ctx.insts.size();
    blocks[block_index].last_index = end_index - 1;
  }

  std::unordered_map<uint64_t, size_t> block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    block_by_offset.emplace(blocks[block_index].offset, block_index);

  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    AnalysisBlock &block = blocks[block_index];
    const Instruction &term = *ctx.insts[block.last_index];
    const uint64_t next_offset = term.src_loc() + static_cast<uint64_t>(term.size());

    if (auto delta = term.branch_offset_bytes()) {
      // Direct branches and direct scalar calls contribute their encoded target
      // to the temporary CFG. Direct calls also keep fallthrough below.
      const int64_t target = static_cast<int64_t>(next_offset) + static_cast<int64_t>(*delta);
      if (target >= 0) {
        if (auto it = block_by_offset.find(static_cast<uint64_t>(target));
            it != block_by_offset.end())
          block.successors.push_back(it->second);
      }
    }

    if (has_no_direct_successor(term) || (is_unconditional_branch(term) && !is_direct_call(term)))
      continue;

    if (auto it = block_by_offset.find(next_offset); it != block_by_offset.end()) {
      if (std::ranges::find(block.successors, it->second) == block.successors.end())
        block.successors.push_back(it->second);
    }
  }

  return blocks;
}

[[nodiscard]] std::vector<uint8_t>
explicit_external_entries(const std::vector<AnalysisBlock> &blocks,
                          std::span<const uint64_t> sorted_extra_leaders) {
  std::vector<uint8_t> entries(blocks.size(), 0);
  if (!entries.empty())
    entries[0] = 1;
  // Analysis blocks are built from the ordered instruction stream. Both input
  // sequences are therefore ascending and can be matched in one merge pass.
  assert(std::ranges::is_sorted(blocks, {}, &AnalysisBlock::offset));
  assert(std::ranges::is_sorted(sorted_extra_leaders));
  auto leader = sorted_extra_leaders.begin();
  for (size_t block_index = 1; block_index < blocks.size(); ++block_index) {
    while (leader != sorted_extra_leaders.end() && *leader < blocks[block_index].offset)
      ++leader;
    if (leader != sorted_extra_leaders.end() && *leader == blocks[block_index].offset)
      entries[block_index] = 1;
  }
  return entries;
}

[[nodiscard]] bool is_analysis_root(size_t block_index, std::span<const uint8_t> external_entries,
                                    const std::vector<std::vector<size_t>> &predecessors,
                                    ExternalEntryPolicy entry_policy) {
  if (external_entries[block_index] != 0)
    return true;
  return entry_policy == ExternalEntryPolicy::InferPredecessorless &&
         predecessors[block_index].empty();
}

void set_kill_transfer(AnalysisBlock &block, uint16_t pair_lo) {
  // KILL is weaker than SET for the same block: if the final state proves a
  // concrete builder, earlier dirty writes in the block should not downgrade it.
  if (pair_lo >= kMaxTrackedSgprPair)
    return;
  auto &transfer = block.transfers[pair_lo];
  if (transfer.kind != PairTransfer::Kind::Set)
    transfer.kind = PairTransfer::Kind::Kill;
}

void finalize_block_transfers(AnalysisBlock &block, const BlockState &state,
                              const std::bitset<REGISTER_SET_MAX_SGPRS> &consumer_pairs) {
  // Phase 2 block-exit summary:
  //
  // 1. Every still-live builder for a consumer pair becomes SET. This
  //    overrides incoming facts for the same pair in Phase 3.
  // 2. Every dirty half overlapping a consumer pair that is not covered by a
  //    SET kills that interpretation: s[N:N+1] or, when N > 0, s[N-1:N].
  // 3. Pairs never mentioned in transfers are implicit PASS.
  for (uint16_t pair_lo : state.active_pairs()) {
    if (!consumer_pairs[pair_lo])
      continue;
    const PcValue *value = state.builder(pair_lo);
    if (value == nullptr)
      continue;
    PairTransfer &transfer = block.transfers[pair_lo];
    transfer.kind = PairTransfer::Kind::Set;
    transfer.value = *value;
  }

  const auto has_set_transfer = [&](uint16_t pair_lo) {
    const auto transfer = block.transfers.find(pair_lo);
    return transfer != block.transfers.end() && transfer->second.kind == PairTransfer::Kind::Set;
  };
  state.for_each_dirty([&](uint16_t half) {
    if (consumer_pairs[half] && !has_set_transfer(half))
      set_kill_transfer(block, half);
    if (half > 0) {
      const uint16_t previous_pair = static_cast<uint16_t>(half - 1);
      if (consumer_pairs[previous_pair] && !has_set_transfer(previous_pair))
        set_kill_transfer(block, previous_pair);
    }
  });
}

std::optional<size_t> try_apply_temp_delta_pattern(AnalysisContext &ctx, const AnalysisBlock &block,
                                                   size_t index, BlockState &state) {
  // The common PC builder sometimes materializes the low-half delta in a
  // temporary SGPR immediately before adding/subtracting it into the tracked
  // pair. Looking at each instruction independently would see the low-half edit
  // as a write from an unknown SGPR and would kill the pair. Matching the whole
  // idiom as one transfer preserves precision while keeping the lattice small:
  // the temporary itself is never added to the dataflow state.
  for (uint16_t pair_lo : state.active_pairs()) {
    const PcValue *value = state.builder(pair_lo);
    if (value == nullptr)
      continue;
    // The matched idiom's first instruction must start where the recorded range
    // ends, or an unmodeled instruction sits inside the range that the patcher
    // would NOP-erase along with the builder. Mark the value non-contiguous so
    // the whole-scope proof declines to rewrite that range.
    const bool adjacent = ctx.insts[index]->src_loc() == value->source_recovery_end_offset;
    if (auto pattern = match_temp_add_pattern(ctx, index, block.last_index, pair_lo)) {
      PcValue updated = *value;
      updated.offset += pattern->delta;
      updated.source_recovery_end_offset = pattern->end_offset;
      updated.contiguous = updated.contiguous && adjacent;

      for (size_t i = 0; i < pattern->instruction_count; ++i)
        invalidate_written_sgprs(ctx, index + i, state, pair_lo);
      state.set_builder(pair_lo, updated);
      return pattern->instruction_count;
    }
    if (auto pattern = match_temp_sub_pattern(ctx, index, block.last_index, pair_lo)) {
      PcValue updated = *value;
      updated.offset += pattern->delta;
      updated.source_recovery_end_offset = pattern->end_offset;
      updated.contiguous = updated.contiguous && adjacent;

      for (size_t i = 0; i < pattern->instruction_count; ++i)
        invalidate_written_sgprs(ctx, index + i, state, pair_lo);
      state.set_builder(pair_lo, updated);
      return pattern->instruction_count;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<size_t> match_signed_delta_add_consumer(const AnalysisContext &ctx,
                                                                    const AnalysisBlock &block,
                                                                    uint16_t pair_lo,
                                                                    uint16_t tmp_sreg) {
  // Match the positive half of the compiler-emitted signed PC-delta template:
  //
  //   s_add_u32  pair_lo, pair_lo, tmp
  //   s_addc_u32 pair_hi, pair_hi, 0
  //   s_setpc_b64 pair
  //
  // The temporary was materialized before the conditional branch that selected
  // this block. We deliberately do not add that temporary to the general
  // lattice; this helper is only for the complete signed-delta template where
  // the sibling subtract block proves both paths are the same static target.
  const auto add_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddU32);
  const auto addc_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddcU32);
  if (!add_u32_opcode || !addc_u32_opcode)
    return std::nullopt;

  const auto is_gfx1250_padding = [&](size_t index) {
    if (ctx.arch != ROCJITSU_CODE_ARCH_GFX1250)
      return false;
    const Instruction &inst = *ctx.insts[index];
    // The gfx1250 sequence drains XCNT before an instruction prefetch. The
    // shader manual defines S_WAIT_XCNT as a counter wait, so neither it nor
    // the prefetch changes the PC pair or the signed-delta temporary. This
    // block is the conditional branch target, so it lies past the recovery
    // range and keeps both instructions verbatim; the predicate only skips them
    // while locating the arithmetic and set-PC consumer. The subtract half sits
    // inside the range and has to reproduce its drain -- see
    // match_signed_delta_sub_consumer.
    if (inst.mnemonic() == "s_wait_xcnt" || inst.mnemonic() == "s_prefetch_inst_pc_rel")
      return true;
    // The compiler also emits a scalar immediate move to configure the
    // prefetch. It is safe to skip only when its destination is outside the
    // getpc pair being proven AND is not tmp_sreg — a move into tmp_sreg would
    // change the value the following s_add/s_abs consumes while recovery keeps
    // computing the target from the original literal.
    if (inst.mnemonic() != "s_mov_b32" || inst.size() != sizeof(uint32_t))
      return false;
    const uint16_t dst = static_cast<uint16_t>((ctx.facts[index].word >> 16) & 0x7fu);
    return dst != pair_lo && dst != static_cast<uint16_t>(pair_lo + 1u) && dst != tmp_sreg;
  };

  size_t low_index = block.first_index;
  while (low_index <= block.last_index && is_gfx1250_padding(low_index))
    ++low_index;
  if (low_index > block.last_index)
    return std::nullopt;
  const Instruction &low_inst = *ctx.insts[low_index];

  size_t high_index = low_index + 1;
  while (high_index <= block.last_index && is_gfx1250_padding(high_index))
    ++high_index;
  if (high_index > block.last_index)
    return std::nullopt;
  const Instruction &high_inst = *ctx.insts[high_index];

  size_t setpc_index = high_index + 1;
  while (setpc_index <= block.last_index && is_gfx1250_padding(setpc_index))
    ++setpc_index;
  if (setpc_index > block.last_index)
    return std::nullopt;
  const Instruction &setpc_inst = *ctx.insts[setpc_index];
  if (!sop2_sreg_inline_to_sreg(low_inst, ctx.facts[low_index].word, *add_u32_opcode, pair_lo,
                                pair_lo, tmp_sreg))
    return std::nullopt;
  if (!sop2_sreg_inline_zero_to_sreg(high_inst, ctx.facts[high_index].word, *addc_u32_opcode,
                                     static_cast<uint16_t>(pair_lo + 1),
                                     static_cast<uint16_t>(pair_lo + 1)))
    return std::nullopt;
  auto setpc_sreg =
      scalar_pc_sreg(ctx.arch, setpc_inst, ctx.facts[setpc_index].word, ScalarPcOp::SetPc64);
  if (!setpc_sreg || *setpc_sreg != pair_lo)
    return std::nullopt;
  return setpc_index;
}

/// @brief Negative half of the signed PC-delta template, as matched in one block.
struct SignedDeltaSubMatch {
  size_t setpc_index = 0;    ///< Index of the subtract half's set-PC consumer.
  uint64_t recovery_end = 0; ///< One-past-end source byte of the recovery range.
  bool skipped_xcnt = false; ///< The range holds an `s_wait_xcnt` the rewrite must reproduce.
};

[[nodiscard]] std::optional<SignedDeltaSubMatch>
match_signed_delta_sub_consumer(const AnalysisContext &ctx, const AnalysisBlock &block,
                                uint16_t pair_lo, uint16_t tmp_sreg) {
  // Match the negative half of the same signed PC-delta template:
  //
  //   s_abs_i32  tmp, tmp
  //   s_sub_u32  pair_lo, pair_lo, tmp
  //   s_subb_u32 pair_hi, pair_hi, 0
  //   s_setpc_b64 pair
  //
  // The recovery range returned here is contiguous from the original getpc
  // through this subtract half. Relocation rewrites that first range once; the
  // add-half fixup shares the range only so translation knows its setpc was
  // statically accounted for.
  const auto sub_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubU32);
  const auto subb_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubbU32);
  if (!sub_u32_opcode || !subb_u32_opcode)
    return std::nullopt;

  const auto is_gfx1250_padding = [&](size_t index) {
    if (ctx.arch != ROCJITSU_CODE_ARCH_GFX1250)
      return false;
    const Instruction &inst = *ctx.insts[index];
    if (inst.mnemonic() == "s_wait_xcnt" || inst.mnemonic() == "s_prefetch_inst_pc_rel")
      return true;
    // Skip a prefetch-config move only when it clobbers neither the getpc pair
    // nor tmp_sreg (whose value s_abs_i32/s_sub_u32 below consume).
    if (inst.mnemonic() != "s_mov_b32" || inst.size() != sizeof(uint32_t))
      return false;
    const uint16_t dst = static_cast<uint16_t>((ctx.facts[index].word >> 16) & 0x7fu);
    return dst != pair_lo && dst != static_cast<uint16_t>(pair_lo + 1u) && dst != tmp_sreg;
  };

  // Padding skipped ahead of the subtract half lies inside the recovery range,
  // which patch_recovered_builder_fixups overwrites with the canonical builder
  // and NOP-fills. Losing the prefetch and its configuration move only costs a
  // hint, but the canonical builder writes the same pair the XCNT drain orders,
  // so the rewrite has to reproduce the drain.
  size_t abs_index = block.first_index;
  bool skipped_xcnt = false;
  while (abs_index <= block.last_index && is_gfx1250_padding(abs_index)) {
    skipped_xcnt = skipped_xcnt || ctx.insts[abs_index]->mnemonic() == "s_wait_xcnt";
    ++abs_index;
  }
  if (abs_index + 3 > block.last_index)
    return std::nullopt;
  const Instruction &abs_inst = *ctx.insts[abs_index];
  const Instruction &low_inst = *ctx.insts[abs_index + 1];
  const Instruction &high_inst = *ctx.insts[abs_index + 2];
  size_t setpc_index = abs_index + 3;
  while (setpc_index <= block.last_index && is_gfx1250_padding(setpc_index))
    ++setpc_index;
  if (setpc_index > block.last_index)
    return std::nullopt;
  const Instruction &setpc_inst = *ctx.insts[setpc_index];
  if (!sop1_same_sreg(abs_inst, ctx.facts[abs_index].word, "s_abs_i32", tmp_sreg))
    return std::nullopt;
  if (!sop2_sreg_inline_to_sreg(low_inst, ctx.facts[abs_index + 1].word, *sub_u32_opcode, pair_lo,
                                pair_lo, tmp_sreg))
    return std::nullopt;
  if (!sop2_sreg_inline_zero_to_sreg(high_inst, ctx.facts[abs_index + 2].word, *subb_u32_opcode,
                                     static_cast<uint16_t>(pair_lo + 1),
                                     static_cast<uint16_t>(pair_lo + 1)))
    return std::nullopt;
  auto setpc_sreg =
      scalar_pc_sreg(ctx.arch, setpc_inst, ctx.facts[setpc_index].word, ScalarPcOp::SetPc64);
  if (!setpc_sreg || *setpc_sreg != pair_lo)
    return std::nullopt;
  return SignedDeltaSubMatch{
      .setpc_index = setpc_index,
      .recovery_end = high_inst.src_loc() + static_cast<uint64_t>(high_inst.size()),
      .skipped_xcnt = skipped_xcnt,
  };
}

bool try_apply_pair_update(AnalysisContext &ctx, size_t index, BlockState &state) {
  for (uint16_t pair_lo : state.active_pairs()) {
    const PcValue *value = state.builder(pair_lo);
    if (value == nullptr)
      continue;
    PcValue updated = *value;
    const Instruction &inst = *ctx.insts[index];
    const uint32_t word = ctx.facts[index].word;
    // RDNA4 canonicalizes the high half of a getpc result before applying a
    // signed 48-bit text-relative delta. This exact self-register operation
    // preserves the local text offset represented by PcValue.
    if (ctx.arch == ROCJITSU_CODE_ARCH_RDNA4 && inst.size() == sizeof(uint32_t) &&
        inst.mnemonic() == "s_sext_i32_i16" && (word >> 23) == kSop1EncodingPrefix) {
      const auto dst = static_cast<uint16_t>((word >> 16) & 0x7fu);
      const auto src = static_cast<uint16_t>(word & 0xffu);
      if (dst == pair_lo + 1 && src == pair_lo + 1) {
        updated.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
        invalidate_written_sgprs(ctx, index, state, pair_lo);
        state.set_builder(pair_lo, updated);
        return true;
      }
      if (dst == pair_lo || dst == pair_lo + 1) {
        state.invalidate_pair(pair_lo);
        return true;
      }
    }
    // A builder step must start exactly where the recorded range ends. If the
    // pair survived an instruction between the previous step and this one, that
    // instruction sits inside [begin, end) and would be NOP-erased by
    // patch_recovered_builder_fixups. Mark the value non-contiguous so the
    // whole-scope proof declines to rewrite the range.
    const bool adjacent = inst.src_loc() == updated.source_recovery_end_offset;
    if (!apply_high_pc_canonicalization(inst, word, pair_lo, updated) &&
        !apply_pair_literal64_update(inst, RegisterRef{RegClass::SGPR, pair_lo, 2}, updated) &&
        !apply_gfx1250_add_nc_u64_update(inst, word, ctx.text, ctx.arch,
                                         RegisterRef{RegClass::SGPR, pair_lo, 2}, updated) &&
        !apply_low_literal_update(inst, word, ctx.text, ctx.arch, pair_lo, updated) &&
        !apply_high_carry_update(inst, word, ctx.text, ctx.arch, pair_lo, updated))
      continue;
    updated.contiguous = updated.contiguous && adjacent;

    invalidate_written_sgprs(ctx, index, state, pair_lo);
    state.set_builder(pair_lo, updated);
    return true;
  }
  return false;
}

void emit_fixups_for_values(const AnalysisContext &ctx, size_t inst_index, uint16_t pair_lo,
                            std::span<const PcValue> values,
                            std::vector<IndirectCallFixup> &recovered, bool incomplete = false) {
  // A complete lattice value can contain multiple concrete targets. That is not
  // an error by itself; it represents a bounded static dispatch where different
  // predecessor paths materialize different PC constants before joining at one
  // setpc/swappc consumer. When @p incomplete, at least one predecessor left the
  // pair unconstrained; the concrete targets are still recorded (for relocation
  // and liveness) but flagged so the translator does not build a direct window.
  for (const PcValue &value : values) {
    if (auto fixup = fixup_for_value(ctx, inst_index, pair_lo, value)) {
      fixup->source_incomplete = incomplete;
      fixup->source_targets_exhaustive = !incomplete;
      append_unique(recovered, *fixup);
    }
  }
}

void scan_block(AnalysisContext &ctx, size_t block_index, std::vector<AnalysisBlock> &blocks,
                std::vector<PendingConsumer> &pending_consumers,
                std::vector<IndirectCallFixup> &recovered, PcAddressBuilderMap &pc_builders) {
  // Phase 2: run local transfer semantics for one straight-line block.
  //
  // This scan has no incoming lattice facts by design. A pair either becomes
  // known because this block builds it, becomes dirty because this block writes
  // it, or remains pristine and can be resolved later from block-entry dataflow.
  // Keeping those cases separate prevents stale predecessor facts from leaking
  // through an unmodeled in-block write.
  AnalysisBlock &block = blocks[block_index];
  BlockState state;

  // Publish every still-live builder's current value. Called only where the
  // tracked pairs are about to stop being tracked, so the published value is
  // the one the original builder range really produces at that point.
  // A chain still waiting on its s_addc_u32 is not the value the range produces, and no rewrite of
  // [begin, end) can be right for it: the regenerated range stops before the carry. Poison instead
  // of publishing, so a whole-scope relocation claim fails closed rather than resting on a value
  // the program never finished computing.
  const auto publish_or_poison = [&](uint16_t pair_lo, const PcValue &value) {
    if (value.pending_high_carry)
      poison_pc_builder(pc_builders, value.source_getpc_offset);
    else
      note_pc_builder(pc_builders, pair_lo, value);
  };

  const auto note_live_pc_builders = [&] {
    for (uint16_t pair_lo : state.active_pairs()) {
      if (const PcValue *value = state.builder(pair_lo))
        publish_or_poison(pair_lo, *value);
    }
  };

  for (size_t index = block.first_index; index <= block.last_index; ++index) {
    const Instruction &inst = *ctx.insts[index];
    const InstructionFacts &facts = ctx.facts[index];
    const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());

    if (facts.getpc_sdst && *facts.getpc_sdst < kMaxTrackedSgprPair) {
      // s_getpc_b64 writes the address of the following instruction. The
      // low/high add sequence edits this base to the eventual branch target.
      const uint16_t pair_lo = *facts.getpc_sdst;
      // A second builder seeding the same pair is a stable point for the first one, in exactly the
      // sense note_pc_builder means: the pair stops being tracked as the old builder here, and the
      // old chain cannot be extended afterwards because any later arithmetic on this pair edits the
      // new builder's value. Whatever the replaced range produces at this instruction is therefore
      // final, and it is precisely what any consumer between the two getpcs read -- so publish it.
      //
      // Poisoning instead used to abandon a completed producer merely because its pair was reused.
      // Code that materializes several function pointers through one scratch pair -- build, spill
      // to a VGPR lane, rebuild -- would leave a poisoned record behind for the first pointer and
      // defeat any whole-scope or whole-object claim that every code address is relocated.
      if (const PcValue *replaced = state.builder(pair_lo))
        publish_or_poison(pair_lo, *replaced);
      seed_pc_builder(pc_builders, inst.src_loc(), pair_lo);
      state.set_builder(pair_lo, PcValue{.offset = static_cast<int64_t>(next_offset),
                                         .source_getpc_offset = inst.src_loc(),
                                         .source_recovery_begin_offset = next_offset,
                                         .source_recovery_end_offset = next_offset});
      continue;
    }
    // A getpc the pass declines to track still produces a PC-derived value.
    // Record it as an unresolvable producer so it cannot be silently omitted
    // from a whole-scope claim.
    if (facts.getpc_sdst) {
      seed_pc_builder(pc_builders, inst.src_loc(), *facts.getpc_sdst);
      poison_pc_builder(pc_builders, inst.src_loc());
    }

    const std::optional<uint16_t> consumer_pair =
        facts.setpc_ssrc ? facts.setpc_ssrc : facts.swappc_ssrc;
    if (consumer_pair) {
      // The consumer terminates this block, and its destination pair write can
      // clobber a tracked builder. Publish the pre-consumer values first.
      note_live_pc_builders();
      // A consumer resolved from local state is the strongest case: the builder
      // and the branch/call through that builder are in the same straight-line
      // block. Emit now so BasicBlock can split at the consumer and target.
      if (PcValue *value = state.builder(*consumer_pair)) {
        emit_fixups_for_values(ctx, index, *consumer_pair, std::span<const PcValue>(value, 1),
                               recovered);
        // A setpc/swappc source operand is a read, not a write. Preserve the
        // source pair unless the instruction also defines it. Real kernels can
        // build one callee address once and issue multiple swappc calls through
        // that same pair on the fallthrough path.
      } else if (*consumer_pair < kMaxTrackedSgprPair && !state.pair_dirty(*consumer_pair)) {
        // The block did not touch the pair, so any useful fact must come from
        // predecessor blocks. Defer classification until the block-entry
        // lattice is available. Out-of-range selectors cannot name a tracked
        // SGPR pair and deliberately remain unresolved.
        assert(ctx.consumer_pairs[*consumer_pair] &&
               "deferred consumer pair must be in the whole-text consumer set");
        pending_consumers.push_back(PendingConsumer{
            .block_index = block_index,
            .inst_index = index,
            .pair_lo = *consumer_pair,
        });
      }

      if (facts.swappc_sdst)
        // swappc writes a return PC to its destination pair. That value is
        // useful to hardware but not a getpc-relative builder for this pass.
        state.invalidate_pair(*facts.swappc_sdst);
      continue;
    }

    if (facts.call_sdst) {
      // A direct s_call can execute arbitrary callee code before control reaches
      // the fallthrough continuation. The temporary analysis CFG has no
      // context-sensitive return edge, so allowing existing builders to PASS
      // through this block would incorrectly preserve values that the callee may
      // clobber. Fail closed by killing every carried builder at the call site.
      // The values are still exactly what the builder ranges produced up to
      // here, so publish them before dropping them.
      note_live_pc_builders();
      const std::vector<uint16_t> active_pairs = state.active_pairs();
      for (uint16_t pair_lo : active_pairs)
        state.invalidate_pair(pair_lo);

      // The call destination receives the hardware return PC. It is meaningful
      // to the callee, but it is not an editable getpc-relative target builder.
      state.invalidate_pair(*facts.call_sdst);
      continue;
    }

    if (auto consumed = try_apply_temp_delta_pattern(ctx, block, index, state)) {
      index += *consumed - 1;
      continue;
    }

    if (try_apply_pair_update(ctx, index, state))
      continue;

    // Anything not modeled above is allowed to read arbitrary registers, but
    // only SGPR writes affect this analysis. Every write to either half of a
    // tracked pair kills that pair; every write to an otherwise-pristine pair
    // marks it dirty so a later consumer cannot incorrectly fall back to
    // predecessor facts.
    invalidate_written_sgprs(ctx, index, state);
  }

  note_live_pc_builders();
  finalize_block_transfers(block, state, ctx.consumer_pairs);
}

[[nodiscard]] std::vector<LatticeFacts> run_block_dataflow(
    const std::vector<AnalysisBlock> &blocks, std::span<const PendingConsumer> pending_consumers,
    std::span<const uint64_t> sorted_extra_leaders, ExternalEntryPolicy entry_policy) {
  // Phase 3: compute block-entry facts to a fixed point.
  //
  // entry[B] = JOIN(exit[P]) for every predecessor P of B.
  //
  // Explicit external entries begin with an empty map. Empty does not mean
  // "known empty set"; at those entries every pair has an unconstrained
  // hardware-supplied value unless a predecessor later mentions it. Under the
  // ExplicitOnly policy, a predecessorless non-entry is instead unreachable
  // (BOTTOM), so its empty map must not participate in a successor join.
  std::vector<std::vector<size_t>> predecessors(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    for (size_t successor : blocks[block_index].successors)
      predecessors[successor].push_back(block_index);
  }
  const std::vector<uint8_t> external_entries =
      explicit_external_entries(blocks, sorted_extra_leaders);

  // Dataflow results are consumed only by pending cross-block branches. A pair
  // that is built or killed somewhere but never reaches such a consumer cannot
  // affect any emitted fixup, so exclude it from the lattice entirely. Local
  // consumers were already resolved during scan_block(). This set must contain
  // every tracked pair used by a cross-block consumer; omitted consumers remain
  // unresolved.
  std::bitset<REGISTER_SET_MAX_SGPRS> relevant_pairs;
  for (const PendingConsumer &consumer : pending_consumers) {
    if (consumer.pair_lo < kMaxTrackedSgprPair)
      relevant_pairs.set(consumer.pair_lo);
  }

  std::vector<LatticeFacts> entry_facts(blocks.size());
  // Reachability is a separate lattice bit. A predecessor that has not become
  // reachable yet is BOTTOM, not an execution path carrying unconstrained
  // kernel-entry SGPRs. This distinction matters for loops: the first worklist
  // visit can see a builder entry edge plus an as-yet-unvisited backedge.
  // Treating that backedge as unconstrained permanently poisons an otherwise
  // dominated PC builder (the RCCL call-loop shape). Section entry and every
  // caller-provided kernel entry are nevertheless external roots even when
  // they have structural predecessors.
  //
  // With ExplicitOnly, do not infer an external entry merely because a block
  // has no predecessor. BinaryTranslator supplies every descriptor- or
  // firmware-visible kernel entry, then walks each entry's reachable CFG and
  // emits shared blocks separately in every kernel-local scope. A callable
  // helper is either an explicit leader itself or has a direct or recovered
  // predecessor edge; a predecessorless block after a non-returning instruction
  // such as s_trap 2 cannot acquire a hidden incoming edge from another kernel
  // scope. Generic callers without a complete entry list use
  // InferPredecessorless to preserve conservative multi-function recovery.
  std::vector<bool> reachable(blocks.size(), false);
  // Keep key presence separate from the sparse fact vectors. The dataflow
  // join needs the union of predecessor keys on every worklist visit; caching
  // that bounded set avoids repeatedly scanning every fact
  // to recover information that changes only when the corresponding vector does.
  std::vector<std::bitset<REGISTER_SET_MAX_SGPRS>> entry_pairs(blocks.size());
  std::deque<size_t> worklist;
  std::vector<bool> on_worklist(blocks.size(), false);
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    worklist.push_back(block_index);
    on_worklist[block_index] = true;
  }

  while (!worklist.empty()) {
    const size_t block_index = worklist.front();
    worklist.pop_front();
    on_worklist[block_index] = false;

    LatticeFacts new_entry;
    std::bitset<REGISTER_SET_MAX_SGPRS> mentioned_pairs;
    const bool new_reachable =
        is_analysis_root(block_index, external_entries, predecessors, entry_policy) ||
        std::ranges::any_of(predecessors[block_index],
                            [&](size_t predecessor) { return reachable[predecessor]; });
    if (new_reachable && !predecessors[block_index].empty()) {
      // A predecessor exit is its sparse entry map with this block's SET/KILL
      // summaries overlaid. The old implementation materialized that complete
      // map for every predecessor on every worklist visit, then allocated a
      // std::set to recover the union of keys. Large generated kernels spend
      // most dataflow time allocating and freeing those short-lived nodes.
      // Track the bounded SGPR-pair key union in fixed storage and evaluate each
      // predecessor's exit value only for the pair currently being joined.
      for (size_t predecessor : predecessors[block_index]) {
        if (!reachable[predecessor])
          continue;
        mentioned_pairs |= entry_pairs[predecessor];
        for (const auto &[pair_lo, _] : blocks[predecessor].transfers) {
          if (relevant_pairs[pair_lo])
            mentioned_pairs.set(pair_lo);
        }
      }

      new_entry.reserve(mentioned_pairs.count());
      for (uint16_t pair_lo = 0; pair_lo < mentioned_pairs.size(); ++pair_lo) {
        if (!mentioned_pairs[pair_lo])
          continue;

        LatticeValue joined;
        for (size_t predecessor : predecessors[block_index]) {
          if (!reachable[predecessor])
            continue;
          const AnalysisBlock &pred_block = blocks[predecessor];
          const auto transfer = pred_block.transfers.find(pair_lo);
          if (transfer != pred_block.transfers.end()) {
            if (transfer->second.kind == PairTransfer::Kind::Set) {
              append_lattice_value(joined, transfer->second.value);
            } else if (transfer->second.kind == PairTransfer::Kind::Kill) {
              joined.incomplete = true;
              joined.killed = true;
            } else {
              // Pass is normally represented by absence from the sparse
              // transfer map, but handle it explicitly to keep this lookup
              // equivalent if a caller ever stores a Pass summary.
              const LatticeValue *entry = entry_facts[predecessor].find(pair_lo);
              if (entry == nullptr)
                joined.incomplete = true;
              else
                join_lattice_value(joined, *entry);
            }
            continue;
          }

          const LatticeValue *entry = entry_facts[predecessor].find(pair_lo);
          if (entry == nullptr) {
            // A missing predecessor fact means the pair is still at its
            // unconstrained kernel-entry value on that path. Joining a concrete
            // PC with an unconstrained value must not create a speculative CFG
            // edge, so the result becomes incomplete.
            joined.incomplete = true;
          } else {
            join_lattice_value(joined, *entry);
          }
        }
        // Every explicit kernel entry has an external path carrying an
        // unconstrained SGPR pair, even if it also has structural
        // predecessors. That path must participate in the join.
        if (external_entries[block_index] != 0)
          joined.incomplete = true;
        new_entry.append(pair_lo, std::move(joined));
      }
    }

    if (new_reachable == reachable[block_index] && new_entry == entry_facts[block_index])
      continue;

    reachable[block_index] = new_reachable;
    entry_facts[block_index] = std::move(new_entry);
    entry_pairs[block_index] = mentioned_pairs;
    for (size_t successor : blocks[block_index].successors) {
      if (on_worklist[successor])
        continue;
      worklist.push_back(successor);
      on_worklist[successor] = true;
    }
  }

  return entry_facts;
}

[[nodiscard]] size_t
classify_pending_consumers(const AnalysisContext &ctx, const std::vector<AnalysisBlock> &blocks,
                           const std::vector<LatticeFacts> &entry_facts,
                           const std::vector<PendingConsumer> &pending_consumers,
                           std::vector<IndirectCallFixup> &recovered) {
  // Phase 4: resolve consumers that were pristine in their own block. A complete
  // entry fact provides concrete getpc-built targets. Missing, empty, or killed
  // facts are unresolved. Incomplete facts are still allowed when the concrete
  // target set is below the cap and no kill participated in the join: the
  // unknown part usually comes from path-insensitive joins in shared helper
  // code, while the concrete values are real return continuations that must be
  // represented for relocation and liveness. A saturated set is left unresolved
  // because the cap may have dropped valid targets.
  size_t unresolved = 0;
  for (const PendingConsumer &consumer : pending_consumers) {
    if (consumer.block_index >= blocks.size()) {
      ++unresolved;
      continue;
    }
    const auto &facts = entry_facts[consumer.block_index];
    const LatticeValue *value = facts.find(consumer.pair_lo);
    if (value == nullptr || value->values.empty() || value->killed) {
      ++unresolved;
      continue;
    }
    if (value->incomplete && value->values.size() >= kMaxIndirectTargetsPerConsumer) {
      ++unresolved;
      continue;
    }

    emit_fixups_for_values(ctx, consumer.inst_index, consumer.pair_lo, value->values, recovered,
                           value->incomplete);
  }
  return unresolved;
}

void add_recovered_leaders(std::vector<uint64_t> &leaders,
                           std::span<const IndirectCallFixup> recovered) {
  for (const IndirectCallFixup &fixup : recovered) {
    leaders.push_back(fixup.source_call_offset);
    leaders.push_back(fixup.source_target_offset);
  }
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
}

void add_recovered_successors(const std::vector<IndirectCallFixup> &recovered,
                              std::vector<AnalysisBlock> &blocks) {
  // Recovered indirect edges are not part of the first direct-CFG graph, but
  // they are real control flow once proven. Feeding them into the next
  // fixed-point round lets facts flow through nested helper returns without
  // speculating about unknown indirect targets.
  std::unordered_map<uint64_t, size_t> block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    block_by_offset.emplace(blocks[block_index].offset, block_index);

  for (const IndirectCallFixup &fixup : recovered) {
    auto source_it = block_by_offset.find(fixup.source_call_offset);
    auto target_it = block_by_offset.find(fixup.source_target_offset);
    if (source_it == block_by_offset.end() || target_it == block_by_offset.end())
      continue;

    std::vector<size_t> &successors = blocks[source_it->second].successors;
    if (std::ranges::find(successors, target_it->second) == successors.end())
      successors.push_back(target_it->second);
  }
}

struct VectorLaneSlot {
  uint16_t vgpr = 0;
  uint16_t lane = 0;

  friend bool operator==(const VectorLaneSlot &, const VectorLaneSlot &) = default;
};

struct VectorLaneSlotHash {
  size_t operator()(const VectorLaneSlot &slot) const {
    return (static_cast<size_t>(slot.vgpr) << 16) | slot.lane;
  }
};

struct StashedPcHalf {
  PcValue value;
  bool high = false;

  friend bool operator==(const StashedPcHalf &, const StashedPcHalf &) = default;
};

/// @brief Copy-on-write lane facts propagated through the CFG.
///
/// @details Most blocks pass an unchanged lane table to their successors. Sharing
/// that immutable table keeps the retained storage proportional to distinct
/// transfer states instead of multiplying every live slot by every CFG block.
class VectorLaneFacts {
  using Map = std::unordered_map<VectorLaneSlot, StashedPcHalf, VectorLaneSlotHash>;

public:
  using ConstIterator = Map::const_iterator;

  [[nodiscard]] ConstIterator find(VectorLaneSlot slot) const { return values().find(slot); }
  [[nodiscard]] ConstIterator end() const { return values().end(); }
  [[nodiscard]] bool empty() const { return !values_ || values_->empty(); }
  [[nodiscard]] size_t size() const { return values_ ? values_->size() : 0; }

  template <typename Visitor> void for_each(Visitor &&visitor) const {
    for (const auto &item : values())
      visitor(item.first, item.second);
  }

  void set(VectorLaneSlot slot, StashedPcHalf half) {
    if (values_) {
      const auto found = values_->find(slot);
      if (found != values_->end() && found->second == half)
        return;
    }
    writable_values().insert_or_assign(slot, std::move(half));
  }

  void erase(VectorLaneSlot slot) {
    if (!values_ || !values_->contains(slot))
      return;
    writable_values().erase(slot);
    reset_if_empty();
  }

  template <typename Predicate> void erase_if(Predicate &&predicate) {
    if (!values_ || std::ranges::none_of(*values_, predicate))
      return;
    std::erase_if(writable_values(), std::forward<Predicate>(predicate));
    reset_if_empty();
  }

  void intersect_with(const VectorLaneFacts &other) {
    if (!values_ || values_ == other.values_)
      return;
    const auto conflicts = [&](const auto &item) {
      const auto incoming = other.find(item.first);
      return incoming == other.end() || incoming->second != item.second;
    };
    erase_if(conflicts);
  }

  void clear() { values_.reset(); }

  friend bool operator==(const VectorLaneFacts &lhs, const VectorLaneFacts &rhs) {
    return lhs.values_ == rhs.values_ || lhs.values() == rhs.values();
  }

private:
  [[nodiscard]] const Map &values() const {
    static const Map empty;
    return values_ ? *values_ : empty;
  }

  [[nodiscard]] Map &writable_values() {
    if (!values_)
      values_ = std::make_shared<Map>();
    else if (!values_.unique())
      values_ = std::make_shared<Map>(*values_);
    return *values_;
  }

  void reset_if_empty() {
    if (values_->empty())
      values_.reset();
  }

  std::shared_ptr<Map> values_;
};

/// @brief One dword and wave lane in a compiler-managed private scratch frame.
///
/// @details GFX1250 uses an explicit scalar frame offset together with an
/// address-free vector operand for ordinary spills. The scalar selector is
/// part of the key: any write to that SGPR invalidates facts based on it.
struct ScratchLaneSlot {
  uint16_t saddr = 0;
  uint32_t byte_offset = 0;
  uint16_t lane = 0;

  friend bool operator==(const ScratchLaneSlot &, const ScratchLaneSlot &) = default;
};

struct ScratchLaneSlotHash {
  size_t operator()(const ScratchLaneSlot &slot) const noexcept {
    size_t seed = std::hash<uint32_t>{}(slot.byte_offset);
    seed ^= std::hash<uint16_t>{}(slot.saddr) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    seed ^= std::hash<uint16_t>{}(slot.lane) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    return seed;
  }
};

/// @brief Copy-on-write facts for proven PC halves spilled to private scratch.
class ScratchLaneFacts {
  using Map = std::unordered_map<ScratchLaneSlot, StashedPcHalf, ScratchLaneSlotHash>;

public:
  [[nodiscard]] bool empty() const { return !values_ || values_->empty(); }
  [[nodiscard]] size_t size() const { return values_ ? values_->size() : 0; }

  void set(ScratchLaneSlot slot, StashedPcHalf half) {
    if (values_) {
      const auto found = values_->find(slot);
      if (found != values_->end() && found->second == half)
        return;
    }
    writable_values().insert_or_assign(slot, std::move(half));
  }

  template <typename Predicate> void erase_if(Predicate &&predicate) {
    if (!values_ || std::ranges::none_of(*values_, predicate))
      return;
    std::erase_if(writable_values(), std::forward<Predicate>(predicate));
    reset_if_empty();
  }

  template <typename Visitor> void for_each(Visitor &&visitor) const {
    for (const auto &item : values())
      visitor(item.first, item.second);
  }

  void intersect_with(const ScratchLaneFacts &other) {
    if (!values_ || values_ == other.values_)
      return;
    const auto conflicts = [&](const auto &item) {
      const auto incoming = other.values().find(item.first);
      return incoming == other.values().end() || incoming->second != item.second;
    };
    erase_if(conflicts);
  }

  void clear() { values_.reset(); }

  friend bool operator==(const ScratchLaneFacts &lhs, const ScratchLaneFacts &rhs) {
    return lhs.values_ == rhs.values_ || lhs.values() == rhs.values();
  }

private:
  [[nodiscard]] const Map &values() const {
    static const Map empty;
    return values_ ? *values_ : empty;
  }

  [[nodiscard]] Map &writable_values() {
    if (!values_)
      values_ = std::make_shared<Map>();
    else if (!values_.unique())
      values_ = std::make_shared<Map>(*values_);
    return *values_;
  }

  void reset_if_empty() {
    if (values_->empty())
      values_.reset();
  }

  std::shared_ptr<Map> values_;
};

// Bound every variable-size fact retained by lane-table recovery. One unit is
// deliberately large enough for a lane-map node or restored-SGPR entry. Exact
// callee summaries are charged by their fixed object size, including a cache
// node allowance. Logical state facts are charged once per entry/exit/call
// state even when their backing storage is shared, so the bound is
// conservative. Recovery is optional and fails closed on exhaustion.
constexpr size_t kRetainedAnalysisUnitBytes = 64;
constexpr size_t kMaxRetainedAnalysisUnits = size_t{1} << 20;
constexpr size_t kMaxCalleeSummaryVariantsPerTarget = 8;
constexpr size_t kCalleeSummaryCacheEntryUnits =
    1 + (sizeof(CalleeSummaryCacheKey) + sizeof(std::optional<CalleeSummary>) + 3 * sizeof(void *) -
         1) /
            kRetainedAnalysisUnitBytes;

/// @brief Sparse, canonically ordered SGPR facts carried between CFG blocks.
///
/// @details Lane-restored PC halves are uncommon and normally occupy one pair.
/// Shared, ordered vectors keep copies of an unchanged flow state cheap while
/// preserving deterministic equality and lookup. A mutation detaches only the
/// affected state.
class RestoredSgprFacts {
  struct Entry {
    uint16_t sgpr = 0;
    StashedPcHalf half;

    friend bool operator==(const Entry &, const Entry &) = default;
  };

  using Storage = std::vector<Entry>;

public:
  [[nodiscard]] const StashedPcHalf *find(uint16_t sgpr) const {
    const Storage &current = values();
    const auto it =
        std::lower_bound(current.begin(), current.end(), sgpr,
                         [](const Entry &entry, uint16_t value) { return entry.sgpr < value; });
    return it != current.end() && it->sgpr == sgpr ? &it->half : nullptr;
  }

  void set(uint16_t sgpr, StashedPcHalf half) {
    const Storage &current = values();
    const auto found =
        std::lower_bound(current.begin(), current.end(), sgpr,
                         [](const Entry &entry, uint16_t value) { return entry.sgpr < value; });
    const size_t index = static_cast<size_t>(found - current.begin());
    if (found != current.end() && found->sgpr == sgpr) {
      if (found->half == half)
        return;
      writable_values()[index].half = std::move(half);
      return;
    }
    Storage &updated = writable_values();
    updated.insert(updated.begin() + static_cast<Storage::difference_type>(index),
                   Entry{.sgpr = sgpr, .half = std::move(half)});
  }

  void erase(uint16_t sgpr) {
    if (!values_)
      return;
    const auto found =
        std::lower_bound(values_->begin(), values_->end(), sgpr,
                         [](const Entry &entry, uint16_t value) { return entry.sgpr < value; });
    if (found == values_->end() || found->sgpr != sgpr)
      return;
    const size_t index = static_cast<size_t>(found - values_->begin());
    Storage &updated = writable_values();
    updated.erase(updated.begin() + static_cast<Storage::difference_type>(index));
    reset_if_empty();
  }

  template <typename Predicate> void erase_if(Predicate &&predicate) {
    if (!values_ ||
        std::ranges::none_of(*values_, [&](const Entry &entry) { return predicate(entry.sgpr); }))
      return;
    std::erase_if(writable_values(), [&](const Entry &entry) { return predicate(entry.sgpr); });
    reset_if_empty();
  }

  void intersect_with(const RestoredSgprFacts &other) {
    if (!values_ || values_ == other.values_)
      return;
    const auto conflicts = [&](const Entry &entry) {
      const StashedPcHalf *incoming = other.find(entry.sgpr);
      return incoming == nullptr || *incoming != entry.half;
    };
    if (std::ranges::none_of(*values_, conflicts))
      return;
    std::erase_if(writable_values(), conflicts);
    reset_if_empty();
  }

  [[nodiscard]] bool empty() const { return !values_ || values_->empty(); }
  [[nodiscard]] size_t size() const { return values_ ? values_->size() : 0; }
  void clear() { values_.reset(); }

  friend bool operator==(const RestoredSgprFacts &lhs, const RestoredSgprFacts &rhs) {
    return lhs.values_ == rhs.values_ || lhs.values() == rhs.values();
  }

private:
  [[nodiscard]] const Storage &values() const {
    static const Storage empty;
    return values_ ? *values_ : empty;
  }

  [[nodiscard]] Storage &writable_values() {
    if (!values_)
      values_ = std::make_shared<Storage>();
    else if (!values_.unique())
      values_ = std::make_shared<Storage>(*values_);
    return *values_;
  }

  void reset_if_empty() {
    if (values_->empty())
      values_.reset();
  }

  std::shared_ptr<Storage> values_;
};

struct VectorLaneFlowState {
  VectorLaneFacts slots;
  ScratchLaneFacts scratch_slots;
  RestoredSgprFacts restored_sgprs;
  std::optional<uint8_t> vgpr_msb_imm;
  std::optional<bool> gpr_idx_enabled;

  friend bool operator==(const VectorLaneFlowState &, const VectorLaneFlowState &) = default;
};

/// @brief Call-only dataflow metadata, allocated only for blocks with call edges.
struct CallEdgeInfo {
  std::unordered_set<size_t> target_successors;
  std::optional<size_t> continuation_successor;
  std::optional<VectorLaneFlowState> entry_state;
};

// The pass also allocates dense per-instruction and per-block scaffolding before
// retaining any logical facts. Bound that fixed footprint independently of the
// fact budget below. The estimates use the actual value sizes plus conservative
// pointer allowances for hash nodes, buckets, predecessor storage, and the
// worklist. Optional recovery fails closed when either budget would be exceeded.
constexpr size_t kMaxAnalysisScaffoldingBytes = size_t{256} << 20;
constexpr size_t kInstructionScaffoldingBytes =
    sizeof(std::pair<const uint64_t, size_t>) + 4 * sizeof(void *) + sizeof(size_t);
constexpr size_t kBlockScaffoldingBytes =
    sizeof(std::vector<size_t>) + 2 * sizeof(VectorLaneFlowState) + sizeof(CallEdgeInfo) +
    6 * sizeof(void *) + 3 * sizeof(size_t) + 2 * sizeof(uint8_t);

[[nodiscard]] constexpr bool hwreg_slice_overlaps_vgpr_msb(uint16_t hwreg) {
  const amdgpu::HwregSlice slice = amdgpu::decode_vgpr_msb_hwreg(hwreg);
  if (slice.id != amdgpu::MODE_HWREG)
    return false;
  const uint16_t end = static_cast<uint16_t>(
      std::min<uint32_t>(32, static_cast<uint32_t>(slice.begin) + slice.width));
  return slice.begin < amdgpu::VGPR_MSB_MODE_SHIFT + 8 && end > amdgpu::VGPR_MSB_MODE_SHIFT;
}

[[nodiscard]] constexpr bool hwreg_slice_overlaps_mode_bit(uint16_t hwreg, uint16_t bit) {
  const amdgpu::HwregSlice slice = amdgpu::decode_vgpr_msb_hwreg(hwreg);
  return slice.id == amdgpu::MODE_HWREG && slice.begin <= bit &&
         static_cast<uint32_t>(slice.begin) + slice.width > bit;
}

[[nodiscard]] std::optional<uint8_t> vgpr_bank_for_role(std::optional<uint8_t> mode,
                                                        amdgpu::VgprMsbRole role) {
  return amdgpu::vgpr_msb_bank_for_role(mode, role);
}

[[nodiscard]] std::optional<uint32_t> instruction_literal(const Instruction &inst,
                                                          std::span<const uint8_t> text);

void update_vgpr_mode(std::optional<uint8_t> &mode, const Instruction &inst,
                      std::span<const uint8_t> text) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_set_vgpr_msb") {
    if (const Operand *imm = inst.src_operand(0))
      mode = static_cast<uint8_t>(imm->encoding_value() & 0xffu);
    else
      mode = std::nullopt;
    return;
  }
  if (mnemonic != "s_setreg_b32" && mnemonic != "s_setreg_imm32_b32")
    return;
  const Operand *hwreg_operand = inst.dst_operand(0);
  if (hwreg_operand == nullptr) {
    mode = std::nullopt;
    return;
  }
  const uint16_t hwreg = static_cast<uint16_t>(hwreg_operand->encoding_value());
  if (mnemonic == "s_setreg_b32") {
    if (hwreg_slice_overlaps_vgpr_msb(hwreg))
      mode = std::nullopt;
    return;
  }
  amdgpu::VgprMsbBanks banks = amdgpu::unpack_vgpr_msb_banks(mode);
  amdgpu::apply_vgpr_msb_mode_write(banks, hwreg, instruction_literal(inst, text));
  mode = amdgpu::pack_vgpr_msb_banks(banks);
}

void update_gpr_idx_enabled(std::optional<bool> &enabled, const Instruction &inst,
                            std::span<const uint8_t> text) {
  constexpr uint16_t kGprIdxEnableBit = 27;
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_set_gpr_idx_on") {
    enabled = true;
    return;
  }
  if (mnemonic == "s_set_gpr_idx_off") {
    enabled = false;
    return;
  }
  if (mnemonic != "s_setreg_b32" && mnemonic != "s_setreg_imm32_b32")
    return;
  const Operand *hwreg_operand = inst.dst_operand(0);
  if (hwreg_operand == nullptr) {
    enabled = std::nullopt;
    return;
  }
  const uint16_t hwreg = static_cast<uint16_t>(hwreg_operand->encoding_value());
  if (!hwreg_slice_overlaps_mode_bit(hwreg, kGprIdxEnableBit))
    return;
  if (mnemonic == "s_setreg_b32") {
    enabled = std::nullopt;
    return;
  }
  const amdgpu::HwregSlice slice = amdgpu::decode_vgpr_msb_hwreg(hwreg);
  const auto literal = instruction_literal(inst, text);
  enabled = literal
                ? std::optional<bool>{((*literal >> (kGprIdxEnableBit - slice.begin)) & 1u) != 0}
                : std::nullopt;
}

[[nodiscard]] bool has_explicit_vgpr_destination(const Instruction &inst) {
  for (int index = 0; index < inst.num_dst_operands(); ++index) {
    const Operand *dst = inst.dst_operand(index);
    const auto ref = dst ? dst->to_register_ref() : std::nullopt;
    if (ref && ref->cls == RegClass::VGPR)
      return true;
  }
  return false;
}

[[nodiscard]] bool has_runtime_relative_destination(std::string_view mnemonic) {
  return mnemonic.starts_with("s_movreld") || mnemonic.starts_with("s_movrelsd") ||
         mnemonic.starts_with("v_movreld") || mnemonic.starts_with("v_movrelsd") ||
         mnemonic.starts_with("v_swaprel");
}

[[nodiscard]] std::optional<uint32_t> instruction_literal(const Instruction &inst,
                                                          std::span<const uint8_t> text) {
  const uint64_t literal_offset = inst.src_loc() + sizeof(uint32_t);
  if (inst.size() < 2 * static_cast<int>(sizeof(uint32_t)) || literal_offset > text.size() ||
      sizeof(uint32_t) > text.size() - literal_offset) {
    return std::nullopt;
  }
  uint32_t literal = 0;
  std::memcpy(&literal, text.data() + literal_offset, sizeof(literal));
  return literal;
}

[[nodiscard]] std::optional<uint16_t> inline_lane(const Operand *operand) {
  if (operand == nullptr || operand->encoding_value() < kInlineInt0 ||
      operand->encoding_value() >= kInlineInt0 + 32)
    return std::nullopt;
  return static_cast<uint16_t>(operand->encoding_value() - kInlineInt0);
}

[[nodiscard]] std::optional<RegisterRef> operand_register(const Operand *operand, RegClass cls) {
  if (operand == nullptr)
    return std::nullopt;
  auto ref = operand->to_register_ref();
  if (!ref || ref->cls != cls || ref->width != 1)
    return std::nullopt;
  return ref;
}

struct Gfx1250ScratchDword {
  uint16_t saddr = 0;
  uint32_t byte_offset = 0;
  uint16_t vgpr = 0;
  amdgpu::VgprMsbRole vgpr_role = amdgpu::VgprMsbRole::None;
  bool load = false;
};

struct Gfx1250ScratchRange {
  uint16_t saddr = 0;
  uint32_t byte_offset = 0;
  uint32_t byte_size = 0;
};

/// @brief Whether a GFX1250 scratch scalar base is restored by the device-call ABI.
///
/// @details LLVM reserves s32 and s33 as its stack- and frame-offset registers.
/// A generated nested callee may establish its own frame from them, but must
/// restore both values before returning. Other scalar scratch bases have no
/// such contract and require an exact nested-callee preservation proof.
[[nodiscard]] bool gfx1250_call_preserves_scratch_base(uint16_t saddr) {
  return saddr == 32 || saddr == 33;
}

/// @brief Decode a fixed-offset scratch store without a vector address.
///
/// @details Callee-preservation analysis only needs to know whether an
/// unrelated store overlaps a previously saved dword. It can therefore model
/// the complete family of fixed-width stores without interpreting their data
/// operands.
[[nodiscard]] std::optional<Gfx1250ScratchRange>
gfx1250_scratch_store_range(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  uint32_t byte_size = 0;
  if (mnemonic == "scratch_store_b8" || mnemonic == "scratch_store_d16_hi_b8")
    byte_size = 1;
  else if (mnemonic == "scratch_store_b16" || mnemonic == "scratch_store_d16_hi_b16")
    byte_size = 2;
  else if (mnemonic == "scratch_store_b32")
    byte_size = 4;
  else if (mnemonic == "scratch_store_b64")
    byte_size = 8;
  else if (mnemonic == "scratch_store_b96")
    byte_size = 12;
  else if (mnemonic == "scratch_store_b128")
    byte_size = 16;
  else
    return std::nullopt;

  if (inst.raw_encoding() == nullptr ||
      inst.size() != static_cast<int>(sizeof(cdna5::VscratchMachineInst)))
    return std::nullopt;
  std::array<uint32_t, 3> words;
  std::memcpy(words.data(), inst.raw_encoding(), sizeof(words));
  const cdna5::VscratchMachineInst raw = std::bit_cast<cdna5::VscratchMachineInst>(words);
  if (raw.vaddr != 0 || raw.scale_offset != 0)
    return std::nullopt;
  return Gfx1250ScratchRange{.saddr = static_cast<uint16_t>(raw.saddr),
                             .byte_offset = raw.ioffset,
                             .byte_size = byte_size};
}

/// @brief Decode the narrow scratch form used for compiler VGPR spills.
///
/// @details Only an address-free vector operand and an unscaled immediate are
/// accepted. A vector address or scaled offset would need value analysis before
/// two encoded addresses could be proven equal.
[[nodiscard]] std::optional<Gfx1250ScratchDword> gfx1250_scratch_dword(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic != "scratch_store_b32" && mnemonic != "scratch_load_b32")
    return std::nullopt;
  if (inst.raw_encoding() == nullptr ||
      inst.size() != static_cast<int>(sizeof(cdna5::VscratchMachineInst)))
    return std::nullopt;

  static_assert(sizeof(cdna5::VscratchMachineInst) == 3 * sizeof(uint32_t));
  std::array<uint32_t, 3> words;
  std::memcpy(words.data(), inst.raw_encoding(), sizeof(words));
  const cdna5::VscratchMachineInst raw = std::bit_cast<cdna5::VscratchMachineInst>(words);
  if (raw.vaddr != 0 || raw.scale_offset != 0)
    return std::nullopt;

  const bool load = mnemonic == "scratch_load_b32";
  const Operand *vgpr_operand = load ? inst.dst_operand(0) : inst.src_operand(1);
  const auto vgpr = operand_register(vgpr_operand, RegClass::VGPR);
  if (!vgpr)
    return std::nullopt;
  return Gfx1250ScratchDword{.saddr = static_cast<uint16_t>(raw.saddr),
                             .byte_offset = raw.ioffset,
                             .vgpr = vgpr->index,
                             .vgpr_role = vgpr_operand->vgpr_msb_role(),
                             .load = load};
}

/// @brief Prove that a scratch instruction executes for every Wave32 lane and
/// completes before the compiler restores the old EXEC mask.
[[nodiscard]] bool gfx1250_full_exec_scratch_transfer(const AnalysisContext &ctx,
                                                      const AnalysisBlock &block, size_t index) {
  if (index < block.first_index || index > block.last_index)
    return false;

  // LLVM groups adjacent callee-saved spills under one temporary all-EXEC
  // region. Find the complete group rather than requiring this dword transfer
  // to be its only member.
  const auto is_scratch_transfer = [&](size_t candidate) {
    return ctx.insts[candidate]->mnemonic().starts_with("scratch_");
  };
  size_t first_transfer = index;
  while (first_transfer > block.first_index && is_scratch_transfer(first_transfer - 1))
    --first_transfer;
  size_t last_transfer = index;
  while (last_transfer < block.last_index && is_scratch_transfer(last_transfer + 1))
    ++last_transfer;
  if (first_transfer == block.first_index || last_transfer + 2 > block.last_index)
    return false;

  const Instruction &save = *ctx.insts[first_transfer - 1];
  const Instruction &wait = *ctx.insts[last_transfer + 1];
  const Instruction &restore = *ctx.insts[last_transfer + 2];
  if (save.mnemonic() != "s_or_saveexec_b32" || save.num_dst_operands() < 1 ||
      save.num_src_operands() < 1 || (wait.flags() & WAITCNT) == 0 ||
      wait.raw_encoding() == nullptr || (wait.raw_encoding()[0] & 0xffffu) != 0 ||
      restore.mnemonic() != "s_mov_b32" || restore.num_dst_operands() != 1 ||
      restore.num_src_operands() != 1)
    return false;

  const auto saved_exec = operand_register(save.dst_operand(0), RegClass::SGPR);
  const auto restored_exec = operand_register(restore.dst_operand(0), RegClass::EXEC);
  const auto restore_source = operand_register(restore.src_operand(0), RegClass::SGPR);
  const auto all_ones = save.src_operand(0)->const_value();
  return saved_exec && restored_exec && restore_source && restored_exec->index == 0 &&
         saved_exec->index == restore_source->index && all_ones &&
         static_cast<uint32_t>(*all_ones) == UINT32_MAX;
}

// Whether a physical VGPR is callee-saved under the AMDGPU device calling
// convention (CSR_AMDGPU_VGPRs). The callee-saved VGPRs are interleaved with
// scratch registers in stripes of eight at a stride of sixteen starting at
// v40: v40-47, v56-63, v72-79, ... A conforming callee must preserve these
// across a call, so a PC stashed in one survives an intervening call even
// though the analysis does not descend into the callee body.
//
// TODO: Replace this calling-convention assumption with analysis that proves
// every reachable callee preserves the stashed physical VGPR before allowing
// the stash to survive a call. A compiler-generated callee violating the ABI is
// highly unlikely, but hand-written or otherwise non-conforming code may still
// do so. See the LLVM AMDGPU User Guide and AMDGPUCallingConv.td.
//
// @p phys_vgpr is the resolved physical index, which for gfx1250 VGPR_MSB
// banking may exceed 255 (bank*256 + selector). The ABI table only defines the
// convention for v0-255, so a banked register above that range is NOT proven
// callee-saved and must fail closed rather than be masked down to its selector.

void recover_vector_lane_stashed_pcs(AnalysisContext &ctx, const std::vector<AnalysisBlock> &blocks,
                                     std::span<const IndirectCallFixup> known_fixups,
                                     std::vector<IndirectCallFixup> &recovered,
                                     std::span<const uint64_t> sorted_extra_leaders,
                                     ExternalEntryPolicy entry_policy) {
  // gfx1250 device functions sometimes keep a small static call set in one
  // VGPR: getpc-built low/high halves are written to fixed lanes, then read
  // back into an SGPR pair before swappc. Track fixed-lane writelane/readlane
  // transport and compiler-generated, full-EXEC private-scratch spills of that
  // table. Any ordinary write to the physical VGPR invalidates every recorded
  // lane, and any SGPR write invalidates a reconstructed half or a scratch
  // address based on that scalar frame offset.
  //
  // RCCL carries this stash across branches and repeatedly changes the operand
  // bank selectors in between. VGPR contents do not disappear when MODE changes:
  // S_SET_VGPR_MSB only changes how later low eight-bit selectors are mapped.
  // Consequently slots are keyed by the resolved physical VGPR and propagated
  // with a must-reaching-definition dataflow. A slot reaches a block only when
  // every reachable predecessor carries the identical PcValue. A bypassed stash,
  // conflicting definition, unknown bank, or overlapping VGPR write therefore
  // still fails closed.
  //
  // The gfx1250 A0 profile uses S_SET_VGPR_MSB SIMM16[15:8] for the previous
  // bank state. The bank update remains SIMM16[7:0], so analysis ignores the
  // profile metadata byte.
  // This pass only observes MODE; it never inserts or reorders
  // S_SETREG/S_SET_VGPR_MSB and therefore cannot violate the required co-issue
  // spacing.
  if (ctx.arch != ROCJITSU_CODE_ARCH_GFX1250)
    return;

  if (blocks.empty())
    return;
  if (ctx.insts.size() > kMaxAnalysisScaffoldingBytes / kInstructionScaffoldingBytes)
    return;
  const size_t instruction_scaffolding_bytes = ctx.insts.size() * kInstructionScaffoldingBytes;
  const size_t remaining_scaffolding_bytes =
      kMaxAnalysisScaffoldingBytes - instruction_scaffolding_bytes;
  if (blocks.size() > remaining_scaffolding_bytes / kBlockScaffoldingBytes)
    return;
  const size_t initial_recovered_size = recovered.size();
  const std::vector<uint8_t> external_entries =
      explicit_external_entries(blocks, sorted_extra_leaders);

  std::unordered_map<uint64_t, size_t> instruction_by_offset;
  instruction_by_offset.reserve(ctx.insts.size());
  for (size_t index = 0; index < ctx.insts.size(); ++index)
    instruction_by_offset.emplace(ctx.insts[index]->src_loc(), index);

  std::vector<size_t> block_for_instruction(ctx.insts.size(), blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    for (size_t index = blocks[block_index].first_index; index <= blocks[block_index].last_index;
         ++index)
      block_for_instruction[index] = block_index;
  }

  // Cache exact register clobbers for a statically recovered helper whose
  // complete local CFG ends in returns through the call's destination pair (or
  // in paths that do not return). Nested or still-unresolved control transfers
  // fall back to the architecture's call-preserved register set below. The
  // cache key also includes the caller's VGPR-MSB mode and GPR-index enable
  // state because both can change the physical register named by an operand.
  size_t retained_state_units = 0;
  size_t retained_summary_units = 0;
  bool analysis_budget_exhausted = false;
  std::unordered_map<CalleeSummaryCacheKey, std::optional<CalleeSummary>, CalleeSummaryCacheKeyHash>
      callee_summary_cache;
  std::unordered_map<CalleeSummaryGroupKey, size_t, CalleeSummaryGroupKeyHash>
      callee_summary_variants;
  const auto statically_bounded_callee_summary =
      [&](uint64_t target, uint16_t return_pair, std::optional<uint8_t> initial_mode,
          std::optional<bool> initial_gpr_idx_enabled) -> std::optional<CalleeSummary> {
    const CalleeSummaryCacheKey key{.target = target,
                                    .return_pair = return_pair,
                                    .mode = initial_mode,
                                    .gpr_idx_enabled = initial_gpr_idx_enabled};
    const CalleeSummaryGroupKey group{.target = target, .return_pair = return_pair};
    auto group_variants = callee_summary_variants.find(group);
    if (group_variants != callee_summary_variants.end() &&
        group_variants->second > kMaxCalleeSummaryVariantsPerTarget)
      return std::nullopt;
    if (auto cached = callee_summary_cache.find(key); cached != callee_summary_cache.end())
      return cached->second;
    // Once a target/return pair needs more than the bounded number of MODE and
    // GPR-index variants, permanently use the conservative ABI summary for the
    // group. This loss of precision is monotone across the dataflow fixed point.
    if (group_variants != callee_summary_variants.end() &&
        group_variants->second == kMaxCalleeSummaryVariantsPerTarget) {
      ++group_variants->second;
      return std::nullopt;
    }

    std::optional<CalleeSummary> result;
    auto start = instruction_by_offset.find(target);
    if (start != instruction_by_offset.end() &&
        block_for_instruction[start->second] != blocks.size() &&
        blocks[block_for_instruction[start->second]].first_index == start->second) {
      CalleeSummary summary;
      struct WorkItem {
        size_t block_index;
        std::optional<uint8_t> mode;
        std::optional<bool> gpr_idx_enabled;
      };
      std::vector<WorkItem> worklist{
          {block_for_instruction[start->second], initial_mode, initial_gpr_idx_enabled}};
      std::unordered_set<uint64_t> visited;
      bool saw_return = false;
      std::optional<uint8_t> return_mode;
      std::optional<bool> return_gpr_idx_enabled;
      bool unsupported = false;
      while (!worklist.empty() && !unsupported) {
        const WorkItem item = worklist.back();
        worklist.pop_back();
        const uint64_t mode_key = item.mode ? *item.mode : uint64_t{256};
        const uint64_t gpr_idx_key = item.gpr_idx_enabled ? (*item.gpr_idx_enabled ? 1u : 0u) : 2u;
        const uint64_t visit_key =
            (static_cast<uint64_t>(item.block_index) << 11) | (gpr_idx_key << 9) | mode_key;
        if (!visited.insert(visit_key).second)
          continue;
        // Bound malformed or unexpectedly broad callees. Falling back to the
        // ABI is always safe and avoids turning one recovery into a whole-text
        // traversal.
        if (visited.size() > 4096) {
          unsupported = true;
          break;
        }

        const AnalysisBlock &block = blocks[item.block_index];
        std::optional<uint8_t> mode = item.mode;
        std::optional<bool> gpr_idx_enabled = item.gpr_idx_enabled;
        const auto record_all_banks = [&](uint16_t selector) {
          for (uint16_t bank = 0; bank < 4; ++bank)
            summary.vgprs.set(static_cast<uint16_t>((selector & 0xffu) + bank * 256u));
        };
        for (size_t index = block.first_index; index <= block.last_index; ++index) {
          const Instruction &inst = *ctx.insts[index];
          const std::string_view mnemonic = inst.mnemonic();
          // Relative-register operations can read or write a register selected
          // at runtime. Likewise, an ordinary VGPR destination is not a static
          // destination while GPR indexing may be enabled. An exact clobber
          // summary cannot represent either case, so use the ABI fallback.
          if (mnemonic.starts_with("s_movrel") || mnemonic.starts_with("v_movrel") ||
              mnemonic.starts_with("v_swaprel") ||
              (gpr_idx_enabled != std::optional<bool>{false} &&
               has_explicit_vgpr_destination(inst))) {
            unsupported = true;
            break;
          }
          ensure_written_sgprs(ctx, index);
          ctx.facts[index].written_sgprs.for_each([&](uint16_t sgpr) { summary.sgprs.set(sgpr); });
          RegisterSet implicit_defs;
          inst.implicit_defs(implicit_defs);
          implicit_defs.for_each([&](RegisterRef ref) {
            if (ref.cls == RegClass::VGPR)
              for (uint16_t lane = 0; lane < std::max<uint16_t>(1, ref.width); ++lane)
                record_all_banks(static_cast<uint16_t>(ref.index + lane));
          });
          for (int dst_index = 0; dst_index < inst.num_dst_operands(); ++dst_index) {
            const Operand *dst = inst.dst_operand(dst_index);
            if (dst == nullptr)
              continue;
            auto ref = dst->to_register_ref();
            if (!ref || ref->cls != RegClass::VGPR)
              continue;
            const auto bank = vgpr_bank_for_role(mode, dst->vgpr_msb_role());
            for (uint16_t lane = 0; lane < std::max<uint16_t>(1, ref->width); ++lane) {
              const uint16_t selector = static_cast<uint16_t>(ref->index + lane);
              if (!bank || selector >= 256) {
                record_all_banks(selector);
              } else {
                summary.vgprs.set(
                    static_cast<uint16_t>(selector + static_cast<uint16_t>(*bank) * 256u));
              }
            }
          }
          update_vgpr_mode(mode, inst, ctx.text);
          update_gpr_idx_enabled(gpr_idx_enabled, inst, ctx.text);
        }
        if (unsupported)
          break;

        const Instruction &term = *ctx.insts[block.last_index];
        const InstructionFacts &facts = ctx.facts[block.last_index];
        if (facts.setpc_ssrc) {
          if (*facts.setpc_ssrc != return_pair)
            unsupported = true;
          else if (!saw_return) {
            return_mode = mode;
            return_gpr_idx_enabled = gpr_idx_enabled;
            saw_return = true;
          } else {
            if (return_mode != mode)
              return_mode = std::nullopt;
            if (return_gpr_idx_enabled != gpr_idx_enabled)
              return_gpr_idx_enabled = std::nullopt;
          }
          continue;
        }
        if (facts.call_sdst || facts.swappc_ssrc || is_indirect_branch(term)) {
          unsupported = true;
          continue;
        }
        if (block.successors.empty()) {
          if (!is_program_path_terminator(term))
            unsupported = true;
          continue;
        }
        for (size_t successor : block.successors)
          worklist.push_back({successor, mode, gpr_idx_enabled});
      }
      // A transfer through the nominal return pair is only a proven return if
      // the callee never repurposed either half. This deliberately rejects
      // save-and-restore sequences that this summary does not value-track.
      if (!unsupported && saw_return && !summary.sgprs.test(return_pair) &&
          !summary.sgprs.test(static_cast<uint16_t>(return_pair + 1))) {
        summary.return_mode = return_mode;
        summary.return_gpr_idx_enabled = return_gpr_idx_enabled;
        result = summary;
      }
    }
    const size_t retained_units =
        kCalleeSummaryCacheEntryUnits + (group_variants == callee_summary_variants.end() ? 1 : 0);
    const size_t available_units =
        kMaxRetainedAnalysisUnits - retained_state_units - retained_summary_units;
    if (retained_units > available_units) {
      analysis_budget_exhausted = true;
      return std::nullopt;
    }
    callee_summary_cache.emplace(key, result);
    if (group_variants == callee_summary_variants.end())
      callee_summary_variants.emplace(group, 1);
    else
      ++group_variants->second;
    retained_summary_units += retained_units;
    return result;
  };

  struct ControlModeState {
    std::optional<uint8_t> vgpr_msb_imm;
    std::optional<bool> gpr_idx_enabled;

    bool operator==(const ControlModeState &) const = default;
  };
  struct ControlModeSummary {
    std::optional<ControlModeState> normal_return;
    std::optional<ControlModeState> parent_return;
  };
  struct ControlModeSummaryKey {
    CalleeSummaryCacheKey callee;
    std::optional<uint16_t> parent_return_pair;

    bool operator==(const ControlModeSummaryKey &) const = default;
  };
  struct ControlModeSummaryKeyHash {
    size_t operator()(const ControlModeSummaryKey &key) const {
      size_t hash = CalleeSummaryCacheKeyHash{}(key.callee);
      hash ^= std::hash<std::optional<uint16_t>>{}(key.parent_return_pair) + 0x9e3779b9u +
              (hash << 6) + (hash >> 2);
      return hash;
    }
  };
  std::unordered_map<ControlModeSummaryKey, std::optional<ControlModeSummary>,
                     ControlModeSummaryKeyHash>
      control_mode_summary_cache;
  std::unordered_set<ControlModeSummaryKey, ControlModeSummaryKeyHash>
      active_control_mode_summaries;

  // The full clobber summary intentionally declines complex callees. Recover a
  // smaller, context-sensitive summary of just the two control fields needed
  // to interpret vector-register operands. Unlike a write-free proof, this
  // accepts instrumentation helpers that deterministically set a field to its
  // incoming value (notably `s_set_vgpr_msb 0`) or restore it before return.
  // A nested helper may tail-return through its caller's return pair; keep
  // that distinct from an ordinary return so the local continuation is not
  // considered on that path. Unresolved calls and any other return pair still
  // fail closed.
  std::function<std::optional<ControlModeSummary>(uint64_t, uint16_t, std::optional<uint16_t>,
                                                  ControlModeState, size_t &)>
      summarize_callee_control_mode;
  summarize_callee_control_mode =
      [&](uint64_t target, uint16_t return_pair, std::optional<uint16_t> parent_return_pair,
          ControlModeState initial_state, size_t &work) -> std::optional<ControlModeSummary> {
    constexpr size_t kMaxControlModeSummaryBlocks = 4096;
    constexpr size_t kMaxControlModeSummaryCacheEntries = 4096;
    const ControlModeSummaryKey key{.callee = {.target = target,
                                               .return_pair = return_pair,
                                               .mode = initial_state.vgpr_msb_imm,
                                               .gpr_idx_enabled = initial_state.gpr_idx_enabled},
                                    .parent_return_pair = parent_return_pair};
    if (const auto cached = control_mode_summary_cache.find(key);
        cached != control_mode_summary_cache.end())
      return cached->second;
    if (control_mode_summary_cache.size() >= kMaxControlModeSummaryCacheEntries)
      return std::nullopt;
    if (!active_control_mode_summaries.insert(key).second) {
      // A recursive edge with the identical abstract input is the backedge of
      // this finite-state dataflow problem. Use that input as the coinductive
      // value for the edge; concrete return paths in the active invocation are
      // still scanned and merged, so a mode-changing base case weakens or
      // replaces the result rather than being hidden by the cycle.
      return ControlModeSummary{.normal_return = initial_state, .parent_return = std::nullopt};
    }

    struct WorkItem {
      size_t block_index;
      ControlModeState state;
    };
    ControlModeSummary result;
    const auto merge_return_state = [](std::optional<ControlModeState> &combined,
                                       const ControlModeState &incoming) {
      if (!combined) {
        combined = incoming;
        return;
      }
      if (combined->vgpr_msb_imm != incoming.vgpr_msb_imm)
        combined->vgpr_msb_imm = std::nullopt;
      if (combined->gpr_idx_enabled != incoming.gpr_idx_enabled)
        combined->gpr_idx_enabled = std::nullopt;
    };
    bool unsupported = false;
    std::vector<WorkItem> worklist;
    const auto start = instruction_by_offset.find(target);
    if (start == instruction_by_offset.end() ||
        block_for_instruction[start->second] == blocks.size() ||
        blocks[block_for_instruction[start->second]].first_index != start->second) {
      unsupported = true;
    } else {
      worklist.push_back({block_for_instruction[start->second], initial_state});
    }

    std::unordered_set<uint64_t> visited;
    while (!unsupported && !worklist.empty()) {
      WorkItem item = worklist.back();
      worklist.pop_back();
      const uint64_t mode_key = item.state.vgpr_msb_imm ? *item.state.vgpr_msb_imm : uint64_t{256};
      const uint64_t gpr_idx_key =
          item.state.gpr_idx_enabled ? (*item.state.gpr_idx_enabled ? 1u : 0u) : 2u;
      const uint64_t visit_key =
          (static_cast<uint64_t>(item.block_index) << 11) | (gpr_idx_key << 9) | mode_key;
      if (!visited.insert(visit_key).second)
        continue;
      if (++work > kMaxControlModeSummaryBlocks) {
        unsupported = true;
        break;
      }

      const AnalysisBlock &callee_block = blocks[item.block_index];
      for (size_t index = callee_block.first_index; index <= callee_block.last_index; ++index) {
        const Instruction &candidate = *ctx.insts[index];
        update_vgpr_mode(item.state.vgpr_msb_imm, candidate, ctx.text);
        update_gpr_idx_enabled(item.state.gpr_idx_enabled, candidate, ctx.text);
      }

      const Instruction &term = *ctx.insts[callee_block.last_index];
      const InstructionFacts &term_facts = ctx.facts[callee_block.last_index];
      if (term_facts.setpc_ssrc) {
        if (*term_facts.setpc_ssrc == return_pair) {
          merge_return_state(result.normal_return, item.state);
        } else if (parent_return_pair && *term_facts.setpc_ssrc == *parent_return_pair) {
          merge_return_state(result.parent_return, item.state);
        } else {
          // ConSan probe relays tail-transfer through a temporary PC pair into
          // an out-of-line helper, which ultimately returns through the
          // original call pair. Follow only a complete target set recovered by
          // the preceding scalar-PC analysis; an unresolved or incomplete
          // transfer remains indistinguishable from a mismatched return.
          bool complete = true;
          std::vector<uint64_t> tail_targets;
          for (const IndirectCallFixup &fixup : known_fixups) {
            if (fixup.source_call_offset != term.src_loc())
              continue;
            complete &= !fixup.source_incomplete;
            tail_targets.push_back(fixup.source_target_offset);
          }
          if (!complete || tail_targets.empty()) {
            unsupported = true;
          } else {
            std::ranges::sort(tail_targets);
            tail_targets.erase(std::ranges::unique(tail_targets).begin(), tail_targets.end());
            for (uint64_t tail_target : tail_targets) {
              const auto tail_entry = instruction_by_offset.find(tail_target);
              if (tail_entry == instruction_by_offset.end() ||
                  block_for_instruction[tail_entry->second] == blocks.size() ||
                  blocks[block_for_instruction[tail_entry->second]].first_index !=
                      tail_entry->second) {
                unsupported = true;
                break;
              }
              worklist.push_back({block_for_instruction[tail_entry->second], item.state});
            }
          }
        }
        continue;
      }

      std::vector<uint64_t> nested_targets;
      std::optional<uint16_t> nested_return_pair;
      if (term_facts.call_sdst) {
        nested_return_pair = *term_facts.call_sdst;
        const uint64_t next_offset = term.src_loc() + static_cast<uint64_t>(term.size());
        const auto delta = term.branch_offset_bytes();
        if (!delta || static_cast<int64_t>(next_offset) + *delta < 0) {
          unsupported = true;
        } else {
          nested_targets.push_back(
              static_cast<uint64_t>(static_cast<int64_t>(next_offset) + *delta));
        }
      } else if (term_facts.swappc_sdst) {
        nested_return_pair = *term_facts.swappc_sdst;
        bool complete = true;
        for (const IndirectCallFixup &fixup : known_fixups) {
          if (fixup.source_call_offset != term.src_loc())
            continue;
          complete &= !fixup.source_incomplete;
          nested_targets.push_back(fixup.source_target_offset);
        }
        if (!complete || nested_targets.empty()) {
          unsupported = true;
        }
      } else if (is_indirect_branch(term)) {
        unsupported = true;
      }
      if (unsupported)
        break;

      if (nested_return_pair) {
        std::ranges::sort(nested_targets);
        nested_targets.erase(std::ranges::unique(nested_targets).begin(), nested_targets.end());
        ControlModeSummary nested_result;
        const bool recursive_return_pair = *nested_return_pair == return_pair;
        const std::optional<uint16_t> nested_parent_return_pair =
            recursive_return_pair ? parent_return_pair : std::optional<uint16_t>{return_pair};
        for (uint64_t nested_target : nested_targets) {
          const auto target_result = summarize_callee_control_mode(
              nested_target, *nested_return_pair, nested_parent_return_pair, item.state, work);
          if (!target_result) {
            unsupported = true;
            break;
          }
          if (target_result->normal_return)
            merge_return_state(nested_result.normal_return, *target_result->normal_return);
          if (target_result->parent_return)
            merge_return_state(nested_result.parent_return, *target_result->parent_return);
        }
        if (unsupported)
          break;
        if (nested_result.parent_return) {
          // A conventional nested call's parent is this callee. A recursive
          // or coroutine-style edge reusing the current return pair inherits
          // this callee's parent instead.
          merge_return_state(recursive_return_pair ? result.parent_return : result.normal_return,
                             *nested_result.parent_return);
        }
        if (!nested_result.normal_return)
          continue;
        const uint64_t continuation = term.src_loc() + static_cast<uint64_t>(term.size());
        const auto continuation_inst = instruction_by_offset.find(continuation);
        if (continuation_inst == instruction_by_offset.end() ||
            block_for_instruction[continuation_inst->second] == blocks.size()) {
          unsupported = true;
          break;
        }
        worklist.push_back(
            {block_for_instruction[continuation_inst->second], *nested_result.normal_return});
        continue;
      }

      if (callee_block.successors.empty()) {
        if (!is_program_path_terminator(term)) {
          unsupported = true;
        }
        continue;
      }
      for (size_t successor : callee_block.successors)
        worklist.push_back({successor, item.state});
    }

    active_control_mode_summaries.erase(key);
    std::optional<ControlModeSummary> summary_result = result;
    if (unsupported || (!result.normal_return && !result.parent_return)) {
      summary_result = std::nullopt;
    }
    control_mode_summary_cache.emplace(key, summary_result);
    return summary_result;
  };

  const auto summarize_call_targets = [&](std::span<const uint64_t> targets, uint16_t return_pair,
                                          ControlModeState initial_state) {
    std::optional<ControlModeState> combined;
    size_t work = 0;
    for (uint64_t target : targets) {
      const auto result =
          summarize_callee_control_mode(target, return_pair, std::nullopt, initial_state, work);
      if (!result || !result->normal_return)
        return std::optional<ControlModeState>{};
      if (!combined) {
        combined = *result->normal_return;
      } else {
        if (combined->vgpr_msb_imm != result->normal_return->vgpr_msb_imm)
          combined->vgpr_msb_imm = std::nullopt;
        if (combined->gpr_idx_enabled != result->normal_return->gpr_idx_enabled)
          combined->gpr_idx_enabled = std::nullopt;
      }
    }
    return combined;
  };

  const auto scan_block = [&](const AnalysisBlock &block, VectorLaneFlowState state,
                              bool emit_fixups,
                              std::optional<VectorLaneFlowState> *call_entry_state) {
    BlockState builders;
    const auto publish_builders = [&]() {
      for (uint16_t pair_lo : builders.active_pairs()) {
        const PcValue *value = builders.builder(pair_lo);
        if (value == nullptr || static_cast<size_t>(pair_lo) + 1 >= REGISTER_SET_MAX_SGPRS)
          continue;
        state.restored_sgprs.set(pair_lo, StashedPcHalf{.value = *value, .high = false});
        state.restored_sgprs.set(static_cast<uint16_t>(pair_lo + 1),
                                 StashedPcHalf{.value = *value, .high = true});
      }
    };

    const auto physical_vgpr = [&](uint16_t low,
                                   amdgpu::VgprMsbRole role) -> std::optional<uint16_t> {
      const auto bank = amdgpu::vgpr_msb_bank_for_role(state.vgpr_msb_imm, role);
      if (!bank)
        return std::nullopt;
      return static_cast<uint16_t>(low + static_cast<uint16_t>(*bank) * 256u);
    };

    for (size_t index = block.first_index; index <= block.last_index; ++index) {
      const Instruction &inst = *ctx.insts[index];
      const InstructionFacts &facts = ctx.facts[index];
      const std::string_view mnemonic = inst.mnemonic();
      // Large dispatchers keep getpc-built targets in long-lived SGPRs, then
      // copy selected pairs into short-lived call operands. Capture the source
      // before generic destination invalidation and publish the copy only when
      // both halves carry the same proven PC value.
      std::optional<std::pair<uint16_t, PcValue>> copied_pair;
      if (mnemonic == "s_mov_b64" && inst.num_dst_operands() == 1 && inst.num_src_operands() == 1) {
        const Operand *dst_operand = inst.dst_operand(0);
        const Operand *src_operand = inst.src_operand(0);
        const auto dst = dst_operand ? dst_operand->to_register_ref() : std::nullopt;
        const auto src = src_operand ? src_operand->to_register_ref() : std::nullopt;
        if (dst && src && dst->cls == RegClass::SGPR && src->cls == RegClass::SGPR &&
            dst->width == 2 && src->width == 2 && dst->index < kMaxTrackedSgprPair &&
            src->index < kMaxTrackedSgprPair) {
          const StashedPcHalf *lo = state.restored_sgprs.find(src->index);
          const StashedPcHalf *hi =
              state.restored_sgprs.find(static_cast<uint16_t>(src->index + 1));
          if (lo != nullptr && hi != nullptr && !lo->high && hi->high && lo->value == hi->value) {
            copied_pair = std::pair{dst->index, lo->value};
          } else if (const PcValue *value = builders.builder(src->index)) {
            copied_pair = std::pair{dst->index, *value};
          }
        }
      }
      if (!state.restored_sgprs.empty() || !state.scratch_slots.empty()) {
        ensure_written_sgprs(ctx, index);
        ctx.facts[index].written_sgprs.for_each([&](uint16_t sgpr) {
          state.restored_sgprs.erase(sgpr);
          state.scratch_slots.erase_if([&](const auto &item) { return item.first.saddr == sgpr; });
        });
      }

      if (facts.getpc_sdst && *facts.getpc_sdst < kMaxTrackedSgprPair) {
        const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
        builders.set_builder(*facts.getpc_sdst, PcValue{.offset = static_cast<int64_t>(next_offset),
                                                        .source_getpc_offset = inst.src_loc(),
                                                        .source_recovery_begin_offset = next_offset,
                                                        .source_recovery_end_offset = next_offset});
        continue;
      }
      if (try_apply_pair_update(ctx, index, builders))
        continue;

      // Preserve proven lane-table entries across the exact scratch idiom used
      // by LLVM's GFX1250 dynamic-stack prologue. Both transfer instructions
      // are wrapped in `s_or_saveexec_b32 ..., -1`, a zero wait, and an exact
      // EXEC restore, so every Wave32 lane is transferred and complete before
      // the old mask is reinstated. Partial-EXEC, vector-addressed, scaled, or
      // aliased accesses deliberately fail closed.
      const auto scratch = gfx1250_scratch_dword(inst);
      if (mnemonic.starts_with("scratch_store")) {
        if (!scratch) {
          state.scratch_slots.clear();
        } else {
          bool possibly_aliased_base = false;
          state.scratch_slots.for_each([&](const ScratchLaneSlot &slot, const StashedPcHalf &) {
            possibly_aliased_base |= slot.saddr != scratch->saddr;
          });
          if (possibly_aliased_base)
            state.scratch_slots.clear();
          state.scratch_slots.erase_if([&](const auto &item) {
            return item.first.saddr == scratch->saddr &&
                   item.first.byte_offset == scratch->byte_offset;
          });
          const auto source_phys = physical_vgpr(scratch->vgpr, scratch->vgpr_role);
          if (source_phys && gfx1250_full_exec_scratch_transfer(ctx, block, index)) {
            state.slots.for_each([&](VectorLaneSlot slot, const StashedPcHalf &half) {
              if (slot.vgpr == *source_phys) {
                state.scratch_slots.set(ScratchLaneSlot{.saddr = scratch->saddr,
                                                        .byte_offset = scratch->byte_offset,
                                                        .lane = slot.lane},
                                        half);
              }
            });
          }
        }
        continue;
      }
      if (!state.scratch_slots.empty() &&
          (mnemonic.starts_with("scratch_atomic") || mnemonic.starts_with("flat_store") ||
           mnemonic.starts_with("flat_atomic")))
        state.scratch_slots.clear();
      if (scratch && scratch->load) {
        const auto destination_phys = physical_vgpr(scratch->vgpr, scratch->vgpr_role);
        if (destination_phys) {
          state.slots.erase_if(
              [&](const auto &item) { return item.first.vgpr == *destination_phys; });
          if (gfx1250_full_exec_scratch_transfer(ctx, block, index)) {
            state.scratch_slots.for_each([&](const ScratchLaneSlot &slot,
                                             const StashedPcHalf &half) {
              if (slot.saddr == scratch->saddr && slot.byte_offset == scratch->byte_offset) {
                state.slots.set(VectorLaneSlot{.vgpr = *destination_phys, .lane = slot.lane}, half);
              }
            });
          }
        } else {
          state.slots.erase_if([&](const auto &item) {
            return (item.first.vgpr & 0xffu) == (scratch->vgpr & 0xffu);
          });
        }
        continue;
      }

      if (facts.call_sdst) {
        // A direct call can clobber any caller-saved VGPR before its
        // fallthrough continuation executes, and the temporary CFG has no
        // context-sensitive return edge. Drop every stash in a caller-saved
        // VGPR; a conforming callee must preserve a callee-saved VGPR, so a
        // stash there survives (see is_callee_saved_vgpr).
        const uint16_t destination = *facts.call_sdst;
        builders.invalidate_pair(destination);
        state.restored_sgprs.erase(destination);
        state.restored_sgprs.erase(static_cast<uint16_t>(destination + 1));
        publish_builders();
        if (call_entry_state != nullptr)
          *call_entry_state = state;
        const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
        const auto delta = inst.branch_offset_bytes();
        std::vector<uint64_t> call_targets;
        if (delta && static_cast<int64_t>(next_offset) + *delta >= 0)
          call_targets.push_back(static_cast<uint64_t>(static_cast<int64_t>(next_offset) + *delta));
        const std::optional<CalleeSummary> callee_summary =
            call_targets.empty()
                ? std::nullopt
                : statically_bounded_callee_summary(call_targets.front(), *facts.call_sdst,
                                                    state.vgpr_msb_imm, state.gpr_idx_enabled);
        const auto survives_call = [&](uint16_t sgpr) {
          return callee_summary ? !callee_summary->sgprs.test(sgpr) : is_callee_saved_sgpr(sgpr);
        };
        state.slots.erase_if([&](const auto &item) {
          return callee_summary ? callee_summary->vgprs.test(item.first.vgpr)
                                : !is_callee_saved_vgpr(item.first.vgpr);
        });
        state.restored_sgprs.erase_if([&](uint16_t sgpr) { return !survives_call(sgpr); });
        const std::vector<uint16_t> active_pairs = builders.active_pairs();
        for (uint16_t pair_lo : active_pairs) {
          if (!survives_call(pair_lo) || !survives_call(static_cast<uint16_t>(pair_lo + 1)))
            builders.invalidate_pair(pair_lo);
        }
        if (callee_summary) {
          state.vgpr_msb_imm = callee_summary->return_mode;
          state.gpr_idx_enabled = callee_summary->return_gpr_idx_enabled;
        } else {
          const auto control_mode =
              summarize_call_targets(call_targets, *facts.call_sdst,
                                     ControlModeState{.vgpr_msb_imm = state.vgpr_msb_imm,
                                                      .gpr_idx_enabled = state.gpr_idx_enabled});
          if (control_mode) {
            state.vgpr_msb_imm = control_mode->vgpr_msb_imm;
            state.gpr_idx_enabled = control_mode->gpr_idx_enabled;
          } else {
            state.vgpr_msb_imm = std::nullopt;
            state.gpr_idx_enabled = std::nullopt;
          }
        }
      }

      if (mnemonic == "v_writelane_b32") {
        const auto dst = operand_register(inst.dst_operand(0), RegClass::VGPR);
        const auto src = operand_register(inst.src_operand(0), RegClass::SGPR);
        const auto lane = inline_lane(inst.src_operand(1));
        const auto dst_phys =
            dst ? physical_vgpr(dst->index, inst.dst_operand(0)->vgpr_msb_role()) : std::nullopt;
        if (dst && lane && state.gpr_idx_enabled == std::optional<bool>{false}) {
          if (dst_phys) {
            const VectorLaneSlot written_slot{*dst_phys, *lane};
            state.slots.erase(written_slot);
            if (src) {
              const StashedPcHalf *restored = state.restored_sgprs.find(src->index);
              if (restored != nullptr) {
                // Dispatchers can move a proven target from one lane table to
                // callee-saved SGPRs and then re-stash it in another table.
                state.slots.set(written_slot, *restored);
              } else {
                for (uint16_t pair_lo : builders.active_pairs()) {
                  const PcValue *value = builders.builder(pair_lo);
                  if (value == nullptr || (src->index != pair_lo && src->index != pair_lo + 1))
                    continue;
                  state.slots.set(written_slot, StashedPcHalf{.value = *value,
                                                              .high = src->index == pair_lo + 1});
                  break;
                }
              }
            }
          } else {
            // The destination bank is unknown. It may overwrite any physical
            // register with this low selector, so invalidate all four banks.
            state.slots.erase_if([&](const auto &item) {
              return (item.first.vgpr & 0xffu) == (dst->index & 0xffu) && item.first.lane == *lane;
            });
          }
        } else if (dst && lane) {
          state.slots.clear();
        }
        invalidate_written_sgprs(ctx, index, builders);
        continue;
      }

      if (mnemonic == "v_readlane_b32") {
        const auto dst = operand_register(inst.dst_operand(0), RegClass::SGPR);
        const auto src = operand_register(inst.src_operand(0), RegClass::VGPR);
        const auto lane = inline_lane(inst.src_operand(1));
        invalidate_written_sgprs(ctx, index, builders);
        const auto src_phys =
            src ? physical_vgpr(src->index, inst.src_operand(0)->vgpr_msb_role()) : std::nullopt;
        if (dst && lane && src_phys && state.gpr_idx_enabled == std::optional<bool>{false}) {
          auto slot = state.slots.find(VectorLaneSlot{*src_phys, *lane});
          if (slot != state.slots.end())
            state.restored_sgprs.set(dst->index, slot->second);
        }
        continue;
      }

      if (emit_fixups && is_lane_fixup_consumer(facts)) {
        const uint16_t pair_lo = *facts.swappc_ssrc;
        const StashedPcHalf *lo = state.restored_sgprs.find(pair_lo);
        const StashedPcHalf *hi = state.restored_sgprs.find(static_cast<uint16_t>(pair_lo + 1));
        if (lo != nullptr && hi != nullptr && !lo->high && hi->high && lo->value == hi->value) {
          if (auto fixup = fixup_for_value(ctx, index, pair_lo, lo->value))
            append_unique(recovered, *fixup);
        }
      }
      if (facts.swappc_sdst) {
        // A returning indirect call may execute arbitrary callee code before
        // the fallthrough continuation. Resolve this call from the pre-call
        // state above, then drop every stash in a caller-saved VGPR before
        // publishing the block exit so a callee-clobbered value cannot reach
        // the continuation. A callee-saved VGPR is preserved by a conforming
        // callee, so a stash there survives (see is_callee_saved_vgpr).
        const uint16_t pair_lo = *facts.swappc_ssrc;
        std::vector<uint64_t> targets;
        const StashedPcHalf *restored_lo = state.restored_sgprs.find(pair_lo);
        const StashedPcHalf *restored_hi =
            state.restored_sgprs.find(static_cast<uint16_t>(pair_lo + 1));
        if (restored_lo != nullptr && restored_hi != nullptr && !restored_lo->high &&
            restored_hi->high && restored_lo->value == restored_hi->value) {
          if (restored_lo->value.offset >= 0)
            targets.push_back(static_cast<uint64_t>(restored_lo->value.offset));
        } else if (const PcValue *builder = builders.builder(pair_lo)) {
          if (builder->offset >= 0)
            targets.push_back(static_cast<uint64_t>(builder->offset));
        }
        if (targets.empty()) {
          bool complete = true;
          for (const IndirectCallFixup &fixup : known_fixups) {
            if (fixup.source_call_offset != inst.src_loc())
              continue;
            if (fixup.source_incomplete) {
              complete = false;
              break;
            }
            targets.push_back(fixup.source_target_offset);
          }
          if (!complete)
            targets.clear();
        }
        const uint16_t destination = *facts.swappc_sdst;
        builders.invalidate_pair(destination);
        state.restored_sgprs.erase(destination);
        state.restored_sgprs.erase(static_cast<uint16_t>(destination + 1));
        publish_builders();
        if (call_entry_state != nullptr)
          *call_entry_state = state;
        std::ranges::sort(targets);
        targets.erase(std::ranges::unique(targets).begin(), targets.end());
        std::optional<CalleeSummary> callee_summary;
        if (!targets.empty()) {
          std::optional<CalleeSummary> combined;
          bool all_bounded = true;
          for (uint64_t target : targets) {
            auto summary = statically_bounded_callee_summary(
                target, *facts.swappc_sdst, state.vgpr_msb_imm, state.gpr_idx_enabled);
            if (!summary) {
              all_bounded = false;
              break;
            }
            if (combined)
              *combined |= *summary;
            else
              combined = *summary;
          }
          if (all_bounded)
            callee_summary = combined;
        }
        const auto survives_call = [&](uint16_t sgpr) {
          return callee_summary ? !callee_summary->sgprs.test(sgpr) : is_callee_saved_sgpr(sgpr);
        };
        state.slots.erase_if([&](const auto &item) {
          return callee_summary ? callee_summary->vgprs.test(item.first.vgpr)
                                : !is_callee_saved_vgpr(item.first.vgpr);
        });
        state.restored_sgprs.erase_if([&](uint16_t sgpr) { return !survives_call(sgpr); });
        const std::vector<uint16_t> active_pairs = builders.active_pairs();
        for (uint16_t active_pair : active_pairs) {
          if (!survives_call(active_pair) || !survives_call(static_cast<uint16_t>(active_pair + 1)))
            builders.invalidate_pair(active_pair);
        }
        if (callee_summary) {
          state.vgpr_msb_imm = callee_summary->return_mode;
          state.gpr_idx_enabled = callee_summary->return_gpr_idx_enabled;
        } else {
          const auto control_mode =
              summarize_call_targets(targets, *facts.swappc_sdst,
                                     ControlModeState{.vgpr_msb_imm = state.vgpr_msb_imm,
                                                      .gpr_idx_enabled = state.gpr_idx_enabled});
          if (control_mode) {
            state.vgpr_msb_imm = control_mode->vgpr_msb_imm;
            state.gpr_idx_enabled = control_mode->gpr_idx_enabled;
          } else {
            state.vgpr_msb_imm = std::nullopt;
            state.gpr_idx_enabled = std::nullopt;
          }
        }
      }

      // VGPR defs are decoded only to invalidate tracked slots; no slots makes
      // this entire region a no-op. Explicit destinations use MODE's DST bank,
      // so a bank-zero scratch address in v[0:1] does not clobber a lane table
      // in v[256:257]. Implicit definitions have no operand role from which to
      // select a bank and therefore conservatively invalidate every bank with
      // the same low selector.
      if (!state.slots.empty()) {
        if ((has_runtime_relative_destination(mnemonic) && mnemonic.starts_with("v_")) ||
            (state.gpr_idx_enabled != std::optional<bool>{false} &&
             has_explicit_vgpr_destination(inst))) {
          state.slots.clear();
        }
        const auto erase_selector_in_all_banks = [&](uint16_t selector) {
          state.slots.erase_if(
              [&](const auto &item) { return (item.first.vgpr & 0xffu) == (selector & 0xffu); });
        };
        for (int dst_index = 0; dst_index < inst.num_dst_operands(); ++dst_index) {
          const Operand *op = inst.dst_operand(dst_index);
          if (op == nullptr)
            continue;
          auto ref = op->to_register_ref();
          if (!ref || ref->cls != RegClass::VGPR)
            continue;
          for (uint16_t lane = 0; lane < std::max<uint16_t>(1, ref->width); ++lane) {
            const uint32_t selector = static_cast<uint32_t>(ref->index) + lane;
            if (selector >= 256) {
              erase_selector_in_all_banks(static_cast<uint16_t>(selector));
              continue;
            }
            const auto physical =
                physical_vgpr(static_cast<uint16_t>(selector), op->vgpr_msb_role());
            if (physical) {
              state.slots.erase_if([&](const auto &item) { return item.first.vgpr == *physical; });
            } else {
              erase_selector_in_all_banks(static_cast<uint16_t>(selector));
            }
          }
        }
        RegisterSet implicit_defs;
        inst.implicit_defs(implicit_defs);
        implicit_defs.for_each([&](RegisterRef ref) {
          if (ref.cls != RegClass::VGPR)
            return;
          erase_selector_in_all_banks(ref.index);
        });
      }

      if (!builders.active_pairs().empty())
        invalidate_written_sgprs(ctx, index, builders);

      if (has_runtime_relative_destination(mnemonic) && mnemonic.starts_with("s_")) {
        state.restored_sgprs.clear();
        for (uint16_t pair_lo : builders.active_pairs())
          builders.invalidate_pair(pair_lo);
      }

      if (copied_pair) {
        const uint16_t pair_lo = copied_pair->first;
        state.restored_sgprs.set(pair_lo,
                                 StashedPcHalf{.value = copied_pair->second, .high = false});
        state.restored_sgprs.set(static_cast<uint16_t>(pair_lo + 1),
                                 StashedPcHalf{.value = copied_pair->second, .high = true});
      }

      update_vgpr_mode(state.vgpr_msb_imm, inst, ctx.text);
      update_gpr_idx_enabled(state.gpr_idx_enabled, inst, ctx.text);
    }
    publish_builders();
    return state;
  };

  std::vector<std::vector<size_t>> predecessors(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    for (size_t successor : blocks[block_index].successors)
      predecessors[successor].push_back(block_index);
  }

  // A call block has two different outgoing states: the callee sees the
  // pre-call register state, while the fallthrough continuation sees the
  // summarized return state. AnalysisBlock stores both as ordinary successors,
  // so classify target edges here and select the matching state during meet.
  std::unordered_map<size_t, CallEdgeInfo> call_edges;
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const AnalysisBlock &block = blocks[block_index];
    const Instruction &term = *ctx.insts[block.last_index];
    const InstructionFacts &facts = ctx.facts[block.last_index];
    if (!facts.call_sdst && !facts.swappc_sdst)
      continue;
    CallEdgeInfo &call = call_edges[block_index];
    const uint64_t next_offset = term.src_loc() + static_cast<uint64_t>(term.size());
    if (const auto next = instruction_by_offset.find(next_offset);
        next != instruction_by_offset.end() && block_for_instruction[next->second] != blocks.size())
      call.continuation_successor = block_for_instruction[next->second];
    if (!facts.call_sdst)
      continue;
    const auto delta = term.branch_offset_bytes();
    if (!delta || static_cast<int64_t>(next_offset) + *delta < 0)
      continue;
    const uint64_t target = static_cast<uint64_t>(static_cast<int64_t>(next_offset) + *delta);
    if (const auto entry = instruction_by_offset.find(target);
        entry != instruction_by_offset.end() &&
        block_for_instruction[entry->second] != blocks.size())
      call.target_successors.insert(block_for_instruction[entry->second]);
  }
  // Every known source offset was added as a leader before these blocks were
  // built, so a recovered call is the first instruction in its source block
  // and this classification matches add_recovered_successors(). Incomplete
  // targets are still useful here: extra predecessor edges only weaken the
  // entry-state meet.
  for (const IndirectCallFixup &fixup : known_fixups) {
    if (!fixup.source_is_call)
      continue;
    const auto source = instruction_by_offset.find(fixup.source_call_offset);
    const auto target = instruction_by_offset.find(fixup.source_target_offset);
    if (source == instruction_by_offset.end() || target == instruction_by_offset.end())
      continue;
    const size_t source_block = block_for_instruction[source->second];
    const size_t target_block = block_for_instruction[target->second];
    if (source_block < blocks.size() && target_block < blocks.size())
      call_edges[source_block].target_successors.insert(target_block);
  }

  std::vector<VectorLaneFlowState> entry_states(blocks.size());
  std::vector<VectorLaneFlowState> exit_states(blocks.size());
  std::vector<uint8_t> reachable(blocks.size(), 0);
  std::vector<uint8_t> on_worklist(blocks.size(), 1);
  std::deque<size_t> worklist;
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    worklist.push_back(block_index);

  const auto state_units = [](const VectorLaneFlowState &state) {
    return state.slots.size() + state.scratch_slots.size() + state.restored_sgprs.size();
  };
  size_t worklist_visits = 0;
  const size_t max_worklist_visits = std::max<size_t>(4096, blocks.size() * 64);
  while (!worklist.empty()) {
    // Exact-summary selection is intentionally fail-closed, but switching
    // between a state-derived target and the ABI fallback is not monotone.
    // Abandon this optional recovery if malformed control flow oscillates.
    if (++worklist_visits > max_worklist_visits)
      return;
    const size_t block_index = worklist.front();
    worklist.pop_front();
    on_worklist[block_index] = 0;

    VectorLaneFlowState new_entry;
    bool new_reachable = false;
    bool have_predecessor_state = false;
    const auto meet_predecessor = [&](const VectorLaneFlowState &incoming) {
      new_reachable = true;
      if (!have_predecessor_state) {
        new_entry = incoming;
        have_predecessor_state = true;
        return;
      }
      new_entry.slots.intersect_with(incoming.slots);
      new_entry.scratch_slots.intersect_with(incoming.scratch_slots);
      new_entry.restored_sgprs.intersect_with(incoming.restored_sgprs);
      if (new_entry.vgpr_msb_imm != incoming.vgpr_msb_imm)
        new_entry.vgpr_msb_imm = std::nullopt;
      if (new_entry.gpr_idx_enabled != incoming.gpr_idx_enabled)
        new_entry.gpr_idx_enabled = std::nullopt;
    };
    for (size_t predecessor : predecessors[block_index]) {
      if (!reachable[predecessor])
        continue;
      const auto call = call_edges.find(predecessor);
      const bool call_target =
          call != call_edges.end() && call->second.target_successors.contains(block_index);
      if (call_target) {
        if (call->second.entry_state)
          meet_predecessor(*call->second.entry_state);
        else
          meet_predecessor(VectorLaneFlowState{});
      }
      if (!call_target ||
          (call != call_edges.end() && call->second.continuation_successor == block_index))
        meet_predecessor(exit_states[predecessor]);
    }

    // Generic callers conservatively infer predecessorless device-function
    // entries; callers with a complete entry list leave unlisted blocks at
    // BOTTOM. Explicit entries remain roots even with structural predecessors.
    // Meet a root's external state with any reachable predecessor: it
    // contributes no lane stash, and explicit entries begin in bank zero
    // according to the entry contract.
    if (is_analysis_root(block_index, external_entries, predecessors, entry_policy)) {
      new_reachable = true;
      VectorLaneFlowState external_entry;
      if (external_entries[block_index] != 0) {
        external_entry.vgpr_msb_imm = uint8_t{0};
        external_entry.gpr_idx_enabled = false;
      }
      if (!have_predecessor_state) {
        new_entry = std::move(external_entry);
      } else {
        new_entry.slots.clear();
        new_entry.scratch_slots.clear();
        new_entry.restored_sgprs.clear();
        if (new_entry.vgpr_msb_imm != external_entry.vgpr_msb_imm)
          new_entry.vgpr_msb_imm = std::nullopt;
        if (new_entry.gpr_idx_enabled != external_entry.gpr_idx_enabled)
          new_entry.gpr_idx_enabled = std::nullopt;
      }
    }
    if (!new_reachable)
      continue;

    std::optional<VectorLaneFlowState> new_call_entry;
    VectorLaneFlowState new_exit =
        scan_block(blocks[block_index], new_entry, false, &new_call_entry);
    if (analysis_budget_exhausted)
      return;
    auto call = call_edges.find(block_index);
    const bool call_entry_unchanged =
        new_call_entry ? call != call_edges.end() && call->second.entry_state == new_call_entry
                       : call == call_edges.end() || !call->second.entry_state;
    if (reachable[block_index] && entry_states[block_index] == new_entry &&
        exit_states[block_index] == new_exit && call_entry_unchanged)
      continue;

    // Count logical facts rather than unique COW allocations. This makes the
    // limit independent of sharing details and bounds both lane and restored
    // SGPR state together with the summary-cache units already retained.
    size_t next_retained_state_units = retained_state_units;
    const auto replace_fact_count = [&](size_t old_count, size_t new_count) {
      assert(next_retained_state_units >= old_count);
      next_retained_state_units -= old_count;
      if (new_count >
          kMaxRetainedAnalysisUnits - retained_summary_units - next_retained_state_units)
        return false;
      next_retained_state_units += new_count;
      return true;
    };
    if (!replace_fact_count(state_units(entry_states[block_index]), state_units(new_entry)) ||
        !replace_fact_count(state_units(exit_states[block_index]), state_units(new_exit)))
      return;
    if (call != call_edges.end()) {
      const size_t old_call_count =
          call->second.entry_state ? state_units(*call->second.entry_state) : 0;
      const size_t new_call_count = new_call_entry ? state_units(*new_call_entry) : 0;
      if (!replace_fact_count(old_call_count, new_call_count))
        return;
    }

    reachable[block_index] = 1;
    entry_states[block_index] = std::move(new_entry);
    exit_states[block_index] = std::move(new_exit);
    if (call != call_edges.end())
      call->second.entry_state = std::move(new_call_entry);
    else
      assert(!new_call_entry);
    retained_state_units = next_retained_state_units;
    for (size_t successor : blocks[block_index].successors) {
      if (on_worklist[successor])
        continue;
      worklist.push_back(successor);
      on_worklist[successor] = 1;
    }
  }

  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    if (!reachable[block_index])
      continue;
    const AnalysisBlock &block = blocks[block_index];
    bool has_consumer = false;
    for (size_t index = block.first_index; index <= block.last_index; ++index) {
      if (is_lane_fixup_consumer(ctx.facts[index])) {
        has_consumer = true;
        break;
      }
    }
    if (has_consumer) {
      (void)scan_block(block, entry_states[block_index], true, nullptr);
      if (analysis_budget_exhausted) {
        recovered.resize(initial_recovered_size);
        return;
      }
    }
  }
}

using AnalysisBlockIndex = std::unordered_map<uint64_t, size_t>;

[[nodiscard]] AnalysisBlockIndex index_analysis_blocks(const std::vector<AnalysisBlock> &blocks) {
  AnalysisBlockIndex block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    block_by_offset.emplace(blocks[block_index].offset, block_index);
  return block_by_offset;
}

struct AnalysisDominators {
  std::vector<size_t> block_for_instruction;
  std::vector<std::vector<uint64_t>> blocks;

  [[nodiscard]] bool instruction_dominates(size_t definition, size_t consumer) const {
    if (definition >= block_for_instruction.size() || consumer >= block_for_instruction.size())
      return false;
    const size_t definition_block = block_for_instruction[definition];
    const size_t consumer_block = block_for_instruction[consumer];
    if (definition_block >= blocks.size() || consumer_block >= blocks.size())
      return false;
    if (definition_block == consumer_block)
      return definition <= consumer;
    const size_t word = definition_block / 64u;
    const uint64_t mask = uint64_t{1} << (definition_block % 64u);
    return word < blocks[consumer_block].size() && (blocks[consumer_block][word] & mask) != 0;
  }
};

[[nodiscard]] AnalysisDominators
compute_analysis_dominators(const AnalysisContext &ctx, const std::vector<AnalysisBlock> &blocks,
                            const AnalysisBlockIndex &block_by_offset,
                            std::span<const IndirectCallFixup> recovered) {
  AnalysisDominators result;
  result.block_for_instruction.assign(ctx.insts.size(), blocks.size());
  const size_t word_count = (blocks.size() + 63u) / 64u;
  result.blocks.assign(blocks.size(), std::vector<uint64_t>(word_count, 0));
  std::vector<std::vector<size_t>> predecessors(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    for (size_t successor : blocks[block_index].successors) {
      if (successor < blocks.size())
        predecessors[successor].push_back(block_index);
    }
    for (size_t index = blocks[block_index].first_index;
         index <= blocks[block_index].last_index && index < result.block_for_instruction.size();
         ++index) {
      result.block_for_instruction[index] = block_index;
    }
  }

  // A recovered swappc call has both a callee edge and a return continuation.
  // The temporary CFG stores only the callee edge because setpc returns are
  // context-sensitive. Add the proven continuation solely for the dominance
  // graph so a valid save/restore chain can cross such a call.
  for (const IndirectCallFixup &fixup : recovered) {
    if (!fixup.source_is_call)
      continue;
    const auto source = block_by_offset.find(fixup.source_call_offset);
    const auto call_index = instruction_index_for_offset(ctx.insts, fixup.source_call_offset);
    if (source == block_by_offset.end() || !call_index)
      continue;
    const Instruction &call = *ctx.insts[*call_index];
    const uint64_t continuation_offset = call.src_loc() + static_cast<uint64_t>(call.size());
    const auto continuation = block_by_offset.find(continuation_offset);
    if (continuation == block_by_offset.end())
      continue;
    std::vector<size_t> &incoming = predecessors[continuation->second];
    if (std::ranges::find(incoming, source->second) == incoming.end())
      incoming.push_back(source->second);
  }

  std::vector<bool> roots(blocks.size(), false);
  std::vector<uint64_t> all(word_count, UINT64_MAX);
  if (!all.empty() && blocks.size() % 64u != 0)
    all.back() = (uint64_t{1} << (blocks.size() % 64u)) - 1u;
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    roots[block_index] = blocks[block_index].external_entry || predecessors[block_index].empty();
    result.blocks[block_index] = roots[block_index] ? std::vector<uint64_t>(word_count, 0) : all;
    result.blocks[block_index][block_index / 64u] |= uint64_t{1} << (block_index % 64u);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      if (roots[block_index])
        continue;
      std::vector<uint64_t> next = all;
      for (size_t predecessor : predecessors[block_index]) {
        for (size_t word = 0; word < word_count; ++word)
          next[word] &= result.blocks[predecessor][word];
      }
      next[block_index / 64u] |= uint64_t{1} << (block_index % 64u);
      if (next != result.blocks[block_index]) {
        result.blocks[block_index] = std::move(next);
        changed = true;
      }
    }
  }
  return result;
}

[[nodiscard]] bool register_ref_contains_vgpr(RegisterRef ref, uint16_t vgpr) {
  if (ref.cls != RegClass::VGPR)
    return false;
  const uint32_t width = std::max<uint16_t>(1, ref.width);
  return vgpr >= ref.index && static_cast<uint32_t>(vgpr) < ref.index + width;
}

[[nodiscard]] bool instruction_preserves_vgpr_lane(const AnalysisContext &ctx, size_t inst_index,
                                                   VgprLane tracked) {
  const Instruction &inst = *ctx.insts[inst_index];

  // Relative VGPR addressing can name a register not represented by the
  // decoder's nominal operand. It is outside this finite proof, even when the
  // displayed base register differs from the carrier.
  if (inst.mnemonic().starts_with("v_movrel"))
    return false;

  if (inst.mnemonic() == "v_writelane_b32") {
    const Operand *dst = inst.dst_operand(0);
    const auto dst_ref = dst == nullptr ? std::optional<RegisterRef>{} : dst->to_register_ref();
    if (dst_ref && register_ref_contains_vgpr(*dst_ref, tracked.vgpr)) {
      const auto transfer = fixed_lane_transfer(inst, ctx.wavefront_size);
      return transfer && transfer->kind == FixedLaneTransfer::Kind::Write &&
             transfer->lane != tracked;
    }
  }

  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *operand = inst.dst_operand(i);
    if (operand == nullptr)
      continue;
    const auto ref = operand->to_register_ref();
    if (ref && register_ref_contains_vgpr(*ref, tracked.vgpr))
      return false;
  }

  RegisterSet implicit_defs;
  inst.implicit_defs(implicit_defs);
  return !implicit_defs.contains({RegClass::VGPR, tracked.vgpr, 1});
}

[[nodiscard]] bool instruction_writes_sgpr(AnalysisContext &ctx, size_t inst_index, uint16_t sgpr) {
  ensure_written_sgprs(ctx, inst_index);
  return ctx.facts[inst_index].written_sgprs.contains(sgpr);
}

[[nodiscard]] bool instruction_range_preserves_lane(const AnalysisContext &ctx, size_t begin,
                                                    size_t end, VgprLane lane) {
  for (size_t index = begin; index < end; ++index) {
    if (!instruction_preserves_vgpr_lane(ctx, index, lane))
      return false;
  }
  return true;
}

[[nodiscard]] bool instruction_range_preserves_sgpr(AnalysisContext &ctx, size_t begin, size_t end,
                                                    uint16_t sgpr) {
  for (size_t index = begin; index < end; ++index) {
    if (instruction_writes_sgpr(ctx, index, sgpr))
      return false;
  }
  return true;
}

/// @brief Proven copies of abstract values used by a callee-preservation proof.
///
/// @details Values zero and one are the halves of the incoming return PC. The
/// remaining values are caller VGPR lanes that must be restored unchanged.
/// Keeping exact SGPR, VGPR-lane, and proven private-scratch locations explicit
/// proves the actual save/restore chain instead of assuming a named return pair
/// or a register-preservation ABI. Joins retain only copies proven on every
/// incoming path. EXEC-gated or otherwise unproven memory transfers fail closed.
struct ValueCopies {
  std::array<bool, REGISTER_SET_MAX_SGPRS> sgprs{};
  std::vector<VgprLane> vgpr_lanes;
  std::vector<ScratchLaneSlot> scratch_lanes;

  friend bool operator==(const ValueCopies &, const ValueCopies &) = default;
};

struct CalleeValueState {
  std::vector<ValueCopies> values;

  friend bool operator==(const CalleeValueState &, const CalleeValueState &) = default;
};

template <typename T> [[nodiscard]] bool contains_copy(const std::vector<T> &copies, T copy) {
  return std::ranges::find(copies, copy) != copies.end();
}

template <typename T> void add_copy(std::vector<T> &copies, T copy) {
  if (!contains_copy(copies, copy))
    copies.push_back(copy);
}

/// @pre Both states originate from the same callee entry and therefore track
/// the same two return-PC halves followed by the same protected carrier lanes.
void meet_callee_value_state(CalleeValueState &dst, const CalleeValueState &src) {
  assert(dst.values.size() == src.values.size());
  for (size_t value = 0; value < dst.values.size(); ++value) {
    for (uint16_t sgpr = 0; sgpr < REGISTER_SET_MAX_SGPRS; ++sgpr)
      dst.values[value].sgprs[sgpr] =
          dst.values[value].sgprs[sgpr] && src.values[value].sgprs[sgpr];
    std::erase_if(dst.values[value].vgpr_lanes, [&](VgprLane lane) {
      return !contains_copy(src.values[value].vgpr_lanes, lane);
    });
    std::erase_if(dst.values[value].scratch_lanes, [&](ScratchLaneSlot slot) {
      return !contains_copy(src.values[value].scratch_lanes, slot);
    });
  }
}

[[nodiscard]] bool state_has_sgpr_pair(const CalleeValueState &state, uint16_t pair_lo) {
  if (state.values.size() < 2u || pair_lo >= kMaxTrackedSgprPair)
    return false;
  return state.values[0].sgprs.at(pair_lo) && state.values[1].sgprs.at(pair_lo + 1u);
}

void transfer_callee_value_state(AnalysisContext &ctx, const AnalysisBlock &block,
                                 size_t inst_index, CalleeValueState &state) {
  const Instruction &inst = *ctx.insts[inst_index];
  const auto lane_transfer = fixed_lane_transfer(*ctx.insts[inst_index], ctx.wavefront_size);
  const auto scratch =
      ctx.arch == ROCJITSU_CODE_ARCH_GFX1250 ? gfx1250_scratch_dword(inst) : std::nullopt;
  const auto scratch_store =
      ctx.arch == ROCJITSU_CODE_ARCH_GFX1250 ? gfx1250_scratch_store_range(inst) : std::nullopt;
  const bool full_exec_scratch =
      scratch && gfx1250_full_exec_scratch_transfer(ctx, block, inst_index);

  std::vector<bool> lane_sources(state.values.size(), false);
  std::vector<std::vector<uint8_t>> scratch_store_lanes(state.values.size());
  std::vector<std::vector<uint8_t>> scratch_load_lanes(state.values.size());
  for (size_t value = 0; value < state.values.size(); ++value) {
    if (lane_transfer) {
      lane_sources[value] =
          lane_transfer->kind == FixedLaneTransfer::Kind::Write
              ? lane_transfer->sgpr < REGISTER_SET_MAX_SGPRS &&
                    state.values[value].sgprs[lane_transfer->sgpr]
              : contains_copy(state.values[value].vgpr_lanes, lane_transfer->lane);
    }
    if (!full_exec_scratch)
      continue;
    if (!scratch->load) {
      for (VgprLane lane : state.values[value].vgpr_lanes) {
        if (lane.vgpr == scratch->vgpr)
          scratch_store_lanes[value].push_back(lane.lane);
      }
    } else {
      for (ScratchLaneSlot slot : state.values[value].scratch_lanes) {
        if (slot.saddr == scratch->saddr && slot.byte_offset == scratch->byte_offset)
          scratch_load_lanes[value].push_back(slot.lane);
      }
    }
  }

  ensure_written_sgprs(ctx, inst_index);
  for (ValueCopies &copies : state.values) {
    for (uint16_t sgpr = 0; sgpr < REGISTER_SET_MAX_SGPRS; ++sgpr) {
      if (ctx.facts[inst_index].written_sgprs.contains(sgpr))
        copies.sgprs[sgpr] = false;
    }
    std::erase_if(copies.vgpr_lanes, [&](VgprLane lane) {
      return !instruction_preserves_vgpr_lane(ctx, inst_index, lane);
    });
    std::erase_if(copies.scratch_lanes, [&](ScratchLaneSlot slot) {
      return ctx.facts[inst_index].written_sgprs.contains(slot.saddr);
    });
    if (inst.mnemonic().starts_with("scratch_store")) {
      if (!scratch_store) {
        copies.scratch_lanes.clear();
      } else {
        std::erase_if(copies.scratch_lanes, [&](ScratchLaneSlot slot) {
          const uint64_t slot_begin = slot.byte_offset;
          const uint64_t slot_end = slot_begin + sizeof(uint32_t);
          const uint64_t store_begin = scratch_store->byte_offset;
          const uint64_t store_end = store_begin + scratch_store->byte_size;
          return slot.saddr != scratch_store->saddr ||
                 (slot_begin < store_end && store_begin < slot_end);
        });
      }
    } else if (inst.mnemonic().starts_with("scratch_atomic")) {
      copies.scratch_lanes.clear();
    }
  }

  for (size_t value = 0; value < state.values.size(); ++value) {
    ValueCopies &copies = state.values[value];
    if (lane_transfer && lane_sources[value]) {
      if (lane_transfer->kind == FixedLaneTransfer::Kind::Write) {
        add_copy(copies.vgpr_lanes, lane_transfer->lane);
      } else if (lane_transfer->sgpr < REGISTER_SET_MAX_SGPRS) {
        copies.sgprs[lane_transfer->sgpr] = true;
      }
    }
    if (full_exec_scratch && !scratch->load) {
      for (uint8_t lane : scratch_store_lanes[value]) {
        add_copy(copies.scratch_lanes, ScratchLaneSlot{.saddr = scratch->saddr,
                                                       .byte_offset = scratch->byte_offset,
                                                       .lane = lane});
      }
    }
    if (full_exec_scratch && scratch->load) {
      for (uint8_t lane : scratch_load_lanes[value])
        add_copy(copies.vgpr_lanes, VgprLane{.vgpr = scratch->vgpr, .lane = lane});
    }
  }
}

struct CalleeCallProtection {
  std::vector<VgprLane> vgpr_lanes;
  std::vector<uint16_t> scratch_saddrs;
};

[[nodiscard]] std::optional<CalleeCallProtection>
prepare_callee_value_state_for_call(CalleeValueState &state) {
  CalleeCallProtection protection;
  for (ValueCopies &copies : state.values) {
    copies.sgprs.fill(false);
    if (copies.vgpr_lanes.empty() && copies.scratch_lanes.empty())
      return std::nullopt;
    // A proven private-scratch copy survives the nested call independently of
    // every live VGPR copy of the same value, so no callee-body proof is needed
    // for those redundant registers. Without a scratch copy, recursively prove
    // every live carrier lane; do not substitute an architecture-specific ABI
    // assumption into this exact cross-architecture analysis.
    if (!copies.scratch_lanes.empty()) {
      for (ScratchLaneSlot slot : copies.scratch_lanes) {
        if (!gfx1250_call_preserves_scratch_base(slot.saddr) &&
            !contains_copy(protection.scratch_saddrs, slot.saddr)) {
          protection.scratch_saddrs.push_back(slot.saddr);
        }
      }
      continue;
    }
    for (VgprLane lane : copies.vgpr_lanes) {
      if (!contains_copy(protection.vgpr_lanes, lane))
        protection.vgpr_lanes.push_back(lane);
    }
  }
  return protection;
}

[[nodiscard]] bool
callee_preserves_vgpr_lanes(AnalysisContext &ctx, const std::vector<AnalysisBlock> &blocks,
                            const AnalysisBlockIndex &block_by_offset,
                            std::span<const IndirectCallFixup> recovered, uint64_t target_offset,
                            uint16_t return_sreg, std::span<const VgprLane> lanes,
                            std::span<const uint16_t> protected_sgprs,
                            std::vector<std::pair<uint64_t, uint16_t>> &active_callees) {
  const std::pair key{target_offset, return_sreg};
  if (std::ranges::find(active_callees, key) != active_callees.end())
    return false;
  active_callees.push_back(key);

  const bool result = [&] {
    const auto entry_it = block_by_offset.find(target_offset);
    if (entry_it == block_by_offset.end())
      return false;

    CalleeValueState entry_state;
    if (return_sreg >= kMaxTrackedSgprPair)
      return false;
    entry_state.values.resize(2u + lanes.size() + protected_sgprs.size());
    entry_state.values[0].sgprs[return_sreg] = true;
    entry_state.values[1].sgprs[return_sreg + 1u] = true;
    for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index)
      entry_state.values[2u + lane_index].vgpr_lanes.push_back(lanes[lane_index]);
    for (size_t sgpr_index = 0; sgpr_index < protected_sgprs.size(); ++sgpr_index) {
      const uint16_t sgpr = protected_sgprs[sgpr_index];
      if (sgpr >= REGISTER_SET_MAX_SGPRS)
        return false;
      entry_state.values[2u + lanes.size() + sgpr_index].sgprs[sgpr] = true;
    }

    std::vector<size_t> worklist{entry_it->second};
    std::vector<std::optional<CalleeValueState>> incoming(blocks.size());
    incoming[entry_it->second] = entry_state;
    bool found_return = false;

    const auto propagate = [&](size_t successor, const CalleeValueState &state) {
      if (successor >= blocks.size())
        return false;
      if (!incoming[successor]) {
        incoming[successor] = state;
        worklist.push_back(successor);
        return true;
      }
      CalleeValueState joined = *incoming[successor];
      meet_callee_value_state(joined, state);
      if (joined != *incoming[successor]) {
        incoming[successor] = std::move(joined);
        worklist.push_back(successor);
      }
      return true;
    };

    while (!worklist.empty()) {
      const size_t block_index = worklist.back();
      worklist.pop_back();
      if (block_index >= blocks.size() || !incoming[block_index])
        continue;

      const AnalysisBlock &block = blocks[block_index];
      CalleeValueState state = *incoming[block_index];
      for (size_t index = block.first_index; index <= block.last_index; ++index)
        transfer_callee_value_state(ctx, block, index, state);

      const size_t term_index = block.last_index;
      const Instruction &term = *ctx.insts[term_index];
      const InstructionFacts &facts = ctx.facts[term_index];
      const uint64_t continuation_offset = term.src_loc() + static_cast<uint64_t>(term.size());

      if (facts.call_sdst) {
        const auto delta = term.branch_offset_bytes();
        if (!delta)
          return false;
        const int64_t nested_target =
            static_cast<int64_t>(continuation_offset) + static_cast<int64_t>(*delta);
        const auto protection = prepare_callee_value_state_for_call(state);
        if (!protection || nested_target < 0 ||
            ((!protection->vgpr_lanes.empty() || !protection->scratch_saddrs.empty()) &&
             !callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, recovered,
                                          static_cast<uint64_t>(nested_target), *facts.call_sdst,
                                          protection->vgpr_lanes, protection->scratch_saddrs,
                                          active_callees))) {
          return false;
        }
        const auto continuation = block_by_offset.find(continuation_offset);
        if (continuation == block_by_offset.end() || !propagate(continuation->second, state))
          return false;
        continue;
      }

      if (facts.swappc_sdst) {
        bool found_target = false;
        const auto protection = prepare_callee_value_state_for_call(state);
        if (!protection)
          return false;
        for (const IndirectCallFixup &fixup : recovered) {
          if (fixup.source_call_offset != term.src_loc() || !fixup.source_is_call ||
              fixup.source_return_selector != *facts.swappc_sdst)
            continue;
          found_target = true;
          if ((!protection->vgpr_lanes.empty() || !protection->scratch_saddrs.empty()) &&
              !callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, recovered,
                                           fixup.source_target_offset, fixup.source_return_selector,
                                           protection->vgpr_lanes, protection->scratch_saddrs,
                                           active_callees)) {
            return false;
          }
        }
        if (!found_target)
          return false;
        const auto continuation = block_by_offset.find(continuation_offset);
        if (continuation == block_by_offset.end() || !propagate(continuation->second, state))
          return false;
        continue;
      }

      if (facts.setpc_ssrc) {
        // A recovered SETPC inside a callee is an internal branch, not the
        // callee's return.  Instrumentation detours use exactly this shape:
        // a direct branch reaches a getpc/add/setpc island, which transfers to
        // an out-of-line relay and eventually rejoins the original function.
        // Follow every proven target while carrying the preservation state
        // through the relay.  Treating the island SETPC as a return would ask
        // whether its temporary target pair still held the caller's return PC
        // and conservatively reject otherwise-preserving instrumented callees.
        bool recovered_internal_branch = false;
        for (const IndirectCallFixup &fixup : recovered) {
          if (fixup.source_call_offset != term.src_loc() || fixup.source_is_call ||
              fixup.source_call_selector != *facts.setpc_ssrc)
            continue;
          recovered_internal_branch = true;
          if (fixup.source_incomplete || !fixup.source_targets_exhaustive)
            return false;
          const auto target = block_by_offset.find(fixup.source_target_offset);
          if (target == block_by_offset.end() || !propagate(target->second, state))
            return false;
        }
        if (recovered_internal_branch)
          continue;

        if (!state_has_sgpr_pair(state, *facts.setpc_ssrc)) {
          return false;
        }
        for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
          if (!contains_copy(state.values[2u + lane_index].vgpr_lanes, lanes[lane_index])) {
            return false;
          }
        }
        for (size_t sgpr_index = 0; sgpr_index < protected_sgprs.size(); ++sgpr_index) {
          const uint16_t sgpr = protected_sgprs[sgpr_index];
          if (!state.values[2u + lanes.size() + sgpr_index].sgprs[sgpr])
            return false;
        }
        found_return = true;
        continue;
      }

      if ((term.flags() & (INDIRECT_BRANCH | PROGRAM_TERMINATOR)) != 0 || block.successors.empty())
        return false;
      for (size_t successor : block.successors) {
        if (!propagate(successor, state))
          return false;
      }
    }
    return found_return;
  }();

  active_callees.pop_back();
  return result;
}

void recover_lane_saved_call_targets(AnalysisContext &ctx, const std::vector<AnalysisBlock> &blocks,
                                     const AnalysisBlockIndex &block_by_offset,
                                     std::span<const IndirectCallFixup> known_recovered,
                                     std::vector<IndirectCallFixup> &recovered) {
  // Some device-call sequences preserve a reusable static target in exact VGPR
  // lanes while the target SGPR pair is used for call arguments:
  //
  //   getpc/add target
  //   writelane carrier, target_lo, lane_a
  //   writelane carrier, target_hi, lane_b
  //   swappc return, target
  //   ...
  //   readlane target_lo, carrier, lane_a
  //   readlane target_hi, carrier, lane_b
  //   swappc return, target
  //
  // Recover the later call only when an earlier proven call establishes the
  // value saved in both lanes, the caller does not clobber either lane, and
  // every intervening decoded callee returns without clobbering it. This is an
  // ISA dataflow proof; register numbers, targets, symbols, and instruction
  // offsets are all discovered from the code object.

  struct IndexedLaneRead {
    size_t index = 0;
    VgprLane lane;
  };
  struct IndexedLaneWrite {
    size_t index = 0;
    uint16_t sgpr = 0;
  };

  std::array<std::vector<IndexedLaneRead>, REGISTER_SET_MAX_SGPRS> reads_by_sgpr;
  std::unordered_map<uint32_t, std::vector<IndexedLaneWrite>> writes_by_lane;
  static_assert(std::numeric_limits<decltype(VgprLane::lane)>::max() < (1u << 8u),
                "lane_key packs the lane index into the low eight bits");
  const auto lane_key = [](VgprLane lane) {
    return (static_cast<uint32_t>(lane.vgpr) << 8u) | lane.lane;
  };
  for (size_t index = 0; index < ctx.insts.size(); ++index) {
    const auto transfer = fixed_lane_transfer(*ctx.insts[index], ctx.wavefront_size);
    if (!transfer)
      continue;
    if (transfer->kind == FixedLaneTransfer::Kind::Read) {
      // SGPRs outside the tracked write mask cannot be proven preserved. Drop
      // those reads conservatively instead of indexing them as valid restores.
      if (transfer->sgpr < REGISTER_SET_MAX_SGPRS)
        reads_by_sgpr[transfer->sgpr].push_back({.index = index, .lane = transfer->lane});
      continue;
    }
    writes_by_lane[lane_key(transfer->lane)].push_back({.index = index, .sgpr = transfer->sgpr});
  }

  const auto previous_read = [&](uint16_t sgpr, size_t before) -> const IndexedLaneRead * {
    if (sgpr >= REGISTER_SET_MAX_SGPRS)
      return nullptr;
    const auto &reads = reads_by_sgpr[sgpr];
    auto it = std::lower_bound(
        reads.begin(), reads.end(), before,
        [](const IndexedLaneRead &read, size_t value) { return read.index < value; });
    return it == reads.begin() ? nullptr : &*std::prev(it);
  };
  const auto previous_write = [&](VgprLane lane, size_t before) -> const IndexedLaneWrite * {
    const auto found = writes_by_lane.find(lane_key(lane));
    if (found == writes_by_lane.end())
      return nullptr;
    const auto &writes = found->second;
    auto it = std::lower_bound(
        writes.begin(), writes.end(), before,
        [](const IndexedLaneWrite &write, size_t value) { return write.index < value; });
    return it == writes.begin() ? nullptr : &*std::prev(it);
  };

  const AnalysisDominators dominators =
      compute_analysis_dominators(ctx, blocks, block_by_offset, known_recovered);
  // Revisit consumers already present in known_recovered. Each fixed-point
  // round may add call edges and therefore change dominance or preservation;
  // a still-valid proof must be re-emitted in this round, while a proof made
  // stale by the larger CFG must disappear and be downgraded by the caller.
  for (size_t consumer_index = 0; consumer_index < ctx.insts.size(); ++consumer_index) {
    const InstructionFacts &consumer_facts = ctx.facts[consumer_index];
    if (!consumer_facts.swappc_ssrc || !consumer_facts.swappc_sdst)
      continue;
    const uint64_t consumer_offset = ctx.insts[consumer_index]->src_loc();

    std::array<size_t, 2> restore_indices{};
    std::array<VgprLane, 2> saved_lanes{};
    bool complete = true;
    for (uint16_t half = 0; half < 2; ++half) {
      const uint16_t sgpr = static_cast<uint16_t>(*consumer_facts.swappc_ssrc + half);
      const IndexedLaneRead *restore = previous_read(sgpr, consumer_index);
      if (restore == nullptr) {
        complete = false;
        break;
      }
      restore_indices[half] = restore->index;
      saved_lanes[half] = restore->lane;
      if (!instruction_range_preserves_sgpr(ctx, restore_indices[half] + 1, consumer_index, sgpr)) {
        complete = false;
        break;
      }
    }
    if (!complete || saved_lanes[0] == saved_lanes[1])
      continue;

    std::array<size_t, 2> save_indices{};
    std::array<uint16_t, 2> saved_sgprs{};
    for (uint16_t half = 0; half < 2; ++half) {
      const IndexedLaneWrite *save = previous_write(saved_lanes[half], restore_indices[half]);
      if (save == nullptr) {
        complete = false;
        break;
      }
      save_indices[half] = save->index;
      saved_sgprs[half] = save->sgpr;
      if (!instruction_range_preserves_lane(ctx, save_indices[half] + 1, restore_indices[half],
                                            saved_lanes[half])) {
        complete = false;
        break;
      }
    }
    if (!complete || saved_sgprs[0] >= kMaxTrackedSgprPair ||
        saved_sgprs[1] != static_cast<uint16_t>(saved_sgprs[0] + 1))
      continue;
    // This recovery intentionally models one 64-bit PC pair. A future
    // lane-carried scalar analysis can generalize the same dominance and
    // preservation machinery to an arbitrary consecutive register run.
    for (uint16_t half = 0; half < 2; ++half) {
      if (!dominators.instruction_dominates(save_indices[half], consumer_index) ||
          !dominators.instruction_dominates(restore_indices[half], consumer_index)) {
        complete = false;
        break;
      }
    }
    if (!complete)
      continue;

    const size_t first_save = std::min(save_indices[0], save_indices[1]);
    const size_t last_save = std::max(save_indices[0], save_indices[1]);
    for (size_t index = first_save; index < last_save; ++index) {
      if (is_block_terminator(*ctx.insts[index]) ||
          ctx.insts[index]->src_loc() + static_cast<uint64_t>(ctx.insts[index]->size()) !=
              ctx.insts[index + 1]->src_loc()) {
        complete = false;
        break;
      }
    }
    if (!complete)
      continue;

    std::optional<size_t> origin_call_index;
    uint64_t origin_call_offset = 0;
    for (const IndirectCallFixup &fixup : known_recovered) {
      if (!fixup.source_is_call || fixup.source_call_selector != saved_sgprs[0] ||
          fixup.source_call_carrier != RegisterRef{RegClass::SGPR, saved_sgprs[0], 2})
        continue;
      const auto call_index = instruction_index_for_offset(ctx.insts, fixup.source_call_offset);
      if (!call_index || *call_index <= last_save || *call_index >= consumer_index)
        continue;
      if (fixup.source_recovery_end_offset > ctx.insts[save_indices[0]]->src_loc() ||
          fixup.source_recovery_end_offset > ctx.insts[save_indices[1]]->src_loc() ||
          !instruction_range_preserves_sgpr(ctx, save_indices[0] + 1, *call_index,
                                            saved_sgprs[0]) ||
          !instruction_range_preserves_sgpr(ctx, save_indices[1] + 1, *call_index, saved_sgprs[1]))
        continue;
      if (!origin_call_index || *call_index < *origin_call_index) {
        origin_call_index = *call_index;
        origin_call_offset = fixup.source_call_offset;
      }
    }
    if (!origin_call_index)
      continue;

    std::vector<const IndirectCallFixup *> origin_fixups;
    for (const IndirectCallFixup &fixup : known_recovered) {
      if (fixup.source_call_offset == origin_call_offset && fixup.source_is_call &&
          fixup.source_call_selector == saved_sgprs[0] &&
          fixup.source_call_carrier == RegisterRef{RegClass::SGPR, saved_sgprs[0], 2})
        origin_fixups.push_back(&fixup);
    }
    if (origin_fixups.empty())
      continue;

    const std::array<VgprLane, 2> lanes{saved_lanes[0], saved_lanes[1]};
    bool path_preserves = true;
    for (size_t index = last_save + 1; index < consumer_index; ++index) {
      if (index + 1 < ctx.insts.size() &&
          ctx.insts[index]->src_loc() + static_cast<uint64_t>(ctx.insts[index]->size()) !=
              ctx.insts[index + 1]->src_loc()) {
        path_preserves = false;
        break;
      }

      for (VgprLane lane : lanes) {
        if (!instruction_preserves_vgpr_lane(ctx, index, lane)) {
          path_preserves = false;
          break;
        }
      }
      if (!path_preserves)
        break;

      const Instruction &inst = *ctx.insts[index];
      const InstructionFacts &facts = ctx.facts[index];
      if (facts.call_sdst) {
        const auto delta = inst.branch_offset_bytes();
        const int64_t target = delta ? static_cast<int64_t>(inst.src_loc() + inst.size()) +
                                           static_cast<int64_t>(*delta)
                                     : -1;
        std::vector<std::pair<uint64_t, uint16_t>> active_callees;
        if (target < 0 ||
            !callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, known_recovered,
                                         static_cast<uint64_t>(target), *facts.call_sdst, lanes,
                                         std::span<const uint16_t>{}, active_callees)) {
          path_preserves = false;
          break;
        }
      } else if (facts.swappc_sdst) {
        bool found_target = false;
        for (const IndirectCallFixup &fixup : known_recovered) {
          if (fixup.source_call_offset != inst.src_loc() || !fixup.source_is_call ||
              fixup.source_return_selector != *facts.swappc_sdst)
            continue;
          found_target = true;
          std::vector<std::pair<uint64_t, uint16_t>> active_callees;
          if (!callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, known_recovered,
                                           fixup.source_target_offset, fixup.source_return_selector,
                                           lanes, std::span<const uint16_t>{}, active_callees)) {
            path_preserves = false;
            break;
          }
        }
        if (!found_target || !path_preserves) {
          path_preserves = false;
          break;
        }
      } else if (is_block_terminator(inst)) {
        path_preserves = false;
        break;
      }
    }
    if (!path_preserves)
      continue;

    for (const IndirectCallFixup *fixup : origin_fixups) {
      std::vector<std::pair<uint64_t, uint16_t>> active_callees;
      if (!callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, known_recovered,
                                       fixup->source_target_offset, *consumer_facts.swappc_sdst,
                                       lanes, std::span<const uint16_t>{}, active_callees)) {
        complete = false;
        break;
      }
    }
    if (!complete)
      continue;

    for (const IndirectCallFixup *fixup : origin_fixups) {
      IndirectCallFixup restored = *fixup;
      restored.source_call_offset = consumer_offset;
      restored.source_call_sreg = *consumer_facts.swappc_ssrc;
      restored.source_call_selector = *consumer_facts.swappc_ssrc;
      restored.source_call_carrier = RegisterRef{RegClass::SGPR, *consumer_facts.swappc_ssrc, 2};
      restored.source_is_call = true;
      restored.source_return_sreg = *consumer_facts.swappc_sdst;
      restored.source_return_selector = *consumer_facts.swappc_sdst;
      append_unique(recovered, restored);
    }
  }
}

void recover_signed_delta_templates(const AnalysisContext &ctx,
                                    const std::vector<AnalysisBlock> &blocks,
                                    std::vector<IndirectCallFixup> &recovered) {
  // Some compiler output uses one signed literal template for a PC-relative
  // setpc:
  //
  //   s_getpc_b64 pair
  //   s_add_i32 tmp, literal, 4
  //   s_cmp_ge_i32 tmp, 0
  //   s_cbranch_scc1 add_half
  // sub_half:
  //   s_abs_i32 tmp, tmp
  //   s_sub_u32 pair_lo, pair_lo, tmp
  //   s_subb_u32 pair_hi, pair_hi, 0
  //   s_setpc_b64 pair
  // add_half:
  //   s_add_u32 pair_lo, pair_lo, tmp
  //   s_addc_u32 pair_hi, pair_hi, 0
  //   s_setpc_b64 pair
  //
  // The temporary crosses a conditional branch, but only inside this closed
  // template. Tracking arbitrary temporary SGPR values would enlarge the
  // lattice and make every scalar write relevant. Instead, match the whole
  // template structurally and recover both consumers to the same target. The
  // add-half fixup intentionally reuses the subtract-half recovery range; final
  // relocation rewrites that contiguous range once and then ignores the
  // duplicate builder range.
  std::unordered_map<uint64_t, size_t> block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    block_by_offset.emplace(blocks[block_index].offset, block_index);

  const auto add_i32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddI32);
  if (!add_i32_opcode)
    return;

  for (const AnalysisBlock &entry : blocks) {
    if (entry.last_index < entry.first_index + 3)
      continue;

    const Instruction &term = *ctx.insts[entry.last_index];
    if ((term.flags() & COND_BRANCH) == 0)
      continue;
    const auto branch_delta = term.branch_offset_bytes();
    if (!branch_delta)
      continue;

    const size_t getpc_index = entry.last_index - 3;
    const size_t temp_index = entry.last_index - 2;
    const Instruction &getpc_inst = *ctx.insts[getpc_index];
    const Instruction &temp_inst = *ctx.insts[temp_index];
    auto pair_lo =
        scalar_pc_sreg(ctx.arch, getpc_inst, ctx.facts[getpc_index].word, ScalarPcOp::GetPc64);
    if (!pair_lo)
      continue;

    const uint32_t temp_word = ctx.facts[temp_index].word;
    const auto tmp_sreg = static_cast<uint16_t>((temp_word >> 16) & 0x7fu);
    uint32_t literal = 0;
    if (!sop2_literal_inline_to_sreg(temp_inst, temp_word,
                                     text_word_at(ctx.text, temp_inst.src_loc() + sizeof(uint32_t)),
                                     *add_i32_opcode, tmp_sreg, kInlineInt4, literal))
      continue;

    const uint64_t fallthrough_offset = term.src_loc() + static_cast<uint64_t>(term.size());
    const int64_t branch_target =
        static_cast<int64_t>(fallthrough_offset) + static_cast<int64_t>(*branch_delta);
    if (branch_target < 0)
      continue;

    auto sub_block_it = block_by_offset.find(fallthrough_offset);
    auto add_block_it = block_by_offset.find(static_cast<uint64_t>(branch_target));
    if (sub_block_it == block_by_offset.end() || add_block_it == block_by_offset.end())
      continue;

    auto sub_consumer =
        match_signed_delta_sub_consumer(ctx, blocks[sub_block_it->second], *pair_lo, tmp_sreg);
    auto add_consumer =
        match_signed_delta_add_consumer(ctx, blocks[add_block_it->second], *pair_lo, tmp_sreg);
    if (!sub_consumer || !add_consumer)
      continue;

    const uint64_t getpc_next = getpc_inst.src_loc() + static_cast<uint64_t>(getpc_inst.size());
    PcValue value{
        .offset = static_cast<int64_t>(getpc_next) +
                  static_cast<int64_t>(static_cast<int32_t>(literal)) + 4,
        .source_getpc_offset = getpc_inst.src_loc(),
        .source_recovery_begin_offset = getpc_next,
        .source_recovery_end_offset = sub_consumer->recovery_end,
    };

    // Both consumers name the same range, so both must ask for the same
    // replacement: patch_recovered_builder_fixups rewrites it once and requires
    // the duplicate to agree.
    if (auto fixup = fixup_for_value(ctx, sub_consumer->setpc_index, *pair_lo, value)) {
      fixup->source_requires_xcnt_drain = sub_consumer->skipped_xcnt;
      append_unique(recovered, *fixup);
    }
    if (auto fixup = fixup_for_value(ctx, *add_consumer, *pair_lo, value)) {
      fixup->source_requires_xcnt_drain = sub_consumer->skipped_xcnt;
      append_unique(recovered, *fixup);
    }
  }
}

[[nodiscard]] bool is_pc_dependency_wait(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  return mnemonic == "s_wait_alu" || mnemonic == "s_delay_alu" || mnemonic == "s_nop";
}

void recover_special_pair_pc_templates(const AnalysisContext &ctx,
                                       const std::vector<AnalysisBlock> &blocks,
                                       std::vector<IndirectCallFixup> &recovered) {
  // Special architectural pairs are deliberately excluded from the ordinary
  // SGPR lattice. Prove only a closed straight-line getpc/update/setpc template
  // using decoded register identities; any unmodeled instruction ends it.
  for (const AnalysisBlock &block : blocks) {
    for (size_t getpc_index = block.first_index; getpc_index <= block.last_index; ++getpc_index) {
      const Instruction &getpc_inst = *ctx.insts[getpc_index];
      auto carrier =
          scalar_pc_carrier(ctx.arch, getpc_inst, ctx.facts[getpc_index].word, ScalarPcOp::GetPc64);
      if (!carrier || carrier->ref.cls == RegClass::SGPR ||
          !is_supported_special_pc_carrier(carrier->ref))
        continue;

      const uint64_t getpc_next = getpc_inst.src_loc() + static_cast<uint64_t>(getpc_inst.size());
      PcValue value{
          .offset = static_cast<int64_t>(getpc_next),
          .source_getpc_offset = getpc_inst.src_loc(),
          .source_recovery_begin_offset = getpc_next,
          .source_recovery_end_offset = getpc_next,
      };

      for (size_t index = getpc_index + 1; index <= block.last_index; ++index) {
        const Instruction &inst = *ctx.insts[index];
        auto consumer =
            scalar_pc_carrier(ctx.arch, inst, ctx.facts[index].word, ScalarPcOp::SetPc64);
        if (consumer) {
          if (consumer->selector == carrier->selector && consumer->ref == carrier->ref) {
            if (auto fixup = fixup_for_value(ctx, index, carrier->selector, carrier->ref, value))
              append_unique(recovered, *fixup);
          }
          break;
        }

        if (auto pattern =
                match_temp_add_pattern(ctx, index, block.last_index, carrier->selector)) {
          value.offset += pattern->delta;
          value.source_recovery_end_offset = pattern->end_offset;
          index += pattern->instruction_count - 1;
          continue;
        }
        if (auto pattern =
                match_temp_sub_pattern(ctx, index, block.last_index, carrier->selector)) {
          value.offset += pattern->delta;
          value.source_recovery_end_offset = pattern->end_offset;
          index += pattern->instruction_count - 1;
          continue;
        }
        if (apply_high_pc_canonicalization(inst, ctx.facts[index].word, carrier->selector, value) ||
            apply_gfx1250_add_nc_u64_update(inst, ctx.facts[index].word, ctx.text, ctx.arch,
                                            carrier->ref, value) ||
            apply_pair_literal64_update(inst, carrier->ref, value) ||
            apply_low_literal_update(inst, ctx.facts[index].word, ctx.text, ctx.arch,
                                     carrier->selector, value) ||
            apply_high_carry_update(inst, ctx.facts[index].word, ctx.text, ctx.arch,
                                    carrier->selector, value) ||
            is_pc_dependency_wait(inst))
          continue;

        break;
      }
    }
  }
}

std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((inst.flags() & INDIRECT_CALL) == 0 || !inst.branch_offset_bytes())
    return std::nullopt;
  return static_cast<uint16_t>((word >> 16) & 0x7fu);
}

[[nodiscard]] std::vector<IndirectCallFixup> discover_indirect_branch_edges_unfiltered(
    std::span<const Instruction *const> insts, std::span<const uint8_t> text, rj_code_arch_t arch,
    std::span<const uint64_t> extra_leaders, ExternalEntryPolicy entry_policy,
    std::vector<PcAddressBuilder> *pc_builders, std::span<const uint64_t> extra_split_points,
    uint32_t wavefront_size, std::span<const uint64_t> external_entries) {
  std::vector<IndirectCallFixup> recovered;
  FixupIndex recovered_index;
  AnalysisContext ctx = build_context(insts, text, arch, wavefront_size);
  std::vector<uint64_t> sorted_extra_leaders(extra_leaders.begin(), extra_leaders.end());
  std::ranges::sort(sorted_extra_leaders);
  sorted_extra_leaders.erase(std::ranges::unique(sorted_extra_leaders).begin(),
                             sorted_extra_leaders.end());
  std::vector<uint64_t> sorted_external_entries(external_entries.begin(), external_entries.end());
  std::ranges::sort(sorted_external_entries);
  sorted_external_entries.erase(std::ranges::unique(sorted_external_entries).begin(),
                                sorted_external_entries.end());
  std::vector<uint64_t> leaders(sorted_extra_leaders);
  leaders.insert(leaders.end(), sorted_external_entries.begin(), sorted_external_entries.end());
  // A split point shapes the block graph but says nothing about how a block is entered. Keeping
  // these out of sorted_extra_leaders is the whole point of the distinction: under ExplicitOnly
  // every explicit entry is treated as externally entered, so promoting an ordinary helper to one
  // would discard the incoming SGPR-pair facts its real callers establish and leave otherwise
  // recoverable getpc flows unresolved.
  leaders.insert(leaders.end(), extra_split_points.begin(), extra_split_points.end());
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());

  PcAddressBuilderMap round_builders;
  for (size_t iteration = 0; iteration < kMaxIndirectDiscoveryIterations; ++iteration) {
    add_recovered_leaders(leaders, recovered);

    std::vector<AnalysisBlock> blocks =
        build_analysis_blocks(ctx, leaders, sorted_external_entries);
    add_recovered_successors(recovered, blocks);

    std::vector<PendingConsumer> pending_consumers;
    std::vector<IndirectCallFixup> iteration_recovered;
    // Lane-stash recovery consumes the same graph as scalar recovery, including
    // edges proven in earlier rounds. Keep the sorted external entries separate
    // from leaders: recovered targets become reachable through those edges, not
    // by being promoted to external roots.
    recover_vector_lane_stashed_pcs(ctx, blocks, recovered, iteration_recovered,
                                    sorted_external_entries, entry_policy);
    // Recovered leaders can split a block between rounds, which changes where a
    // builder's block-exit value is observed. Keep only the final round's view
    // so the published records are internally consistent with one CFG.
    round_builders.clear();
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
      scan_block(ctx, block_index, blocks, pending_consumers, iteration_recovered, round_builders);
    recover_signed_delta_templates(ctx, blocks, iteration_recovered);
    recover_special_pair_pc_templates(ctx, blocks, iteration_recovered);
    const AnalysisBlockIndex block_by_offset = index_analysis_blocks(blocks);
    std::vector<IndirectCallFixup> known_recovered = recovered;
    for (const IndirectCallFixup &fixup : iteration_recovered)
      append_unique(known_recovered, fixup);
    recover_lane_saved_call_targets(ctx, blocks, block_by_offset, known_recovered,
                                    iteration_recovered);

    if (!pending_consumers.empty()) {
      const auto entry_facts =
          run_block_dataflow(blocks, pending_consumers, sorted_external_entries, entry_policy);
      (void)classify_pending_consumers(ctx, blocks, entry_facts, pending_consumers,
                                       iteration_recovered);
    }

    const auto same_observation = [](const IndirectCallFixup &left,
                                     const IndirectCallFixup &right) {
      return left.source_call_offset == right.source_call_offset &&
             left.source_target_offset == right.source_target_offset &&
             left.source_call_selector == right.source_call_selector &&
             left.source_call_carrier == right.source_call_carrier &&
             left.source_getpc_offset == right.source_getpc_offset &&
             left.source_recovery_begin_offset == right.source_recovery_begin_offset &&
             left.source_recovery_end_offset == right.source_recovery_end_offset;
    };
    // Earlier rounds may have recovered a closed target before another
    // recovered edge exposed an additional predecessor. Retain that concrete
    // target for reachability and relocation, but never retain the stale claim
    // that it is exhaustive when the fixed-point graph no longer observes it.
    for (IndirectCallFixup &existing : recovered) {
      if (std::ranges::none_of(iteration_recovered, [&](const IndirectCallFixup &candidate) {
            return same_observation(existing, candidate);
          })) {
        const auto consumer_block = std::ranges::find_if(blocks, [&](const AnalysisBlock &block) {
          const uint64_t begin = ctx.insts[block.first_index]->src_loc();
          const Instruction &last = *ctx.insts[block.last_index];
          const uint64_t end = last.src_loc() + static_cast<uint64_t>(last.size());
          return existing.source_call_offset >= begin && existing.source_call_offset < end;
        });
        if (consumer_block != blocks.end() && !consumer_block->external_entry) {
          const size_t consumer_index =
              static_cast<size_t>(std::distance(blocks.begin(), consumer_block));
          const auto predecessor = std::ranges::find_if(blocks, [&](const AnalysisBlock &block) {
            return std::ranges::find(block.successors, consumer_index) != block.successors.end();
          });
          const bool has_second_predecessor =
              predecessor != blocks.end() &&
              std::ranges::find_if(std::next(predecessor), blocks.end(),
                                   [&](const AnalysisBlock &block) {
                                     return std::ranges::find(block.successors, consumer_index) !=
                                            block.successors.end();
                                   }) != blocks.end();
          if (predecessor != blocks.end() && !has_second_predecessor) {
            if (existing.source_call_carrier.cls != RegClass::SGPR) {
              // Adding a recovered consumer leader splits an otherwise closed
              // special-pair template at the transfer itself. That mechanical
              // split is not a new runtime path; preserve the proof unless the
              // consumer has become an external root or gained another incoming
              // edge.
              continue;
            }

            // A scalar builder and consumer can likewise be split between
            // fixed-point rounds. A predecessorless device function is not a
            // hardware entry, so its block-entry fact remains dataflow BOTTOM;
            // nevertheless, the sole predecessor's SET transfer proves that
            // every runtime path to the consumer overwrites the pair with this
            // exact builder. Preserve only that exact observation. A kill,
            // pass-through, different builder, external consumer, or additional
            // predecessor still takes the conservative downgrade below.
            const auto transfer = predecessor->transfers.find(existing.source_call_sreg);
            if (transfer != predecessor->transfers.end() &&
                transfer->second.kind == PairTransfer::Kind::Set) {
              const PcValue &value = transfer->second.value;
              if (value.offset >= 0 &&
                  static_cast<uint64_t>(value.offset) == existing.source_target_offset &&
                  value.source_getpc_offset == existing.source_getpc_offset &&
                  value.source_recovery_begin_offset == existing.source_recovery_begin_offset &&
                  value.source_recovery_end_offset == existing.source_recovery_end_offset) {
                continue;
              }
            }
          }
        }
        existing.source_incomplete = true;
        existing.source_targets_exhaustive = false;
      }
    }

    bool changed = false;
    for (const IndirectCallFixup &fixup : iteration_recovered)
      changed |= append_unique_indexed(recovered, recovered_index, fixup);
    if (!changed)
      break;
  }

  if (pc_builders != nullptr) {
    pc_builders->clear();
    pc_builders->reserve(round_builders.size());
    for (const auto &[getpc_offset, entry] : round_builders) {
      // Publish the disagreement flag on the copy that leaves this pass. It is kept off the stored
      // record so the equality test above, which decides whether a second observation conflicts,
      // keeps comparing only the observed value.
      PcAddressBuilder published = entry.record;
      published.poisoned = entry.poisoned;
      pc_builders->push_back(published);
    }
    std::ranges::sort(*pc_builders, {}, &PcAddressBuilder::source_getpc_offset);
  }

  std::ranges::sort(recovered, {}, &IndirectCallFixup::source_call_offset);
  return recovered;
}

} // namespace

bool is_callee_saved_vgpr(uint16_t phys_vgpr) {
  return phys_vgpr >= 40 && phys_vgpr <= 255 && ((phys_vgpr - 40) % 16) < 8;
}

bool is_callee_saved_sgpr(uint16_t sgpr) {
  // Intersection of CSR_AMDGPU_SGPRs and CSR_AMDGPU_SI_Gfx_SGPRs.
  return (sgpr >= 30 && sgpr <= 31) || (sgpr >= 64 && sgpr <= 71) || (sgpr >= 80 && sgpr <= 87) ||
         (sgpr >= 96 && sgpr <= 105);
}

std::vector<IndirectCallFixup> discover_indirect_branch_edges(
    std::span<const Instruction *const> insts, std::span<const uint8_t> text, rj_code_arch_t arch,
    std::span<const uint64_t> extra_leaders, ExternalEntryPolicy entry_policy,
    std::vector<PcAddressBuilder> *pc_builders, std::span<const uint64_t> extra_split_points) {
  if (pc_builders != nullptr)
    pc_builders->clear();
  if (insts.empty())
    return {};

  // Every recoverable edge ends at an indirect branch/call consumer. Most
  // generated kernels have none, so avoid building the auxiliary CFG and
  // running its dataflow passes when no fixup can possibly be produced. A
  // section with no dynamic transfer also has no consumer whose target could be
  // a stale PC, so leaving pc_builders empty here withholds a claim rather than
  // making a false one.
  const bool has_indirect_consumer = std::ranges::any_of(
      insts, [](const Instruction *inst) { return is_recoverable_indirect_consumer(*inst); });
  if (!has_indirect_consumer) {
#ifndef NDEBUG
    // Keep the cheap predicate coupled to every fixup producer. A future
    // recovery path for another consumer kind must extend the predicate above.
    const auto unfiltered = discover_indirect_branch_edges_unfiltered(
        insts, text, arch, extra_leaders, entry_policy, nullptr, extra_split_points,
        /*wavefront_size=*/0, extra_leaders);
    assert(unfiltered.empty() && "indirect-recovery prefilter skipped a fixup-producing consumer");
#endif
    return {};
  }

  return discover_indirect_branch_edges_unfiltered(insts, text, arch, extra_leaders, entry_policy,
                                                   pc_builders, extra_split_points,
                                                   /*wavefront_size=*/0, extra_leaders);
}

std::vector<IndirectCallFixup>
discover_indirect_branch_edges(std::span<const Instruction *const> insts,
                               std::span<const uint8_t> text, rj_code_arch_t arch,
                               std::span<const uint64_t> extra_leaders, uint32_t wavefront_size,
                               std::span<const uint64_t> external_entries) {
  if (insts.empty())
    return {};
  const bool has_indirect_consumer = std::ranges::any_of(
      insts, [](const Instruction *inst) { return is_recoverable_indirect_consumer(*inst); });
  if (!has_indirect_consumer)
    return {};
  const bool has_explicit_entries = !external_entries.empty();
  const std::span<const uint64_t> roots = has_explicit_entries ? external_entries : extra_leaders;
  return discover_indirect_branch_edges_unfiltered(insts, text, arch, extra_leaders,
                                                   has_explicit_entries
                                                       ? ExternalEntryPolicy::ExplicitOnly
                                                       : ExternalEntryPolicy::InferPredecessorless,
                                                   nullptr, {}, wavefront_size, roots);
}

} // namespace rocjitsu
