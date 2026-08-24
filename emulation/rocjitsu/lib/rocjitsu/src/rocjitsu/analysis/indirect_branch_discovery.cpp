// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/indirect_branch_discovery.h"

#include "rocjitsu/analysis/control_flow.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
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
  /// @brief Low SGPR of the pair the producing `s_getpc_b64` wrote.
  ///
  /// @details Not always the pair the eventual consumer reads. A lane-banked dispatcher builds
  /// the address in a scratch pair, stashes it through `v_writelane_b32`, and restores it into a
  /// different pair before the transfer. patch_recovered_builder_fixups regenerates only the add
  /// half and leaves the original getpc in place, so its replacement has to name this pair:
  /// naming the consumer's would pair a fresh add against a getpc that writes something else and
  /// materialize an address derived from an unrelated register. Fully determined by
  /// source_getpc_offset, so carrying it splits no lattice value that was previously merged.
  uint16_t source_sreg = 0;
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

  [[nodiscard]] bool test(uint16_t sgpr) const {
    if (sgpr < 64)
      return (lo & (uint64_t{1} << sgpr)) != 0;
    if (sgpr < REGISTER_SET_MAX_SGPRS)
      return (hi & (uint64_t{1} << (sgpr - 64))) != 0;
    return false;
  }

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
  bool setreg_vgpr_msb_fixup = false;
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
  case ROCJITSU_CODE_ARCH_CDNA5:
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
  case ROCJITSU_CODE_ARCH_CDNA5:
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

[[nodiscard]] std::optional<uint16_t> scalar_pc_sreg(rj_code_arch_t arch, const Instruction &inst,
                                                     uint32_t word, ScalarPcOp op) {
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
  if (op == ScalarPcOp::GetPc64)
    return static_cast<uint16_t>((word >> 16) & 0x7fu);
  return static_cast<uint16_t>(word & 0xffu);
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

void record_written_sgprs(const Instruction &inst, InstructionFacts &facts) {
  // This analysis only needs SGPR defs. Avoid the heavier def-use helper here:
  // computing use sets and vector metadata for every instruction was a major
  // cost on large generated kernels, and none of that information participates
  // in this lattice.
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
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
    record_written_sgprs(*ctx.insts[index], facts);
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
                                                   rj_code_arch_t arch, uint16_t pair_lo,
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
  if (arch != ROCJITSU_CODE_ARCH_CDNA5 || inst.mnemonic() != "s_add_nc_u64" ||
      inst.num_dst_operands() != 1 || inst.num_src_operands() != 2)
    return false;

  const Operand *dst = inst.dst_operand(0);
  const Operand *src0 = inst.src_operand(0);
  const Operand *src1 = inst.src_operand(1);
  if (dst == nullptr || src0 == nullptr || src1 == nullptr)
    return false;
  const auto dst_ref = dst->to_register_ref();
  const auto src0_ref = src0->to_register_ref();
  if (!dst_ref || !src0_ref || dst_ref->cls != RegClass::SGPR || src0_ref->cls != RegClass::SGPR ||
      dst_ref->index != pair_lo || src0_ref->index != pair_lo || dst_ref->width != 2 ||
      src0_ref->width != 2)
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

[[nodiscard]] std::optional<IndirectCallFixup> fixup_for_value(const AnalysisContext &ctx,
                                                               size_t inst_index, uint16_t pair_lo,
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
      .source_call_sreg = pair_lo,
      .source_builder_sreg = value.source_sreg,
      .source_is_call = ctx.facts[inst_index].swappc_sdst.has_value(),
      .source_return_sreg = ctx.facts[inst_index].swappc_sdst.value_or(0),
  };
}

/// @brief Identity of a fixup for deduplication: exactly the fields append_unique() compares.
struct FixupIdentity {
  uint64_t call;
  uint64_t target;
  uint64_t getpc;
  uint64_t begin;
  uint64_t end;
  uint16_t sreg;
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
    return h;
  }
};

[[nodiscard]] FixupIdentity fixup_identity(const IndirectCallFixup &fixup) {
  return {fixup.source_call_offset,         fixup.source_target_offset,
          fixup.source_getpc_offset,        fixup.source_recovery_begin_offset,
          fixup.source_recovery_end_offset, fixup.source_call_sreg};
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
           existing.source_call_sreg == fixup.source_call_sreg &&
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
                                            rj_code_target_id_t target_id) {
  // Phase 1a: collect cheap facts that are independent of CFG. We do not build
  // full def-use information here. Generic writes are intentionally lazy because
  // many instructions never interact with a PC-builder pair, and decoding all
  // their operands dominated runtime on large code objects.
  AnalysisContext ctx;
  ctx.insts = insts;
  ctx.text = text;
  ctx.arch = arch;
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *architecture = registry.find(arch);
  const IsaTargetDescriptor *target_architecture = registry.find(target_id);
  const IsaGpuTargetDescription *target =
      target_architecture != nullptr && target_architecture->architecture_id == arch
          ? registry.find_gpu_target(target_id)
          : nullptr;
  if (target == nullptr && architecture != nullptr)
    target = registry.find_default_gpu_target(*architecture);
  ctx.setreg_vgpr_msb_fixup = target != nullptr && target->capabilities.setreg_vgpr_msb_fixup;

  ctx.facts.resize(insts.size());
  for (size_t i = 0; i < insts.size(); ++i) {
    const Instruction &inst = *insts[i];
    InstructionFacts &facts = ctx.facts[i];
    facts.word = text_word_at(text, inst.src_loc());
    facts.getpc_sdst = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::GetPc64);
    facts.setpc_ssrc = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::SetPc64);
    facts.swappc_ssrc = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::SwapPc64);
    if (facts.setpc_ssrc && *facts.setpc_ssrc < kMaxTrackedSgprPair)
      ctx.consumer_pairs.set(*facts.setpc_ssrc);
    if (facts.swappc_ssrc && *facts.swappc_ssrc < kMaxTrackedSgprPair)
      ctx.consumer_pairs.set(*facts.swappc_ssrc);
    if (facts.swappc_ssrc)
      facts.swappc_sdst = static_cast<uint16_t>((facts.word >> 16) & 0x7fu);
    facts.call_sdst = s_call_sdst(inst, facts.word);
  }

  return ctx;
}

[[nodiscard]] std::vector<AnalysisBlock>
build_analysis_blocks(const AnalysisContext &ctx, std::span<const uint64_t> extra_leaders) {
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
    if (ctx.arch != ROCJITSU_CODE_ARCH_CDNA5)
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
    if (ctx.arch != ROCJITSU_CODE_ARCH_CDNA5)
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
    // A builder step must start exactly where the recorded range ends. If the
    // pair survived an instruction between the previous step and this one, that
    // instruction sits inside [begin, end) and would be NOP-erased by
    // patch_recovered_builder_fixups. Mark the value non-contiguous so the
    // whole-scope proof declines to rewrite the range.
    const bool adjacent = inst.src_loc() == updated.source_recovery_end_offset;
    if (!apply_gfx1250_add_nc_u64_update(inst, word, ctx.text, ctx.arch, pair_lo, updated) &&
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
                                         .source_sreg = pair_lo,
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

  // Visit blocks in reverse postorder. The least fixed point of this monotone
  // join does not depend on the visit order, but the order decides how many
  // times it is recomputed. A recovered indirect call gives its shared callee
  // block one predecessor per call site -- thousands of them on a large device
  // library -- and popping in block-index order then re-evaluates that
  // O(predecessors x pairs) join once for every predecessor that happens to be
  // processed after it. Reverse postorder evaluates a predecessor before its
  // successors wherever the graph is acyclic, so a block is normally recomputed
  // only for its back edges. On the largest hipTensor object this is the
  // difference between 84.8M block visits and 558k, and between 264 s and 0.7 s
  // per round.
  //
  // The order is a pure function of the block graph: roots are tried in
  // ascending block index, successors in their stored order, and the root loop
  // covers every block, so blocks unreachable from any earlier root still
  // receive a position and are still seeded onto the worklist below.
  std::vector<size_t> rpo_order;
  rpo_order.reserve(blocks.size());
  {
    std::vector<uint8_t> visited(blocks.size(), 0);
    std::vector<std::pair<size_t, size_t>> stack;
    for (size_t root = 0; root < blocks.size(); ++root) {
      if (visited[root])
        continue;
      visited[root] = 1;
      stack.emplace_back(root, 0);
      while (!stack.empty()) {
        const size_t node = stack.back().first;
        size_t &next_successor = stack.back().second;
        if (next_successor < blocks[node].successors.size()) {
          const size_t successor = blocks[node].successors[next_successor++];
          if (successor < blocks.size() && !visited[successor]) {
            visited[successor] = 1;
            stack.emplace_back(successor, 0);
          }
          continue;
        }
        rpo_order.push_back(node);
        stack.pop_back();
      }
    }
  }
  std::ranges::reverse(rpo_order);
  std::vector<size_t> rpo_position(blocks.size(), 0);
  for (size_t position = 0; position < rpo_order.size(); ++position)
    rpo_position[rpo_order[position]] = position;

  // Keyed by reverse-postorder position so the lowest position is always popped
  // next. Positions are unique, so this is a total order and the pop sequence is
  // fully determined by the block graph.
  std::set<size_t> worklist;
  std::vector<bool> on_worklist(blocks.size(), false);
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    worklist.insert(rpo_position[block_index]);
    on_worklist[block_index] = true;
  }

  while (!worklist.empty()) {
    const size_t block_index = rpo_order[*worklist.begin()];
    worklist.erase(worklist.begin());
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
      worklist.insert(rpo_position[successor]);
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

// Bound every variable-size fact retained by lane-table recovery. One unit is
// deliberately large enough for a lane-map node or restored-SGPR entry. Exact
// callee summaries are charged by their fixed object size, including a cache
// node allowance. Logical state facts are charged once per entry/exit/call
// state even when their backing storage is shared, so the bound is
// conservative. Recovery is optional and fails closed on exhaustion.
constexpr size_t kRetainedAnalysisUnitBytes = 64;
// The retained charge scales with block count, not with how many lane stashes an object actually
// uses: hipTensor's contraction kernels reach ~2.9M units across ~55k blocks and exhausted a 1<<20
// bound partway through, which abandons lane recovery for the WHOLE object and leaves every
// s_swap_pc_i64 unrecovered for the by-construction gate to refuse. The charge is deliberately
// conservative -- states share copy-on-write backing and are billed once per entry/exit/call even
// when identical -- so the nominal figure overstates real memory several times over. Measured peak
// RSS across the hipTensor objects this admits is 243 MB, 273 MB, 288 MB and 653 MB for a 3.4 MB
// input, against the corpus per-object budget of 4096 MiB, and the slowest takes 4.4 s of a 30 s
// budget. Exhaustion remains fail-closed, so a still larger object loses recovery rather than
// memory.
//
// Sized to admit the objects real workloads dispatch rather than to leave headroom. hipTensor's
// scale_contraction and trinary_scale_contraction tests dispatch 4.65 MB and 4.70 MB objects that
// 1<<24 still refused; at this bound they translate in 7.6 s and 1.06 GB, against a 30 s and
// 4096 MiB per-object budget.
//
// Raising it does NOT expose the loader to the multi-minute translations some very large objects
// take. Those are pre-existing and not governed by this constant: the 8.3 MB hipTensor object costs
// 510 s / 2.4 GB on unmodified develop at 1<<20, 541 s at 1<<24 and 552 s at 1<<26 -- about 6%
// across a 64x change in the bound, because the time is spent outside lane recovery almost
// entirely. Exhaustion stays fail-closed: an object that outgrows this loses recovery, not memory.
constexpr size_t kMaxRetainedAnalysisUnits = size_t{1} << 26;
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
  RestoredSgprFacts restored_sgprs;
  std::optional<uint8_t> vgpr_msb_imm;
  std::optional<bool> setreg_vgpr_msb_hazard;
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

void update_vgpr_mode(std::optional<uint8_t> &mode, std::optional<bool> &adjacent_hazard,
                      const Instruction &inst, std::span<const uint8_t> text,
                      bool setreg_vgpr_msb_fixup) {
  const std::optional<bool> incoming_hazard = adjacent_hazard;
  adjacent_hazard = false;
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_set_vgpr_msb") {
    if (setreg_vgpr_msb_fixup && incoming_hazard == std::optional<bool>{true})
      return;
    if (setreg_vgpr_msb_fixup && !incoming_hazard) {
      mode = std::nullopt;
      return;
    }
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
  const bool writes_mode = amdgpu::decode_vgpr_msb_hwreg(hwreg).id == amdgpu::MODE_HWREG;
  if (setreg_vgpr_msb_fixup && writes_mode) {
    if (mnemonic == "s_setreg_b32") {
      mode = std::nullopt;
      return;
    }
    if (const auto literal = instruction_literal(inst, text)) {
      const uint8_t mode_layout = static_cast<uint8_t>((*literal & amdgpu::VGPR_MSB_MODE_MASK) >>
                                                       amdgpu::VGPR_MSB_MODE_SHIFT);
      mode = amdgpu::mode_layout_to_set_vgpr_msb(mode_layout);
    } else {
      mode = std::nullopt;
    }
    adjacent_hazard = true;
    return;
  }
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
                            std::span<const uint8_t> text, rj_code_arch_t arch) {
  if (!isa_properties(arch).mode_has_gpr_idx_en)
    return;
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
  // back into an SGPR pair before swappc. Track only fixed-lane
  // writelane/readlane transport. Any ordinary write to the physical VGPR
  // invalidates every recorded lane, and any SGPR write invalidates a
  // reconstructed half.
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
  if (ctx.arch != ROCJITSU_CODE_ARCH_CDNA5)
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
        std::optional<bool> setreg_vgpr_msb_hazard;
        std::optional<bool> gpr_idx_enabled;
      };
      std::vector<WorkItem> worklist{
          {block_for_instruction[start->second], initial_mode, false, initial_gpr_idx_enabled}};
      std::unordered_set<uint64_t> visited;
      bool saw_return = false;
      std::optional<uint8_t> return_mode;
      std::optional<bool> return_gpr_idx_enabled;
      bool unsupported = false;
      while (!worklist.empty() && !unsupported) {
        const WorkItem item = worklist.back();
        worklist.pop_back();
        const uint64_t mode_key = item.mode ? *item.mode : uint64_t{256};
        const uint64_t hazard_key =
            item.setreg_vgpr_msb_hazard ? (*item.setreg_vgpr_msb_hazard ? 1u : 0u) : 2u;
        const uint64_t gpr_idx_key = item.gpr_idx_enabled ? (*item.gpr_idx_enabled ? 1u : 0u) : 2u;
        const uint64_t visit_key = (static_cast<uint64_t>(item.block_index) << 13) |
                                   (gpr_idx_key << 11) | (hazard_key << 9) | mode_key;
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
        std::optional<bool> setreg_vgpr_msb_hazard = item.setreg_vgpr_msb_hazard;
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
          update_vgpr_mode(mode, setreg_vgpr_msb_hazard, inst, ctx.text, ctx.setreg_vgpr_msb_fixup);
          update_gpr_idx_enabled(gpr_idx_enabled, inst, ctx.text, ctx.arch);
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
          worklist.push_back({successor, mode, setreg_vgpr_msb_hazard, gpr_idx_enabled});
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
      if (!state.restored_sgprs.empty()) {
        ensure_written_sgprs(ctx, index);
        ctx.facts[index].written_sgprs.for_each(
            [&](uint16_t sgpr) { state.restored_sgprs.erase(sgpr); });
      }

      if (facts.getpc_sdst && *facts.getpc_sdst < kMaxTrackedSgprPair) {
        const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
        builders.set_builder(*facts.getpc_sdst, PcValue{.offset = static_cast<int64_t>(next_offset),
                                                        .source_getpc_offset = inst.src_loc(),
                                                        .source_sreg = *facts.getpc_sdst,
                                                        .source_recovery_begin_offset = next_offset,
                                                        .source_recovery_end_offset = next_offset});
        continue;
      }
      if (try_apply_pair_update(ctx, index, builders))
        continue;

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
        const std::optional<CalleeSummary> callee_summary =
            delta && static_cast<int64_t>(next_offset) + *delta >= 0
                ? statically_bounded_callee_summary(
                      static_cast<uint64_t>(static_cast<int64_t>(next_offset) + *delta),
                      *facts.call_sdst, state.vgpr_msb_imm, state.gpr_idx_enabled)
                : std::nullopt;
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
        state.vgpr_msb_imm =
            callee_summary ? callee_summary->return_mode : std::optional<uint8_t>{};
        state.gpr_idx_enabled =
            callee_summary ? callee_summary->return_gpr_idx_enabled : std::optional<bool>{};
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
        state.vgpr_msb_imm =
            callee_summary ? callee_summary->return_mode : std::optional<uint8_t>{};
        state.gpr_idx_enabled =
            callee_summary ? callee_summary->return_gpr_idx_enabled : std::optional<bool>{};
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

      update_vgpr_mode(state.vgpr_msb_imm, state.setreg_vgpr_msb_hazard, inst, ctx.text,
                       ctx.setreg_vgpr_msb_fixup);
      update_gpr_idx_enabled(state.gpr_idx_enabled, inst, ctx.text, ctx.arch);
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
    return state.slots.size() + state.restored_sgprs.size();
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
      new_entry.restored_sgprs.intersect_with(incoming.restored_sgprs);
      if (new_entry.vgpr_msb_imm != incoming.vgpr_msb_imm)
        new_entry.vgpr_msb_imm = std::nullopt;
      if (new_entry.setreg_vgpr_msb_hazard != incoming.setreg_vgpr_msb_hazard)
        new_entry.setreg_vgpr_msb_hazard = std::nullopt;
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
        external_entry.setreg_vgpr_msb_hazard = false;
        external_entry.gpr_idx_enabled = false;
      }
      if (!have_predecessor_state) {
        new_entry = std::move(external_entry);
      } else {
        new_entry.slots.clear();
        new_entry.restored_sgprs.clear();
        if (new_entry.vgpr_msb_imm != external_entry.vgpr_msb_imm)
          new_entry.vgpr_msb_imm = std::nullopt;
        if (new_entry.setreg_vgpr_msb_hazard != external_entry.setreg_vgpr_msb_hazard)
          new_entry.setreg_vgpr_msb_hazard = std::nullopt;
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
        .source_sreg = *pair_lo,
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
    rj_code_target_id_t target) {
  std::vector<IndirectCallFixup> recovered;
  FixupIndex recovered_index;
  AnalysisContext ctx = build_context(insts, text, arch, target);
  std::vector<uint64_t> sorted_extra_leaders(extra_leaders.begin(), extra_leaders.end());
  std::ranges::sort(sorted_extra_leaders);
  sorted_extra_leaders.erase(std::ranges::unique(sorted_extra_leaders).begin(),
                             sorted_extra_leaders.end());
  // A split point shapes the block graph but says nothing about how a block is entered. Keeping
  // these out of sorted_extra_leaders is the whole point of the distinction: under ExplicitOnly
  // every explicit entry is treated as externally entered, so promoting an ordinary helper to one
  // would discard the incoming SGPR-pair facts its real callers establish and leave otherwise
  // recoverable getpc flows unresolved.
  std::vector<uint64_t> leaders(sorted_extra_leaders);
  leaders.insert(leaders.end(), extra_split_points.begin(), extra_split_points.end());
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());

  PcAddressBuilderMap round_builders;
  for (size_t iteration = 0; iteration < kMaxIndirectDiscoveryIterations; ++iteration) {
    add_recovered_leaders(leaders, recovered);

    std::vector<AnalysisBlock> blocks = build_analysis_blocks(ctx, leaders);
    add_recovered_successors(recovered, blocks);

    std::vector<PendingConsumer> pending_consumers;
    std::vector<IndirectCallFixup> iteration_recovered;
    // Lane-stash recovery consumes the same graph as scalar recovery, including
    // edges proven in earlier rounds. Keep the sorted explicit entries separate
    // from leaders: recovered targets become reachable through those edges, not
    // by being promoted to external roots.
    recover_vector_lane_stashed_pcs(ctx, blocks, recovered, iteration_recovered,
                                    sorted_extra_leaders, entry_policy);
    // Recovered leaders can split a block between rounds, which changes where a
    // builder's block-exit value is observed. Keep only the final round's view
    // so the published records are internally consistent with one CFG.
    round_builders.clear();
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
      scan_block(ctx, block_index, blocks, pending_consumers, iteration_recovered, round_builders);
    recover_signed_delta_templates(ctx, blocks, iteration_recovered);

    if (!pending_consumers.empty()) {
      const auto entry_facts =
          run_block_dataflow(blocks, pending_consumers, sorted_extra_leaders, entry_policy);
      (void)classify_pending_consumers(ctx, blocks, entry_facts, pending_consumers,
                                       iteration_recovered);
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
    std::vector<PcAddressBuilder> *pc_builders, std::span<const uint64_t> extra_split_points,
    rj_code_target_id_t target) {
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
        insts, text, arch, extra_leaders, entry_policy, nullptr, extra_split_points, target);
    assert(unfiltered.empty() && "indirect-recovery prefilter skipped a fixup-producing consumer");
#endif
    return {};
  }

  return discover_indirect_branch_edges_unfiltered(insts, text, arch, extra_leaders, entry_policy,
                                                   pc_builders, extra_split_points, target);
}

} // namespace rocjitsu
