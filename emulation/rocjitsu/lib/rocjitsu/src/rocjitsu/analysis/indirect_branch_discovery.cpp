// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/indirect_branch_discovery.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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
// The pass is split into five phases.
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
// Phase 5 - Recover exact 64-bit call targets that compiler code saved in two
// fixed VGPR lanes and later restored to a consecutive SGPR pair. This separate
// interprocedural proof starts only from calls already recovered by Phases 2-4,
// requires each save and restore to dominate the later consumer from every
// supplied entry, and follows the carrier lanes plus every callee return pair
// through exact SGPR and VGPR-lane copies across all intervening calls and CFG
// paths. EXEC-gated vector and memory copies are deliberately outside this
// proof: accepting them would require full-EXEC and private-memory alias
// analysis, and a conservative miss is preferable to an unproven CFG edge.
// Follow-up bd-1w9.9.6.1.1 records the semantic prerequisites for safely
// admitting either carrier class in the future.
//
// The five phases run to a small fixed point. The first round uses only direct
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

  friend bool operator==(const PcValue &, const PcValue &) = default;
};

struct VgprLane {
  uint16_t vgpr = 0;
  uint8_t lane = 0;

  friend bool operator==(const VgprLane &, const VgprLane &) = default;
};

struct FixedLaneTransfer {
  enum class Kind {
    Write,
    Read,
  };

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

struct AnalysisContext {
  std::span<const Instruction *const> insts;
  std::span<const uint8_t> text;
  rj_code_arch_t arch;
  uint32_t wavefront_size = 0;
  std::vector<InstructionFacts> facts;
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

  /// The section start or a caller-supplied entry. Such a block is a root even
  /// when lexical/direct CFG construction also gives it an incoming edge.
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

  // The lane selector is the raw VOP3 src1 field on every AMDGPU encoding
  // supported here. Decoder operand order differs because some targets expose
  // the tied destination before the scalar value and others expose it after.
  // Reading the encoded selector also distinguishes an inline lane from a
  // literal whose decoded numeric value happens to fall in the inline range.
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
  const uint16_t pair_lo = op == ScalarPcOp::GetPc64 ? static_cast<uint16_t>((word >> 16) & 0x7fu)
                                                     : static_cast<uint16_t>(word & 0xffu);
  // SOP1 source fields also encode inline constants and special registers.
  // Only a complete pair in the tracked general-purpose SGPR file can carry a
  // PC. Reject other selectors at the decoding boundary so no later analysis
  // can accidentally use the raw selector as an array index.
  if (pair_lo >= kMaxTrackedSgprPair)
    return std::nullopt;
  return pair_lo;
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
    if (auto ref = op->to_register_ref()) {
      const std::string_view mnemonic = inst.mnemonic();
      // LLVM models wave-mask operands using one physical SGPR in Wave32 and
      // an SGPR pair in Wave64. The generated decoder retains the maximum
      // encoded width for these operands, so normalize it before deciding
      // whether a PC-builder pair was clobbered. Without this, a Wave32 VOPC
      // destination in s3 falsely kills an adjacent builder in s[4:5].
      if (wavefront_size != 0 && ref->cls == RegClass::SGPR &&
          ((i == 0 && mnemonic.starts_with("v_cmp")) ||
           (i == 1 && mnemonic.starts_with("v_div_scale_"))))
        ref->width = wavefront_size == 32 ? 1 : 2;
      record_written_sgpr_ref(facts, *ref);
    }
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

[[nodiscard]] bool is_program_terminator(const Instruction &inst) {
  return (inst.flags() & PROGRAM_TERMINATOR) != 0;
}

[[nodiscard]] bool is_unconditional_branch(const Instruction &inst) {
  return (inst.flags() & BRANCH) && !(inst.flags() & COND_BRANCH);
}

[[nodiscard]] bool is_indirect_branch(const Instruction &inst) {
  return (inst.flags() & INDIRECT_BRANCH) != 0;
}

[[nodiscard]] bool is_block_terminator(const Instruction &inst) {
  return inst.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR);
}

[[nodiscard]] bool is_direct_call(const Instruction &inst) {
  return (inst.flags() & INDIRECT_CALL) != 0 && inst.branch_offset_bytes().has_value();
}

[[nodiscard]] bool has_no_direct_successor(const Instruction &inst) {
  // Indirect branches have no known target until this analysis recovers one.
  // Indirect calls still expose their ordinary fallthrough/return continuation
  // in the direct CFG, which is required for liveness and for callers that do
  // not care about the callee body.
  return is_program_terminator(inst) || is_indirect_branch(inst);
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

[[nodiscard]] bool sop1_same_sreg(const Instruction &inst, uint32_t word, std::string_view mnemonic,
                                  uint16_t sreg);

[[nodiscard]] std::optional<TempDeltaPattern> match_temp_add_pattern(const AnalysisContext &ctx,
                                                                     size_t index,
                                                                     size_t last_index,
                                                                     uint16_t pair_lo) {
  // Match:
  //   s_add_i32 tmp, literal, 4
  //   s_add_u32 pair_lo, pair_lo, tmp
  //   s_addc_u32 pair_hi, pair_hi, 0
  //
  // A one-instruction transfer cannot model this because the low-half add reads
  // a temporary whose value is not part of the lattice. Recognizing the compact
  // idiom as a single transfer lets the block scan keep tracking the pair
  // without adding arbitrary scalar-value analysis.
  if (index + 2 > last_index)
    return std::nullopt;

  const auto add_i32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddI32);
  const auto add_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddU32);
  const auto addc_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddcU32);
  if (!add_i32_opcode || !add_u32_opcode || !addc_u32_opcode)
    return std::nullopt;

  const Instruction &temp_inst = *ctx.insts[index];
  const Instruction &low_inst = *ctx.insts[index + 1];
  const Instruction &high_inst = *ctx.insts[index + 2];
  const uint32_t temp_word = ctx.facts[index].word;
  const uint32_t low_word = ctx.facts[index + 1].word;
  const uint32_t high_word = ctx.facts[index + 2].word;
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
      .instruction_count = 3,
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

[[nodiscard]] bool apply_high_pc_canonicalization(const Instruction &inst, uint32_t word,
                                                  uint16_t pair_lo, PcValue &value) {
  // LLVM canonicalizes the high half of a getpc result before applying some
  // negative PC-relative deltas:
  //
  //   s_getpc_b64       pair
  //   s_sext_i32_i16    pair_hi, pair_hi
  //   s_add_co_u32      pair_lo, pair_lo, literal
  //   s_add_co_ci_u32   pair_hi, pair_hi, -1
  //
  // GPU virtual addresses are canonical 48-bit values, so sign-extending the
  // already-known PC high half preserves the local text offset tracked here.
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
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool apply_pair_literal64_update(const Instruction &inst, uint16_t pair_lo,
                                               PcValue &value) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic != "s_add_nc_u64" && mnemonic != "s_sub_nc_u64")
    return false;

  const Operand *dst = inst.dst_operand(0);
  const Operand *src0 = inst.src_operand(0);
  const Operand *src1 = inst.src_operand(1);
  if (dst == nullptr || src0 == nullptr || src1 == nullptr)
    return false;

  const auto is_tracked_pair = [pair_lo](const Operand &operand) {
    const auto ref = operand.to_register_ref();
    return ref && ref->cls == RegClass::SGPR && ref->index == pair_lo && ref->width == 2;
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

  // Scalar pair arithmetic wraps modulo 2^64. Perform the update on the bit
  // representation so extreme literals cannot trigger signed-overflow UB.
  uint64_t offset_bits = 0;
  static_assert(sizeof(offset_bits) == sizeof(value.offset));
  std::memcpy(&offset_bits, &value.offset, sizeof(offset_bits));
  offset_bits = subtract ? offset_bits - *literal : offset_bits + *literal;
  std::memcpy(&value.offset, &offset_bits, sizeof(value.offset));
  value.source_recovery_end_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
  return true;
}

[[nodiscard]] std::optional<IndirectCallFixup> fixup_for_value(const AnalysisContext &ctx,
                                                               size_t inst_index, uint16_t pair_lo,
                                                               const PcValue &value,
                                                               bool targets_exhaustive = true) {
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
      .source_is_call = ctx.facts[inst_index].swappc_sdst.has_value(),
      .source_targets_exhaustive = targets_exhaustive,
      .source_return_sreg = ctx.facts[inst_index].swappc_sdst.value_or(0),
  };
}

bool append_unique(std::vector<IndirectCallFixup> &out, IndirectCallFixup fixup) {
  const auto duplicate = std::ranges::find_if(out, [&](const IndirectCallFixup &existing) {
    return existing.source_call_offset == fixup.source_call_offset &&
           existing.source_target_offset == fixup.source_target_offset &&
           existing.source_call_sreg == fixup.source_call_sreg;
  });
  if (duplicate != out.end()) {
    if (fixup.source_targets_exhaustive && !duplicate->source_targets_exhaustive) {
      duplicate->source_targets_exhaustive = true;
      return true;
    }
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
      if (return_pair < kMaxTrackedSgprPair) {
        facts.swappc_sdst = return_pair;
      } else {
        // A malformed/non-general destination cannot be modeled as either a
        // returning call or an ordinary recovered branch without inventing
        // hardware semantics, so reject the whole consumer.
        facts.swappc_ssrc.reset();
      }
    }
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
  // membership check on very large kernels.
  std::vector<uint8_t> leaders(ctx.insts.size(), 0);
  leaders.front() = 1;

  const uint64_t section_end =
      ctx.insts.back()->src_loc() + static_cast<uint64_t>(ctx.insts.back()->size());
  for (uint64_t leader : extra_leaders) {
    if (leader >= section_end)
      continue;
    if (auto index = instruction_index_for_offset(ctx.insts, leader))
      leaders[*index] = 1;
  }

  for (size_t i = 0; i < ctx.insts.size(); ++i) {
    const Instruction &inst = *ctx.insts[i];
    const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
    if (is_block_terminator(inst) && next_offset < section_end) {
      // Splitting after terminators makes each setpc/swappc the last
      // instruction in its analysis block. That is important because the block
      // transfer must summarize the state at the control-transfer boundary, not
      // after unrelated fallthrough instructions.
      if (i + 1 < ctx.insts.size() && ctx.insts[i + 1]->src_loc() == next_offset)
        leaders[i + 1] = 1;
    }

    if (auto delta = inst.branch_offset_bytes()) {
      const int64_t target = static_cast<int64_t>(next_offset) + static_cast<int64_t>(*delta);
      if (target >= 0 && static_cast<uint64_t>(target) < section_end) {
        if (auto index = instruction_index_for_offset(ctx.insts, static_cast<uint64_t>(target)))
          leaders[*index] = 1;
      }
    }
  }

  std::vector<AnalysisBlock> blocks;
  blocks.reserve(std::ranges::count(leaders, uint8_t{1}));
  const std::unordered_set<uint64_t> external_entry_offsets(external_entries.begin(),
                                                            external_entries.end());
  for (size_t i = 0; i < ctx.insts.size(); ++i) {
    if (leaders[i] == 0)
      continue;
    AnalysisBlock block;
    block.offset = ctx.insts[i]->src_loc();
    block.first_index = i;
    block.last_index = i;
    block.external_entry = i == 0 || external_entry_offsets.contains(block.offset);
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

void set_kill_transfer(AnalysisBlock &block, uint16_t pair_lo) {
  // KILL is weaker than SET for the same block: if the final state proves a
  // concrete builder, earlier dirty writes in the block should not downgrade it.
  if (pair_lo >= kMaxTrackedSgprPair)
    return;
  auto &transfer = block.transfers[pair_lo];
  if (transfer.kind != PairTransfer::Kind::Set)
    transfer.kind = PairTransfer::Kind::Kill;
}

void finalize_block_transfers(AnalysisBlock &block, const BlockState &state) {
  // Phase 2 block-exit summary:
  //
  // 1. Every still-live builder becomes SET. This overrides incoming facts for
  //    the same pair in Phase 3.
  // 2. Every dirty half that is not covered by a SET kills both possible pair
  //    interpretations: s[N:N+1] and, when N > 0, s[N-1:N].
  // 3. Pairs never mentioned in transfers are implicit PASS.
  for (uint16_t pair_lo : state.active_pairs()) {
    const PcValue *value = state.builder(pair_lo);
    if (value == nullptr)
      continue;
    PairTransfer &transfer = block.transfers[pair_lo];
    transfer.kind = PairTransfer::Kind::Set;
    transfer.value = *value;
  }

  for (uint16_t half = 0; half < REGISTER_SET_MAX_SGPRS; ++half) {
    if (!state.dirty(half))
      continue;
    if (!block.transfers.contains(half) || block.transfers[half].kind != PairTransfer::Kind::Set)
      set_kill_transfer(block, half);
    if (half > 0) {
      const uint16_t previous_pair = static_cast<uint16_t>(half - 1);
      if (!block.transfers.contains(previous_pair) ||
          block.transfers[previous_pair].kind != PairTransfer::Kind::Set)
        set_kill_transfer(block, previous_pair);
    }
  }
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
    if (auto pattern = match_temp_add_pattern(ctx, index, block.last_index, pair_lo)) {
      PcValue updated = *value;
      updated.offset += pattern->delta;
      updated.source_recovery_end_offset = pattern->end_offset;

      for (size_t i = 0; i < pattern->instruction_count; ++i)
        invalidate_written_sgprs(ctx, index + i, state, pair_lo);
      state.set_builder(pair_lo, updated);
      return pattern->instruction_count;
    }
    if (auto pattern = match_temp_sub_pattern(ctx, index, block.last_index, pair_lo)) {
      PcValue updated = *value;
      updated.offset += pattern->delta;
      updated.source_recovery_end_offset = pattern->end_offset;

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
  if (block.first_index + 2 > block.last_index)
    return std::nullopt;

  const auto add_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddU32);
  const auto addc_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::AddcU32);
  if (!add_u32_opcode || !addc_u32_opcode)
    return std::nullopt;

  const Instruction &low_inst = *ctx.insts[block.first_index];
  const Instruction &high_inst = *ctx.insts[block.first_index + 1];
  const Instruction &setpc_inst = *ctx.insts[block.first_index + 2];
  if (!sop2_sreg_inline_to_sreg(low_inst, ctx.facts[block.first_index].word, *add_u32_opcode,
                                pair_lo, pair_lo, tmp_sreg))
    return std::nullopt;
  if (!sop2_sreg_inline_zero_to_sreg(high_inst, ctx.facts[block.first_index + 1].word,
                                     *addc_u32_opcode, static_cast<uint16_t>(pair_lo + 1),
                                     static_cast<uint16_t>(pair_lo + 1)))
    return std::nullopt;
  auto setpc_sreg = scalar_pc_sreg(ctx.arch, setpc_inst, ctx.facts[block.first_index + 2].word,
                                   ScalarPcOp::SetPc64);
  if (!setpc_sreg || *setpc_sreg != pair_lo)
    return std::nullopt;
  return block.first_index + 2;
}

[[nodiscard]] std::optional<std::pair<size_t, uint64_t>>
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
  if (block.first_index + 3 > block.last_index)
    return std::nullopt;

  const auto sub_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubU32);
  const auto subb_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubbU32);
  if (!sub_u32_opcode || !subb_u32_opcode)
    return std::nullopt;

  const Instruction &abs_inst = *ctx.insts[block.first_index];
  const Instruction &low_inst = *ctx.insts[block.first_index + 1];
  const Instruction &high_inst = *ctx.insts[block.first_index + 2];
  const Instruction &setpc_inst = *ctx.insts[block.first_index + 3];
  if (!sop1_same_sreg(abs_inst, ctx.facts[block.first_index].word, "s_abs_i32", tmp_sreg))
    return std::nullopt;
  if (!sop2_sreg_inline_to_sreg(low_inst, ctx.facts[block.first_index + 1].word, *sub_u32_opcode,
                                pair_lo, pair_lo, tmp_sreg))
    return std::nullopt;
  if (!sop2_sreg_inline_zero_to_sreg(high_inst, ctx.facts[block.first_index + 2].word,
                                     *subb_u32_opcode, static_cast<uint16_t>(pair_lo + 1),
                                     static_cast<uint16_t>(pair_lo + 1)))
    return std::nullopt;
  auto setpc_sreg = scalar_pc_sreg(ctx.arch, setpc_inst, ctx.facts[block.first_index + 3].word,
                                   ScalarPcOp::SetPc64);
  if (!setpc_sreg || *setpc_sreg != pair_lo)
    return std::nullopt;
  return std::pair{block.first_index + 3,
                   high_inst.src_loc() + static_cast<uint64_t>(high_inst.size())};
}

bool try_apply_pair_update(AnalysisContext &ctx, size_t index, BlockState &state) {
  for (uint16_t pair_lo : state.active_pairs()) {
    const PcValue *value = state.builder(pair_lo);
    if (value == nullptr)
      continue;
    PcValue updated = *value;
    const Instruction &inst = *ctx.insts[index];
    const uint32_t word = ctx.facts[index].word;
    if (!apply_high_pc_canonicalization(inst, word, pair_lo, updated) &&
        !apply_pair_literal64_update(inst, pair_lo, updated) &&
        !apply_low_literal_update(inst, word, ctx.text, ctx.arch, pair_lo, updated) &&
        !apply_high_carry_update(inst, word, ctx.text, ctx.arch, pair_lo, updated))
      continue;

    invalidate_written_sgprs(ctx, index, state, pair_lo);
    state.set_builder(pair_lo, updated);
    return true;
  }
  return false;
}

void emit_fixups_for_values(const AnalysisContext &ctx, size_t inst_index, uint16_t pair_lo,
                            std::span<const PcValue> values,
                            std::vector<IndirectCallFixup> &recovered,
                            bool targets_exhaustive = true) {
  // A complete lattice value can contain multiple concrete targets. That is not
  // an error by itself; it represents a bounded static dispatch where different
  // predecessor paths materialize different PC constants before joining at one
  // setpc/swappc consumer.
  // A complete lattice is exhaustive only when every value names a target in
  // this text section. Dropping an out-of-section value is safe for CFG
  // construction, but consumers that prove a closed owner scope must still see
  // that the emitted local edges are only a subset of the runtime target set.
  targets_exhaustive =
      targets_exhaustive && std::ranges::all_of(values, [&](const PcValue &value) {
        return value.offset >= 0 && static_cast<uint64_t>(value.offset) < ctx.text.size();
      });
  for (const PcValue &value : values) {
    if (auto fixup = fixup_for_value(ctx, inst_index, pair_lo, value, targets_exhaustive))
      append_unique(recovered, *fixup);
  }
}

void scan_block(AnalysisContext &ctx, size_t block_index, std::vector<AnalysisBlock> &blocks,
                std::vector<PendingConsumer> &pending_consumers,
                std::vector<IndirectCallFixup> &recovered) {
  // Phase 2: run local transfer semantics for one straight-line block.
  //
  // This scan has no incoming lattice facts by design. A pair either becomes
  // known because this block builds it, becomes dirty because this block writes
  // it, or remains pristine and can be resolved later from block-entry dataflow.
  // Keeping those cases separate prevents stale predecessor facts from leaking
  // through an unmodeled in-block write.
  AnalysisBlock &block = blocks[block_index];
  BlockState state;

  for (size_t index = block.first_index; index <= block.last_index; ++index) {
    const Instruction &inst = *ctx.insts[index];
    const InstructionFacts &facts = ctx.facts[index];
    const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());

    if (facts.getpc_sdst && *facts.getpc_sdst < kMaxTrackedSgprPair) {
      // s_getpc_b64 writes the address of the following instruction. The
      // low/high add sequence edits this base to the eventual branch target.
      const uint16_t pair_lo = *facts.getpc_sdst;
      state.set_builder(pair_lo, PcValue{.offset = static_cast<int64_t>(next_offset),
                                         .source_getpc_offset = inst.src_loc(),
                                         .source_recovery_begin_offset = next_offset,
                                         .source_recovery_end_offset = next_offset});
      continue;
    }

    const std::optional<uint16_t> consumer_pair =
        facts.setpc_ssrc ? facts.setpc_ssrc : facts.swappc_ssrc;
    if (consumer_pair) {
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
      } else if (!state.pair_dirty(*consumer_pair)) {
        // The block did not touch the pair, so any useful fact must come from
        // predecessor blocks. Defer classification until the block-entry
        // lattice is available.
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

  finalize_block_transfers(block, state);
}

[[nodiscard]] std::unordered_map<uint16_t, LatticeValue>
compute_exit_facts(const std::unordered_map<uint16_t, LatticeValue> &entry,
                   const AnalysisBlock &block) {
  // Apply one block's transfer summary to an entry fact map. SET and KILL are
  // strong updates for their pair; PASS is represented by absence from the
  // transfer map, so the incoming entry value remains in `exit`.
  std::unordered_map<uint16_t, LatticeValue> exit = entry;
  for (const auto &[pair_lo, transfer] : block.transfers) {
    if (transfer.kind == PairTransfer::Kind::Set) {
      exit[pair_lo] =
          LatticeValue{.values = {transfer.value}, .incomplete = false, .killed = false};
    } else if (transfer.kind == PairTransfer::Kind::Kill) {
      exit[pair_lo] = LatticeValue{.values = {}, .incomplete = true, .killed = true};
    }
  }
  return exit;
}

[[nodiscard]] std::vector<std::unordered_map<uint16_t, LatticeValue>>
run_block_dataflow(const std::vector<AnalysisBlock> &blocks) {
  // Phase 3: compute block-entry facts to a fixed point.
  //
  // entry[B] = JOIN(exit[P]) for every predecessor P of B.
  //
  // Blocks with no predecessors keep an empty entry map. Empty does not mean
  // "known empty set"; it means every pair is at unconstrained kernel-entry
  // state unless a predecessor later mentions that pair. Consumers interpret a
  // missing fact as unresolved.
  std::vector<std::vector<size_t>> predecessors(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    for (size_t successor : blocks[block_index].successors)
      predecessors[successor].push_back(block_index);
  }

  std::vector<std::unordered_map<uint16_t, LatticeValue>> entry_facts(blocks.size());
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

    std::unordered_map<uint16_t, LatticeValue> new_entry;
    if (!predecessors[block_index].empty()) {
      std::vector<std::unordered_map<uint16_t, LatticeValue>> pred_exits;
      pred_exits.reserve(predecessors[block_index].size());
      std::set<uint16_t> mentioned_pairs;
      for (size_t predecessor : predecessors[block_index]) {
        pred_exits.push_back(compute_exit_facts(entry_facts[predecessor], blocks[predecessor]));
        for (const auto &[pair_lo, _] : pred_exits.back())
          mentioned_pairs.insert(pair_lo);
      }

      for (uint16_t pair_lo : mentioned_pairs) {
        LatticeValue joined;
        for (const auto &pred_exit : pred_exits) {
          auto it = pred_exit.find(pair_lo);
          if (it == pred_exit.end()) {
            // A missing predecessor fact means the pair is still at its
            // unconstrained kernel-entry value on that path. Joining a concrete
            // PC with an unconstrained value must not create a speculative CFG
            // edge, so the result becomes incomplete.
            joined.incomplete = true;
          } else {
            join_lattice_value(joined, it->second);
          }
        }
        new_entry.emplace(pair_lo, std::move(joined));
      }
    }

    if (new_entry == entry_facts[block_index])
      continue;

    entry_facts[block_index] = std::move(new_entry);
    for (size_t successor : blocks[block_index].successors) {
      if (on_worklist[successor])
        continue;
      worklist.push_back(successor);
      on_worklist[successor] = true;
    }
  }

  return entry_facts;
}

[[nodiscard]] size_t classify_pending_consumers(
    const AnalysisContext &ctx, const std::vector<AnalysisBlock> &blocks,
    const std::vector<std::unordered_map<uint16_t, LatticeValue>> &entry_facts,
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
    auto it = facts.find(consumer.pair_lo);
    if (it == facts.end() || it->second.values.empty() || it->second.killed) {
      ++unresolved;
      continue;
    }
    if (it->second.incomplete && it->second.values.size() >= kMaxIndirectTargetsPerConsumer) {
      ++unresolved;
      continue;
    }

    emit_fixups_for_values(ctx, consumer.inst_index, consumer.pair_lo, it->second.values, recovered,
                           !it->second.incomplete);
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

using AnalysisBlockIndex = std::unordered_map<uint64_t, size_t>;

[[nodiscard]] AnalysisBlockIndex index_analysis_blocks(const std::vector<AnalysisBlock> &blocks) {
  AnalysisBlockIndex block_by_offset;
  block_by_offset.reserve(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    block_by_offset.emplace(blocks[block_index].offset, block_index);
  return block_by_offset;
}

void add_recovered_successors(const std::vector<IndirectCallFixup> &recovered,
                              std::vector<AnalysisBlock> &blocks,
                              const AnalysisBlockIndex &block_by_offset) {
  // Recovered indirect edges are not part of the first direct-CFG graph, but
  // they are real control flow once proven. Feeding them into the next
  // fixed-point round lets facts flow through nested helper returns without
  // speculating about unknown indirect targets.
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
/// Keeping exact SGPR and VGPR-lane locations explicit proves the actual
/// save/restore chain instead of assuming a named return pair or a
/// register-preservation ABI. Joins retain only copies proven on every incoming
/// path. EXEC-gated or memory-backed transfers fail closed.
struct ValueCopies {
  std::array<bool, REGISTER_SET_MAX_SGPRS> sgprs{};
  std::vector<VgprLane> vgpr_lanes;

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
  }
}

[[nodiscard]] bool state_has_sgpr_pair(const CalleeValueState &state, uint16_t pair_lo) {
  if (state.values.size() < 2u || pair_lo >= kMaxTrackedSgprPair)
    return false;
  return state.values[0].sgprs.at(pair_lo) && state.values[1].sgprs.at(pair_lo + 1u);
}

void transfer_callee_value_state(AnalysisContext &ctx, size_t inst_index, CalleeValueState &state) {
  const auto lane_transfer = fixed_lane_transfer(*ctx.insts[inst_index], ctx.wavefront_size);

  std::vector<bool> lane_sources(state.values.size(), false);
  for (size_t value = 0; value < state.values.size(); ++value) {
    if (lane_transfer) {
      lane_sources[value] =
          lane_transfer->kind == FixedLaneTransfer::Kind::Write
              ? lane_transfer->sgpr < REGISTER_SET_MAX_SGPRS &&
                    state.values[value].sgprs[lane_transfer->sgpr]
              : contains_copy(state.values[value].vgpr_lanes, lane_transfer->lane);
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
  }
}

[[nodiscard]] std::optional<std::vector<VgprLane>>
prepare_callee_value_state_for_call(CalleeValueState &state) {
  std::vector<VgprLane> protected_lanes;
  for (ValueCopies &copies : state.values) {
    copies.sgprs.fill(false);
    if (copies.vgpr_lanes.empty())
      return std::nullopt;
    for (VgprLane lane : copies.vgpr_lanes) {
      if (!contains_copy(protected_lanes, lane))
        protected_lanes.push_back(lane);
    }
  }
  return protected_lanes;
}

[[nodiscard]] bool
callee_preserves_vgpr_lanes(AnalysisContext &ctx, const std::vector<AnalysisBlock> &blocks,
                            const AnalysisBlockIndex &block_by_offset,
                            std::span<const IndirectCallFixup> recovered, uint64_t target_offset,
                            uint16_t return_sreg, std::span<const VgprLane> lanes,
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
    entry_state.values.resize(2u + lanes.size());
    entry_state.values[0].sgprs[return_sreg] = true;
    entry_state.values[1].sgprs[return_sreg + 1u] = true;
    for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index)
      entry_state.values[2u + lane_index].vgpr_lanes.push_back(lanes[lane_index]);

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
        transfer_callee_value_state(ctx, index, state);

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
        const auto protected_lanes = prepare_callee_value_state_for_call(state);
        if (!protected_lanes || nested_target < 0 ||
            !callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, recovered,
                                         static_cast<uint64_t>(nested_target), *facts.call_sdst,
                                         *protected_lanes, active_callees)) {
          return false;
        }
        const auto continuation = block_by_offset.find(continuation_offset);
        if (continuation == block_by_offset.end() || !propagate(continuation->second, state))
          return false;
        continue;
      }

      if (facts.swappc_sdst) {
        bool found_target = false;
        const auto protected_lanes = prepare_callee_value_state_for_call(state);
        if (!protected_lanes)
          return false;
        for (const IndirectCallFixup &fixup : recovered) {
          if (fixup.source_call_offset != term.src_loc() || !fixup.source_is_call ||
              fixup.source_return_sreg != *facts.swappc_sdst)
            continue;
          found_target = true;
          if (!callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, recovered,
                                           fixup.source_target_offset, fixup.source_return_sreg,
                                           *protected_lanes, active_callees)) {
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
        if (!state_has_sgpr_pair(state, *facts.setpc_ssrc)) {
          return false;
        }
        for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
          if (!contains_copy(state.values[2u + lane_index].vgpr_lanes, lanes[lane_index])) {
            return false;
          }
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

  const AnalysisDominators dominators =
      compute_analysis_dominators(ctx, blocks, block_by_offset, known_recovered);
  for (size_t consumer_index = 0; consumer_index < ctx.insts.size(); ++consumer_index) {
    const InstructionFacts &consumer_facts = ctx.facts[consumer_index];
    if (!consumer_facts.swappc_ssrc || !consumer_facts.swappc_sdst)
      continue;
    const uint64_t consumer_offset = ctx.insts[consumer_index]->src_loc();
    if (std::ranges::any_of(known_recovered, [&](const IndirectCallFixup &fixup) {
          return fixup.source_call_offset == consumer_offset;
        }))
      continue;

    std::array<size_t, 2> restore_indices{};
    std::array<VgprLane, 2> saved_lanes{};
    bool complete = true;
    for (uint16_t half = 0; half < 2; ++half) {
      const uint16_t sgpr = static_cast<uint16_t>(*consumer_facts.swappc_ssrc + half);

      bool found_restore = false;
      for (size_t index = consumer_index; index-- > 0;) {
        const auto transfer = fixed_lane_transfer(*ctx.insts[index], ctx.wavefront_size);
        if (!transfer || transfer->kind != FixedLaneTransfer::Kind::Read || transfer->sgpr != sgpr)
          continue;
        restore_indices[half] = index;
        saved_lanes[half] = transfer->lane;
        found_restore = true;
        break;
      }

      if (!found_restore ||
          !instruction_range_preserves_sgpr(ctx, restore_indices[half] + 1, consumer_index, sgpr)) {
        complete = false;
        break;
      }
    }
    if (!complete || saved_lanes[0] == saved_lanes[1])
      continue;

    std::array<size_t, 2> save_indices{};
    std::array<uint16_t, 2> saved_sgprs{};
    for (uint16_t half = 0; half < 2; ++half) {
      bool found_save = false;
      for (size_t index = restore_indices[half]; index-- > 0;) {
        const auto transfer = fixed_lane_transfer(*ctx.insts[index], ctx.wavefront_size);
        if (!transfer || transfer->kind != FixedLaneTransfer::Kind::Write ||
            transfer->lane != saved_lanes[half])
          continue;
        save_indices[half] = index;
        saved_sgprs[half] = transfer->sgpr;
        found_save = true;
        break;
      }
      if (!found_save ||
          !instruction_range_preserves_lane(ctx, save_indices[half] + 1, restore_indices[half],
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
      if (!fixup.source_is_call || fixup.source_call_sreg != saved_sgprs[0])
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
          fixup.source_call_sreg == saved_sgprs[0])
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
                                         active_callees)) {
          path_preserves = false;
          break;
        }
      } else if (facts.swappc_sdst) {
        bool found_target = false;
        for (const IndirectCallFixup &fixup : known_recovered) {
          if (fixup.source_call_offset != inst.src_loc() || !fixup.source_is_call ||
              fixup.source_return_sreg != *facts.swappc_sdst)
            continue;
          found_target = true;
          std::vector<std::pair<uint64_t, uint16_t>> active_callees;
          if (!callee_preserves_vgpr_lanes(ctx, blocks, block_by_offset, known_recovered,
                                           fixup.source_target_offset, fixup.source_return_sreg,
                                           lanes, active_callees)) {
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
                                       lanes, active_callees)) {
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
      restored.source_is_call = true;
      restored.source_return_sreg = *consumer_facts.swappc_sdst;
      append_unique(recovered, restored);
    }
  }
}

void recover_signed_delta_templates(const AnalysisContext &ctx,
                                    const std::vector<AnalysisBlock> &blocks,
                                    const AnalysisBlockIndex &block_by_offset,
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
        .source_recovery_end_offset = sub_consumer->second,
    };

    if (auto fixup = fixup_for_value(ctx, sub_consumer->first, *pair_lo, value))
      append_unique(recovered, *fixup);
    if (auto fixup = fixup_for_value(ctx, *add_consumer, *pair_lo, value))
      append_unique(recovered, *fixup);
  }
}

std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((inst.flags() & INDIRECT_CALL) == 0 || !inst.branch_offset_bytes())
    return std::nullopt;
  return static_cast<uint16_t>((word >> 16) & 0x7fu);
}

} // namespace

std::vector<IndirectCallFixup>
discover_indirect_branch_edges(std::span<const Instruction *const> insts,
                               std::span<const uint8_t> text, rj_code_arch_t arch,
                               std::span<const uint64_t> extra_leaders, uint32_t wavefront_size,
                               std::span<const uint64_t> external_entries) {
  std::vector<IndirectCallFixup> recovered;
  if (insts.empty())
    return recovered;

  AnalysisContext ctx = build_context(insts, text, arch, wavefront_size);
  std::vector<uint64_t> leaders(extra_leaders.begin(), extra_leaders.end());
  if (external_entries.empty())
    external_entries = extra_leaders;

  for (size_t iteration = 0; iteration < kMaxIndirectDiscoveryIterations; ++iteration) {
    add_recovered_leaders(leaders, recovered);

    std::vector<AnalysisBlock> blocks = build_analysis_blocks(ctx, leaders, external_entries);
    const AnalysisBlockIndex block_by_offset = index_analysis_blocks(blocks);
    add_recovered_successors(recovered, blocks, block_by_offset);

    std::vector<PendingConsumer> pending_consumers;
    std::vector<IndirectCallFixup> iteration_recovered;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
      scan_block(ctx, block_index, blocks, pending_consumers, iteration_recovered);
    recover_signed_delta_templates(ctx, blocks, block_by_offset, iteration_recovered);
    std::vector<IndirectCallFixup> known_recovered = recovered;
    for (const IndirectCallFixup &fixup : iteration_recovered)
      append_unique(known_recovered, fixup);
    recover_lane_saved_call_targets(ctx, blocks, block_by_offset, known_recovered,
                                    iteration_recovered);

    size_t unresolved_consumers = 0;
    if (!pending_consumers.empty()) {
      const auto entry_facts = run_block_dataflow(blocks);
      unresolved_consumers = classify_pending_consumers(ctx, blocks, entry_facts, pending_consumers,
                                                        iteration_recovered);
    }

    bool changed = false;
    for (const IndirectCallFixup &fixup : iteration_recovered)
      changed |= append_unique(recovered, fixup);
    if (!changed || unresolved_consumers == 0)
      break;
  }

  std::ranges::sort(recovered, {}, &IndirectCallFixup::source_call_offset);
  return recovered;
}

} // namespace rocjitsu
