// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/indirect_branch_discovery.h"

#include "rocjitsu/analysis/control_flow.h"
#include "rocjitsu/code/patch/instruction_builder.h"
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
#include <limits>
#include <optional>
#include <set>
#include <string_view>
#include <unordered_map>
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
  uint64_t source_recovery_begin_offset = 0;
  uint64_t source_recovery_end_offset = 0;

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
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || inst.mnemonic() != "s_add_nc_u64" ||
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
      .source_is_call = ctx.facts[inst_index].swappc_sdst.has_value(),
      .source_return_sreg = ctx.facts[inst_index].swappc_sdst.value_or(0),
  };
}

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
    // Incompleteness is monotonic: if any iteration observes this fixup as
    // incomplete, the merged record must stay incomplete. Otherwise a later
    // fixed-point pass that rediscovers an earlier-complete fact as incomplete
    // would be dropped here, leaving the consumer wrongly eligible for a direct
    // window.
    duplicate->source_incomplete = duplicate->source_incomplete || fixup.source_incomplete;
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
                                            std::span<const uint8_t> text, rj_code_arch_t arch) {
  // Phase 1a: collect cheap facts that are independent of CFG. We do not build
  // full def-use information here. Generic writes are intentionally lazy because
  // many instructions never interact with a PC-builder pair, and decoding all
  // their operands dominated runtime on large code objects.
  AnalysisContext ctx;
  ctx.insts = insts;
  ctx.text = text;
  ctx.arch = arch;

  ctx.facts.resize(insts.size());
  for (size_t i = 0; i < insts.size(); ++i) {
    const Instruction &inst = *insts[i];
    InstructionFacts &facts = ctx.facts[i];
    facts.word = text_word_at(text, inst.src_loc());
    facts.getpc_sdst = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::GetPc64);
    facts.setpc_ssrc = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::SetPc64);
    facts.swappc_ssrc = scalar_pc_sreg(arch, inst, facts.word, ScalarPcOp::SwapPc64);
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
                          std::span<const uint64_t> extra_leaders) {
  std::vector<uint8_t> entries(blocks.size(), 0);
  if (!entries.empty())
    entries[0] = 1;
  for (size_t block_index = 1; block_index < blocks.size(); ++block_index) {
    if (std::ranges::find(extra_leaders, blocks[block_index].offset) != extra_leaders.end())
      entries[block_index] = 1;
  }
  return entries;
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
  const auto is_gfx1250_padding = [&](size_t index) {
    if (ctx.arch != ROCJITSU_CODE_ARCH_GFX1250)
      return false;
    const Instruction &inst = *ctx.insts[index];
    if (inst.mnemonic() == "s_prefetch_inst_pc_rel")
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

  size_t high_index = block.first_index + 1;
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
  if (!sop2_sreg_inline_to_sreg(low_inst, ctx.facts[block.first_index].word, *add_u32_opcode,
                                pair_lo, pair_lo, tmp_sreg))
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
  const auto sub_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubU32);
  const auto subb_u32_opcode = scalar_sop2_opcode(ctx.arch, ScalarSop2Op::SubbU32);
  if (!sub_u32_opcode || !subb_u32_opcode)
    return std::nullopt;

  const auto is_gfx1250_padding = [&](size_t index) {
    if (ctx.arch != ROCJITSU_CODE_ARCH_GFX1250)
      return false;
    const Instruction &inst = *ctx.insts[index];
    if (inst.mnemonic() == "s_prefetch_inst_pc_rel")
      return true;
    // Skip a prefetch-config move only when it clobbers neither the getpc pair
    // nor tmp_sreg (whose value s_abs_i32/s_sub_u32 below consume).
    if (inst.mnemonic() != "s_mov_b32" || inst.size() != sizeof(uint32_t))
      return false;
    const uint16_t dst = static_cast<uint16_t>((ctx.facts[index].word >> 16) & 0x7fu);
    return dst != pair_lo && dst != static_cast<uint16_t>(pair_lo + 1u) && dst != tmp_sreg;
  };

  size_t abs_index = block.first_index;
  while (abs_index <= block.last_index && is_gfx1250_padding(abs_index))
    ++abs_index;
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
  return std::pair{setpc_index, high_inst.src_loc() + static_cast<uint64_t>(high_inst.size())};
}

bool try_apply_pair_update(AnalysisContext &ctx, size_t index, BlockState &state) {
  for (uint16_t pair_lo : state.active_pairs()) {
    const PcValue *value = state.builder(pair_lo);
    if (value == nullptr)
      continue;
    PcValue updated = *value;
    const Instruction &inst = *ctx.insts[index];
    const uint32_t word = ctx.facts[index].word;
    if (!apply_gfx1250_add_nc_u64_update(inst, word, ctx.text, ctx.arch, pair_lo, updated) &&
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
      } else if (*consumer_pair < kMaxTrackedSgprPair && !state.pair_dirty(*consumer_pair)) {
        // The block did not touch the pair, so any useful fact must come from
        // predecessor blocks. Defer classification until the block-entry
        // lattice is available. Out-of-range selectors cannot name a tracked
        // SGPR pair and deliberately remain unresolved.
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

[[nodiscard]] std::vector<LatticeFacts>
run_block_dataflow(const std::vector<AnalysisBlock> &blocks,
                   std::span<const PendingConsumer> pending_consumers,
                   std::span<const uint64_t> extra_leaders) {
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
  const std::vector<uint8_t> external_entries = explicit_external_entries(blocks, extra_leaders);

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
  // they have structural predecessors. Blocks with no predecessors remain
  // conservative analysis roots because they may also be externally entered.
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
        external_entries[block_index] != 0 || predecessors[block_index].empty() ||
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

struct VectorLaneFlowState {
  std::unordered_map<VectorLaneSlot, StashedPcHalf, VectorLaneSlotHash> slots;
  std::optional<uint8_t> vgpr_msb_imm;

  friend bool operator==(const VectorLaneFlowState &, const VectorLaneFlowState &) = default;
};

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

void recover_vector_lane_stashed_pcs(AnalysisContext &ctx,
                                     std::vector<IndirectCallFixup> &recovered,
                                     std::span<const uint64_t> extra_leaders) {
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
  // The gfx1250 A0 trap-recovery workaround requires S_SET_VGPR_MSB SIMM16[15:8]
  // to carry the previous bank state. The architectural bank update remains
  // SIMM16[7:0], so analysis intentionally ignores the workaround metadata byte.
  // This pass only observes MODE; it never inserts or reorders
  // S_SETREG/S_SET_VGPR_MSB and therefore cannot violate the required co-issue
  // spacing.
  if (ctx.arch != ROCJITSU_CODE_ARCH_GFX1250)
    return;

  const std::vector<AnalysisBlock> blocks = build_analysis_blocks(ctx, extra_leaders);
  if (blocks.empty())
    return;
  const std::vector<uint8_t> external_entries = explicit_external_entries(blocks, extra_leaders);

  const auto changes_vgpr_msb_bank = [](std::string_view mnemonic) {
    return mnemonic == "s_set_vgpr_msb" || mnemonic == "s_setreg_b32" ||
           mnemonic == "s_setreg_imm32_b32";
  };

  constexpr unsigned kDstBankShift = 6;  // s_set_vgpr_msb immediate DST field.
  constexpr unsigned kSrc0BankShift = 0; // s_set_vgpr_msb immediate SRC0 field.

  const auto scan_block = [&](const AnalysisBlock &block, VectorLaneFlowState state,
                              bool emit_fixups) {
    BlockState builders;
    std::array<std::optional<StashedPcHalf>, REGISTER_SET_MAX_SGPRS> read_halves;
    std::set<uint16_t> active_read_halves;

    const auto physical_vgpr = [&](uint16_t low, unsigned bank_shift) -> std::optional<uint16_t> {
      if (!state.vgpr_msb_imm)
        return std::nullopt;
      const uint8_t bank = static_cast<uint8_t>((*state.vgpr_msb_imm >> bank_shift) & 0x3u);
      return static_cast<uint16_t>(low + static_cast<uint16_t>(bank) * 256u);
    };

    for (size_t index = block.first_index; index <= block.last_index; ++index) {
      const Instruction &inst = *ctx.insts[index];
      const InstructionFacts &facts = ctx.facts[index];
      const std::string_view mnemonic = inst.mnemonic();

      if (!active_read_halves.empty()) {
        ensure_written_sgprs(ctx, index);
        ctx.facts[index].written_sgprs.for_each([&](uint16_t sgpr) {
          read_halves[sgpr].reset();
          active_read_halves.erase(sgpr);
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

      if (facts.call_sdst) {
        // A direct call can clobber any caller-saved VGPR before its
        // fallthrough continuation executes. The temporary CFG has no
        // context-sensitive return edge, so no pre-call lane stash is proven
        // to survive into either successor.
        state.slots.clear();
      }

      if (mnemonic == "v_writelane_b32") {
        const auto dst = operand_register(inst.dst_operand(0), RegClass::VGPR);
        const auto src = operand_register(inst.src_operand(0), RegClass::SGPR);
        const auto lane = inline_lane(inst.src_operand(1));
        const auto dst_phys = dst ? physical_vgpr(dst->index, kDstBankShift) : std::nullopt;
        if (dst && lane) {
          if (dst_phys) {
            const VectorLaneSlot written_slot{*dst_phys, *lane};
            state.slots.erase(written_slot);
            if (src) {
              for (uint16_t pair_lo : builders.active_pairs()) {
                const PcValue *value = builders.builder(pair_lo);
                if (value == nullptr || (src->index != pair_lo && src->index != pair_lo + 1))
                  continue;
                state.slots[written_slot] =
                    StashedPcHalf{.value = *value, .high = src->index == pair_lo + 1};
                break;
              }
            }
          } else {
            // The destination bank is unknown. It may overwrite any physical
            // register with this low selector, so invalidate all four banks.
            std::erase_if(state.slots, [&](const auto &item) {
              return (item.first.vgpr & 0xffu) == (dst->index & 0xffu) && item.first.lane == *lane;
            });
          }
        }
        invalidate_written_sgprs(ctx, index, builders);
        continue;
      }

      if (mnemonic == "v_readlane_b32") {
        const auto dst = operand_register(inst.dst_operand(0), RegClass::SGPR);
        const auto src = operand_register(inst.src_operand(0), RegClass::VGPR);
        const auto lane = inline_lane(inst.src_operand(1));
        invalidate_written_sgprs(ctx, index, builders);
        const auto src_phys = src ? physical_vgpr(src->index, kSrc0BankShift) : std::nullopt;
        if (dst && lane && src_phys) {
          auto slot = state.slots.find(VectorLaneSlot{*src_phys, *lane});
          if (slot != state.slots.end()) {
            read_halves[dst->index] = slot->second;
            active_read_halves.insert(dst->index);
          }
        }
        continue;
      }

      if (emit_fixups && facts.swappc_ssrc &&
          static_cast<size_t>(*facts.swappc_ssrc + 1) < read_halves.size()) {
        const uint16_t pair_lo = *facts.swappc_ssrc;
        const auto &lo = read_halves[pair_lo];
        const auto &hi = read_halves[pair_lo + 1];
        if (lo && hi && !lo->high && hi->high && lo->value == hi->value) {
          if (auto fixup = fixup_for_value(ctx, index, pair_lo, lo->value))
            append_unique(recovered, *fixup);
        }
      }
      if (facts.swappc_sdst) {
        // A returning indirect call may execute arbitrary callee code before
        // the fallthrough continuation. Resolve this call from the pre-call
        // state above, then discard every lane stash before publishing the
        // block exit so a callee-clobbered value cannot reach the continuation.
        state.slots.clear();
      }

      RegisterSet vgpr_defs;
      for (int dst_index = 0; dst_index < inst.num_dst_operands(); ++dst_index) {
        const Operand *op = inst.dst_operand(dst_index);
        if (op == nullptr)
          continue;
        if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::VGPR)
          vgpr_defs.expand(*ref);
      }
      inst.implicit_defs(vgpr_defs);
      vgpr_defs.for_each([&](RegisterRef ref) {
        if (ref.cls != RegClass::VGPR)
          return;
        // Operand metadata does not expose a role for every implicit/wide def.
        // Conservatively invalidate every physical bank sharing this selector.
        std::erase_if(state.slots, [&](const auto &item) {
          return (item.first.vgpr & 0xffu) == (ref.index & 0xffu);
        });
      });

      if (!builders.active_pairs().empty())
        invalidate_written_sgprs(ctx, index, builders);

      if (changes_vgpr_msb_bank(mnemonic)) {
        if (mnemonic == "s_set_vgpr_msb") {
          if (const auto *imm = inst.src_operand(0))
            state.vgpr_msb_imm = static_cast<uint8_t>(imm->encoding_value() & 0xffu);
          else
            state.vgpr_msb_imm = std::nullopt;
        } else {
          // Without scalar constant propagation, a SETREG write to MODE makes
          // the operand-bank mapping unknown. Physical slots remain intact.
          state.vgpr_msb_imm = std::nullopt;
        }
      }
    }
    return state;
  };

  std::vector<std::vector<size_t>> predecessors(blocks.size());
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    for (size_t successor : blocks[block_index].successors)
      predecessors[successor].push_back(block_index);
  }

  std::vector<VectorLaneFlowState> entry_states(blocks.size());
  std::vector<VectorLaneFlowState> exit_states(blocks.size());
  std::vector<uint8_t> reachable(blocks.size(), 0);
  std::vector<uint8_t> on_worklist(blocks.size(), 1);
  std::deque<size_t> worklist;
  for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    worklist.push_back(block_index);

  while (!worklist.empty()) {
    const size_t block_index = worklist.front();
    worklist.pop_front();
    on_worklist[block_index] = 0;

    VectorLaneFlowState new_entry;
    bool new_reachable = false;
    bool have_predecessor_state = false;
    for (size_t predecessor : predecessors[block_index]) {
      if (!reachable[predecessor])
        continue;
      new_reachable = true;
      if (!have_predecessor_state) {
        new_entry = exit_states[predecessor];
        have_predecessor_state = true;
        continue;
      }

      for (auto it = new_entry.slots.begin(); it != new_entry.slots.end();) {
        auto incoming = exit_states[predecessor].slots.find(it->first);
        if (incoming == exit_states[predecessor].slots.end() || incoming->second != it->second)
          it = new_entry.slots.erase(it);
        else
          ++it;
      }
      if (new_entry.vgpr_msb_imm != exit_states[predecessor].vgpr_msb_imm)
        new_entry.vgpr_msb_imm = std::nullopt;
    }

    // A block with no direct predecessors is a possible device-function entry.
    // Section entry and every caller-provided kernel entry are explicit
    // external entries even when they have structural predecessors. Meet the
    // corresponding external state with any reachable predecessor: it
    // contributes no lane stash, and explicit kernel entries begin in bank
    // zero according to the entry contract.
    if (predecessors[block_index].empty() || external_entries[block_index] != 0) {
      new_reachable = true;
      VectorLaneFlowState external_entry;
      if (external_entries[block_index] != 0)
        external_entry.vgpr_msb_imm = uint8_t{0};
      if (!have_predecessor_state) {
        new_entry = std::move(external_entry);
      } else {
        new_entry.slots.clear();
        if (new_entry.vgpr_msb_imm != external_entry.vgpr_msb_imm)
          new_entry.vgpr_msb_imm = std::nullopt;
      }
    }
    if (!new_reachable)
      continue;

    VectorLaneFlowState new_exit = scan_block(blocks[block_index], new_entry, false);
    if (reachable[block_index] && entry_states[block_index] == new_entry &&
        exit_states[block_index] == new_exit)
      continue;

    reachable[block_index] = 1;
    entry_states[block_index] = std::move(new_entry);
    exit_states[block_index] = std::move(new_exit);
    for (size_t successor : blocks[block_index].successors) {
      if (on_worklist[successor])
        continue;
      worklist.push_back(successor);
      on_worklist[successor] = 1;
    }
  }

  for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    if (reachable[block_index])
      (void)scan_block(blocks[block_index], entry_states[block_index], true);
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

[[nodiscard]] std::vector<IndirectCallFixup>
discover_indirect_branch_edges_unfiltered(std::span<const Instruction *const> insts,
                                          std::span<const uint8_t> text, rj_code_arch_t arch,
                                          std::span<const uint64_t> extra_leaders) {
  std::vector<IndirectCallFixup> recovered;
  AnalysisContext ctx = build_context(insts, text, arch);
  recover_vector_lane_stashed_pcs(ctx, recovered, extra_leaders);
  std::vector<uint64_t> leaders(extra_leaders.begin(), extra_leaders.end());

  for (size_t iteration = 0; iteration < kMaxIndirectDiscoveryIterations; ++iteration) {
    add_recovered_leaders(leaders, recovered);

    std::vector<AnalysisBlock> blocks = build_analysis_blocks(ctx, leaders);
    add_recovered_successors(recovered, blocks);

    std::vector<PendingConsumer> pending_consumers;
    std::vector<IndirectCallFixup> iteration_recovered;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
      scan_block(ctx, block_index, blocks, pending_consumers, iteration_recovered);
    recover_signed_delta_templates(ctx, blocks, iteration_recovered);

    size_t unresolved_consumers = 0;
    if (!pending_consumers.empty()) {
      const auto entry_facts = run_block_dataflow(blocks, pending_consumers, extra_leaders);
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

} // namespace

std::vector<IndirectCallFixup>
discover_indirect_branch_edges(std::span<const Instruction *const> insts,
                               std::span<const uint8_t> text, rj_code_arch_t arch,
                               std::span<const uint64_t> extra_leaders) {
  if (insts.empty())
    return {};

  // Every recoverable edge ends at an indirect branch/call consumer. Most
  // generated kernels have none, so avoid building the auxiliary CFG and
  // running its dataflow passes when no fixup can possibly be produced.
  const bool has_indirect_consumer = std::ranges::any_of(
      insts, [](const Instruction *inst) { return is_recoverable_indirect_consumer(*inst); });
  if (!has_indirect_consumer) {
#ifndef NDEBUG
    // Keep the cheap predicate coupled to every fixup producer. A future
    // recovery path for another consumer kind must extend the predicate above.
    const auto unfiltered =
        discover_indirect_branch_edges_unfiltered(insts, text, arch, extra_leaders);
    assert(unfiltered.empty() && "indirect-recovery prefilter skipped a fixup-producing consumer");
#endif
    return {};
  }

  return discover_indirect_branch_edges_unfiltered(insts, text, arch, extra_leaders);
}

} // namespace rocjitsu
