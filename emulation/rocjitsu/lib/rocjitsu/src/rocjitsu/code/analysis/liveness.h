// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file liveness.h
/// @brief Kernel-scoped CFG-aware register liveness for DBT/DBI analyses.
///
/// @details Each LivenessAnalysis instance models one kernel CFG scope: callers
/// provide only the BasicBlocks reachable from a single kernel descriptor entry,
/// and successor/predecessor edges outside that scope are ignored. The analysis
/// currently tracks only ordinary SGPRs, VGPRs, and ACC_VGPRs through
/// RegisterSet; special state such as EXEC, SCC, VCC, and FLAT_SCRATCH is not
/// part of the dataflow model. Semantic translation uses this to verify that
/// scratch registers introduced by lowerings do not clobber live values.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {

class BasicBlock;
class ExecMaskAnalysis;
class Gfx1250VgprMsbAnalysis;
class Instruction;

/// @brief The basic blocks reachable from one kernel entry.
using KernelBlockScope = std::span<BasicBlock *const>;

/// @brief Extra edge in a kernel-scoped analysis graph.
///
/// @details BasicBlock::successors() stores context-free local CFG edges only.
/// DBT can provide scoped call and return edges here when translating one
/// kernel body, so liveness sees the callee and the correct call-site return
/// continuation without making those edges globally visible to other kernels.
struct ScopedCfgEdge {
  BasicBlock *from = nullptr;
  BasicBlock *to = nullptr;
};

/// @brief Block-level dataflow state for one kernel scope.
///
/// @details `gen` is the upward-exposed use set: registers read in the block
/// before any local definition. `kill` is the set of registers defined in the
/// block. The standard backward equations are:
///   live_out(B) = union(live_in(S) for S in successors(B))
///   live_in(B)  = gen(B) union (live_out(B) - kill(B))
struct BlockLiveness {
  RegisterSet live_in;
  RegisterSet live_out;
  RegisterSet gen;
  RegisterSet kill;
};

/// @brief Optional controls for liveness construction.
struct LivenessAnalysisOptions {
  /// @brief Architecture whose register semantics should be analyzed.
  ///
  /// @details INVALID preserves the legacy ISA-independent behavior. DBT sets
  /// this to gfx1250 together with entry_block so VGPR_MSB state can resolve
  /// encoded vector operands to physical VGPRs.
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;

  /// @brief Kernel entry block where architectural VGPR_MSB state is zero.
  BasicBlock *entry_block = nullptr;

  /// @brief Further blocks entered with the same architectural state as entry_block.
  ///
  /// @details These are address-taken device functions adopted into this scope. They are reached
  /// by a call through a pointer rather than by a decoded edge, so without seeding their blocks
  /// stay unreachable and every VGPR_MSB query over them answers nullopt. The state to seed them
  /// with is the ABI's, which is the same one entry_block gets.
  std::span<BasicBlock *const> additional_entry_blocks;

  /// @brief Lowest VGPR index that a VGPR scratch query may return.
  ///
  /// @details This is a debug-oriented allocation floor, not a dataflow fact.
  /// The computed live-before sets remain the normal kernel liveness result,
  /// while scratch allocation can be forced above a descriptor-declared VGPR
  /// range to test whether semantic lowerings clobber guest registers.
  uint16_t min_free_vgpr = 0;

  /// @brief Exclusive destination-ISA limit for all VGPR scratch queries.
  ///
  /// @details RegisterSet may track more VGPR indices than a particular
  /// destination encoding can name. Keeping this allocation ceiling separate
  /// prevents 8-bit destination fields from truncating v256 and above.
  uint16_t max_free_vgpr = static_cast<uint16_t>(
      std::min(amdgpu::CdnaIsaBase::MAX_VGPRS_PER_WF, amdgpu::RdnaIsaBase::MAX_VGPRS_PER_WF));

  /// @brief Restrict instruction-level live-before materialization to selected instructions.
  ///
  /// @details When a query first requests CFG liveness, block-level dataflow
  /// analyzes every instruction in the kernel scope. This option only controls
  /// which per-instruction RegisterSet snapshots are then stored for
  /// live_before()/find_free_*() queries.
  bool restrict_live_before_to_instructions = false;

  /// @brief Instruction pointers that need live-before snapshots when filtering is enabled.
  ///
  /// @details Semantic DBT lowerings need live-before snapshots only at the
  /// handful of instructions that may allocate scratch registers. Materializing
  /// snapshots for every instruction in a large kernel is expensive and does
  /// not help lowerings that never query them. Callers can pass an empty span
  /// together with restrict_live_before_to_instructions=true to request no
  /// instruction-level snapshots. Pointers outside the analyzed block scope are
  /// ignored.
  std::span<const Instruction *const> live_before_instructions = {};

  /// @brief Raw .text image for the analyzed kernel scope.
  ///
  /// @details Forwarded to the gfx1250 VGPR_MSB analysis so it can read
  /// S_SETREG_IMM32_B32 literals safely from the instruction stream (at
  /// src_loc()+4) instead of indexing past a decoded instruction's encoding word.
  /// Empty is tolerated: such writes then mark the affected banks ambiguous.
  std::span<const uint8_t> text = {};
};

/// @brief Reverse-post-order traversal of one kernel's implicit CFG.
///
/// @details The CFG is embedded in the BasicBlock objects returned by
/// BasicBlock::build(); no separate graph object is needed. Traversal is
/// constrained to the block span, so callers can analyze one kernel without
/// walking into other decoded code.
[[nodiscard]] std::vector<const BasicBlock *> reverse_post_order(KernelBlockScope blocks);

/// @brief Kernel-global register usage plus deferred backward CFG liveness.
///
/// @details Construction eagerly resolves gfx1250 VGPR banks and scans global
/// register usage. The backward CFG fixed point and live-before snapshots are
/// materialized on the first query that needs them. Queries on one analysis
/// object are not thread-safe because const query methods may populate that
/// deferred cache.
class LivenessAnalysis {
public:
  /// @brief Prepare register analysis for one kernel's block set.
  ///
  /// @details Successor/predecessor edges that leave @p blocks are ignored.
  /// DBT callers must pass only the blocks reachable from the kernel descriptor
  /// entry being translated, not every block decoded from the containing code
  /// object. The pointed-to BasicBlocks and their Instructions must outlive this
  /// analysis because deferred CFG queries retain and later dereference them.
  /// @param blocks Blocks in one kernel CFG scope.
  /// @param exec Optional program-point EXEC-state analysis over the same
  /// @p blocks scope; lets EXEC-masked vector defs count as kills where EXEC is
  /// provably full. Ownership is moved into the analysis. Defaults to null: with
  /// no EXEC analysis every EXEC-masked vector def is treated as `Unknown`, so
  /// the conservative behavior (never promote such a def to a kill) is preserved.
  LivenessAnalysis(KernelBlockScope blocks, std::unique_ptr<ExecMaskAnalysis> exec = nullptr,
                   LivenessAnalysisOptions options = {},
                   std::span<const ScopedCfgEdge> extra_edges = {});
  ~LivenessAnalysis();

  LivenessAnalysis(const LivenessAnalysis &) = delete;
  LivenessAnalysis &operator=(const LivenessAnalysis &) = delete;
  LivenessAnalysis(LivenessAnalysis &&) noexcept;
  LivenessAnalysis &operator=(LivenessAnalysis &&) noexcept;

  /// @brief Create a fail-closed sentinel for a scope that does not need liveness.
  ///
  /// @details BinaryTranslator uses this when every matching semantic expansion
  /// rule is marked liveness-free. Any query on the returned object throws
  /// std::logic_error so an incorrectly classified rule cannot silently use
  /// missing liveness data.
  [[nodiscard]] static LivenessAnalysis unavailable();

  /// @brief Whether a query has materialized the deferred backward CFG state.
  ///
  /// @details This is useful for qualification and tests that must verify the
  /// kernel-unused allocation path did not force the expensive fixed point.
  [[nodiscard]] bool has_materialized_cfg_liveness() const { return analyzed_; }

  /// @brief Block liveness by block object.
  [[nodiscard]] const BlockLiveness &block_liveness(const BasicBlock &block) const;

  /// @brief Whether a live-before snapshot was materialized for @p inst.
  /// @details Returns false when @p inst was not part of the analyzed scope.
  ///          Throws std::logic_error when the analysis is unavailable,
  ///          matching live_before().
  [[nodiscard]] bool has_live_before(const Instruction &inst) const;

  /// @brief Registers live immediately before @p inst executes.
  [[nodiscard]] const RegisterSet &live_before(const Instruction &inst) const;

  /// @brief Convenience predicate for one register reference.
  [[nodiscard]] bool is_live_before(const Instruction &inst, RegisterRef ref) const;

  /// @brief gfx1250 VGPR bank selected for @p role before @p inst.
  /// @returns nullopt when this is not a gfx1250 analysis or the state is ambiguous.
  /// @brief The VGPR_MSB analysis this liveness owns, or nullptr when it has none.
  ///
  /// @details Exposed so a caller that needs the same dataflow does not build a second one over
  /// the same scope: the fixpoint is superlinear in scope size, and on a large object the two
  /// copies were the same computation run twice.
  [[nodiscard]] const Gfx1250VgprMsbAnalysis *gfx1250_vgpr_msb() const {
    return gfx1250_vgpr_msb_.get();
  }

  [[nodiscard]] std::optional<uint8_t> vgpr_msb_bank_before(const Instruction &inst,
                                                            amdgpu::VgprMsbRole role) const;

  /// @brief Whether statically decoded operands cover every VGPR access in the scope.
  ///
  /// @details False when relative addressing or runtime GPR indexing can redirect
  /// an encoded VGPR operand. Transformations that compare or reuse physical
  /// VGPR ranges must fail closed when this query returns false.
  [[nodiscard]] bool global_vgpr_usage_is_complete() const;

  /// @brief Find a VGPR tuple that no instruction in the kernel reads or writes.
  ///
  /// @details This query is cheaper and stronger than point liveness: the
  /// selected tuple is unused everywhere, so it is dead at every instruction
  /// in the decoded source scope. It does not force the deferred CFG
  /// live-before computation. This conclusion requires a complete decoded
  /// kernel scope, including reachable callees supplied through scoped edges.
  /// Scopes with relative or GPR-indexed VGPR access fail closed because encoded
  /// operands do not identify every physical register they may touch.
  ///
  /// @param inst Instruction that will use the tuple. It must belong to the
  ///        analyzed scope.
  /// @param count Number of consecutive VGPRs required. Runs wider than a
  ///        RegisterRef can name (255) fail closed, since the search tests a
  ///        candidate run as one RegisterRef.
  /// @param search_start Lowest candidate base requested by the caller.
  /// @param base_alignment Required tuple-base alignment.
  /// @param available_count Exclusive upper bound imposed by the current
  ///        descriptor allocation.
  [[nodiscard]] std::optional<uint16_t> find_globally_unused_vgpr_run(
      const Instruction *inst, uint16_t count, uint16_t search_start = 0,
      uint16_t base_alignment = 1,
      uint16_t available_count = static_cast<uint16_t>(REGISTER_SET_MAX_VGPRS)) const;

  /// @brief Find N consecutive dead VGPRs immediately before an instruction.
  ///
  /// @details Semantic lowerings use this to allocate temporary VGPRs while
  /// replacing one guest instruction with a host instruction sequence. The
  /// selected registers are dead at the replacement point according to this
  /// kernel-scope live-before set. Some host operands also require the base of
  /// a register tuple to be aligned; @p base_alignment lets those lowerings ask
  /// liveness for a power-of-two-aligned dead run that is also encodable for
  /// the target instruction. Runs wider than a RegisterRef can name (255) fail
  /// closed, since the search tests a candidate run as one RegisterRef.
  [[nodiscard]] std::optional<uint16_t> find_free_run(const Instruction *inst, uint16_t count,
                                                      uint16_t search_start = 0,
                                                      uint16_t base_alignment = 1) const;

  /// @brief Find an even-aligned dead SGPR pair immediately before an instruction.
  ///
  /// @details Even alignment is required for pair operations such as saving EXEC
  /// with an s_mov_b64-style scalar move.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr_pair(const Instruction *inst,
                                                            uint16_t search_start = 0) const;

  /// @brief Find one dead SGPR immediately before an instruction.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr(const Instruction *inst,
                                                       uint16_t search_start = 0) const;

private:
  /// @brief Tag selecting the unavailable sentinel constructor.
  struct UnavailableTag {};

  /// @brief Construct an unavailable sentinel without running dataflow.
  explicit LivenessAnalysis(UnavailableTag);

  /// @brief Reject a query when this object is the unavailable sentinel.
  /// @throws std::logic_error if liveness data was intentionally not built.
  void require_available() const;

  /// @brief Materialize CFG live-before state on the first query that needs it.
  void ensure_analyzed() const;

  /// @brief Collect whole-kernel register use without running backward dataflow.
  void collect_global_register_usage(KernelBlockScope blocks, std::span<const uint8_t> text,
                                     rj_code_arch_t arch);

  void analyze(KernelBlockScope blocks, bool restrict_live_before_to_instructions,
               std::span<const Instruction *const> live_before_instructions,
               std::span<const ScopedCfgEdge> extra_edges) const;

  bool available_ = true;
  mutable bool analyzed_ = false;
  bool global_vgpr_usage_is_complete_ = true;
  uint16_t min_free_vgpr_ = 0;
  uint16_t max_free_vgpr_ = 0;
  std::unique_ptr<Gfx1250VgprMsbAnalysis> gfx1250_vgpr_msb_;
  // EXEC-state analysis over the same scope, captured at construction so the
  // deferred backward dataflow can treat an EXEC-masked vector def as a kill
  // where EXEC is provably full.
  std::unique_ptr<ExecMaskAnalysis> exec_;
  RegisterSet globally_used_registers_;
  std::vector<BasicBlock *> deferred_blocks_;
  std::unordered_set<const BasicBlock *> scoped_blocks_;
  bool deferred_restrict_live_before_to_instructions_ = false;
  std::vector<const Instruction *> deferred_live_before_instructions_;
  std::vector<ScopedCfgEdge> deferred_extra_edges_;
  mutable std::vector<BlockLiveness> liveness_;
  mutable std::unordered_map<const BasicBlock *, size_t> block_index_;
  mutable std::unordered_map<const Instruction *, RegisterSet> live_before_;
  static constexpr RegisterSet empty_{};
};

} // namespace rocjitsu
