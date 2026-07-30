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
#include <vector>

namespace rocjitsu {

class BasicBlock;
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

  /// @brief Lowest VGPR index that find_free_run() may return.
  ///
  /// @details This is a debug-oriented allocation floor, not a dataflow fact.
  /// The computed live-before sets remain the normal kernel liveness result,
  /// while scratch allocation can be forced above a descriptor-declared VGPR
  /// range to test whether semantic lowerings clobber guest registers.
  uint16_t min_free_vgpr = 0;

  /// @brief Exclusive destination-ISA limit for VGPR scratch allocation.
  ///
  /// @details RegisterSet may track more VGPR indices than a particular
  /// destination encoding can name. Keeping this allocation ceiling separate
  /// prevents 8-bit destination fields from truncating v256 and above.
  uint16_t max_free_vgpr = static_cast<uint16_t>(
      std::min(amdgpu::CdnaIsaBase::MAX_VGPRS_PER_WF, amdgpu::RdnaIsaBase::MAX_VGPRS_PER_WF));

  /// @brief Restrict instruction-level live-before materialization to selected instructions.
  ///
  /// @details Block-level dataflow still analyzes every instruction in the
  /// kernel scope. This option only controls which per-instruction RegisterSet
  /// snapshots are stored for live_before()/find_free_*() queries.
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

/// @brief Backward SGPR/VGPR/ACC_VGPR liveness over one decoded kernel CFG scope.
class LivenessAnalysis {
public:
  /// @brief Compute liveness for one kernel's block set.
  ///
  /// @details Successor/predecessor edges that leave @p blocks are ignored.
  /// DBT callers must pass only the blocks reachable from the kernel descriptor
  /// entry being translated, not every block decoded from the containing code
  /// object.
  /// @param blocks Blocks in one kernel CFG scope.
  LivenessAnalysis(KernelBlockScope blocks, LivenessAnalysisOptions options = {},
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

  /// @brief Block liveness by block object.
  [[nodiscard]] const BlockLiveness &block_liveness(const BasicBlock &block) const;

  /// @brief Registers live immediately before @p inst executes.
  [[nodiscard]] const RegisterSet &live_before(const Instruction &inst) const;

  /// @brief Convenience predicate for one register reference.
  [[nodiscard]] bool is_live_before(const Instruction &inst, RegisterRef ref) const;

  /// @brief gfx1250 VGPR bank selected for @p role before @p inst.
  /// @returns nullopt when this is not a gfx1250 analysis or the state is ambiguous.
  [[nodiscard]] std::optional<uint8_t> vgpr_msb_bank_before(const Instruction &inst,
                                                            amdgpu::VgprMsbRole role) const;

  /// @brief Find N consecutive dead VGPRs immediately before an instruction.
  ///
  /// @details Semantic lowerings use this to allocate temporary VGPRs while
  /// replacing one guest instruction with a host instruction sequence. The
  /// selected registers are dead at the replacement point according to this
  /// kernel-scope live-before set. Some host operands also require the base of
  /// a register tuple to be aligned; @p base_alignment lets those lowerings ask
  /// liveness for a power-of-two-aligned dead run that is also encodable for
  /// the target instruction.
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

  void analyze(KernelBlockScope blocks, const LivenessAnalysisOptions &options,
               std::span<const ScopedCfgEdge> extra_edges);

  bool available_ = true;
  uint16_t min_free_vgpr_ = 0;
  uint16_t max_free_vgpr_ = 0;
  std::unique_ptr<Gfx1250VgprMsbAnalysis> gfx1250_vgpr_msb_;
  std::vector<BlockLiveness> liveness_;
  std::unordered_map<const BasicBlock *, size_t> block_index_;
  std::unordered_map<const Instruction *, RegisterSet> live_before_;
  static constexpr RegisterSet empty_{};
};

} // namespace rocjitsu
