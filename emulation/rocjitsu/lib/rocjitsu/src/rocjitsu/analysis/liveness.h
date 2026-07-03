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

#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class BasicBlock;
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
  /// @brief Lowest VGPR index that find_free_run() may return.
  ///
  /// @details This is a debug-oriented allocation floor, not a dataflow fact.
  /// The computed live-before sets remain the normal kernel liveness result,
  /// while scratch allocation can be forced above a descriptor-declared VGPR
  /// range to test whether semantic lowerings clobber guest registers.
  uint16_t min_free_vgpr = 0;

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

  /// @brief Block liveness by block object.
  [[nodiscard]] const BlockLiveness &block_liveness(const BasicBlock &block) const;

  /// @brief Registers live immediately before @p inst executes.
  [[nodiscard]] const RegisterSet &live_before(const Instruction &inst) const;

  /// @brief Reuse @p anchor's live-before set for a decoded temporary instruction.
  ///
  /// @details DBT remapping can decode a short-lived instruction with adjusted
  /// operands and then run a normal semantic lowering on it. The temporary is
  /// not part of the original CFG, so it needs the source instruction's
  /// liveness for scratch allocation.
  [[nodiscard]] bool alias_live_before(const Instruction &alias, const Instruction &anchor);

  /// @brief Convenience predicate for one register reference.
  [[nodiscard]] bool is_live_before(const Instruction &inst, RegisterRef ref) const;

  /// @brief Encoded VGPR base for a caller-provided high-bank scratch window.
  [[nodiscard]] std::optional<uint16_t> high_vgpr_scratch_base() const {
    return high_vgpr_scratch_base_;
  }

  /// @brief Set the encoded base of a scratch window reached through VGPR_MSB.
  void set_high_vgpr_scratch_base(std::optional<uint16_t> base) { high_vgpr_scratch_base_ = base; }

  /// @brief Byte offset of a caller-reserved LDS spill zone, if available.
  [[nodiscard]] std::optional<uint32_t> lds_spill_base() const { return lds_spill_base_; }

  /// @brief Size in bytes of the caller-reserved LDS spill zone.
  [[nodiscard]] uint32_t lds_spill_bytes() const { return lds_spill_bytes_; }

  /// @brief Set the caller-reserved LDS spill zone used by semantic lowerings.
  void set_lds_spill_zone(std::optional<uint32_t> base, uint32_t bytes) {
    lds_spill_base_ = base;
    lds_spill_bytes_ = bytes;
  }

  /// @brief Byte offset of a caller-reserved private scratch spill zone, if available.
  [[nodiscard]] std::optional<uint32_t> private_spill_base() const { return private_spill_base_; }

  /// @brief Size in bytes of the caller-reserved private scratch spill zone.
  [[nodiscard]] uint32_t private_spill_bytes() const { return private_spill_bytes_; }

  /// @brief Set the caller-reserved private scratch spill zone used by semantic lowerings.
  void set_private_spill_zone(std::optional<uint32_t> base, uint32_t bytes) {
    private_spill_base_ = base;
    private_spill_bytes_ = bytes;
  }

  /// @brief Exclude a register range from temporary scratch allocation.
  ///
  /// @details These reservations are not liveness facts. They model registers
  /// that semantic lowerings must not borrow even if the original instruction
  /// stream does not make them live at a particular program point, such as
  /// kernel-entry ABI SGPRs materialized by the descriptor translator.
  void reserve_scratch_registers(RegisterRef ref);

  /// @brief Set the highest ordinary SGPR index considered for scratch allocation.
  ///
  /// @details The default remains conservative across all supported families.
  /// Target-specific translators may raise this when the host ISA exposes more
  /// ordinary SGPRs.
  void set_allocatable_sgpr_limit(uint16_t limit) {
    allocatable_sgpr_limit_ =
        std::min<uint16_t>(limit, static_cast<uint16_t>(REGISTER_SET_MAX_SGPRS));
  }

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
  void analyze(KernelBlockScope blocks, const LivenessAnalysisOptions &options,
               std::span<const ScopedCfgEdge> extra_edges);

  uint16_t min_free_vgpr_ = 0;
  std::vector<BlockLiveness> liveness_;
  std::unordered_map<const BasicBlock *, size_t> block_index_;
  std::unordered_map<const Instruction *, RegisterSet> live_before_;
  std::optional<uint16_t> high_vgpr_scratch_base_;
  std::optional<uint32_t> lds_spill_base_;
  uint32_t lds_spill_bytes_ = 0;
  std::optional<uint32_t> private_spill_base_;
  uint32_t private_spill_bytes_ = 0;
  RegisterSet scratch_reserved_;
  uint16_t allocatable_sgpr_limit_ = static_cast<uint16_t>(REGISTER_SET_ALLOCATABLE_SGPRS);
  static constexpr RegisterSet empty_{};
};

} // namespace rocjitsu
