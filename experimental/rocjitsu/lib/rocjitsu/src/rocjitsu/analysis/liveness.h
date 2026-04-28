// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file liveness.h
/// @brief CFG-aware register liveness for DBT/DBI analyses.
///
/// @details This is the single liveness implementation used by DBT/DBI. It
/// follows docs/dbt_dbi_plan.md: it works over BasicBlock successor edges, uses
/// RegisterSet for all tracked register classes, and computes live-before
/// information for every decoded instruction.

#pragma once

#include "rocjitsu/analysis/register_set.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class BasicBlock;
class Instruction;

/// @brief Block-level dataflow state.
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

/// @brief Reverse-post-order traversal of the implicit CFG.
///
/// @details The CFG is embedded in the BasicBlock list returned by
/// BasicBlock::build(); no separate graph object is needed. Unreachable blocks
/// are appended in their original order so callers still get a complete result
/// for every decoded block.
[[nodiscard]] std::vector<const BasicBlock *>
reverse_post_order(const std::vector<std::unique_ptr<BasicBlock>> &blocks);

/// @brief Backward liveness analysis over a decoded basic-block graph.
class LivenessAnalysis {
public:
  /// @brief Compute liveness for the full block list.
  /// @param blocks Blocks returned by BasicBlock::build(), with CFG edges populated.
  /// @param wf_size Wavefront width in lanes; forwarded to operand/implicit register mapping.
  LivenessAnalysis(const std::vector<std::unique_ptr<BasicBlock>> &blocks, uint8_t wf_size);

  /// @brief Block liveness by block object.
  [[nodiscard]] const BlockLiveness &block_liveness(const BasicBlock &block) const;

  /// @brief Registers live immediately before @p inst executes.
  [[nodiscard]] const RegisterSet &live_before(const Instruction &inst) const;

  /// @brief Convenience predicate for one register reference.
  [[nodiscard]] bool is_live_before(const Instruction &inst, RegisterRef ref) const;

  /// @brief Find N consecutive dead VGPRs immediately before an instruction offset.
  ///
  /// @details Semantic lowerings use this to allocate temporary VGPRs while
  /// replacing one guest instruction with a host instruction sequence. The
  /// selected registers are dead at the replacement point according to the full
  /// CFG-aware live-before set.
  [[nodiscard]] std::optional<uint16_t> find_free_run(uint64_t offset, uint16_t count,
                                                      uint16_t search_start = 0) const;

  /// @brief Find an even-aligned dead SGPR pair immediately before an instruction offset.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr_pair(uint64_t offset,
                                                            uint16_t search_start = 0) const;

  /// @brief Find one dead SGPR immediately before an instruction offset.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr(uint64_t offset,
                                                       uint16_t search_start = 0) const;

private:
  std::vector<BlockLiveness> liveness_;
  std::unordered_map<const BasicBlock *, size_t> block_index_;
  std::unordered_map<const Instruction *, RegisterSet> live_before_;
  std::unordered_map<uint64_t, const Instruction *> instruction_by_offset_;
  RegisterSet empty_;
};

} // namespace rocjitsu
