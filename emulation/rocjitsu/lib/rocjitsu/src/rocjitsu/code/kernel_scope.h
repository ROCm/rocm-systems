// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_scope.h
/// @brief Shared block-offset indices and the kernel reachability walk that
///        forms one kernel's block scope.
///
/// @details A kernel scope is the set of decoded blocks reachable from one
/// kernel entry, which is the unit `LivenessAnalysis` and the EXEC-state
/// analyses model (see `analysis/liveness.h`: edges leaving the scope are
/// silently ignored, so passing every decoded block in a code object gives
/// wrong liveness). DBT and DBI both need that set and must agree on which
/// blocks belong to a kernel, so the walk lives here rather than inside either
/// one. DBT wraps it with its own `KdTranslation`-keyed scope construction;
/// this header knows only about block offsets.

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

class BasicBlock;

/// @brief Sorted index from source .text byte offsets to decoded blocks.
///
/// @details DBT relocation repeatedly maps descriptor entries, branch targets,
/// and recovered indirect targets back to the BasicBlock that owns a source
/// offset. Keeping this compact sorted index avoids rebuilding that lookup while
/// preserving BasicBlock ownership in the vector returned by BasicBlock::build().
using BlockOffsetIndex = std::vector<std::pair<uint64_t, BasicBlock *>>;
using BlockPositionIndex = std::unordered_map<const BasicBlock *, size_t>;

[[nodiscard]] BlockOffsetIndex
build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks);

[[nodiscard]] BlockPositionIndex
build_block_position_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks);

/// @brief The block containing @p offset, or nullptr when no block covers it.
///
/// @details Answers with the block CONTAINING the offset, not the block starting
/// at it: an offset in the middle of a block resolves to that block.
[[nodiscard]] BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset);

/// @brief Entry-offset sets that bound one kernel reachability walk.
struct KernelScopeSpec {
  /// @brief Every hardware kernel entry in the code object.
  ///
  /// @details A successor or callee that is another kernel's entry stops the
  /// walk, so one kernel's scope never absorbs another's body.
  std::unordered_set<uint64_t> kernel_entries;

  /// @brief Entries this scope owns, including its own kernel entry.
  ///
  /// @details Membership here overrides the kernel_entries stop, which is what
  /// lets a scope keep its kernarg-preload firmware entry and any adopted
  /// device-function roots.
  std::unordered_set<uint64_t> own_entries;

  /// @brief Device-function entries whose address is taken.
  ///
  /// @details Such a body is emitted once as an adopted root and is not cloned
  /// into the scope of an indirect caller; see the comment at the call-edge walk
  /// in kernel_scope.cpp for why cloning it diverges under repeat translation.
  /// A consumer that does not relocate `.text` can leave this empty.
  std::unordered_set<uint64_t> address_taken_entries;
};

/// @brief Blocks reachable from @p entry, in source .text order.
///
/// @details Walks ordinary CFG successors plus call edges, stopping at entries
/// owned by another scope per @p spec. Returned in ascending source order, which
/// callers rely on to emit a scope's body with fallthrough preserved.
[[nodiscard]] std::vector<BasicBlock *> reachable_kernel_blocks(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, const BlockOffsetIndex &block_index,
    const BlockPositionIndex &block_positions, BasicBlock &entry, const KernelScopeSpec &spec);

} // namespace rocjitsu
