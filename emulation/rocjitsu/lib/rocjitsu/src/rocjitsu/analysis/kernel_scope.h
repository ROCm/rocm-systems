// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_scope.h
/// @brief Shared kernel-local CFG ownership and call/return analysis.

#pragma once

#include "rocjitsu/analysis/liveness.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

class BasicBlock;

/// @brief Sorted source-offset index for decoded basic blocks.
using BlockOffsetIndex = std::vector<std::pair<uint64_t, BasicBlock *>>;

/// @brief Entry points that belong to one kernel descriptor.
struct KernelScopeRequest {
  uint64_t entry_offset = 0;
  std::vector<uint64_t> additional_entry_offsets;
};

/// @brief Kernel-local control-flow scope shared by DBT and DBI clients.
struct KernelCfgScope {
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
  std::vector<ScopedCfgEdge> liveness_edges;
  std::unordered_set<uint64_t> call_return_offsets;
};

/// @brief Build a sorted lookup from source text offsets to decoded blocks.
[[nodiscard]] BlockOffsetIndex
build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks);

/// @brief Return the decoded block containing @p offset, or nullptr.
[[nodiscard]] BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset);

/// @brief Build one descriptor-owned CFG scope without entering other kernels.
///
/// @details The walk follows ordinary CFG successors and validated call edges,
/// but stops at entry points owned by other descriptors. Additional hardware
/// entry points (for example, kernarg-preload entries) are seeded explicitly.
/// Context-sensitive call and return edges are materialized for liveness
/// without mutating the process-wide BasicBlock graph.
///
/// @param blocks All decoded blocks in source order.
/// @param block_index Sorted lookup built from @p blocks.
/// @param request Primary and additional entries owned by this descriptor.
/// @param all_kernel_entries Every hardware kernel entry in the code object.
/// @param text Original text bytes used to validate return instructions.
/// @returns A complete kernel-local scope, or nullopt when an owned entry is not
/// decoded.
[[nodiscard]] std::optional<KernelCfgScope>
build_kernel_cfg_scope(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                       const BlockOffsetIndex &block_index, const KernelScopeRequest &request,
                       std::span<const uint64_t> all_kernel_entries, std::span<const uint8_t> text);

} // namespace rocjitsu
