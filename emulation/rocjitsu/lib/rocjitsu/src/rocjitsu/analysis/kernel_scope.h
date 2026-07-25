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
#include <unordered_map>
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

/// @brief Strongest validated path by which one kernel reaches a block.
enum class KernelCfgOwnerProofKind : uint8_t {
  KernelLocal,
  DirectCall,
  RecoveredIndirectCall,
};

/// @brief Kernel-local control-flow scope shared by DBT and DBI clients.
struct KernelCfgScope {
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
  std::vector<ScopedCfgEdge> liveness_edges;
  /// Return terminators reachable only through validated call contexts.
  std::unordered_set<uint64_t> call_return_offsets;
  std::unordered_map<const BasicBlock *, KernelCfgOwnerProofKind> owner_proofs;
};

/// @brief One descriptor and its complete validated execution scope.
struct KernelCfgOwnerScope {
  uint64_t descriptor_file_offset = 0;
  KernelCfgScope scope;
};

/// @brief One descriptor proven to execute a queried instruction.
struct KernelCfgOwnerProof {
  uint64_t descriptor_file_offset = 0;
  KernelCfgOwnerProofKind kind = KernelCfgOwnerProofKind::KernelLocal;
};

/// @brief Build a sorted lookup from source text offsets to decoded blocks.
[[nodiscard]] BlockOffsetIndex
build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks);

/// @brief Return the decoded block containing @p offset, or nullptr.
[[nodiscard]] BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset);

/// @brief Return every descriptor whose validated scope contains @p offset.
[[nodiscard]] std::vector<KernelCfgOwnerProof>
kernel_cfg_owners_for_offset(std::span<const KernelCfgOwnerScope> owner_scopes,
                             const BlockOffsetIndex &block_index, uint64_t offset);

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
