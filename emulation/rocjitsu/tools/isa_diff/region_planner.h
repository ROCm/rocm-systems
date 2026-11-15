// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file region_planner.h
/// @brief Conservative, explainable diff -> contiguous region -> contract planning.
///
/// @details P0 accepts only single-entry, single-exit, straight-line regions inside one
/// basic block on each side. Starting from a diff hunk, the planner:
///   - merges every later hunk in the same block pair, together with the common code
///     between them, because a mismatch in an intermediate value can be compensated by
///     a later change and must not be judged on its own;
///   - optionally extends backward over N common instructions (`context_insts`), so a
///     prerequisite that the common prefix establishes is executed, not assumed;
///   - stops with a typed reason on a non-P0 instruction, a PC-derived input, a block
///     boundary, or the length or expansion limits.
/// Every hunk ends up either in a READY region or in the unresolved list.
#pragma once

#include "diff.h"
#include "kernel_model.h"
#include "region.h"

#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

struct PlanOptions {
  size_t max_region_insts = 256;
  size_t max_expansions = 64; ///< Common instructions added per direction, at most.
  size_t context_insts = 0;   ///< Manual backward context for every base region.
  bool merge_same_block = true;
  /// Precompute expansion variants (backward and/or forward common context) that `test`
  /// tries automatically when a base region shows a local mismatch.
  bool auto_expand = true;
  size_t max_variants = 12;
};

struct UnresolvedItem {
  std::string kind; ///< change kind or "region"
  std::string what;
  std::string region; ///< Region id when it comes from an unsupported region.
};

struct PlanResult {
  std::vector<RegionPair> regions;
  /// Expansion variants of READY base regions, smallest first per parent.
  std::vector<RegionPair> variants;
  std::vector<UnresolvedItem> unresolved;
  std::vector<ArtifactNote> notes; ///< Non-semantic notes (layout etc.), informational.
};

[[nodiscard]] PlanResult plan_regions(const KernelModel &old_model, const KernelModel &new_model,
                                      const DiffResult &diff, const PlanOptions &options);

} // namespace rocjitsu::isa_diff
