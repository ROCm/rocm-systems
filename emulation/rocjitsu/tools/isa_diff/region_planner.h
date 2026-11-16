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

#include <optional>
#include <string>
#include <string_view>
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
  /// Derive known-bits constraints for entry locations from the common prefix
  /// (source "common_prefix_analysis"); `test` also reports the unrestricted verdict.
  bool derive_constraints = true;
  /// Hand-written region "old:B-E,new:B-E" (see parse_manual_region); replaces automatic
  /// planning when set. Resolved against the kernel models by analyze_artifacts.
  std::string manual_region;
  /// Register maps "s4=s8,..." (old=new) for the manual region; require manual_region.
  /// register_map sets both the entry and the exit map; entry_map/exit_map set one each.
  std::string register_map, entry_map, exit_map;
  /// Handoff 5.3 stop rule on planning time; 0 = unlimited. When reached, optional work
  /// (variants, derived constraints) is skipped and, if the composition check has not run,
  /// every contract stays UNRESOLVED and the scope partial.
  uint64_t max_analysis_us = 0; ///< CLI: --max-analysis-ms (x1000).
  /// Handoff 5.3 explicit diagnostic path: also build, per READY region in the kernel's
  /// entry block, the region extended back to the kernel entry. `test` runs these
  /// separately; they never enter the verdict or the per-case hot path of other regions.
  bool diagnostic_entry_prefix = false;
  /// Also build the whole-function region "f0": both functions from their entry to their
  /// final return (or s_endpgm), with no cut, so no composition or exit mapping. `test`
  /// runs it when the regions alone leave the verdict open (README "Whole functions").
  bool whole_function = true;
  size_t max_function_insts = 4096; ///< Per side, for f0 (max_region_insts for the others).
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
  /// Planning steps skipped because max_analysis_ms was reached.
  std::vector<std::string> skipped_for_time;
  /// Diagnostic entry-prefix regions (diagnostic_entry_prefix), parent = base region id.
  std::vector<RegionPair> diagnostics;
  /// The whole-function region (whole_function), also when it cannot run (with its reason).
  std::optional<RegionPair> whole_function;
};

[[nodiscard]] PlanResult plan_regions(const KernelModel &old_model, const KernelModel &new_model,
                                      const DiffResult &diff, const PlanOptions &options);

/// A hand-written region (handoff 12): half-open instruction index ranges on both sides
/// and an optional register map (handoff 3.1). Its contract is never upgraded beyond
/// ASSUMED, and every diff hunk outside the ranges stays unresolved.
struct ManualRegion {
  size_t old_begin = 0, old_end = 0;
  size_t new_begin = 0, new_end = 0;
  /// old -> new at the entry and at the exit; empty = identity.
  std::vector<std::pair<unsigned, unsigned>> entry_map, exit_map;
};

/// @brief Parse "old:B-E,new:B-E". B and E are decimal instruction indices or 0x-prefixed
/// PCs (E exclusive: the PC after the last region instruction).
[[nodiscard]] std::optional<ManualRegion> parse_manual_region(std::string_view spec,
                                                              const KernelModel &old_model,
                                                              const KernelModel &new_model,
                                                              std::string &error);

/// @brief Build the single region "m0" from a hand-written specification. Same P0
/// admissibility checks as automatic planning; no expansion variants, no derived
/// constraints and no composition check.
[[nodiscard]] PlanResult plan_manual_region(const KernelModel &old_model,
                                            const KernelModel &new_model, const DiffResult &diff,
                                            const ManualRegion &manual, const PlanOptions &options);

} // namespace rocjitsu::isa_diff
