// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

struct TextRange {
  uint64_t start = 0;
  uint64_t end = 0;
};

struct LocalTextCaveRequest {
  TextRange source;
  uint64_t body_size_bytes = 0;
  uint64_t cave_size_bytes = 0;
};

struct LocalBranchIslandRequest {
  TextRange source;
  uint64_t island_target = 0;
};

struct BranchableTextPlacement {
  uint64_t offset = 0;
  int16_t entry_branch_dwords = 0;
  int16_t exit_branch_dwords = 0;
};

struct ExpandedTextScopePlacementRequest {
  uint64_t original_text_size_bytes = 0;
  uint64_t current_tail_size_bytes = 0;
  uint64_t original_entry_offset = 0;
  uint64_t translated_entry_offset = 0;
  uint64_t prologue_size_bytes = 0;
  uint64_t entry_alignment_bytes = 256;
};

struct ExpandedTextScopePlacement {
  uint64_t padding_bytes = 0;
  std::optional<uint64_t> launch_stub_offset;
  uint64_t body_offset = 0;
  uint64_t descriptor_entry_offset = 0;
  std::optional<int16_t> prologue_branch_dwords;
};

struct ExpandedTextBranchFixup {
  uint64_t word_index = 0;
  uint64_t target_offset = 0;
  std::string mnemonic;
  std::optional<uint16_t> sgpr_pair;
};

struct ExpandedTextBranchRelocationRequest {
  std::span<const uint32_t> words;
  std::span<const ExpandedTextBranchFixup> branches;
  std::span<const std::pair<uint64_t, uint64_t>> offset_map;
};

struct ExpandedTextBranchRelocation {
  bool success = false;
  std::vector<uint32_t> words;
  std::vector<std::pair<uint64_t, uint64_t>> offset_map;
  std::string message;
};

struct ExpandedTextPcRelativeFixup {
  uint64_t getpc_offset = 0;
  uint64_t add_tmp_offset = 0;
  uint64_t target_offset = 0;
  std::string kind;
};

struct ExpandedTextPcRelativeRelocationRequest {
  std::span<uint32_t> words;
  std::span<const std::pair<uint64_t, uint64_t>> offset_map;
  std::span<const ExpandedTextPcRelativeFixup> fixups;
  uint64_t original_text_size_bytes = 0;
  uint64_t scope_base_bytes = 0;
};

struct ExpandedTextPcRelativeRelocation {
  bool success = false;
  std::string message;
};

struct ExpandedTextDescriptorEntry {
  uint64_t descriptor_file_offset = 0;
  uint64_t entry_text_offset = 0;
};

struct ExpandedTextDescriptorRedirection {
  uint64_t descriptor_file_offset = 0;
  uint64_t original_entry_offset = 0;
  uint64_t redirected_entry_offset = 0;
};

struct ExpandedTextDescriptorRedirectionPlan {
  bool success = false;
  std::vector<ExpandedTextDescriptorRedirection> redirections;
  std::string message;
};

[[nodiscard]] bool compute_sopp_branch_offset(uint64_t branch_pc, uint64_t target,
                                              int16_t &offset_dwords);

[[nodiscard]] bool ranges_overlap(uint64_t lhs_start, uint64_t lhs_end, uint64_t rhs_start,
                                  uint64_t rhs_end);

/// Returns true when [start, end) overlaps one of the sorted, non-overlapping
/// ranges.
[[nodiscard]] bool overlaps_any_range(uint64_t start, uint64_t end,
                                      std::span<const std::pair<uint64_t, uint64_t>> ranges);

[[nodiscard]] std::optional<BranchableTextPlacement>
find_local_text_cave(std::span<const uint8_t> text, const LocalTextCaveRequest &request,
                     std::span<const std::pair<uint64_t, uint64_t>> reserved_ranges,
                     std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
                     bool allow_unreachable_text_caves);

[[nodiscard]] std::optional<BranchableTextPlacement>
find_local_branch_island(std::span<const uint8_t> text, const LocalBranchIslandRequest &request,
                         std::span<const std::pair<uint64_t, uint64_t>> reserved_ranges,
                         std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
                         bool allow_unreachable_text_caves);

[[nodiscard]] std::optional<ExpandedTextScopePlacement>
plan_expanded_text_scope_placement(const ExpandedTextScopePlacementRequest &request);

[[nodiscard]] ExpandedTextBranchRelocation
relocate_expanded_text_branches(const ExpandedTextBranchRelocationRequest &request);

[[nodiscard]] ExpandedTextPcRelativeRelocation
relocate_expanded_text_pc_relative_fixups(const ExpandedTextPcRelativeRelocationRequest &request);

[[nodiscard]] ExpandedTextDescriptorRedirectionPlan plan_expanded_text_descriptor_redirections(
    std::span<const ExpandedTextDescriptorEntry> descriptors,
    std::span<const std::pair<uint64_t, uint64_t>> copied_entry_offsets,
    uint64_t original_text_size_bytes);

} // namespace rocjitsu
