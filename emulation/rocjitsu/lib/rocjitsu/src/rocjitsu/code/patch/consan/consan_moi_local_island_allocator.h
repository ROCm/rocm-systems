// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_local_island_allocator.h
/// @brief Owner-aware allocation of pristine local branch islands.

#pragma once

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "util/bit.h"

#include <cstring>
#include <limits>
#include <ranges>
#include <set>
#include <unordered_map>

namespace rocjitsu::consan_moi_impl {

class MoiLocalNopIslandAllocator {
public:
  MoiLocalNopIslandAllocator(std::span<const uint8_t> text,
                             const ProgramInventory &program_inventory,
                             std::span<const ConSanPatchInfo> existing_patches, rj_code_arch_t arch,
                             uint32_t island_words,
                             std::span<const ConSanPreappliedReservedRange> reserved_ranges = {})
      : text_(text), nop_(build_s_nop(0, arch)), island_words_(island_words) {
    if (island_words == 0)
      return;
    struct OwnerRange {
      uint64_t entry = 0;
      uint64_t code_end = 0;
    };
    std::vector<OwnerRange> owners;
    owners.reserve(program_inventory.kernels().size() + program_inventory.functions().size());
    const auto append_owner = [&](uint64_t entry, uint64_t code_size) {
      if (code_size == 0 || entry >= text.size())
        return;
      owners.push_back({entry, entry + std::min<uint64_t>(code_size, text.size() - entry)});
    };
    for (const ConSanKernelInfo &kernel : program_inventory.kernels()) {
      if (kernel.has_text_range)
        append_owner(kernel.entry_text_offset, kernel.code_size);
    }
    for (const ConSanFunctionInfo &function : program_inventory.functions())
      append_owner(function.entry_text_offset, function.code_size);
    std::ranges::sort(owners, {}, &OwnerRange::entry);
    std::vector<OwnerRange> unique;
    for (const OwnerRange &owner : owners) {
      if (!unique.empty() && unique.back().entry == owner.entry) {
        unique.back().code_end = std::max(unique.back().code_end, owner.code_end);
      } else {
        unique.push_back(owner);
      }
    }

    const uint64_t island_bytes = static_cast<uint64_t>(island_words) * sizeof(uint32_t);
    std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges;
    occupied_ranges.reserve(existing_patches.size() * 2u + reserved_ranges.size());
    for (const ConSanPatchInfo &patch : existing_patches) {
      occupied_ranges.emplace_back(
          patch.anchor_offset, patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u));
      if (patch.trampoline_size != 0u) {
        occupied_ranges.emplace_back(patch.trampoline_offset,
                                     patch.trampoline_offset + patch.trampoline_size);
      }
    }
    for (const ConSanPreappliedReservedRange &reserved : reserved_ranges) {
      if (reserved.size != 0u)
        occupied_ranges.emplace_back(reserved.text_offset, reserved.text_offset + reserved.size);
    }
    std::ranges::sort(occupied_ranges);
    std::vector<std::pair<uint64_t, uint64_t>> merged_occupied_ranges;
    for (const auto &range : occupied_ranges) {
      if (merged_occupied_ranges.empty() || merged_occupied_ranges.back().second < range.first) {
        merged_occupied_ranges.push_back(range);
      } else {
        merged_occupied_ranges.back().second =
            std::max(merged_occupied_ranges.back().second, range.second);
      }
    }
    const auto overlaps_patch = [&](uint64_t begin, uint64_t end) {
      const auto it = std::ranges::lower_bound(merged_occupied_ranges, end, {},
                                               &std::pair<uint64_t, uint64_t>::first);
      return it != merged_occupied_ranges.begin() && std::prev(it)->second > begin;
    };
    for (size_t owner_index = 0; owner_index < unique.size(); ++owner_index) {
      const uint64_t gap_begin = util::align_up(unique[owner_index].code_end, uint64_t{4});
      const uint64_t gap_end =
          owner_index + 1u < unique.size() ? unique[owner_index + 1u].entry : text.size();
      for (uint64_t word_offset = gap_begin; word_offset + sizeof(uint32_t) <= gap_end;
           word_offset += sizeof(uint32_t)) {
        uint32_t word = 0;
        std::memcpy(&word, text.data() + word_offset, sizeof(word));
        if (word == nop_ && !overlaps_patch(word_offset, word_offset + sizeof(uint32_t))) {
          available_word_offsets_.insert(word_offset);
          available_word_owner_entries_.emplace(word_offset, unique[owner_index].entry);
        }
      }
      uint64_t cursor = gap_begin;
      while (cursor + island_bytes <= gap_end) {
        bool all_nops = !overlaps_patch(cursor, cursor + island_bytes);
        for (uint64_t offset = 0; all_nops && offset < island_bytes; offset += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, text.data() + cursor + offset, sizeof(word));
          all_nops = word == nop_;
        }
        if (all_nops) {
          available_offsets_.insert(cursor);
          available_owner_entries_.emplace(cursor, unique[owner_index].entry);
          cursor += island_bytes;
        } else {
          cursor += sizeof(uint32_t);
        }
      }
    }
  }

  [[nodiscard]] std::optional<uint64_t> claim_reachable(uint64_t anchor_offset,
                                                        uint32_t claim_words = 0u) {
    claim_words = claim_words == 0u ? island_words_ : claim_words;
    auto best = available_offsets_.end();
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();
    const auto [first_reachable, last_reachable] = reachable_target_bounds(anchor_offset);
    for (auto candidate = available_offsets_.lower_bound(first_reachable);
         candidate != available_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      if (!claim_fits(*candidate, claim_words) ||
          !compute_sopp_branch_simm16(anchor_offset, *candidate))
        continue;
      const uint64_t distance =
          anchor_offset < *candidate ? *candidate - anchor_offset : anchor_offset - *candidate;
      if (distance < best_distance) {
        best = candidate;
        best_distance = distance;
      }
    }
    if (best == available_offsets_.end())
      return std::nullopt;
    const uint64_t offset = *best;
    retire_overlaps(offset, claim_words);
    return offset;
  }

  [[nodiscard]] std::optional<uint64_t> claim_reachable_for_owner(uint64_t anchor_offset,
                                                                  uint64_t owner_entry,
                                                                  uint32_t claim_words = 0u) {
    claim_words = claim_words == 0u ? island_words_ : claim_words;
    auto best = available_offsets_.end();
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();
    const auto [first_reachable, last_reachable] = reachable_target_bounds(anchor_offset);
    for (auto candidate = available_offsets_.lower_bound(first_reachable);
         candidate != available_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      const auto owner = available_owner_entries_.find(*candidate);
      if (owner == available_owner_entries_.end() || owner->second != owner_entry ||
          !claim_fits(*candidate, claim_words) ||
          !compute_sopp_branch_simm16(anchor_offset, *candidate)) {
        continue;
      }
      const uint64_t distance =
          anchor_offset < *candidate ? *candidate - anchor_offset : anchor_offset - *candidate;
      if (distance < best_distance) {
        best = candidate;
        best_distance = distance;
      }
    }
    if (best == available_offsets_.end())
      return std::nullopt;
    const uint64_t offset = *best;
    retire_overlaps(offset, claim_words);
    return offset;
  }

  [[nodiscard]] bool has_reachable(uint64_t anchor_offset, uint32_t claim_words = 0u) const {
    claim_words = claim_words == 0u ? island_words_ : claim_words;
    const auto [first_reachable, last_reachable] = reachable_target_bounds(anchor_offset);
    for (auto candidate = available_offsets_.lower_bound(first_reachable);
         candidate != available_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      if (claim_fits(*candidate, claim_words) &&
          compute_sopp_branch_simm16(anchor_offset, *candidate))
        return true;
    }
    return false;
  }

  [[nodiscard]] std::optional<uint64_t>
  claim_reachable_from_all(std::span<const uint64_t> anchor_offsets, uint32_t claim_words = 0u) {
    if (anchor_offsets.empty())
      return std::nullopt;
    claim_words = claim_words == 0u ? island_words_ : claim_words;
    uint64_t first_reachable = 0u;
    uint64_t last_reachable = std::numeric_limits<uint64_t>::max();
    for (uint64_t anchor_offset : anchor_offsets) {
      const auto [first, last] = reachable_target_bounds(anchor_offset);
      first_reachable = std::max(first_reachable, first);
      last_reachable = std::min(last_reachable, last);
    }
    if (first_reachable > last_reachable)
      return std::nullopt;
    auto best = available_offsets_.end();
    uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
    for (auto candidate = available_offsets_.lower_bound(first_reachable);
         candidate != available_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      if (!claim_fits(*candidate, claim_words))
        continue;
      uint64_t max_distance = 0;
      bool reachable = true;
      for (uint64_t anchor_offset : anchor_offsets) {
        if (!compute_sopp_branch_simm16(anchor_offset, *candidate)) {
          reachable = false;
          break;
        }
        max_distance =
            std::max(max_distance, anchor_offset < *candidate ? *candidate - anchor_offset
                                                              : anchor_offset - *candidate);
      }
      if (reachable && max_distance < best_max_distance) {
        best = candidate;
        best_max_distance = max_distance;
      }
    }
    if (best == available_offsets_.end())
      return std::nullopt;
    const uint64_t offset = *best;
    retire_overlaps(offset, claim_words);
    return offset;
  }

  [[nodiscard]] std::optional<uint64_t>
  claim_word_reachable_from_all_for_owner(std::span<const uint64_t> anchor_offsets,
                                          uint64_t owner_entry) {
    if (anchor_offsets.empty())
      return std::nullopt;
    uint64_t first_reachable = 0u;
    uint64_t last_reachable = std::numeric_limits<uint64_t>::max();
    for (uint64_t anchor_offset : anchor_offsets) {
      const auto [first, last] = reachable_target_bounds(anchor_offset);
      first_reachable = std::max(first_reachable, first);
      last_reachable = std::min(last_reachable, last);
    }
    if (first_reachable > last_reachable)
      return std::nullopt;
    auto best = available_word_offsets_.end();
    uint64_t best_max_distance = std::numeric_limits<uint64_t>::max();
    for (auto candidate = available_word_offsets_.lower_bound(first_reachable);
         candidate != available_word_offsets_.end() && *candidate <= last_reachable; ++candidate) {
      const auto owner = available_word_owner_entries_.find(*candidate);
      if (owner == available_word_owner_entries_.end() || owner->second != owner_entry ||
          !claim_fits(*candidate, /*claim_words=*/1u)) {
        continue;
      }
      uint64_t max_distance = 0u;
      bool reachable = true;
      for (uint64_t anchor_offset : anchor_offsets) {
        if (!compute_sopp_branch_simm16(anchor_offset, *candidate)) {
          reachable = false;
          break;
        }
        max_distance =
            std::max(max_distance, anchor_offset < *candidate ? *candidate - anchor_offset
                                                              : anchor_offset - *candidate);
      }
      if (reachable && max_distance < best_max_distance) {
        best = candidate;
        best_max_distance = max_distance;
      }
    }
    if (best == available_word_offsets_.end())
      return std::nullopt;
    const uint64_t offset = *best;
    retire_overlaps(offset, /*claim_words=*/1u);
    return offset;
  }

  [[nodiscard]] std::vector<uint64_t> available_word_offsets() const {
    return {available_word_offsets_.begin(), available_word_offsets_.end()};
  }

  void reserve(uint64_t offset, uint32_t words) { retire_overlaps(offset, words); }

private:
  [[nodiscard]] static constexpr std::pair<uint64_t, uint64_t>
  reachable_target_bounds(uint64_t anchor_offset) {
    constexpr uint64_t kBackwardReach =
        static_cast<uint64_t>(-static_cast<int64_t>(std::numeric_limits<int16_t>::min())) *
        sizeof(uint32_t);
    constexpr uint64_t kForwardReach =
        static_cast<uint64_t>(std::numeric_limits<int16_t>::max()) * sizeof(uint32_t);
    const uint64_t branch_base =
        anchor_offset > std::numeric_limits<uint64_t>::max() - sizeof(uint32_t)
            ? std::numeric_limits<uint64_t>::max()
            : anchor_offset + sizeof(uint32_t);
    const uint64_t first = branch_base > kBackwardReach ? branch_base - kBackwardReach : 0u;
    const uint64_t last = branch_base > std::numeric_limits<uint64_t>::max() - kForwardReach
                              ? std::numeric_limits<uint64_t>::max()
                              : branch_base + kForwardReach;
    return {first, last};
  }

  [[nodiscard]] bool claim_fits(uint64_t offset, uint32_t claim_words) const {
    const uint64_t claim_bytes = static_cast<uint64_t>(claim_words) * sizeof(uint32_t);
    if (offset > text_.size() || claim_bytes > text_.size() - offset)
      return false;
    const uint64_t claim_end = offset + claim_bytes;
    if (std::ranges::any_of(claimed_ranges_, [&](const auto &range) {
          return range.first < claim_end && offset < range.second;
        })) {
      return false;
    }
    for (uint64_t byte = 0; byte < claim_bytes; byte += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, text_.data() + offset + byte, sizeof(word));
      if (word != nop_)
        return false;
    }
    return true;
  }

  void retire_overlaps(uint64_t offset, uint32_t claim_words) {
    const uint64_t claim_end = offset + static_cast<uint64_t>(claim_words) * sizeof(uint32_t);
    claimed_ranges_.emplace_back(offset, claim_end);
    available_word_offsets_.erase(available_word_offsets_.lower_bound(offset),
                                  available_word_offsets_.lower_bound(claim_end));
    const uint64_t island_bytes = static_cast<uint64_t>(island_words_) * sizeof(uint32_t);
    const uint64_t first_possible_overlap =
        offset >= island_bytes ? offset - island_bytes + 1u : 0u;
    auto candidate = available_offsets_.lower_bound(first_possible_overlap);
    while (candidate != available_offsets_.end() && *candidate < claim_end) {
      if (*candidate + island_bytes > offset) {
        candidate = available_offsets_.erase(candidate);
      } else {
        ++candidate;
      }
    }
  }

  std::span<const uint8_t> text_;
  uint32_t nop_ = 0;
  uint32_t island_words_ = 0;
  std::set<uint64_t> available_offsets_;
  std::unordered_map<uint64_t, uint64_t> available_owner_entries_;
  std::set<uint64_t> available_word_offsets_;
  std::unordered_map<uint64_t, uint64_t> available_word_owner_entries_;
  std::vector<std::pair<uint64_t, uint64_t>> claimed_ranges_;
};

} // namespace rocjitsu::consan_moi_impl
