// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/code_placement.h"

#include "rocjitsu/code/patch/instruction_builder.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {
namespace {

constexpr uint32_t kSCodeEnd = 0xBF9F0000u;
constexpr uint32_t kSNop0 = 0xBF800000u;

[[nodiscard]] uint32_t build_sopp_with_simm(uint32_t opcode_word, int16_t simm16) {
  return (opcode_word & 0xFFFF0000u) | static_cast<uint16_t>(simm16);
}

[[nodiscard]] bool is_s_branch_word(uint32_t word) { return (word & 0xFFFF0000u) == 0xBFA00000u; }

[[nodiscard]] std::optional<uint32_t> inverse_conditional_sopp(uint32_t word) {
  switch (word & 0xFFFF0000u) {
  case 0xBFA10000u:
    return 0xBFA20000u; // s_cbranch_scc0 -> s_cbranch_scc1
  case 0xBFA20000u:
    return 0xBFA10000u; // s_cbranch_scc1 -> s_cbranch_scc0
  case 0xBFA30000u:
    return 0xBFA40000u; // s_cbranch_vccz -> s_cbranch_vccnz
  case 0xBFA40000u:
    return 0xBFA30000u; // s_cbranch_vccnz -> s_cbranch_vccz
  case 0xBFA50000u:
    return 0xBFA60000u; // s_cbranch_execz -> s_cbranch_execnz
  case 0xBFA60000u:
    return 0xBFA50000u; // s_cbranch_execnz -> s_cbranch_execz
  default:
    return std::nullopt;
  }
}

[[nodiscard]] uint64_t align_up_to_word(uint64_t offset) {
  constexpr uint64_t kWordMask = sizeof(uint32_t) - 1;
  return (offset + kWordMask) & ~kWordMask;
}

[[nodiscard]] bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

[[nodiscard]] std::optional<uint64_t> padding_to_preserve_residue(uint64_t value,
                                                                  uint64_t residue,
                                                                  uint64_t alignment) {
  if (alignment == 0)
    return std::nullopt;
  const uint64_t desired = residue % alignment;
  const uint64_t actual = value % alignment;
  if (desired >= actual)
    return desired - actual;
  return alignment - (actual - desired);
}

[[nodiscard]] uint32_t read_u32(std::span<const uint8_t> bytes, uint64_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

[[nodiscard]] bool is_local_cave_padding_run(std::span<const uint8_t> text, uint64_t start,
                                             uint64_t size) {
  for (uint64_t off = start; off < start + size; off += sizeof(uint32_t)) {
    const uint32_t word = read_u32(text, off);
    if (word != kSCodeEnd && word != kSNop0)
      return false;
  }
  return true;
}

[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>>
unprotected_text_gaps(uint64_t text_size,
                      std::span<const std::pair<uint64_t, uint64_t>> protected_ranges) {
  std::vector<std::pair<uint64_t, uint64_t>> gaps;
  uint64_t cursor = 0;
  for (const auto &[range_start, range_end] : protected_ranges) {
    if (range_start > cursor)
      gaps.emplace_back(cursor, range_start);
    cursor = std::max(cursor, range_end);
  }
  if (cursor < text_size)
    gaps.emplace_back(cursor, text_size);
  return gaps;
}

[[nodiscard]] std::optional<uint64_t>
find_offset_mapping(std::span<const std::pair<uint64_t, uint64_t>> offset_map,
                    uint64_t source_offset) {
  for (const auto &[source, translated] : offset_map) {
    if (source == source_offset)
      return translated;
  }
  return std::nullopt;
}

} // namespace

bool compute_sopp_branch_offset(uint64_t branch_pc, uint64_t target, int16_t &offset_dwords) {
  // SOPP branches encode a signed dword offset from the next instruction. Keep
  // the range check shared so both cave entry and return branches fail closed.
  constexpr int64_t kBranchPcBiasBytes = static_cast<int64_t>(sizeof(uint32_t));
  constexpr uint64_t kMaxSignedTarget = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  constexpr uint64_t kMaxSignedBranchPc =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - kBranchPcBiasBytes);
  // The PCs are unsigned until this check passes. Compare against the casted
  // signed int64_t limits so the later signed conversion, and branch_pc + 4,
  // cannot overflow.
  if (branch_pc > kMaxSignedBranchPc || target > kMaxSignedTarget)
    return false;

  const int64_t delta_bytes = static_cast<int64_t>(target) - (static_cast<int64_t>(branch_pc) + 4);
  if (delta_bytes % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return false;

  const int64_t delta_dwords = delta_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (delta_dwords < std::numeric_limits<int16_t>::min() ||
      delta_dwords > std::numeric_limits<int16_t>::max())
    return false;

  offset_dwords = static_cast<int16_t>(delta_dwords);
  return true;
}

bool ranges_overlap(uint64_t lhs_start, uint64_t lhs_end, uint64_t rhs_start, uint64_t rhs_end) {
  return lhs_start < rhs_end && rhs_start < lhs_end;
}

bool overlaps_any_range(uint64_t start, uint64_t end,
                        std::span<const std::pair<uint64_t, uint64_t>> ranges) {
  return std::ranges::any_of(ranges, [&](const auto &range) {
    return ranges_overlap(start, end, range.first, range.second);
  });
}

std::optional<BranchableTextPlacement> find_local_text_cave(
    std::span<const uint8_t> text, const LocalTextCaveRequest &request,
    std::span<const std::pair<uint64_t, uint64_t>> reserved_ranges,
    std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
    bool allow_unreachable_text_caves) {
  const uint64_t cave_size = request.cave_size_bytes;
  if (cave_size == 0 || cave_size % sizeof(uint32_t) != 0 || cave_size > text.size())
    return std::nullopt;

  const uint64_t stub_next = request.source.end;
  const uint64_t last_start = text.size() - cave_size;

  auto try_candidate = [&](uint64_t candidate) -> std::optional<BranchableTextPlacement> {
    const uint64_t candidate_end = candidate + cave_size;
    if (ranges_overlap(candidate, candidate_end, request.source.start, request.source.end))
      return std::nullopt;
    if (overlaps_any_range(candidate, candidate_end, reserved_ranges) ||
        overlaps_any_range(candidate, candidate_end, protected_ranges)) {
      return std::nullopt;
    }
    if (!allow_unreachable_text_caves && !is_local_cave_padding_run(text, candidate, cave_size))
      return std::nullopt;

    BranchableTextPlacement placement;
    if (!compute_sopp_branch_offset(request.source.start, candidate, placement.entry_branch_dwords))
      return std::nullopt;

    const uint64_t return_branch_pc = candidate + request.body_size_bytes;
    if (!compute_sopp_branch_offset(return_branch_pc, stub_next, placement.exit_branch_dwords))
      return std::nullopt;

    placement.offset = candidate;
    return placement;
  };

  if (allow_unreachable_text_caves) {
    const auto gaps = unprotected_text_gaps(text.size(), protected_ranges);

    auto scan_forward = [&](uint64_t start, uint64_t end) -> std::optional<BranchableTextPlacement> {
      if (end < start || end - start < cave_size)
        return std::nullopt;
      for (uint64_t candidate = align_up_to_word(start); candidate + cave_size <= end;
           candidate += sizeof(uint32_t)) {
        if (auto cave = try_candidate(candidate))
          return cave;
      }
      return std::nullopt;
    };

    auto scan_backward = [&](uint64_t start,
                             uint64_t end) -> std::optional<BranchableTextPlacement> {
      if (end < start || end - start < cave_size || request.source.start < cave_size)
        return std::nullopt;
      uint64_t candidate = std::min(end - cave_size, request.source.start - cave_size);
      candidate -= candidate % sizeof(uint32_t);
      while (candidate >= start) {
        if (auto cave = try_candidate(candidate))
          return cave;
        if (candidate < start + sizeof(uint32_t))
          break;
        candidate -= sizeof(uint32_t);
      }
      return std::nullopt;
    };

    const uint64_t forward_start = align_up_to_word(request.source.end);
    for (const auto &[gap_start, gap_end] : gaps) {
      if (gap_end <= forward_start)
        continue;
      if (auto cave = scan_forward(std::max(gap_start, forward_start), gap_end))
        return cave;
    }

    for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
      if (it->first >= request.source.start)
        continue;
      if (auto cave = scan_backward(it->first, std::min(it->second, request.source.start)))
        return cave;
    }

    return std::nullopt;
  }

  for (uint64_t candidate = align_up_to_word(request.source.end); candidate <= last_start;
       candidate += sizeof(uint32_t)) {
    if (auto cave = try_candidate(candidate))
      return cave;
  }

  if (request.source.start >= cave_size) {
    uint64_t candidate = std::min(last_start, request.source.start - cave_size);
    candidate -= candidate % sizeof(uint32_t);
    while (true) {
      if (auto cave = try_candidate(candidate))
        return cave;
      if (candidate < sizeof(uint32_t))
        break;
      candidate -= sizeof(uint32_t);
    }
  }

  return std::nullopt;
}

std::optional<BranchableTextPlacement> find_local_branch_island(
    std::span<const uint8_t> text, const LocalBranchIslandRequest &request,
    std::span<const std::pair<uint64_t, uint64_t>> reserved_ranges,
    std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
    bool allow_unreachable_text_caves) {
  constexpr uint64_t kIslandSize = sizeof(uint32_t);
  if (text.size() < kIslandSize)
    return std::nullopt;

  const uint64_t last_start = text.size() - kIslandSize;

  auto try_candidate = [&](uint64_t candidate) -> std::optional<BranchableTextPlacement> {
    const uint64_t candidate_end = candidate + kIslandSize;
    if (ranges_overlap(candidate, candidate_end, request.source.start, request.source.end))
      return std::nullopt;
    if (overlaps_any_range(candidate, candidate_end, reserved_ranges) ||
        overlaps_any_range(candidate, candidate_end, protected_ranges)) {
      return std::nullopt;
    }
    if (!allow_unreachable_text_caves && !is_local_cave_padding_run(text, candidate, kIslandSize))
      return std::nullopt;

    BranchableTextPlacement placement;
    if (!compute_sopp_branch_offset(request.source.start, candidate, placement.entry_branch_dwords))
      return std::nullopt;
    if (!compute_sopp_branch_offset(candidate, request.island_target,
                                    placement.exit_branch_dwords)) {
      return std::nullopt;
    }

    placement.offset = candidate;
    return placement;
  };

  auto scan_forward = [&](uint64_t start, uint64_t end) -> std::optional<BranchableTextPlacement> {
    if (end < start || end - start < kIslandSize)
      return std::nullopt;
    for (uint64_t candidate = align_up_to_word(start); candidate + kIslandSize <= end;
         candidate += kIslandSize) {
      if (auto island = try_candidate(candidate))
        return island;
    }
    return std::nullopt;
  };

  auto scan_backward = [&](uint64_t start, uint64_t end) -> std::optional<BranchableTextPlacement> {
    if (end < start || end - start < kIslandSize)
      return std::nullopt;
    uint64_t candidate = std::min(end - kIslandSize, last_start);
    candidate -= candidate % kIslandSize;
    while (candidate >= start) {
      if (auto island = try_candidate(candidate))
        return island;
      if (candidate < start + kIslandSize)
        break;
      candidate -= kIslandSize;
    }
    return std::nullopt;
  };

  if (allow_unreachable_text_caves) {
    const auto gaps = unprotected_text_gaps(text.size(), protected_ranges);

    for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
      if (auto island = scan_backward(it->first, it->second))
        return island;
    }
    for (const auto &[gap_start, gap_end] : gaps) {
      if (auto island = scan_forward(gap_start, gap_end))
        return island;
    }
    return std::nullopt;
  }

  if (auto island = scan_backward(0, text.size()))
    return island;
  return scan_forward(0, text.size());
}

std::optional<ExpandedTextScopePlacement> plan_expanded_text_scope_placement(
    const ExpandedTextScopePlacementRequest &request) {
  if (request.entry_alignment_bytes == 0)
    return std::nullopt;

  uint64_t base_without_padding = 0;
  if (!checked_add(request.original_text_size_bytes, request.current_tail_size_bytes,
                   base_without_padding)) {
    return std::nullopt;
  }

  const uint64_t entry_residue = request.original_entry_offset % request.entry_alignment_bytes;
  ExpandedTextScopePlacement placement;

  if (request.prologue_size_bytes == 0) {
    uint64_t entry_without_padding = 0;
    if (!checked_add(base_without_padding, request.translated_entry_offset,
                     entry_without_padding)) {
      return std::nullopt;
    }

    auto padding =
        padding_to_preserve_residue(entry_without_padding, entry_residue,
                                    request.entry_alignment_bytes);
    if (!padding || *padding % sizeof(uint32_t) != 0)
      return std::nullopt;

    placement.padding_bytes = *padding;
    if (!checked_add(request.current_tail_size_bytes, placement.padding_bytes,
                     placement.body_offset)) {
      return std::nullopt;
    }
    if (!checked_add(placement.body_offset, request.translated_entry_offset,
                     placement.descriptor_entry_offset)) {
      return std::nullopt;
    }
    return placement;
  }

  auto padding =
      padding_to_preserve_residue(base_without_padding, entry_residue,
                                  request.entry_alignment_bytes);
  if (!padding || *padding % sizeof(uint32_t) != 0)
    return std::nullopt;

  placement.padding_bytes = *padding;
  uint64_t launch_stub_offset = 0;
  if (!checked_add(request.current_tail_size_bytes, placement.padding_bytes,
                   launch_stub_offset)) {
    return std::nullopt;
  }
  placement.launch_stub_offset = launch_stub_offset;

  uint64_t branch_pc = 0;
  if (!checked_add(launch_stub_offset, request.prologue_size_bytes, branch_pc))
    return std::nullopt;
  if (!checked_add(branch_pc, sizeof(uint32_t), placement.body_offset))
    return std::nullopt;

  uint64_t body_entry = 0;
  if (!checked_add(placement.body_offset, request.translated_entry_offset, body_entry))
    return std::nullopt;

  int16_t entry_branch = 0;
  if (!compute_sopp_branch_offset(branch_pc, body_entry, entry_branch))
    return std::nullopt;

  placement.descriptor_entry_offset = *placement.launch_stub_offset;
  placement.prologue_branch_dwords = entry_branch;
  return placement;
}

ExpandedTextBranchRelocation
relocate_expanded_text_branches(const ExpandedTextBranchRelocationRequest &request) {
  ExpandedTextBranchRelocation result;
  result.words.assign(request.words.begin(), request.words.end());
  result.offset_map.assign(request.offset_map.begin(), request.offset_map.end());
  if (request.branches.empty()) {
    result.success = true;
    return result;
  }

  auto fail = [&](std::string message) {
    result.success = false;
    result.words.clear();
    result.offset_map.clear();
    result.message = std::move(message);
    return result;
  };

  struct PcRelativeSetpc {
    size_t getpc_word = 0;
    size_t target_word = 0;
    uint16_t sgpr_pair = 0;
  };

  const size_t old_word_count = request.words.size();
  std::vector<PcRelativeSetpc> pc_relative_setpcs;
  for (size_t word = 0; word + 5 < old_word_count; ++word) {
    const uint32_t getpc = request.words[word];
    const uint16_t sgpr_pair = static_cast<uint16_t>((getpc >> 16) & 0x7Fu);
    if (sgpr_pair >= 127 || getpc != pack_sop1(71, sgpr_pair, 0))
      continue;
    if (request.words[word + 1] != pack_sop2(0, sgpr_pair, sgpr_pair, 255))
      continue;
    if (request.words[word + 3] !=
        pack_sop2(4, static_cast<uint16_t>(sgpr_pair + 1u),
                  static_cast<uint16_t>(sgpr_pair + 1u), 255)) {
      continue;
    }
    if (request.words[word + 5] != pack_sop1(72, 0, sgpr_pair))
      continue;

    const uint64_t delta_bits =
        static_cast<uint64_t>(request.words[word + 2]) |
        (static_cast<uint64_t>(request.words[word + 4]) << 32);
    const int64_t delta = static_cast<int64_t>(delta_bits);
    const int64_t target =
        static_cast<int64_t>(word * sizeof(uint32_t) + sizeof(uint32_t)) + delta;
    if (target < 0 || target % static_cast<int64_t>(sizeof(uint32_t)) != 0) {
      return fail("expanded text copy cannot relocate unaligned s_setpc target");
    }
    const auto target_word = static_cast<size_t>(target / static_cast<int64_t>(sizeof(uint32_t)));
    if (target_word > old_word_count)
      return fail("expanded text copy s_setpc target is outside the copied CFG");
    pc_relative_setpcs.push_back({word, target_word, sgpr_pair});
  }

  std::unordered_map<uint64_t, const ExpandedTextBranchFixup *> branch_by_word;
  for (const ExpandedTextBranchFixup &branch : request.branches)
    branch_by_word.emplace(branch.word_index, &branch);

  std::unordered_set<uint64_t> long_branches;
  std::vector<size_t> old_to_new(old_word_count + 1);

  auto rebuild_map = [&]() {
    size_t new_word = 0;
    for (size_t old_word = 0; old_word < old_word_count; ++old_word) {
      old_to_new[old_word] = new_word;
      if (long_branches.contains(old_word)) {
        const uint32_t word = request.words[old_word];
        const auto branch_it = branch_by_word.find(old_word);
        const uint32_t long_words =
            branch_it != branch_by_word.end() && is_s_branch_word(word) ? 6u : 7u;
        new_word += long_words;
      } else {
        ++new_word;
      }
    }
    old_to_new[old_word_count] = new_word;
  };

  auto translated_target_word =
      [&](const ExpandedTextBranchFixup &branch) -> std::optional<size_t> {
    const auto translated_offset = find_offset_mapping(request.offset_map, branch.target_offset);
    if (!translated_offset)
      return std::nullopt;
    if (*translated_offset % sizeof(uint32_t) != 0)
      return std::nullopt;
    const size_t target_word = *translated_offset / sizeof(uint32_t);
    if (target_word > old_word_count)
      return std::nullopt;
    return target_word;
  };

  for (;;) {
    rebuild_map();
    bool added_long_branch = false;
    for (const ExpandedTextBranchFixup &branch : request.branches) {
      const auto target_word = translated_target_word(branch);
      if (!target_word)
        return fail("expanded text copy branch target is outside the copied CFG for " +
                    branch.mnemonic);
      if (long_branches.contains(branch.word_index))
        continue;

      int16_t branch_dwords = 0;
      const uint64_t branch_pc = old_to_new[branch.word_index] * sizeof(uint32_t);
      const uint64_t target_pc = old_to_new[*target_word] * sizeof(uint32_t);
      if (compute_sopp_branch_offset(branch_pc, target_pc, branch_dwords))
        continue;

      const uint32_t word = request.words[branch.word_index];
      if (!branch.sgpr_pair)
        return fail("expanded text copy needs a dead SGPR pair for long branch " +
                    branch.mnemonic);
      if (!is_s_branch_word(word) && !inverse_conditional_sopp(word))
        return fail("expanded text copy cannot build long branch for " + branch.mnemonic);

      long_branches.insert(branch.word_index);
      added_long_branch = true;
    }
    if (!added_long_branch)
      break;
  }

  rebuild_map();
  std::vector<uint32_t> relocated_words;
  relocated_words.reserve(old_to_new[old_word_count]);
  for (size_t old_word = 0; old_word < old_word_count; ++old_word) {
    const auto branch_it = branch_by_word.find(old_word);
    if (!long_branches.contains(old_word) || branch_it == branch_by_word.end()) {
      relocated_words.push_back(request.words[old_word]);
      continue;
    }

    const ExpandedTextBranchFixup &branch = *branch_it->second;
    const auto target_word = translated_target_word(branch);
    if (!target_word)
      return fail("expanded text copy branch target is outside the copied CFG for " +
                  branch.mnemonic);

    const uint64_t target_pc = old_to_new[*target_word] * sizeof(uint32_t);
    const uint32_t word = request.words[old_word];
    if (is_s_branch_word(word)) {
      auto long_branch =
          build_s_setpc_long_branch(relocated_words.size() * sizeof(uint32_t), target_pc,
                                    *branch.sgpr_pair);
      if (long_branch.empty())
        return fail("expanded text copy could not build long branch for " + branch.mnemonic);
      relocated_words.insert(relocated_words.end(), long_branch.begin(), long_branch.end());
      continue;
    }

    auto inverse = inverse_conditional_sopp(word);
    auto long_branch =
        build_s_setpc_long_branch((relocated_words.size() + 1) * sizeof(uint32_t), target_pc,
                                  *branch.sgpr_pair);
    if (!inverse || long_branch.empty() ||
        long_branch.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
      return fail("expanded text copy could not build long conditional branch for " +
                  branch.mnemonic);
    }
    relocated_words.push_back(
        build_sopp_with_simm(*inverse, static_cast<int16_t>(long_branch.size())));
    relocated_words.insert(relocated_words.end(), long_branch.begin(), long_branch.end());
  }

  for (const ExpandedTextBranchFixup &branch : request.branches) {
    if (long_branches.contains(branch.word_index))
      continue;
    const auto target_word = translated_target_word(branch);
    if (!target_word)
      return fail("expanded text copy branch target is outside the copied CFG for " +
                  branch.mnemonic);

    int16_t branch_dwords = 0;
    const uint64_t branch_pc = old_to_new[branch.word_index] * sizeof(uint32_t);
    const uint64_t target_pc = old_to_new[*target_word] * sizeof(uint32_t);
    if (!compute_sopp_branch_offset(branch_pc, target_pc, branch_dwords))
      return fail("expanded text copy branch range exceeds s_branch simm16 for " +
                  branch.mnemonic);
    relocated_words[old_to_new[branch.word_index]] =
        build_sopp_with_simm(relocated_words[old_to_new[branch.word_index]], branch_dwords);
  }

  for (const PcRelativeSetpc &pc_branch : pc_relative_setpcs) {
    const size_t new_getpc_word = old_to_new[pc_branch.getpc_word];
    const size_t new_target_word = old_to_new[pc_branch.target_word];
    const int64_t getpc_pc = static_cast<int64_t>(new_getpc_word * sizeof(uint32_t));
    const int64_t target_pc = static_cast<int64_t>(new_target_word * sizeof(uint32_t));
    const int64_t delta = target_pc - (getpc_pc + static_cast<int64_t>(sizeof(uint32_t)));
    const uint64_t delta_bits = static_cast<uint64_t>(delta);
    if (new_getpc_word + 5 >= relocated_words.size())
      return fail("expanded text copy s_setpc relocation is out of range");
    relocated_words[new_getpc_word + 2] = static_cast<uint32_t>(delta_bits & 0xFFFF'FFFFu);
    relocated_words[new_getpc_word + 4] = static_cast<uint32_t>(delta_bits >> 32);
  }

  for (auto &[_, translated_offset] : result.offset_map) {
    if (translated_offset % sizeof(uint32_t) != 0)
      return fail("expanded text copy has an unaligned translated branch target");
    const size_t old_word = translated_offset / sizeof(uint32_t);
    if (old_word > old_word_count)
      return fail("expanded text copy has an out-of-range translated branch target");
    translated_offset = old_to_new[old_word] * sizeof(uint32_t);
  }

  result.success = true;
  result.words = std::move(relocated_words);
  return result;
}

ExpandedTextPcRelativeRelocation relocate_expanded_text_pc_relative_fixups(
    const ExpandedTextPcRelativeRelocationRequest &request) {
  ExpandedTextPcRelativeRelocation result;
  if (request.fixups.empty()) {
    result.success = true;
    return result;
  }

  auto fail = [&](std::string message) {
    result.success = false;
    result.message = std::move(message);
    return result;
  };

  constexpr uint32_t kOpSAddCoI32 = 2;
  for (const ExpandedTextPcRelativeFixup &fixup : request.fixups) {
    const auto getpc_offset = find_offset_mapping(request.offset_map, fixup.getpc_offset);
    const auto add_tmp_offset = find_offset_mapping(request.offset_map, fixup.add_tmp_offset);
    if (!getpc_offset || !add_tmp_offset)
      continue;
    if (*getpc_offset % sizeof(uint32_t) != 0 || *add_tmp_offset % sizeof(uint32_t) != 0) {
      return fail("expanded text copy cannot relocate unaligned pc-relative " + fixup.kind);
    }

    const size_t add_tmp_word = *add_tmp_offset / sizeof(uint32_t);
    if (add_tmp_word + 1 >= request.words.size())
      return fail("expanded text copy pc-relative " + fixup.kind + " literal is out of range");

    const uint32_t add_tmp = request.words[add_tmp_word];
    const uint16_t tmp = static_cast<uint16_t>((add_tmp >> 16) & 0x7Fu);
    if (tmp >= 128 ||
        add_tmp != pack_sop2(kOpSAddCoI32, tmp, 255, scalar_positive_inline_u32(4))) {
      return fail("expanded text copy pc-relative " + fixup.kind +
                  " materialization was rewritten");
    }

    const auto copied_target_offset = find_offset_mapping(request.offset_map, fixup.target_offset);
    const int64_t target =
        copied_target_offset
            ? static_cast<int64_t>(request.original_text_size_bytes + request.scope_base_bytes +
                                   *copied_target_offset)
            : static_cast<int64_t>(fixup.target_offset);
    const int64_t getpc_pc = static_cast<int64_t>(request.original_text_size_bytes +
                                                 request.scope_base_bytes + *getpc_offset);
    const int64_t literal = target - getpc_pc - 2 * static_cast<int64_t>(sizeof(uint32_t));
    if (literal < std::numeric_limits<int32_t>::min() ||
        literal > std::numeric_limits<int32_t>::max()) {
      return fail("expanded text copy pc-relative " + fixup.kind + " exceeds literal range");
    }

    request.words[add_tmp_word + 1] = static_cast<uint32_t>(static_cast<int32_t>(literal));
  }

  result.success = true;
  return result;
}

ExpandedTextDescriptorRedirectionPlan plan_expanded_text_descriptor_redirections(
    std::span<const ExpandedTextDescriptorEntry> descriptors,
    std::span<const std::pair<uint64_t, uint64_t>> copied_entry_offsets,
    uint64_t original_text_size_bytes) {
  ExpandedTextDescriptorRedirectionPlan plan;
  std::unordered_set<uint64_t> applied_descriptors;
  for (const ExpandedTextDescriptorEntry &descriptor : descriptors) {
    if (!applied_descriptors.insert(descriptor.descriptor_file_offset).second)
      continue;

    const auto copied_entry_offset =
        find_offset_mapping(copied_entry_offsets, descriptor.entry_text_offset);
    if (!copied_entry_offset) {
      plan.success = false;
      plan.redirections.clear();
      plan.message =
          "expanded text copy could not map a kernel descriptor entry; leaving code object "
          "unchanged";
      return plan;
    }

    uint64_t redirected_entry_offset = 0;
    if (!checked_add(original_text_size_bytes, *copied_entry_offset, redirected_entry_offset)) {
      plan.success = false;
      plan.redirections.clear();
      plan.message =
          "expanded text copy could not map a kernel descriptor entry; leaving code object "
          "unchanged";
      return plan;
    }

    plan.redirections.push_back({.descriptor_file_offset = descriptor.descriptor_file_offset,
                                 .original_entry_offset = descriptor.entry_text_offset,
                                 .redirected_entry_offset = redirected_entry_offset});
  }

  plan.success = true;
  return plan;
}

} // namespace rocjitsu
