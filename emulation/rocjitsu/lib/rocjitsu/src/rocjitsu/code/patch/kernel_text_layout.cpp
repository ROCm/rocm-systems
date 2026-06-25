// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/kernel_text_layout.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace rocjitsu {

namespace {

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] bool is_nop_padding_gap(std::span<const uint8_t> text, uint64_t begin, uint64_t end,
                                      rj_code_arch_t arch) {
  if (begin > end || end > text.size() || (begin % sizeof(uint32_t)) != 0 ||
      (end % sizeof(uint32_t)) != 0)
    return false;

  const uint32_t nop = build_s_nop(0, arch);
  for (uint64_t off = begin; off < end; off += sizeof(uint32_t)) {
    if (text_word_at(text, off) != nop)
      return false;
  }
  return true;
}

} // namespace

void append_words(std::vector<uint8_t> &text, std::span<const uint32_t> words) {
  if (words.empty())
    return;

  const size_t old_size = text.size();
  const size_t extra_bytes = words.size() * sizeof(uint32_t);
  text.resize(old_size + extra_bytes);
  std::memcpy(text.data() + old_size, words.data(), extra_bytes);
}

void append_nop_padding(std::vector<uint8_t> &text, uint64_t byte_count, rj_code_arch_t arch) {
  assert(byte_count % sizeof(uint32_t) == 0 && "padding must be word-aligned");
  for (uint64_t off = 0; off < byte_count; off += sizeof(uint32_t)) {
    const uint32_t nop = build_s_nop(0, arch);
    append_words(text, std::span<const uint32_t>(&nop, 1));
  }
}

[[nodiscard]] uint64_t padding_for_residue(uint64_t current_offset, uint64_t target_residue,
                                           uint64_t alignment) {
  const uint64_t current_residue = current_offset % alignment;
  return (target_residue + alignment - current_residue) % alignment;
}

[[nodiscard]] std::optional<uint64_t> target_for_source_offset(const KernelTextLayout &layout,
                                                               uint64_t source_offset) {
  if (layout.blocks.empty())
    return std::nullopt;

  // Blocks are emitted in source order and are non-overlapping in source space
  // for the current scope; binary search preserves the prior semantics of the
  // linear scan while reducing lookup complexity to O(log N).
  const auto it = std::upper_bound(layout.blocks.begin(), layout.blocks.end(), source_offset,
                                   [](uint64_t source, const BlockPlacement &placement) {
                                     return source < placement.source_start;
                                   });
  if (it == layout.blocks.begin())
    return std::nullopt;

  const BlockPlacement &placement = *(it - 1);
  if (source_offset < placement.source_start || source_offset >= placement.source_end)
    return std::nullopt;

  return placement.target_start + (source_offset - placement.source_start);
}

[[nodiscard]] std::optional<uint64_t> target_for_source_fallthrough(const KernelTextLayout &layout,
                                                                    std::span<const uint8_t> text,
                                                                    uint64_t source_offset,
                                                                    rj_code_arch_t arch) {
  if (auto target = target_for_source_offset(layout, source_offset))
    return target;

  // Local cave expansions return to the source instruction's next PC. When the
  // compiler placed only s_nop padding between two reachable blocks, that next
  // PC is not present in the compact relocated body. Skipping that source-only
  // padding here keeps the body layout compact while preserving the observable
  // fallthrough target: executing zero or more nops reaches the next block.
  for (const BlockPlacement &placement : layout.blocks) {
    if (source_offset >= placement.source_start)
      continue;
    if (is_nop_padding_gap(text, source_offset, placement.source_start, arch))
      return placement.target_start;
    break;
  }
  return std::nullopt;
}

} // namespace rocjitsu
