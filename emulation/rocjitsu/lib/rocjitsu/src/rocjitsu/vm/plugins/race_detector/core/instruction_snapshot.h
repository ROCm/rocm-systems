// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rocjitsu::plugins::race_detector {

/// @brief A copy of one executed instruction's PC and encoding.
///
/// @details The copy does not depend on code mappings or decoded-instruction
/// lifetimes, so a report can disassemble it after the code is rewritten or
/// unmapped. Every snapshot holds an encoding of 1 to kMaxWords words; the
/// caller must validate this length before construction, so encoding() is never empty.
class InstructionSnapshot {
public:
  /// @brief Maximum encoding length in words, matching the CU decode
  /// window, including the optional GFX10 MIMG extension.
  static constexpr size_t kMaxWords = 5;

  /// @brief Copy the complete encoding of the instruction at `pc`.
  /// @param sync_or_branch Whether the encoding is a wait, barrier or branch.
  /// @pre `encoding` contains between one and kMaxWords words.
  constexpr InstructionSnapshot(uint64_t pc, std::span<const uint32_t> encoding,
                                bool sync_or_branch = false)
      : pc_(pc), word_count_(static_cast<uint8_t>(encoding.size())),
        sync_or_branch_(sync_or_branch) {
    assert(!encoding.empty() && encoding.size() <= kMaxWords);
    std::ranges::copy(encoding, words_.begin());
  }

  constexpr uint64_t pc() const { return pc_; }
  constexpr std::span<const uint32_t> encoding() const {
    return std::span(words_).first(word_count_);
  }

  /// True for a wait, barrier or branch, which reports keep when eliding a trace.
  constexpr bool is_sync_or_branch() const { return sync_or_branch_; }

  /// Equal when the PCs and encodings are equal; unused words are always zero,
  /// and the sync/branch property is derived from the encoding.
  friend constexpr bool operator==(const InstructionSnapshot &,
                                   const InstructionSnapshot &) = default;

private:
  uint64_t pc_;
  std::array<uint32_t, kMaxWords> words_{};
  uint8_t word_count_;
  bool sync_or_branch_;
};

} // namespace rocjitsu::plugins::race_detector
