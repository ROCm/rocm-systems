// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace rocjitsu::plugins::race_detector {

/// @brief A copy of one executed instruction's PC and encoding.
///
/// @details The copy does not depend on code mappings or decoded-instruction
/// lifetimes, so a report can disassemble it after the code is rewritten or
/// unmapped. Every snapshot holds an encoding of 1 to kMaxWords words; the
/// constructor rejects any other length, so encoding() is never empty.
class InstructionSnapshot {
public:
  /// @brief Maximum encoding length in words, matching the CU's 16-byte
  /// instruction fetch window.
  static constexpr size_t kMaxWords = 4;

  /// @brief Copy the complete encoding of the instruction at `pc`.
  /// @throws std::logic_error if `encoding` is empty or longer than kMaxWords.
  constexpr InstructionSnapshot(uint64_t pc, std::span<const uint32_t> encoding,
                                bool keep_in_trace = false)
      : pc_(pc), word_count_(static_cast<uint32_t>(encoding.size())),
        keep_in_trace_(keep_in_trace) {
    if (encoding.empty() || encoding.size() > kMaxWords)
      throw std::logic_error("race detector instruction snapshot has an invalid word count");
    std::ranges::copy(encoding, words_.begin());
  }

  constexpr uint64_t pc() const { return pc_; }
  constexpr std::span<const uint32_t> encoding() const {
    return std::span(words_).first(word_count_);
  }

  // Preserve synchronization and control flow when the middle of a trace is elided.
  constexpr bool keep_in_trace() const { return keep_in_trace_; }

  /// Equal when the PCs and encodings are equal; unused words are always zero.
  friend constexpr bool operator==(const InstructionSnapshot &a, const InstructionSnapshot &b) {
    return a.pc_ == b.pc_ && a.word_count_ == b.word_count_ && a.words_ == b.words_;
  }

private:
  uint64_t pc_;
  std::array<uint32_t, kMaxWords> words_{};
  uint32_t word_count_;
  bool keep_in_trace_;
};

} // namespace rocjitsu::plugins::race_detector
