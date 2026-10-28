// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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
/// unmapped. A snapshot is valid when `word_count` is between 1 and kMaxWords;
/// the encoding is the first `word_count` entries of `words`. A
/// default-constructed snapshot is not valid, so encoding() throws for it
/// instead of returning an empty span.
struct InstructionSnapshot {
  /// @brief Maximum encoding length in words, matching the CU's 16-byte
  /// instruction fetch window.
  static constexpr size_t kMaxWords = 4;
  uint64_t pc = 0;
  std::array<uint32_t, kMaxWords> words{};
  uint32_t word_count = 0;

  /// @throws std::logic_error if `word_count` is 0 or greater than kMaxWords.
  void validateEncoding() const {
    if (word_count == 0 || word_count > kMaxWords)
      throw std::logic_error("race detector instruction snapshot has an invalid word count");
  }

  /// @returns The first `word_count` words.
  /// @throws std::logic_error if `word_count` is 0 or greater than kMaxWords.
  std::span<const uint32_t> encoding() const {
    validateEncoding();
    return std::span(words).first(word_count);
  }
};

} // namespace rocjitsu::plugins::race_detector
