// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace rocjitsu::plugins::race_detector {

/// Owned encoding bytes, independent of code mappings and decoded-object lifetimes.
struct InstructionSnapshot {
  // Matches the CU's 16-byte instruction fetch window.
  static constexpr size_t kMaxWords = 4;
  uint64_t pc = 0;
  std::array<uint32_t, kMaxWords> words{};
  uint32_t word_count = 0;

  void validateEncoding() const {
    if (word_count == 0 || word_count > kMaxWords)
      throw std::logic_error("race detector instruction snapshot has an invalid word count");
  }

  std::span<const uint32_t> encoding() const {
    validateEncoding();
    return std::span(words).first(word_count);
  }
};

} // namespace rocjitsu::plugins::race_detector
