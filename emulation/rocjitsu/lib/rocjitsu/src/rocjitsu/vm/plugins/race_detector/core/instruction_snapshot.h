// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rocjitsu::plugins::race_detector {

/// Owned encoding bytes, independent of code mappings and decoded-object lifetimes.
struct InstructionSnapshot {
  static constexpr size_t kMaxWords = 4;
  uint64_t pc = 0;
  std::array<uint32_t, kMaxWords> words{};
  uint32_t word_count = 0;

  std::span<const uint32_t> encoding() const { return std::span(words).first(word_count); }
};

struct RecordedInstruction {
  InstructionSnapshot snapshot;
  // The sequence identifies one execution, including repeated visits to a PC.
  uint64_t sequence = UINT64_MAX;
};

} // namespace rocjitsu::plugins::race_detector
