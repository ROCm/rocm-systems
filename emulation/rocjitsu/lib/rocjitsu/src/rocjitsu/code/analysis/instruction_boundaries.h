// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {
class Instruction;

/// @brief Linear instruction boundaries and clause/reference facts, without CFG ownership.
/// @details Record instructions in source order, retaining clause state across block
/// boundaries. This index alone is not a source identity or a trusted edit capability.
class InstructionBoundaries {
public:
  struct Entry {
    uint64_t offset;
    uint32_t size;
    bool clause_member;
    bool clause;
  };

  void record(const Instruction &instruction);
  const Entry *find(uint64_t offset) const;
  std::span<const Entry> entries() const { return entries_; }
  std::span<const uint64_t> references() const { return references_; }
  bool reference_overflow() const { return reference_overflow_; }

private:
  std::vector<Entry> entries_;
  std::vector<uint64_t> references_;
  uint32_t clause_remaining_ = 0;
  bool reference_overflow_ = false;
};
} // namespace rocjitsu
