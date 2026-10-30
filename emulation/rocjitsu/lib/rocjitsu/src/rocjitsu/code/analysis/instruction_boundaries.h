// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {
class Instruction;

/// @brief Linear instruction boundaries and clause/reference facts, without CFG ownership.
/// @details Record instructions in increasing src_loc() order, retaining clause state across
/// block boundaries. Offsets and branch destinations use src_loc() byte coordinates
/// (.text-relative for checked edits). This index alone is not a source identity or a
/// trusted edit capability.
class InstructionBoundaries {
public:
  struct Entry {
    uint64_t offset;
    uint32_t size;      // Complete encoded instruction size in bytes, including literals.
    bool clause_member; // Covered by the instruction count of a preceding s_clause.
    bool clause;        // This instruction is s_clause.
  };

  void record(const Instruction &instruction);
  const Entry *find(uint64_t offset) const;
  std::span<const Entry> entries() const { return entries_; }
  std::span<const uint64_t> references() const { return references_; }
  bool reference_overflow() const { return reference_overflow_; }

private:
  std::vector<Entry> entries_;
  std::vector<uint64_t> references_; // Direct branch destinations, including outside the index.
  uint32_t clause_remaining_ = 0;
  bool reference_overflow_ = false; // A branch destination could not be represented as uint64_t.
};
} // namespace rocjitsu
