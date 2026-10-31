// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/instruction_boundaries.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <optional>

namespace rocjitsu {
void InstructionBoundaries::record(const Instruction &instruction) {
  const bool clause = instruction.mnemonic() == "s_clause";
  entries_.push_back({.offset = instruction.src_loc(),
                      .size = static_cast<uint32_t>(instruction.size()),
                      .clause_member = clause_remaining_ != 0,
                      .clause = clause});
  if (clause_remaining_ != 0)
    --clause_remaining_;
  if (clause && instruction.size() == 4 && instruction.num_src_operands() == 1 &&
      instruction.src_operand(0)) {
    // BREAK_SPAN in bits 8:11 is not part of the following-instruction count.
    const uint32_t count =
        (static_cast<uint32_t>(instruction.src_operand(0)->encoding_value()) & 0x3fu) + 1u;
    clause_remaining_ = std::max(clause_remaining_, count);
  }
  if (const std::optional<int64_t> displacement = instruction.branch_offset_bytes()) {
    const std::optional<uint64_t> destination =
        add_signed_offset(instruction.src_loc() + instruction.size(), *displacement);
    if (destination)
      references_.push_back(*destination);
    else
      reference_overflow_ = true;
  }
}

const InstructionBoundaries::Entry *InstructionBoundaries::find(uint64_t offset) const {
  const auto found = std::ranges::lower_bound(entries_, offset, {}, &Entry::offset);
  return found != entries_.end() && found->offset == offset ? &*found : nullptr;
}
} // namespace rocjitsu
