// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/instruction_snapshot.h"

#include <array>
#include <optional>

namespace rocjitsu::plugins::race_detector {

// Core tests construct memory events without executing ISA. A complete s_nop
// encoding supplies their instruction metadata without coupling them to a decoder.
inline InstructionSnapshot testInstruction(uint64_t pc, uint32_t word = 0xbf800000) {
  return {pc, std::array{word}};
}

/// Test shorthand for one or two unit increments in the same completion class.
class CounterObligations {
public:
  CounterObligations(amdgpu::WaitCounterType counter, amdgpu::MemoryCompletionClass order,
                     std::optional<amdgpu::WaitCounterType> additional = std::nullopt)
      : obligations_{amdgpu::MemoryCounterObligation{counter, order}, {}} {
    if (additional)
      obligations_[count_++] = {*additional, order};
  }

  operator std::span<const amdgpu::MemoryCounterObligation>() const {
    return {obligations_.data(), count_};
  }

private:
  std::array<amdgpu::MemoryCounterObligation, 2> obligations_;
  size_t count_ = 1;
};

} // namespace rocjitsu::plugins::race_detector
