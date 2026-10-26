// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_timing.h
/// @brief The static view of a decoded instruction a cycle-level sequencer needs.

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// @brief What a cycle-level sequencer needs to know about one decoded instruction.
/// @details Derived once, when a sequencer decodes the instruction ahead of
/// the CU. The CU decodes it again when the instruction executes, so this
/// carries no reference to the decoded Instruction.
struct InstructionTiming {
  uint64_t pc = 0;
  uint32_t size = 0;
  /// The block the instruction issues to.
  ExecutionBlock block = ExecutionBlock::SALU;
  /// VALU transcendental, issued at a reduced rate.
  bool transcendental = false;
  /// No younger instruction of the wave issues until this one has issued:
  /// control flow, waits, barriers, sleeps and the end of the program.
  bool serializing = false;
  /// Fetch stops after this instruction until it issues and the wave's next
  /// PC is known.
  bool redirects_fetch = false;
  /// Ends the program; fetch never resumes after it.
  bool ends_program = false;
  /// A wait-counter wait; the wave then waits on the sequencer's counters.
  bool waitcnt = false;
  /// Registers whose pending writes this instruction must wait for: its
  /// reads (RAW) and its writes (WAW).
  RegisterSet hazards;
  /// Registers this instruction writes.
  RegisterSet writes;
  /// The wait-counter increments a memory instruction makes at issue, one
  /// entry per increment, as the memory pipelines count them.
  MemoryPipeline::WaitCounterTokens counters;

  /// @brief Classify a decoded instruction.
  /// @details The block is Instruction::amdgpu_execution_block(). Register
  /// hazards come from InstDefUse. On gfx1250 a VGPR's bank depends on the
  /// wave's VGPR-MSB state, which a decoded instruction alone does not know;
  /// writes are therefore expanded to every bank, so a dependency is never
  /// missed but unrelated banks can alias. Selector-encoded special registers
  /// (e.g. `s_mov_b32 exec_lo, ...`) are not visible to InstDefUse. Counters
  /// come from the decoder's memory-issue obligations, or from the block if
  /// none are recorded. These are simulator approximations.
  /// @param inst Decoded instruction.
  /// @param pc Byte address the instruction was fetched from.
  /// @returns The timing view, or nullopt for an instruction in no execution block.
  static std::optional<InstructionTiming> classify(const Instruction &inst, uint64_t pc);
};

} // namespace rocjitsu::amdgpu
