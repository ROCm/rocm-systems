// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_timing.h
/// @brief The static view of a decoded instruction a cycle-level sequencer needs.

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace rocjitsu::amdgpu {

/// @brief What a cycle-level sequencer needs to know about one decoded instruction.
/// @details Derived once, when a sequencer decodes the instruction ahead of
/// the CU, and owns that decoded Instruction. Flags the Instruction already
/// carries are read through it; this keeps only what is costly to derive:
/// register hazards and the wait-counter increments. The CU decodes the
/// instruction again when it executes, so this copy is never executed.
struct InstructionTiming {
  /// The decoded instruction. Null only in a record built by hand, which has
  /// no flags: every accessor below requires it.
  std::unique_ptr<const Instruction> inst;
  /// Byte address the instruction was fetched from.
  uint64_t pc = 0;
  /// Registers whose pending writes this instruction must wait for: its
  /// reads (RAW) and its writes (WAW).
  RegisterSet hazards;
  /// Registers this instruction writes.
  RegisterSet writes;
  /// Every wait-counter increment the instruction can make at issue, one entry
  /// per increment: the counter capacity to reserve before it issues. For FLAT
  /// this is an upper bound covering every memory domain the decoder allows;
  /// the counters actually charged are known only once the CU routes the
  /// addresses, and may be fewer (see WavefrontSlot::issued_memory()).
  MemoryPipeline::WaitCounterTokens counters;

  /// @brief The decoded instruction.
  const Instruction &instruction() const { return *inst; }
  /// @brief Encoded size in bytes.
  uint32_t size() const { return static_cast<uint32_t>(inst->size()); }
  /// @brief The block the instruction issues to.
  ExecutionBlock block() const { return inst->amdgpu_execution_block(); }
  /// @brief Whether fetch stops after this instruction until it issues and the
  /// wave's next PC is known.
  bool redirects_fetch() const { return inst->writes_pc(); }
  /// @brief Whether no younger instruction of the wave issues until this one
  /// has: control flow, waits, barriers, sleeps and the end of the program.
  bool serializing() const {
    return inst->writes_pc() || inst->is_waitcnt() || inst->is_barrier() || inst->suspends_wave();
  }

  /// @brief Classify a decoded instruction, taking ownership of it.
  /// @details Register hazards come from InstDefUse at @p wave_size, so a
  /// wave-mask SGPR operand covers one SGPR in Wave32 and a pair in Wave64.
  /// No VGPR-MSB analysis is applied: on gfx1250 a VGPR is compared by its
  /// encoded index, without its bank. Every access to a physical VGPR encodes
  /// the same index, so a dependency is never missed, but the same index in
  /// different banks aliases. Selector-encoded special registers (e.g.
  /// `s_mov_b32 exec_lo, ...`) are not visible to InstDefUse. Counters come
  /// from the decoder's memory-issue obligations, or from the block if none
  /// are recorded; a message that returns a value (`s_sendmsg_rtn_*`) issues
  /// through SALU but counts two KMCNT units, one for the send and one for the
  /// returned value, as MemoryWaitScoreboard::issue_units() does. These are
  /// simulator approximations.
  /// @param inst Decoded instruction.
  /// @param pc Byte address the instruction was fetched from.
  /// @param wave_size Lanes in the wave that executes it, 32 or 64.
  /// @returns The timing view, or nullopt for an instruction in no execution block.
  static std::optional<InstructionTiming> classify(std::unique_ptr<const Instruction> inst,
                                                   uint64_t pc, uint8_t wave_size);
};

} // namespace rocjitsu::amdgpu
