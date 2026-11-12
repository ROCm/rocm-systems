// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/instruction_timing.h"

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"

#include <cassert>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

/// @brief Append @p count increments of counter @p type to @p tokens.
/// @details WaitCounterTokens holds MAX_WAIT_COUNTER_TOKENS entries, the
/// maximum obligations times the maximum increment of each, so neither a
/// MemoryIssueInfo nor a two-unit message return can overflow it.
/// @param tokens The counter increments collected so far.
/// @param type The counter incremented.
/// @param count How many units of @p type to append.
void add_counter(MemoryPipeline::WaitCounterTokens &tokens, WaitCounterType type,
                 uint8_t count = 1) {
  for (uint8_t i = 0; i < count; ++i) {
    assert(tokens.size < tokens.types.size() && "more counter increments than tokens can hold");
    tokens.types[tokens.size++] = type;
  }
}

} // namespace

std::optional<InstructionTiming>
InstructionTiming::classify(std::unique_ptr<const Instruction> inst, uint64_t pc,
                            uint8_t wave_size) {
  assert(inst != nullptr);
  assert((wave_size == 32 || wave_size == 64) && "a wave runs 32 or 64 lanes");
  const ExecutionBlock block = inst->amdgpu_execution_block();
  if (block == ExecutionBlock::UNKNOWN)
    return std::nullopt;

  InstructionTiming timing;
  timing.pc = pc;

  const InstDefUse def_use(*inst, wave_size);
  timing.writes = def_use.defs;
  timing.hazards = def_use.uses | def_use.defs;

  const bool memory = block == ExecutionBlock::SMEM || block == ExecutionBlock::VMEM ||
                      block == ExecutionBlock::LDS;
  if (memory) {
    if (const auto *issue = inst->amdgpu_memory_issue_info()) {
      for (const auto obligation : issue->counter_obligations())
        add_counter(timing.counters, obligation.wait_counter_type(),
                    obligation.counter_increment());
    } else if (block == ExecutionBlock::SMEM) {
      add_counter(timing.counters, WaitCounterType::KMCNT);
    } else if (block == ExecutionBlock::LDS) {
      add_counter(timing.counters, WaitCounterType::DSCNT);
    } else {
      add_counter(timing.counters, inst->amdgpu_memory_kind() == MemoryKind::Store
                                       ? WaitCounterType::STORECNT
                                       : WaitCounterType::LOADCNT);
    }
  } else if (inst->amdgpu_operation() == Operation::S_SENDMSG_RTN_B32 ||
             inst->amdgpu_operation() == Operation::S_SENDMSG_RTN_B64) {
    // KMCNT also counts against LGKMCNT, the counter legacy waits read.
    add_counter(timing.counters, WaitCounterType::KMCNT, 2);
  }

  timing.inst = std::move(inst);
  return timing;
}

} // namespace rocjitsu::amdgpu
