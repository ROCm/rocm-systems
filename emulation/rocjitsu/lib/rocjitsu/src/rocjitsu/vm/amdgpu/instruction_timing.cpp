// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/instruction_timing.h"

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"

#include <string_view>

namespace rocjitsu::amdgpu {
namespace {

void add_counter(MemoryPipeline::WaitCounterTokens &tokens, WaitCounterType type) {
  if (tokens.size < tokens.types.size())
    tokens.types[tokens.size++] = type;
}

} // namespace

std::optional<InstructionTiming> InstructionTiming::classify(const Instruction &inst, uint64_t pc) {
  InstructionTiming timing;
  timing.pc = pc;
  timing.size = static_cast<uint32_t>(inst.size());
  timing.block = inst.amdgpu_execution_block();
  if (timing.block == ExecutionBlock::UNKNOWN)
    return std::nullopt;
  timing.redirects_fetch = inst.writes_pc();
  timing.transcendental = timing.block == ExecutionBlock::VALU && inst.is_transcendental();
  timing.waitcnt = inst.is_waitcnt();
  timing.ends_program = inst.is_program_terminator();
  timing.serializing =
      timing.redirects_fetch || timing.waitcnt || inst.is_barrier() || inst.suspends_wave();

  const InstDefUse def_use(inst, nullptr, UnknownVgprDefPolicy::ExpandAll);
  timing.writes = def_use.defs;
  timing.hazards = def_use.uses | def_use.defs;

  const bool memory = timing.block == ExecutionBlock::SMEM ||
                      timing.block == ExecutionBlock::VMEM || timing.block == ExecutionBlock::LDS;
  if (memory) {
    if (const auto *issue = inst.amdgpu_memory_issue_info()) {
      for (const auto obligation : issue->counter_obligations())
        for (uint8_t token = 0; token < obligation.counter_increment(); ++token)
          add_counter(timing.counters, obligation.wait_counter_type());
    } else if (timing.block == ExecutionBlock::SMEM) {
      add_counter(timing.counters, WaitCounterType::KMCNT);
    } else if (timing.block == ExecutionBlock::LDS) {
      add_counter(timing.counters, WaitCounterType::DSCNT);
    } else {
      add_counter(timing.counters, inst.mnemonic().find("store") != std::string_view::npos
                                       ? WaitCounterType::STORECNT
                                       : WaitCounterType::LOADCNT);
    }
  }
  return timing;
}

} // namespace rocjitsu::amdgpu
