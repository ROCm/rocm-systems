// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/risc_v/hart.h"
#include "rocjitsu/isa/arch/risc_v/decoder.h"
#include "rocjitsu/isa/arch/risc_v/isa.h"
#include "rocjitsu/isa/instruction.h"

namespace rocjitsu {
namespace risc_v {

Hart::Hart(std::string name, const simdojo::ClockDomain &domain)
    : simdojo::Clocked<simdojo::Component>(std::move(name), domain) {}

bool Hart::advance(simdojo::Tick /*now*/) {
  if (state_.halted)
    return false;

  uint32_t word = memory_.fetch32(state_.pc);
  auto inst = Decoder::decode(word);
  if (!inst) {
    state_.halted = true;
    return false;
  }

  state_.next_pc = state_.pc + static_cast<uint64_t>(inst->size());

  set_current_memory(&memory_);

  inst->execute(*inst, &state_);

  state_.pc = state_.next_pc;
  return !state_.halted;
}

void Hart::shutdown() {}

} // namespace risc_v
} // namespace rocjitsu
