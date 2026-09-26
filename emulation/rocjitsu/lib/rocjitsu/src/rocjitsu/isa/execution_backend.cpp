// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/execution_backend.h"

namespace rocjitsu {
namespace {

thread_local const IsaExecutionBackend *active_backend;

} // namespace

Instruction::ExecuteFn current_instruction_execute(size_t instruction_id) noexcept {
  if (active_backend == nullptr || instruction_id >= active_backend->instruction_callback_count)
    return nullptr;
  return active_backend->instruction_callbacks[instruction_id];
}

const void *current_isa_operand_backend() noexcept {
  return active_backend == nullptr ? nullptr : active_backend->operand_backend;
}

ScopedIsaExecutionBackend::ScopedIsaExecutionBackend(const IsaExecutionBackend *backend) noexcept
    : previous_(active_backend) {
  active_backend = backend;
}

ScopedIsaExecutionBackend::~ScopedIsaExecutionBackend() { active_backend = previous_; }

} // namespace rocjitsu
