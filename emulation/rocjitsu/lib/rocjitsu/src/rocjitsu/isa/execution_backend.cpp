// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/execution_backend.h"

namespace rocjitsu {

constinit thread_local const IsaExecutionBackend *detail::active_isa_execution_backend = nullptr;

Instruction::ExecuteFn current_instruction_execute(size_t instruction_id) noexcept {
  const IsaExecutionBackend *backend = detail::active_isa_execution_backend;
  if (backend == nullptr || instruction_id >= backend->instruction_callback_count)
    return nullptr;
  return backend->instruction_callbacks[instruction_id];
}

const void *current_isa_operand_backend() noexcept {
  return detail::current_isa_operand_backend_inline();
}

} // namespace rocjitsu
