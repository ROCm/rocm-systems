// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_backend.h
/// @brief Immutable per-ISA execution dispatch selected while decoding.

#pragma once

#include "rocjitsu/isa/instruction.h"

#include <cstddef>

namespace rocjitsu {

/// @brief Execution operations contributed by one statically linked ISA.
///
/// Split model/execution ISAs use the dense callback table to select a direct
/// ``Instruction::execute`` function once, while decoding. The optional
/// operand backend is ISA-defined and is deliberately opaque to the registry.
struct IsaExecutionBackend {
  const Instruction::ExecuteFn *instruction_callbacks = nullptr;
  size_t instruction_callback_count = 0;
  const void *operand_backend = nullptr;
};

/// @brief Return the callback at @p instruction_id in the active decode scope.
/// @returns A direct callback, or null for model-only decoding/out-of-range IDs.
Instruction::ExecuteFn current_instruction_execute(size_t instruction_id) noexcept;

/// @brief Return the ISA-defined operand backend active during decoding.
const void *current_isa_operand_backend() noexcept;

/// @brief Temporarily select one ISA backend for generated constructors.
class ScopedIsaExecutionBackend final {
public:
  explicit ScopedIsaExecutionBackend(const IsaExecutionBackend *backend) noexcept;
  ~ScopedIsaExecutionBackend();

  ScopedIsaExecutionBackend(const ScopedIsaExecutionBackend &) = delete;
  ScopedIsaExecutionBackend &operator=(const ScopedIsaExecutionBackend &) = delete;

private:
  const IsaExecutionBackend *previous_;
};

} // namespace rocjitsu
