// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file thread_context.h
/// @brief Base class for per-thread execution context.

#ifndef ROCJITSU_VM_THREAD_CONTEXT_H_
#define ROCJITSU_VM_THREAD_CONTEXT_H_

#include <cstdint>

namespace rocjitsu {

/// @brief Base class for per-thread execution context.
///
/// Passed by reference to Instruction::execute(). ISA-specific code
/// static_casts to the derived type: RISC-V → HartState, GPU → Wavefront.
///
/// The program counter lives here because every execution context -
/// regardless of ISA - tracks the current instruction address.
class ThreadContext {
public:
  virtual ~ThreadContext() = default;

  uint64_t pc = 0; ///< Program counter (byte address).
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_THREAD_CONTEXT_H_
