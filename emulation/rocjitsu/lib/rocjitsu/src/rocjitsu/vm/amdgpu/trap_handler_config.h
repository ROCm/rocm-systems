// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Trap configuration shared by MES and simulated-KFD frontends.
/// @details Both frontends resolve this state for compute-unit trap handling.
struct TrapHandlerConfig {
  uint64_t tba = 0;           ///< GPU virtual address of the trap handler.
  uint64_t tma = 0;           ///< GPU virtual address of the trap memory area.
  bool debug_enabled = false; ///< Process shader-debugger enable state.
};

} // namespace rocjitsu::amdgpu
