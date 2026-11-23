// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file matrix_input_snapshot.h
/// @brief Matrix A/B operands copied when an asynchronous MMA was published.

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Published copies of an MMA's factor registers.
/// @details The issuer may overwrite the live A/B registers while a helper still
/// executes the MMA (write-after-read). Accumulator and result registers are
/// never snapshotted: C is read and D written in the live register file.
struct MatrixInputSnapshot {
  struct Operand {
    uint32_t base = 0; ///< Physical VGPR index as resolved by the handler.
    uint32_t regs = 0; ///< Zero when the operand reads live registers.
    const uint32_t *lanes = nullptr; ///< `regs` registers of wave-size lanes each.
  };
  Operand a, b;
};

/// Set by a helper around each instruction it executes; null on every issuer.
inline thread_local const MatrixInputSnapshot *matrix_input_snapshot = nullptr;

} // namespace rocjitsu::amdgpu
