// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wave-width mask types shared by dispatch and ISA-specific backend declarations.
#pragma once

#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith {

// EXEC has one bit per lane in the instruction wave.
template <int Lanes>
  requires(Lanes == 32 || Lanes == 64)
using ExecMask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;

// Test the sparse-mask cutoff without requiring POPCNT in the baseline ISA.
static inline bool has_more_than_four_lanes(uint64_t mask) {
  for (int i = 0; i < 4; ++i) {
    mask &= mask - 1;
  }
  return mask != 0;
}

} // namespace rocjitsu::arith
