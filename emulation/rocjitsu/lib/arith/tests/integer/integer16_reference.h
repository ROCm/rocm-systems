// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 16-bit integer arithmetic, shifts, and selection.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "integer/packed_integer_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent scalar reference inherited from tests/integer/integer16_reference.h.
// Computed references are exact for these integer ISA operations.
inline uint32_t integer16_reference(HalfBinary operation, uint32_t a, uint32_t b, uint32_t d,
                                    uint32_t mode) {
  uint32_t selected_a = mode & RJ_ARITH_ALU_HIGH_A ? a >> 16 : a & 65535;
  uint32_t selected_b = mode & RJ_ARITH_ALU_HIGH_B ? b >> 16 : b & 65535;
  uint32_t value = packed_integer_reference(operation, selected_a, selected_b,
                                            mode & RJ_ARITH_ALU_CLAMP ? RJ_ARITH_PK_CLAMP : 0) &
                   65535;
  return mode & RJ_ARITH_ALU_HIGH_D ? (d & 65535) | (value << 16) : (d & 0xffff0000u) | value;
}

} // namespace rocjitsu::arith::test
