// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Translation of packed halfword selectors and sign bits into per-result ALU modes.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/isa.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Return ordinary ALU source and clamp flags for one packed result half.
inline uint32_t packed_half_instruction_flags(uint32_t mode, bool high) {
  // Three selector bits name A/B/C. The high-result flags invert the default
  // high-half choice, whereas low-result flags enable high-half selection.
  uint32_t selected = high ? (~(mode >> 10) & 7) : ((mode >> 7) & 7);
  uint32_t negation = (mode >> (high ? 3 : 0)) & 7;
  return negation | (selected << 9) | ((mode & RJ_ARITH_PK_CLAMP) ? RJ_ARITH_ALU_CLAMP : 0);
}

// Return whether a selected active FP16 operand is NaN. Read no inactive lanes.
template <typename Mask, int Count>
inline bool packed_f16_has_nan(Mask exec_mask, uint32_t mode,
                               const uint32_t *const *const (&inputs)[Count]) {
  for (int lane = 0; lane < int(sizeof(Mask) * 8); ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    for (int half = 0; half < 2; ++half) {
      const uint32_t selected = packed_half_instruction_flags(mode, half) >> 9;
      for (int operand = 0; operand < Count; ++operand) {
        const int shift = ((selected >> operand) & 1) * 16;
        if (((inputs[operand][0][lane] >> shift) & 0x7fffU) > 0x7c00U) {
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
