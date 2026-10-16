// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Translation of packed halfword selectors and sign bits into per-result ALU modes.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Return ordinary ALU source and clamp flags for one packed result half.
inline uint32_t packed_half_mode(uint32_t mode, bool high) {
  // Three selector bits name A/B/C. The high-result flags invert the default
  // high-half choice, whereas low-result flags enable high-half selection.
  uint32_t selected = high ? (~(mode >> 10) & 7) : ((mode >> 7) & 7);
  uint32_t negation = (mode >> (high ? 3 : 0)) & 7;
  return negation | (selected << 9) | ((mode & RJ_ARITH_PK_CLAMP) ? RJ_ARITH_ALU_CLAMP : 0);
}

} // namespace rocjitsu::arith
