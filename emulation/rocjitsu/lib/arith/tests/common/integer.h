// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deterministic full-word integer inputs and explicit sign extension.

#pragma once

#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Interpret a width-bit unsigned field as signed two's complement if requested.
inline int64_t extend_bits(uint32_t bits, int width, bool is_signed) {
  return int64_t(bits) - (is_signed && (bits & (1U << (width - 1))) ? (1LL << width) : 0);
}

} // namespace rocjitsu::arith::test
