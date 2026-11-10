// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Low/high-half integer multiplication.

#pragma once

#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
enum class Multiply { LoU32, HiU32, HiI32, I24, HiI24, U24, HiU24 };

// Compute low/high-half integer multiplication with optional saturation.
inline uint32_t integer_mul_reference(Multiply operation, uint32_t a, uint32_t b, bool clamp) {
  int op = int(operation);
  const bool signed_op = op == 2 || op == 3 || op == 4;
  const bool high = op == 1 || op == 2 || op == 4 || op == 6;
  const int width = op < 3 ? 32 : 24;
  uint64_t modulus = 1ULL << width;
  a %= modulus;
  b %= modulus;
  bool negative_a = signed_op && a >= modulus / 2;
  bool negative_b = signed_op && b >= modulus / 2;
  uint64_t magnitude_a = negative_a ? modulus - a : a;
  uint64_t magnitude_b = negative_b ? modulus - b : b;
  // Multiply magnitudes; restore the sign after applying the range limit.
  uint64_t product = magnitude_a * magnitude_b;
  bool negative = negative_a != negative_b;
  if (clamp) {
    product = std::min<uint64_t>(product, signed_op ? (negative ? 0x80000000ULL : 0x7fffffffULL)
                                                    : uint64_t(UINT32_MAX));
  }
  if (negative) {
    product = 0ULL - product;
  }
  return uint32_t(high ? product >> 32 : product);
}

} // namespace rocjitsu::arith::test
