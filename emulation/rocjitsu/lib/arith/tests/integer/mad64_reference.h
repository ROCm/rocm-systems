// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide integer multiply-add and carry results.

#pragma once

#include "common/integer.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

struct Mad64Result {
  uint64_t value;
  bool carry;
};

// Exact 65-bit product-plus-accumulator reference, expressed as low-word
// arithmetic and the single extended bit.
// Signed operands extend with ones, so bit 64 is the XOR of the two extension
// bits and the unsigned low-word carry. It is not a signed-overflow indicator.
inline Mad64Result mad64_reference(bool is_signed, uint32_t a, uint32_t b, uint64_t c, bool clamp) {
  uint64_t product =
      is_signed ? uint64_t(extend_bits(a, 32, true) * extend_bits(b, 32, true)) : uint64_t(a) * b;
  uint64_t sum = product + c;
  bool carry = sum < product;
  if (is_signed) {
    carry ^= bool(product >> 63) ^ bool(c >> 63);
  }
  bool overflow = is_signed ? carry != bool(sum >> 63) : carry;
  if (clamp && overflow) {
    sum = is_signed ? carry ? 1ULL << 63 : UINT64_MAX >> 1 : UINT64_MAX;
  }
  return {sum, carry};
}

} // namespace rocjitsu::arith::test
