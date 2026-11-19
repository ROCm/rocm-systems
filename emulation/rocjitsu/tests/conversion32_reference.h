// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Independent standard-arithmetic references for decoded FP32/integer conversions.
#pragma once

#include "fp_reference.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::test {

// Convert I32/U32 to F32 (0/1), or F32 to saturated I32/U32 (2/3),
// nearest-ties-up I32 (4), or floor I32 (5), including guest modifiers.
inline uint32_t convert32(int operation, uint32_t raw, FloatingOutput output, bool flush_input,
                          bool absolute, bool negate) {
  if (operation < 2) {
    // Double represents every 32-bit integer exactly. Correct the default-nearest
    // float conversion by one representable step for directed guest rounding.
    const double exact = operation == 0 ? double(std::bit_cast<int32_t>(raw)) : double(raw);
    float rounded = static_cast<float>(exact);
    const bool up = output.rounding == 1 || (output.rounding == 3 && exact < 0);
    const bool down = output.rounding == 2 || (output.rounding == 3 && exact > 0);
    if ((up && double(rounded) < exact) || (down && double(rounded) > exact)) {
      rounded = std::nextafter(rounded, up ? std::numeric_limits<float>::infinity()
                                           : -std::numeric_limits<float>::infinity());
    }
    // Nonzero integer inputs and their OMOD-scaled results cannot underflow.
    return std::bit_cast<uint32_t>(finish_floating(rounded, output));
  }
  raw = (absolute ? raw & 0x7fffffffU : raw) ^ (negate ? 0x80000000U : 0);
  if (operation >= 4 && flush_input && (raw & 0x7fffffffU) < 0x00800000U) {
    raw &= 0x80000000U;
  }
  const double value = std::bit_cast<float>(raw);
  if (std::isnan(value)) {
    return operation < 4 ? 0 : (raw >> 31) ? 0x80000000U : 0x7fffffffU;
  }
  const double low = operation == 3 ? 0 : -0x1p31;
  const double high = operation == 3 ? double(UINT32_MAX) : double(INT32_MAX);
  if (value <= low) {
    return operation == 3 ? 0 : 0x80000000U;
  }
  if (value >= high) {
    return operation == 3 ? UINT32_MAX : uint32_t(INT32_MAX);
  }
  const double integral = operation == 4   ? std::floor(value + 0.5)
                          : operation == 5 ? std::floor(value)
                                           : std::trunc(value);
  return operation == 3 ? uint32_t(integral) : uint32_t(int32_t(integral));
}

} // namespace rocjitsu::test
