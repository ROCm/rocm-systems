// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Small bit and value operations with ISA-local definitions, avoiding runtime
// standard-library template symbols compiled with conflicting CPU options.
#pragma once

#include "common/isa.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Copy an object's representation without aliasing or numerical conversion.
template <typename To, typename From> constexpr To bit_cast(const From &value) {
  return __builtin_bit_cast(To, value);
}

// Count leading zero bits, including every bit when value is zero.
template <typename T> constexpr int countl_zero(T value) {
  static_assert(sizeof(T) <= sizeof(unsigned long long));
  return value ? __builtin_clzll(static_cast<unsigned long long>(value)) -
                     int(8 * (sizeof(unsigned long long) - sizeof(T)))
               : int(8 * sizeof(T));
}

// Return the number of bits needed to represent an unsigned value.
template <typename T> constexpr int bit_width(T value) {
  return int(8 * sizeof(T)) - countl_zero(value);
}

// Return the larger value, retaining the first on ties.
template <typename T> constexpr T maximum(T a, T b) { return a < b ? b : a; }

// Constrain a value to the inclusive bounds; callers provide low <= high.
template <typename T> constexpr T clamp(T value, T low, T high) {
  return value < low ? low : high < value ? high : value;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
