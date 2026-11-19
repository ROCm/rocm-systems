// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Predicate encodings and ISA-local mask composition for integer and float comparisons.
#pragma once

#include "common/isa.h"
#include <stdint.h>

namespace rocjitsu::arith {

// ISA predicate order: floating unordered forms complement 15 - Predicate.
enum ComparePredicate : unsigned {
  CompareFalse = 0,
  CompareLess = 1,
  CompareEqual = 2,
  CompareLessEqual = 3,
  CompareGreater = 4,
  CompareNotEqual = 5,
  CompareGreaterEqual = 6,
  CompareOrdered = 7,
  CompareUnordered = 8,
  CompareNotGreaterEqual = 9,
  CompareNotLessGreater = 10,
  CompareNotGreater = 11,
  CompareNotLessEqual = 12,
  CompareNotEqualUnordered = 13,
  CompareNotLess = 14,
  CompareTrue = 15,
};

} // namespace rocjitsu::arith

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Combine less/equal masks into the requested integer predicate mask.
template <unsigned Predicate> uint32_t integer_compare_result(uint32_t less, uint32_t equal) {
  if constexpr (Predicate == CompareLess) {
    return less;
  }
  if constexpr (Predicate == CompareEqual) {
    return equal;
  }
  if constexpr (Predicate == CompareLessEqual) {
    return less | equal;
  }
  if constexpr (Predicate == CompareGreater) {
    return ~(less | equal);
  }
  if constexpr (Predicate == CompareNotEqual) {
    return ~equal;
  }
  return ~less;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
