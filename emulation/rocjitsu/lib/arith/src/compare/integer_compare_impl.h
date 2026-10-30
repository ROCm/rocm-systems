// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local integer compare predicate helpers.
#pragma once

#include "common/isa.h"
#include "compare/integer_compare.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Combine less/equal masks into the requested integer predicate mask.
template <unsigned Predicate> uint32_t integer_compare_result(uint32_t less, uint32_t equal) {
  if constexpr (Predicate == 1) {
    return less;
  }
  if constexpr (Predicate == 2) {
    return equal;
  }
  if constexpr (Predicate == 3) {
    return less | equal;
  }
  if constexpr (Predicate == 4) {
    return ~(less | equal);
  }
  if constexpr (Predicate == 5) {
    return ~equal;
  }
  return ~less;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
