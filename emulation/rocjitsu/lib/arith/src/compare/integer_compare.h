// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Signed and unsigned integer comparisons producing lane masks.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_INTEGER_COMPARE_H_
#define RJ_ARITH_INTEGER_COMPARE_H_

#include <stdint.h>

namespace rocjitsu::arith {

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

// Compare signed/unsigned integers.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v3(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b);

// Compare signed/unsigned integers.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v4(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b);

} // namespace rocjitsu::arith

#endif
