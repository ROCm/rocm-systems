// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Ordered/unordered floating-point predicates and optional FP exception reporting.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_FLOAT_COMPARE_H_
#define RJ_ARITH_FLOAT_COMPARE_H_

#include "compare/integer_compare.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// GFX1201 comparisons: nonfinite operands suppress denormal reporting for the pair.
// Signaling NaNs raise invalid; CLAMP makes quiet NaNs signal too. Input flushing suppresses
// input-denormal reporting. Sign modifiers do not affect this classification.
template <int Bits>
uint32_t float_compare_exceptions(uint64_t a, uint64_t b, bool flush, bool signaling = false) {
  constexpr uint64_t magnitude = (1ULL << (Bits - 1)) - 1;
  constexpr uint64_t infinity = Bits == 16   ? 0x7c00
                                : Bits == 32 ? 0x7f800000
                                             : 0x7ff0000000000000ULL;
  constexpr uint64_t quiet = Bits == 16 ? 0x200 : Bits == 32 ? 0x400000 : 0x8000000000000ULL;
  a &= magnitude;
  b &= magnitude;
  if (a >= infinity || b >= infinity) {
    return ((a > infinity && (signaling || !(a & quiet))) ||
            (b > infinity && (signaling || !(b & quiet))))
               ? RJ_ARITH_EXCEPTION_INVALID
               : 0;
  }
  return !flush && ((a && !(a & infinity)) || (b && !(b & infinity)))
             ? RJ_ARITH_EXCEPTION_INPUT_DENORM
             : 0;
}

// Combine less/equal/ordered masks into the requested floating-point predicate mask.
template <unsigned Predicate>
uint32_t float_compare_result(uint32_t less, uint32_t equal, uint32_t ordered) {
  // Predicates 8..15 complement the corresponding ordered predicate; NaNs then satisfy
  // unordered forms.
  constexpr unsigned base = Predicate >= 8 ? 15 - Predicate : Predicate;
  uint32_t holds;
  if constexpr (base == 7) {
    holds = ordered;
  } else {
    holds = integer_compare_result<base>(less, equal) & ordered;
  }
  if constexpr (Predicate >= 8) {
    return ~holds;
  }
  return holds;
}

// Evaluate ordered/unordered FP predicates.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate>
uint32_t float_compare_x86_64_v3(uint32_t mode, bool flush, const uint32_t *const *a,
                                 const uint32_t *const *b);

// Evaluate ordered/unordered FP predicates.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate>
uint32_t float_compare_x86_64_v4(uint32_t mode, bool flush, const uint32_t *const *a,
                                 const uint32_t *const *b);

} // namespace rocjitsu::arith

#endif
