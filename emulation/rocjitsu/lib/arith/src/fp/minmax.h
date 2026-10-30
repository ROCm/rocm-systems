// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selection operation identities and backend declarations.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Minmax3 {
  Min3Legacy,
  Max3Legacy,
  MinmaxLegacy,
  MaxminLegacy,
  MedianLegacy,
  MedianNum,
  Min3Num,
  Max3Num,
  MinmaxNum,
  MaxminNum,
  Minimum3,
  Maximum3,
  MinimumMaximum,
  MaximumMinimum,
};

// Decode each ternary instruction once for scalar and SIMD implementations.
template <Minmax3 Op> struct MinmaxPolicy {
  static constexpr bool first_maximum = Op == Minmax3::Max3Legacy || Op == Minmax3::MaxminLegacy ||
                                        Op == Minmax3::Max3Num || Op == Minmax3::MaxminNum ||
                                        Op == Minmax3::Maximum3 || Op == Minmax3::MaximumMinimum;
  static constexpr bool second_maximum = Op == Minmax3::Max3Legacy || Op == Minmax3::MinmaxLegacy ||
                                         Op == Minmax3::Max3Num || Op == Minmax3::MinmaxNum ||
                                         Op == Minmax3::Maximum3 || Op == Minmax3::MinimumMaximum;
  static constexpr bool propagate = Op == Minmax3::Minimum3 || Op == Minmax3::Maximum3 ||
                                    Op == Minmax3::MinimumMaximum || Op == Minmax3::MaximumMinimum;
  static constexpr bool median = Op == Minmax3::MedianLegacy || Op == Minmax3::MedianNum;
  static constexpr bool rdna3 = Op == Minmax3::Min3Legacy || Op == Minmax3::Max3Legacy ||
                                Op == Minmax3::MinmaxLegacy || Op == Minmax3::MaxminLegacy ||
                                Op == Minmax3::MedianLegacy;
};

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Select FP32 minima, maxima, or medians. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void minmax3_x86_64_v3(Minmax3 op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c);
#endif

} // namespace rocjitsu::arith
