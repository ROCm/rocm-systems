// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/simd.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#if __has_include(<experimental/simd>)
namespace rocjitsu::amdgpu {

// Independent FP4 output rows share each B load while retaining each output's
// original K order. Decoded FP4 values and their block sums are finite; other
// formats keep their existing FMA loop to preserve competing NaN payloads.
// Expose the row count at compile time so accumulators stay in SIMD registers
// rather than a dynamically indexed array in the K loop.
template <uint32_t Rows>
[[gnu::always_inline]] inline std::array<util::native<float>, Rows>
mfma_fp4_block_product(const float *a, uint32_t a_stride, const float *b, uint32_t b_stride,
                       uint32_t count) {
  std::array<util::native<float>, Rows> sums;
  for (auto &sum : sums)
    sum = util::native<float>(0.0f);
  for (uint32_t k = 0; k < count; ++k) {
    util::native<float> bv;
    bv.copy_from(b + k * b_stride, util::stdx::vector_aligned);
    [&]<std::size_t... Row>(std::index_sequence<Row...>) __attribute__((always_inline)) {
      ((sums[Row] = util::native_fma(util::native<float>(a[Row * a_stride + k]), bv, sums[Row])),
       ...);
    }
    (std::make_index_sequence<Rows>{});
  }
  return sums;
}

} // namespace rocjitsu::amdgpu
#endif
