// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/simd.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#if __has_include(<experimental/simd>)
namespace rocjitsu::amdgpu {

// Exact power-of-two scaling needs no floating-point operation when both the
// input and output are normal. Keep libm for every other nonzero input, so
// denormals, rounding, overflow, NaN payloads, errno and exception flags retain
// the scalar path's behavior under the current host floating-point controls.
inline util::native<float> mfma_scale_e8m0_simd(util::native<float> sums, uint8_t scale_a,
                                                const uint8_t *scale_b) {
  using U = util::native<uint32_t>;
  using F = util::native<float>;
  static_assert(U::size() == F::size());
  constexpr uint32_t QNAN = std::bit_cast<uint32_t>(std::numeric_limits<float>::quiet_NaN());
  if (scale_a == 0xffu)
    return F(std::bit_cast<float>(QNAN));

  const U b([&](auto lane) { return uint32_t(scale_b[lane]); });
  const U bits = std::bit_cast<U>(sums);
  const U exponent = (bits >> 23) & U(0xffu);
  const U scaled_exponent = exponent + U(scale_a) + b - U(254u);
  const auto normal = (exponent > U(0u)) && (exponent < U(255u)) && (scaled_exponent > U(0u)) &&
                      (scaled_exponent < U(255u));
  const auto scale_nan = b == U(255u);
  // Test exponent and mantissa separately: Clang can fold an absolute-bit
  // zero test into an FP comparison, which inspects sNaNs and obeys host DAZ.
  const auto zero = (exponent | (bits & U(0x7fffffu))) == U(0u);
  U result_bits = bits;
  util::stdx::where(normal, result_bits) = (bits & U(0x807fffffu)) | (scaled_exponent << 23);
  util::stdx::where(scale_nan, result_bits) = U(QNAN);
  F result = std::bit_cast<F>(result_bits);
  const auto fallback = !normal && !zero && !scale_nan;
  if (util::stdx::any_of(fallback)) {
    for (uint32_t lane = 0; lane < F::size(); ++lane)
      if (fallback[lane])
        result[lane] = std::ldexp(float(sums[lane]), int(scale_a) + int(scale_b[lane]) - 254);
  }
  return result;
}

} // namespace rocjitsu::amdgpu
#endif
