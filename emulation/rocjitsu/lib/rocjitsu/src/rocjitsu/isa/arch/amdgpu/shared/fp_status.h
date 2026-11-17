// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"

namespace rocjitsu::amdgpu::fp_status {

// Floating-point status bits shared by RDNA3.5 TRAPSTS and RDNA4 WAVE_EXCP_FLAG_USER.
inline constexpr uint32_t kInvalid = 1u << 0;
inline constexpr uint32_t kInputDenorm = 1u << 1;
inline constexpr uint32_t kOverflow = 1u << 3;
inline constexpr uint32_t kUnderflow = 1u << 4;
inline constexpr uint32_t kInexact = 1u << 5;

template <typename Fmt> uint32_t input_causes(uint32_t bits, uint32_t denorm) {
  const uint32_t magnitude = bits & Fmt::kMagnitude;
  if (magnitude > Fmt::kInfinity && !(bits & Fmt::kQuiet))
    return kInvalid;
  if ((denorm & 1u) && magnitude && magnitude < Fmt::kMinNormal)
    return kInputDenorm;
  return 0;
}

template <typename Fmt> double value(uint32_t bits) {
  if constexpr (std::is_same_v<Fmt, fp_format::F16>)
    return double(util::f16_to_f32(uint16_t(bits)));
  return double(std::bit_cast<float>(bits));
}

// The finite intermediate carries more precision than either guest format.
// Round-to-odd retains the side of an F16/F32 rounding boundary even when the
// exact sum requires more than the 53 bits available in a double.
template <typename Fmt> uint32_t result_causes(double exact, uint32_t result, uint32_t round) {
  if (exact == value<Fmt>(result))
    return 0;
  const double magnitude = std::fabs(exact);
  const double limit = std::ldexp(1.0, Fmt::kBias + 1);
  const double ulp = std::ldexp(1.0, Fmt::kBias - int(Fmt::kMantissaBits));
  const bool outward = (round == 1 && exact > 0) || (round == 2 && exact < 0);
  const bool exceeds = round == 0 ? magnitude >= limit - ulp / 2
                       : outward  ? magnitude > limit - ulp
                                  : magnitude >= limit;
  if (exceeds)
    return kOverflow | kInexact;
  return kInexact | ((result & Fmt::kMagnitude) < Fmt::kMinNormal ? kUnderflow : 0);
}

// Classify a single fused product plus addend. ADD uses a multiplier of one;
// MUL uses an addend of zero with the sign of the product.
template <typename Fmt>
uint32_t fma(uint32_t a, uint32_t b, uint32_t c, uint32_t result, uint32_t round, uint32_t denorm) {
  const auto policy = input_denormal::Policy::make(denorm);
  a = input_denormal::prepare<Fmt>(a, policy);
  b = input_denormal::prepare<Fmt>(b, policy);
  c = input_denormal::prepare<Fmt>(c, policy);
  uint32_t causes =
      input_causes<Fmt>(a, denorm) | input_causes<Fmt>(b, denorm) | input_causes<Fmt>(c, denorm);
  const uint32_t x = a & Fmt::kMagnitude, y = b & Fmt::kMagnitude, z = c & Fmt::kMagnitude;
  const bool invalid_product = (x == 0 && y == Fmt::kInfinity) || (x == Fmt::kInfinity && y == 0);
  const bool opposite_infinities = (x == Fmt::kInfinity || y == Fmt::kInfinity) &&
                                   x <= Fmt::kInfinity && y <= Fmt::kInfinity &&
                                   z == Fmt::kInfinity && ((a ^ b ^ c) & Fmt::kSign);
  if (invalid_product || opposite_infinities)
    causes |= kInvalid;
  if (x < Fmt::kInfinity && y < Fmt::kInfinity && z < Fmt::kInfinity) {
    fp_mode::ScopedEnvironment nearest(0);
    const double product = value<Fmt>(a) * value<Fmt>(b);
    const double exact =
        fp_mode::detail::round_to_odd(fp_mode::detail::add_exact(product, value<Fmt>(c)));
    causes |= result_causes<Fmt>(exact, result, round);
  }
  return causes;
}
} // namespace rocjitsu::amdgpu::fp_status
