// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file transcendental_f64.h
/// @brief V_RCP_F64, V_RSQ_F64 and V_SQRT_F64 as evaluated by the TRANS unit.
///
/// The ISA gives the exact functions and bounds the error at 2^29 ULP. The
/// rules below reproduce gfx1201 bit for bit: every captured lane of the VOP1
/// and VOP3 forms (ABS, NEG, CLAMP, OMOD) under every MODE setting probed,
/// exhaustive sweeps of the significand bits each operation reads (at both
/// exponent parities for RSQ and SQRT), and random sources over the whole
/// exponent range. gfx1100 gives the same results.
///
/// The datapath reads the top 29 (RCP, RSQ) or 28 (SQRT) significand bits. It
/// evaluates a piecewise cubic from the coefficient ROM the F32 operations use
/// (util/amdgpu_rcp.h, amdgpu_rsq.h, amdgpu_sqrt.h) in fixed point. RCP returns
/// 34 fraction bits; RSQ and SQRT return 35. The MODE rounding mode does not
/// affect any result. Callers apply OMOD and CLAMP to the returned encoding
/// with the TRANS-unit output policy (transcendental_output_modifier_policy).
///
/// Scalar and SIMD callers use the same implementation on raw encodings; SIMD
/// lanes are evaluated one at a time.

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_sqrt.h"
#include "util/simd.h"

#include <bit>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu::transcendental_f64 {

/// @brief MODE settings of one instruction, fixed before any lane is evaluated.
struct Policy {
  input_denormal::Policy input;
  output_denormal::Policy output;
  /// Whether a signaling NaN source is quieted (see fp_mode::quiets_nan).
  bool quiet_nan = true;

  /// @param denorm_mode MODE.FP_DENORM field for F16/F64.
  static constexpr Policy make(uint32_t denorm_mode, bool quiet_nan) {
    return {input_denormal::Policy::make(denorm_mode), output_denormal::Policy::make(denorm_mode),
            quiet_nan};
  }
};

enum class Kind { RCP, RSQ, SQRT };

namespace detail {

using F64 = fp_format::F64;

constexpr uint64_t kNegativeDefaultNan = F64::kSign | F64::kInfinity | F64::kQuiet;
constexpr uint64_t kFractionMask = F64::kMinNormal - 1;

/// @brief Round x / 2^shift to nearest, ties to even.
constexpr int64_t round_shift(int64_t x, unsigned shift) {
  const int64_t quotient = x >> shift;
  const int64_t remainder = x - (quotient << shift);
  const int64_t half = int64_t{1} << (shift - 1);
  return quotient + (remainder > half || (remainder == half && (quotient & 1)));
}

/// @brief Second- and third-order terms shared by the three operations.
/// @details f is the 24-bit offset into the ROM segment. The square of its top
/// 17 bits keeps 24 bits. The cubic coefficient multiplies the top 20 bits
/// plus one, rounded at cubic_shift; the product rounds one bit higher once it
/// reaches 2^47.
constexpr int64_t higher_order(int64_t quadratic, int64_t cubic, int64_t f,
                               unsigned quadratic_shift, unsigned cubic_shift) {
  const int64_t top = f >> 7;
  const int64_t square = (top * top) >> 10;
  const int64_t inner =
      (quadratic << quadratic_shift) - round_shift(cubic * ((f >> 4) + 1), cubic_shift);
  const int64_t product = inner * square;
  return product >= (int64_t{1} << 47) ? round_shift(product, 24) * 2 : round_shift(product, 23);
}

/// @brief 1/(1 + m29 / 2^29) * 2^35, for m29 in [1, 2^29).
constexpr int64_t rcp_datapath(int64_t m29) {
  const int64_t segment = m29 >> 24;
  const int64_t f = m29 & 0xffffff;
  const auto &c = util::detail::rcp::coefficients[segment];
  const int64_t linear_product = int64_t{c.linear} * f;
  const int64_t linear = linear_product >= (int64_t{1} << 44) ? round_shift(linear_product, 21)
                                                              : (linear_product + 0x17ffff) >> 21;
  // The constant carries a half unit in every segment but the first.
  const int64_t constant = int64_t{c.constant} * 512 + (segment != 0 ? 256 : 0);
  return constant - linear * 64 + higher_order(c.quadratic, c.cubic, f, 8, 12);
}

/// @brief 1/sqrt((1 + m29 / 2^29) * 2^parity) * 2^36, except 2^36 for 1.0.
constexpr int64_t rsq_datapath(unsigned parity, int64_t m29) {
  const int64_t segment = m29 >> 24;
  const int64_t f = m29 & 0xffffff;
  const auto &c = util::detail::rsq::coefficients[parity * 32 + segment];
  // The last segment's linear term does not see offset bits 4 and 5.
  const int64_t linear_input = segment == 31 ? f & ~int64_t{0x30} : f;
  const int64_t linear_product = int64_t{c.linear} * linear_input;
  const int64_t linear =
      (linear_product + (linear_product >= (int64_t{1} << 43) ? 0x17ffff : 0x1bffff)) >> 21;
  const int64_t constant = int64_t{c.constant} * 1024 + (segment != 0 ? 512 : 0);
  return constant - linear * 128 + higher_order(c.quadratic, c.cubic, f, 9, 11);
}

/// @brief sqrt((1 + m28 / 2^28) * 2^parity) * 2^35.
constexpr int64_t sqrt_datapath(unsigned parity, int64_t m28) {
  const int64_t segment = m28 >> 24;
  const int64_t f = m28 & 0xffffff;
  const auto &c = util::detail::sqrt::coefficients[parity * 16 + segment];
  const int64_t linear_product = int64_t{c.linear} * f;
  const int64_t linear = linear_product >= (int64_t{1} << 45) ? round_shift(linear_product, 22)
                                                              : (linear_product + (1 << 20)) >> 22;
  // The constant carries a half unit in every segment but the last.
  const int64_t constant =
      (int64_t{1} << 35) + int64_t{c.constant} * 1024 + (segment != 15 ? 512 : 0);
  return constant + linear * 128 - higher_order(c.quadratic, c.cubic, f, 8, 11);
}

/// @brief Encode (fraction52 + 2^52) * 2^(exponent - 52), which must be exact.
/// @details Exponents below the normal range give subnormals and above it infinity.
constexpr uint64_t encode(int exponent, uint64_t fraction52) {
  if (exponent > 1023)
    return F64::kInfinity;
  if (exponent >= -1022)
    return uint64_t(exponent + 1023) << F64::kMantissaBits | fraction52;
  return (F64::kMinNormal | fraction52) >> (-1022 - exponent);
}

/// @brief Encode the representable value next above 2^exponent.
constexpr uint64_t next_above_power(int exponent) {
  return exponent >= -1022 ? encode(exponent, 1) : encode(exponent, 0) + 1;
}

/// @brief Encode the representable value next below 2^exponent, or infinity
/// when 2^exponent itself overflows.
constexpr uint64_t next_below_power(int exponent) {
  return exponent > 1023 ? F64::kInfinity : encode(exponent, 0) - 1;
}

/// @brief Evaluate one finite, nonzero, positive source after input flushing.
template <Kind K> constexpr uint64_t evaluate_magnitude(uint64_t magnitude) {
  // Normalize: magnitude = (2^52 + fraction) * 2^(exponent - 52).
  int exponent = int(magnitude >> F64::kMantissaBits) - 1023;
  uint64_t fraction = magnitude & kFractionMask;
  if (exponent == -1023) {
    const int shift = std::countl_zero(fraction) - 11;
    fraction = (fraction << shift) & kFractionMask;
    exponent = -1022 - shift;
  }
  if constexpr (K == Kind::RCP) {
    const int64_t m29 = int64_t(fraction >> 23);
    if (fraction == 0)
      return encode(-exponent, 0);
    // A source with every fraction bit set gives the value next above
    // 2^(-e-1). One whose top 29 fraction bits are clear, but not all of its
    // fraction bits, gives the value next below 2^-e.
    if (fraction == kFractionMask)
      return next_above_power(-exponent - 1);
    if (m29 == 0)
      return next_below_power(-exponent);
    const uint64_t y = uint64_t(rcp_datapath(m29));
    return encode(-exponent - 1, (y - (uint64_t{1} << 34)) << 18);
  } else {
    const unsigned parity = unsigned(exponent) & 1;
    const int half = (exponent - int(parity)) / 2;
    if constexpr (K == Kind::RSQ) {
      const int64_t m29 = int64_t(fraction >> 23);
      if (parity == 0 && fraction == 0)
        return encode(-half, 0);
      // As for RCP, a fraction with only bits below the top 29 set gives the
      // value next below 2^-half.
      if (parity == 0 && m29 == 0)
        return next_below_power(-half);
      const uint64_t y = uint64_t(rsq_datapath(parity, m29));
      return encode(-half - 1, (y - (uint64_t{1} << 35)) << 17);
    } else {
      const uint64_t y = uint64_t(sqrt_datapath(parity, int64_t(fraction >> 24)));
      return encode(half, (y - (uint64_t{1} << 35)) << 17);
    }
  }
}

template <Kind K> constexpr uint64_t evaluate_lane(uint64_t source, const Policy &policy) {
  const uint64_t bits = input_denormal::flush_input<F64>(source, policy.input);
  const uint64_t sign = bits & F64::kSign;
  const uint64_t magnitude = bits & F64::kMagnitude;
  if (magnitude > F64::kInfinity)
    return policy.quiet_nan ? bits | F64::kQuiet : bits;
  if constexpr (K == Kind::RCP) {
    if (magnitude == 0)
      return sign | F64::kInfinity;
    if (magnitude == F64::kInfinity)
      return sign;
  } else {
    // Zeros keep their sign: RSQ gives a signed infinity, SQRT the zero.
    if (magnitude == 0)
      return K == Kind::RSQ ? sign | F64::kInfinity : bits;
    if (sign != 0)
      return kNegativeDefaultNan;
    if (magnitude == F64::kInfinity)
      return K == Kind::RSQ ? 0 : F64::kInfinity;
  }
  return output_denormal::flush_output<F64>(sign | evaluate_magnitude<K>(magnitude), policy.output);
}

} // namespace detail

/// @brief The operation, with MODE input and output flushing, on raw encodings.
/// @details floating_operation.h supplies ABS/NEG before it and OMOD/CLAMP after.
template <Kind K> struct Operation {
  Policy policy;

  template <typename V> V operator()(V source) const {
    static_assert(fp_format::is_lane_v<fp_format::F64, V>);
    if constexpr (std::is_same_v<V, uint64_t>)
      return detail::evaluate_lane<K>(source, policy);
    else
      return util::map_native64_scalar(
          source, [this](uint64_t lane) { return detail::evaluate_lane<K>(lane, policy); });
  }
};

/// @brief Evaluate the VOP1 form, or a VOP3 form without modifiers.
template <Kind K, typename V> V evaluate(V source, const Policy &policy) {
  return Operation<K>{policy}(source);
}

} // namespace rocjitsu::amdgpu::transcendental_f64
