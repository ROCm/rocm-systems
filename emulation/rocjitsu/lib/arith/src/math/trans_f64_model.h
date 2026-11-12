// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Empirical FP64 reciprocal, reciprocal-square-root and square-root model.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

// gfx1201 evaluates V_RCP_F64, V_RSQ_F64 and V_SQRT_F64 in the TRANS unit: a
// fixed-point piecewise cubic over the FP32 coefficient ROM, not a correctly
// rounded result. The ISA only bounds the error. The rules below reproduce every
// captured gfx1201 lane of the VOP1 and VOP3 forms under every probed MODE,
// exhaustive sweeps of the significand bits each operation reads (both exponent
// parities for RSQ and SQRT) and random sources over the whole exponent range.
// gfx1100 gives the same results.
//
// Level 1, trans_f64_bits: IEEE special sources, then the magnitude.
// Level 2, trans_f64_magnitude: normalize, apply the power-of-two neighbor
//          cases, run the datapath and encode its result.
// Level 3, the per-operation datapaths and their shared higher-order term.
//
// RCP and RSQ read the top 29 significand bits and SQRT the top 28. RCP returns
// 34 fraction bits; RSQ and SQRT return 35. Guest rounding never affects the
// result. Callers own input flushing, ABS/NEG, OMOD, CLAMP and output flushing.

#include "common/bits.h"
#include "fp/fp64.h"
#include "math/rcp_model.h"
#include "math/rsq_model.h"
#include "math/sqrt_model.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trans_f64 {

inline constexpr uint64_t kSign = 0x8000000000000000ULL;
inline constexpr uint64_t kInfinity = 0x7ff0000000000000ULL;
inline constexpr uint64_t kQuiet = 0x0008000000000000ULL;
inline constexpr uint64_t kMinNormal = 0x0010000000000000ULL;
inline constexpr uint64_t kFraction = kMinNormal - 1;

// Round x / 2^shift to nearest, ties to even.
constexpr uint64_t round_shift(uint64_t x, int shift) {
  const uint64_t quotient = x >> shift, remainder = x & ((uint64_t{1} << shift) - 1);
  const uint64_t half = uint64_t{1} << (shift - 1);
  return quotient + (remainder > half || (remainder == half && (quotient & 1)));
}

// Second- and third-order terms shared by the three datapaths. f is the 24-bit
// offset into the ROM segment. The square of its top 17 bits keeps 24 bits. The
// cubic coefficient multiplies the top 20 bits plus one, rounded at cubic_shift;
// the product rounds one bit higher once it reaches 2^47.
constexpr uint64_t higher_order(uint32_t packed, uint64_t f, int quadratic_shift, int cubic_shift) {
  const uint64_t quadratic = packed & 0xffffU, cubic = packed >> 16;
  const uint64_t top = f >> 7;
  const uint64_t square = (top * top) >> 10;
  const uint64_t inner =
      (quadratic << quadratic_shift) - round_shift(cubic * ((f >> 4) + 1), cubic_shift);
  const uint64_t product = inner * square;
  return product >= (uint64_t{1} << 47) ? round_shift(product, 24) * 2 : round_shift(product, 23);
}

// The quadratic term dominates the cubic one in every segment, so the unsigned
// inner difference above never wraps.
template <int N>
constexpr bool inner_positive(const uint32_t (&packed)[N], int quadratic_shift, int cubic_shift) {
  for (int segment = 0; segment < N; ++segment) {
    if (((packed[segment] & 0xffffU) << quadratic_shift) <=
        round_shift(uint64_t(packed[segment] >> 16) << 20, cubic_shift)) {
      return false;
    }
  }
  return true;
}
static_assert(inner_positive(detail::rcp_coefficients[2], 8, 12));
static_assert(inner_positive(detail::rsq::coefficients[2], 9, 11));
static_assert(inner_positive(detail::sqrt::coefficients[2], 8, 11));

// 1 / (1 + m29 / 2^29) * 2^35, for m29 in [1, 2^29).
constexpr uint64_t rcp_datapath(uint64_t m29) {
  const int segment = int(m29 >> 24);
  const uint64_t f = m29 & 0xffffff;
  const uint64_t linear_product = uint64_t{detail::rcp_coefficients[1][segment]} * f;
  const uint64_t linear = linear_product >= (uint64_t{1} << 44) ? round_shift(linear_product, 21)
                                                                : (linear_product + 0x17ffff) >> 21;
  // The constant carries a half unit in every segment but the first.
  const uint64_t constant =
      uint64_t{detail::rcp_coefficients[0][segment]} * 512 + (segment != 0 ? 256 : 0);
  return constant - linear * 64 + higher_order(detail::rcp_coefficients[2][segment], f, 8, 12);
}

// 1 / sqrt((1 + m29 / 2^29) * 2^parity) * 2^36, for m29 in [1, 2^29) when parity is 0.
constexpr uint64_t rsq_datapath(int parity, uint64_t m29) {
  const int segment = int(m29 >> 24), index = parity * 32 + segment;
  const uint64_t f = m29 & 0xffffff;
  // The last segment's linear term does not see offset bits 4 and 5.
  const uint64_t linear_input = segment == 31 ? f & ~uint64_t{0x30} : f;
  const uint64_t linear_product = uint64_t{detail::rsq::coefficients[1][index]} * linear_input;
  const uint64_t linear =
      (linear_product + (linear_product >= (uint64_t{1} << 43) ? 0x17ffff : 0x1bffff)) >> 21;
  const uint64_t constant =
      uint64_t{detail::rsq::coefficients[0][index]} * 1024 + (segment != 0 ? 512 : 0);
  return constant - linear * 128 + higher_order(detail::rsq::coefficients[2][index], f, 9, 11);
}

// sqrt((1 + m28 / 2^28) * 2^parity) * 2^35.
constexpr uint64_t sqrt_datapath(int parity, uint64_t m28) {
  const int segment = int(m28 >> 24), index = parity * 16 + segment;
  const uint64_t f = m28 & 0xffffff;
  const uint64_t linear_product = uint64_t{detail::sqrt::coefficients[1][index]} * f;
  const uint64_t linear = linear_product >= (uint64_t{1} << 45)
                              ? round_shift(linear_product, 22)
                              : (linear_product + (1 << 20)) >> 22;
  // The constant carries a half unit in every segment but the last.
  const uint64_t constant = (uint64_t{1} << 35) +
                            uint64_t{detail::sqrt::coefficients[0][index]} * 1024 +
                            (segment != 15 ? 512 : 0);
  return constant + linear * 128 - higher_order(detail::sqrt::coefficients[2][index], f, 8, 11);
}

// Encode (2^52 + fraction52) * 2^(exponent - 52), which must be exact. Exponents
// below the normal range give subnormals and above it infinity.
constexpr uint64_t encode(int exponent, uint64_t fraction52) {
  if (exponent > 1023) {
    return kInfinity;
  }
  if (exponent >= -1022) {
    return uint64_t(exponent + 1023) << 52 | fraction52;
  }
  return (kMinNormal | fraction52) >> (-1022 - exponent);
}

// Evaluate one finite, nonzero, positive source after input flushing.
template <Fp64 Op> constexpr uint64_t trans_f64_magnitude(uint64_t magnitude) {
  int exponent = int(magnitude >> 52) - 1023;
  uint64_t fraction = magnitude & kFraction;
  if (exponent == -1023) {
    const int shift = countl_zero(fraction) - 11;
    fraction = (fraction << shift) & kFraction;
    exponent = -1022 - shift;
  }
  if constexpr (Op == Fp64::Rcp) {
    const uint64_t m29 = fraction >> 23;
    if (fraction == 0) {
      return encode(-exponent, 0);
    }
    // Every fraction bit set gives the value next above 2^(-e-1). Only bits
    // below the top 29 set gives the value next below 2^-e.
    if (fraction == kFraction) {
      return encode(-exponent - 1, 0) + 1;
    }
    if (m29 == 0) {
      return -exponent > 1023 ? kInfinity : encode(-exponent, 0) - 1;
    }
    return encode(-exponent - 1, (rcp_datapath(m29) - (uint64_t{1} << 34)) << 18);
  } else {
    const int parity = exponent & 1, half = (exponent - parity) / 2;
    if constexpr (Op == Fp64::Rsq) {
      const uint64_t m29 = fraction >> 23;
      // As for RCP, only bits below the top 29 set gives the value next below 2^-half.
      if (parity == 0 && m29 == 0) {
        return encode(-half, 0) - (fraction != 0);
      }
      return encode(-half - 1, (rsq_datapath(parity, m29) - (uint64_t{1} << 35)) << 17);
    } else {
      return encode(half, (sqrt_datapath(parity, fraction >> 24) - (uint64_t{1} << 35)) << 17);
    }
  }
}

// Evaluate RCP, RSQ or SQRT on a source with ABS/NEG and input flushing applied.
// NaNs keep their sign and payload, quieted when quiet_nan is set. A negative
// RSQ or SQRT source gives -NaN; zeros give signed infinity (RCP, RSQ) or keep
// themselves (SQRT).
template <Fp64 Op> constexpr uint64_t trans_f64_bits(uint64_t bits, bool quiet_nan) {
  const uint64_t sign = bits & kSign, magnitude = bits & ~kSign;
  if (magnitude > kInfinity) {
    return quiet_nan ? bits | kQuiet : bits;
  }
  if constexpr (Op == Fp64::Rcp) {
    if (magnitude == 0) {
      return sign | kInfinity;
    }
    if (magnitude == kInfinity) {
      return sign;
    }
  } else {
    if (magnitude == 0) {
      return Op == Fp64::Rsq ? sign | kInfinity : bits;
    }
    if (sign) {
      return kSign | kInfinity | kQuiet;
    }
    if (magnitude == kInfinity) {
      return Op == Fp64::Rsq ? 0 : kInfinity;
    }
  }
  return sign | trans_f64_magnitude<Op>(magnitude);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trans_f64
