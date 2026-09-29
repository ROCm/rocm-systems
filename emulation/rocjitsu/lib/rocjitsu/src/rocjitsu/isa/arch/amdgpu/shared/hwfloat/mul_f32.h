// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu::hwfloat {

/// Exact mathematical facts of one binary32 multiply. These are not TRAPSTS
/// bits; plain_mul_f32_causes() maps them for the qualified targets.
namespace mul_f32_fact {
inline constexpr uint32_t kInvalid = 1u << 0;
inline constexpr uint32_t kOverflow = 1u << 1;
/// Rounding discarded nonzero bits, or the exponent overflowed.
inline constexpr uint32_t kInexact = 1u << 2;
inline constexpr uint32_t kTinyBeforeRounding = 1u << 3;
inline constexpr uint32_t kTinyAfterRounding = 1u << 4;
/// A source was subnormal before input flushing.
inline constexpr uint32_t kInputDenormal = 1u << 5;
inline constexpr uint32_t kInputFlushed = 1u << 6;
inline constexpr uint32_t kOutputFlushed = 1u << 7;
/// The product rounded to 24 significant bits with an unbounded exponent is
/// smaller in magnitude than the smallest normal binary32 value.
inline constexpr uint32_t kTinyAfterSignificandRounding = 1u << 8;
} // namespace mul_f32_fact

/// TRAPSTS.EXCP cause bits produced by plain_mul_f32_causes().
namespace mul_f32_cause {
inline constexpr uint32_t kInvalid = 1u << 0;
inline constexpr uint32_t kInputDenormal = 1u << 1;
inline constexpr uint32_t kOverflow = 1u << 3;
inline constexpr uint32_t kUnderflow = 1u << 4;
inline constexpr uint32_t kInexact = 1u << 5;
} // namespace mul_f32_cause

/// Guest state that selects one binary32 multiply result.
struct MulF32Policy {
  /// MODE.FP_ROUND single-precision field: nearest-even, +inf, -inf, zero.
  uint32_t round = 0;
  /// MODE.FP_DENORM single-precision field: bit 0 keeps subnormal sources,
  /// bit 1 keeps subnormal results.
  uint32_t denorm = 0;
  /// Set the quiet bit of a propagated signaling NaN.
  bool quiet_nan = false;
};

struct MulF32Result {
  uint32_t bits = 0;
  uint32_t facts = 0;
};

namespace detail {

struct MulF32Rounded {
  uint64_t kept = 0;
  bool inexact = false;
};

constexpr MulF32Rounded round_mul_f32_significand(uint64_t value, int shift, bool negative,
                                                  uint32_t round) {
  if (shift <= 0)
    return {value << -shift, false};
  uint64_t kept = 0;
  bool guard = false;
  bool sticky = false;
  if (shift >= 64) {
    sticky = value != 0;
  } else {
    kept = value >> shift;
    guard = ((value >> (shift - 1)) & 1) != 0;
    sticky = shift > 1 && (value & ((uint64_t{1} << (shift - 1)) - 1)) != 0;
  }
  const bool inexact = guard || sticky;
  if ((round == 0 && guard && (sticky || (kept & 1))) || (round == 1 && !negative && inexact) ||
      (round == 2 && negative && inexact))
    ++kept;
  return {kept, inexact};
}

constexpr MulF32Result mul_f32_overflow(uint32_t sign, uint32_t round, uint32_t facts) {
  const bool infinity = round == 0 || (round == 1 && !sign) || (round == 2 && sign);
  return {sign | (infinity ? 0x7f800000u : 0x7f7fffffu),
          facts | mul_f32_fact::kOverflow | mul_f32_fact::kInexact};
}

} // namespace detail

/// Multiply two binary32 values under an explicit guest policy.
/// Uses only integer operations, so no host FP state is read or written.
/// Invalid operations return the negative default NaN; other NaN results
/// propagate the first NaN source.
constexpr MulF32Result multiply_f32(uint32_t a, uint32_t b, MulF32Policy policy) {
  namespace fact = mul_f32_fact;
  uint32_t facts = 0;
  const uint32_t original_a = a & 0x7fffffffu;
  const uint32_t original_b = b & 0x7fffffffu;
  if ((original_a && original_a < 0x00800000u) || (original_b && original_b < 0x00800000u))
    facts |= fact::kInputDenormal;
  if (!(policy.denorm & 1u)) {
    if ((a & 0x7f800000u) == 0)
      a &= 0x80000000u;
    if ((b & 0x7f800000u) == 0)
      b &= 0x80000000u;
    if (facts & fact::kInputDenormal)
      facts |= fact::kInputFlushed;
  }
  const uint32_t am = a & 0x7fffffffu;
  const uint32_t bm = b & 0x7fffffffu;
  const uint32_t sign = (a ^ b) & 0x80000000u;
  const bool negative = sign != 0;
  if ((am == 0 && bm == 0x7f800000u) || (bm == 0 && am == 0x7f800000u))
    return {0xffc00000u, facts | fact::kInvalid};
  const bool a_nan = am > 0x7f800000u;
  const bool b_nan = bm > 0x7f800000u;
  if (a_nan || b_nan) {
    if ((a_nan && !(a & 0x00400000u)) || (b_nan && !(b & 0x00400000u)))
      facts |= fact::kInvalid;
    return {(a_nan ? a : b) | (policy.quiet_nan ? 0x00400000u : 0u), facts};
  }
  if (am == 0x7f800000u || bm == 0x7f800000u)
    return {sign | 0x7f800000u, facts};
  if (am == 0 || bm == 0)
    return {sign, facts};

  const int ae = static_cast<int>(am >> 23);
  const int be = static_cast<int>(bm >> 23);
  const uint32_t as = (am & 0x007fffffu) | (ae ? 0x00800000u : 0u);
  const uint32_t bs = (bm & 0x007fffffu) | (be ? 0x00800000u : 0u);
  const uint64_t product = uint64_t{as} * bs;
  const int exponent = (ae ? ae : 1) + (be ? be : 1) - 300;
  const int highest = 63 - std::countl_zero(product);
  const int top_exponent = highest + exponent;
  if (top_exponent < -126)
    facts |= fact::kTinyBeforeRounding;
  if (top_exponent > 127)
    return detail::mul_f32_overflow(sign, policy.round, facts);
  const int normal_shift = highest - 23;
  int shift = normal_shift;
  if (policy.denorm & 2u) {
    const int subnormal_shift = -149 - exponent;
    if (shift < subnormal_shift)
      shift = subnormal_shift;
  }
  auto rounded = detail::round_mul_f32_significand(product, shift, negative, policy.round);
  if (rounded.inexact)
    facts |= fact::kInexact;
  if (top_exponent < -126) {
    // A rounding carry to 2^24 raises the magnitude to the next power of two.
    const uint64_t kept24 =
        shift == normal_shift
            ? rounded.kept
            : detail::round_mul_f32_significand(product, normal_shift, negative, policy.round).kept;
    if (top_exponent + (kept24 >= (uint64_t{1} << 24) ? 1 : 0) < -126)
      facts |= fact::kTinyAfterSignificandRounding;
  }
  int result_exponent = exponent + shift + 23;
  if (rounded.kept >= (uint64_t{1} << 24)) {
    rounded.kept >>= 1;
    ++result_exponent;
  }
  if (result_exponent > 127)
    return detail::mul_f32_overflow(sign, policy.round, facts);
  if (!(policy.denorm & 2u) && result_exponent < -126)
    return {sign, facts | fact::kTinyAfterRounding | fact::kOutputFlushed};
  if (rounded.kept < (uint64_t{1} << 23))
    return {sign | static_cast<uint32_t>(rounded.kept), facts | fact::kTinyAfterRounding};
  return {sign | (static_cast<uint32_t>(result_exponent + 127) << 23) |
              (static_cast<uint32_t>(rounded.kept) & 0x007fffffu),
          facts};
}

/// TRAPSTS.EXCP causes of V_MUL_F32 measured on gfx1100 and gfx1201 without
/// OMOD, CLAMP, or MODE.DX10_CLAMP. Input denormal is reported only for a
/// preserved source and never with a NaN result. Underflow requires tininess
/// after significand rounding and an inexact or flushed result.
constexpr uint32_t plain_mul_f32_causes(MulF32Result result, MulF32Policy policy) {
  namespace fact = mul_f32_fact;
  namespace cause = mul_f32_cause;
  uint32_t causes = 0;
  if (result.facts & fact::kInvalid)
    causes |= cause::kInvalid;
  const bool nan_result = (result.bits & 0x7fffffffu) > 0x7f800000u;
  if ((result.facts & fact::kInputDenormal) && (policy.denorm & 1u) && !nan_result)
    causes |= cause::kInputDenormal;
  if (result.facts & fact::kOverflow)
    causes |= cause::kOverflow;
  if (result.facts & (fact::kInexact | fact::kOutputFlushed)) {
    causes |= cause::kInexact;
    if (result.facts & fact::kTinyAfterSignificandRounding)
      causes |= cause::kUnderflow;
  }
  return causes;
}

/// Maximum lanes accepted by the wave entry points.
inline constexpr uint32_t kMulF32MaxLanes = 64;

/// Multiply each active lane of @p lhs and @p rhs into @p result and return
/// the union of their plain_mul_f32_causes(). Inactive result lanes are not
/// written. @p lanes is 32 or 64. Arrays hold at least @p lanes values.
/// Selects the AVX-512 backend when the host supports it; results and causes
/// are identical to multiply_f32_wave_integer().
uint32_t multiply_f32_wave(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result,
                           uint64_t active, uint32_t lanes, MulF32Policy policy);

/// Portable multiply_f32_wave() backend using multiply_f32() for every lane.
uint32_t multiply_f32_wave_integer(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result,
                                   uint64_t active, uint32_t lanes, MulF32Policy policy);

/// Whether multiply_f32_wave_avx512() may run on this host.
bool mul_f32_avx512_available();

/// AVX-512F multiply_f32_wave() backend. Lanes whose sources are normal and
/// whose biased exponent sum is 128..380 use one embedded-rounding multiply
/// with exceptions suppressed: their result is normal and only inexactness,
/// derived from the exact 48-bit significand product, can be a fact. Other
/// active lanes use multiply_f32(). Requires mul_f32_avx512_available();
/// otherwise delegates to multiply_f32_wave_integer().
uint32_t multiply_f32_wave_avx512(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result,
                                  uint64_t active, uint32_t lanes, MulF32Policy policy);

} // namespace rocjitsu::amdgpu::hwfloat
