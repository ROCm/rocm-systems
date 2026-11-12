// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file Integer LOG reduction and staged approximation from RDNA3/4 captures.

#include <bit>
#include <cstdint>

namespace util::detail::log {
// Coefficients and intermediate rounding reproduce the captured instruction,
// including cancellation near one. No host floating arithmetic is required.
struct Coefficient {
  uint32_t constant, linear, quadratic, cubic;
};
// Q30, Q28, Q34, Q37. Entry32 refines the upper half of mantissa interval1.
inline constexpr Coefficient coefficients[] = {
    {15u, 12102179u, 12095174u, 1925858u},       {47667825u, 11735468u, 11378345u, 1799053u},
    {93912517u, 11390291u, 10714741u, 1610016u}, {138816594u, 11064855u, 10111507u, 1477533u},
    {182455591u, 10757500u, 9557803u, 1359325u}, {224898847u, 10466758u, 9048378u, 1253542u},
    {266210154u, 10191316u, 8578611u, 1158417u}, {306448312u, 9930003u, 8144494u, 1072666u},
    {345667679u, 9681753u, 7742500u, 995087u},   {383918558u, 9445614u, 7369567u, 924970u},
    {421247645u, 9220718u, 7022931u, 861214u},   {457698312u, 9006284u, 6700175u, 803096u},
    {493310964u, 8801596u, 6399165u, 750066u},   {528123258u, 8606005u, 6118007u, 701755u},
    {562170399u, 8418918u, 5854955u, 657310u},   {595485263u, 8239793u, 5608532u, 616715u},
    {628098719u, 8068131u, 5377341u, 579376u},   {660039693u, 7903475u, 5160150u, 544955u},
    {691335339u, 7745406u, 4955875u, 513361u},   {722011229u, 7593536u, 4763477u, 484038u},
    {752091435u, 7447506u, 4582059u, 456830u},   {781598650u, 7306987u, 4410807u, 431594u},
    {810554297u, 7171673u, 4248984u, 408222u},   {838978616u, 7041279u, 4095937u, 386761u},
    {866890752u, 6915542u, 3950988u, 366593u},   {894308853u, 6794217u, 3813578u, 347636u},
    {921250083u, 6677075u, 3683220u, 330396u},   {947730761u, 6563905u, 3559471u, 313859u},
    {973766367u, 6454506u, 3441820u, 298469u},   {999371610u, 6348694u, 3329908u, 284397u},
    {1024560488u, 6246298u, 3223482u, 270890u},  {1049346329u, 6147160u, 3121894u, 258238u},
    {47667912u, 11735350u, 11363743u, 1719989u},
};

inline uint64_t round_even(uint64_t x, unsigned shift) {
  const uint64_t whole = x >> shift, rest = x & ((uint64_t{1} << shift) - 1),
                 half = uint64_t{1} << (shift - 1);
  return whole + (rest > half || (rest == half && (whole & 1)));
}

// Below one, k measures the distance in units of 2^-24. Normalize the
// linear and squared products separately in four-bit groups. The inner
// coefficient uses the ones-complement distance; its result is rounded to
// Q24. The second coefficient row starts at distance 1/128 from one.
// Keep Q50 guard bits for the smallest inputs, then round magnitude upward.
inline uint32_t near_one_negative(uint32_t bits) {
  const uint32_t k = 0x3f800000u - bits;
  const bool upper = k >= 0x20000u;
  const uint32_t c1 = upper ? 12102221u : 12102203u;
  const uint32_t c2 = upper ? 12097323u : 12101979u;
  const uint32_t c3 = upper ? 2088428u : 2035248u;
  const unsigned width = std::bit_width(k);
  const unsigned normalization = width < 18 ? 4 * ((18 - width) / 4) : 0;
  const unsigned linear_shift = normalization < 14 ? 14 - normalization : 0;
  const unsigned square_shift = normalization < 12 ? 12 - normalization : 0;
  const uint64_t linear = ((uint64_t{2} * c1 * k) >> linear_shift) << linear_shift;
  const uint64_t square = ((uint64_t{k} * k) >> square_shift) << square_shift;
  const uint64_t inner = round_even((uint64_t{c2} << 22) + uint64_t{c3} * (k - 1), 22);
  const unsigned product_shift = 38 - normalization;
  const uint64_t quadratic =
      (inner * square + ((uint64_t{1} << product_shift) - 1)) >> product_shift;
  const uint64_t fixed =
      (linear << 2) + (quadratic << (16 - normalization)) - (upper ? (uint64_t{100} << 16) : 0);
  unsigned leading = std::bit_width(fixed) - 1;
  const unsigned shift = leading - 23;
  uint64_t mantissa = (fixed + ((uint64_t{1} << shift) - 1)) >> shift;
  if (mantissa == (uint64_t{1} << 24)) {
    mantissa >>= 1;
    ++leading;
  }
  return 0x80000000u | ((leading - 50 + 127) << 23) | (static_cast<uint32_t>(mantissa) & 0x7fffffu);
}
// Above one, i measures the distance in units of 2^-23. Constants use Q35;
// the zero-origin row gains precision through four-bit normalization. The
// rounded quadratic product retains at most 24 significant bits. Packing
// advances the truncated positive significand, including exact grid hits.
inline uint32_t near_one_positive(uint32_t bits) {
  const uint32_t i = bits - 0x3f800000u;
  const unsigned row = i < 0x10000u ? 0 : i < 0x20000u ? 1 : 2;
  constexpr uint32_t constants[] = {0u, 192u, 2856u};
  constexpr uint32_t linears[] = {12102203u, 12102186u, 12102072u};
  constexpr uint32_t quadratics[] = {12102094u, 12097614u, 12084310u};
  constexpr uint32_t cubics[] = {1999052u, 1948887u, 1883467u};
  const unsigned width = std::bit_width(i);
  const unsigned normalization = row ? 0 : 4 * ((18 - width) / 4);
  const uint64_t normalized = uint64_t{i} << normalization;
  const uint64_t linear = ((uint64_t{linears[row]} * normalized) >> 13) << 2;
  const uint64_t square = (normalized * i) >> 12;
  const uint64_t inner =
      round_even((uint64_t{quadratics[row]} << 22) - uint64_t{cubics[row]} * (2 * i + 1), 22);
  const uint64_t product = inner * square;
  const uint64_t quadratic =
      product >= (uint64_t{1} << 47) ? round_even(product, 24) * 2 : round_even(product, 23);
  const uint64_t fixed = constants[row] + linear - quadratic;
  unsigned leading = std::bit_width(fixed) - 1;
  const unsigned shift = leading - 23;
  uint64_t mantissa = (fixed >> shift) + 1;
  if (mantissa == (uint64_t{1} << 24)) {
    mantissa >>= 1;
    ++leading;
  }
  return ((leading - 35 - normalization + 127) << 23) |
         (static_cast<uint32_t>(mantissa) & 0x7fffffu);
}
// The ordinary path retains the extracted exponent until the final signed
// accumulation. Its quadratic product is rounded to Q35, capped at 24 bits;
// the final conversion rounds downward, including negative results.
inline uint32_t ordinary(uint32_t bits) {
  const uint32_t fraction = bits & 0x3ffffu;
  unsigned index = (bits >> 18) & 31;
  if (index == 1 && fraction >= 0x20000u)
    index = 32;
  const auto c = coefficients[index];
  const uint64_t linear = (uint64_t{c.linear} * fraction) >> 16;
  const uint64_t inner =
      round_even((uint64_t{c.quadratic} << 22) - uint64_t{c.cubic} * (2 * fraction + 1), 22);
  const uint64_t square = (uint64_t{fraction} * fraction) >> 12, product = inner * square;
  const uint64_t quadratic =
      product >= (uint64_t{1} << 47) ? round_even(product, 24) * 2 : round_even(product, 23);
  const int exponent = static_cast<int>(bits >> 23) - 127;
  const int64_t fixed = int64_t{exponent} * (int64_t{1} << 35) +
                        static_cast<int64_t>(((uint64_t{c.constant} + linear) << 5) - quadratic);
  const bool negative = fixed < 0;
  const uint64_t magnitude =
      negative ? static_cast<uint64_t>(-fixed) : static_cast<uint64_t>(fixed);
  unsigned leading = 63 - std::countl_zero(magnitude), shift = leading - 23;
  uint64_t mantissa = (magnitude + (negative ? ((uint64_t{1} << shift) - 1) : 0)) >> shift;
  if (mantissa == (1u << 24)) {
    mantissa >>= 1;
    ++leading;
  }
  return (negative ? 0x80000000u : 0) | ((leading - 35 + 127) << 23) |
         (static_cast<uint32_t>(mantissa) & 0x7fffffu);
}
inline uint32_t evaluate(uint32_t bits, bool quiet_snan = true) {
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return bits | (quiet_snan ? 0x00400000u : 0u);
  // LOG flushes either sign of subnormal input independently of MODE.
  if (magnitude < 0x00800000u)
    return 0xff800000u;
  if (bits & 0x80000000u)
    return 0xffc00000u;
  if (magnitude == 0x7f800000u)
    return 0x7f800000u;
  if (bits == 0x3f800000u)
    return 0;
  if (bits >= 0x3f7c0000u && bits < 0x3f800000u)
    return near_one_negative(bits);
  if (bits > 0x3f800000u && bits < 0x3f840000u)
    return near_one_positive(bits);
  return ordinary(bits);
}
} // namespace util::detail::log

namespace util {
/// @brief Base-2 logarithm matching the captured RDNA3/4 instruction mapping.
inline float amdgpu_log_f32(float value, bool quiet_snan = true) {
  return std::bit_cast<float>(detail::log::evaluate(std::bit_cast<uint32_t>(value), quiet_snan));
}
} // namespace util
