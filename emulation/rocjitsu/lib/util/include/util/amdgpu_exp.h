// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file Integer EXP reduction and staged approximation from RDNA3/4 captures.

#include <bit>
#include <cstdint>

namespace util::detail::exp {
// Constant and linear coefficients use Q28, quadratic Q35, and cubic Q24.
// The 32 intervals cover the fractional part of x. Evaluation uses only
// integer operations, preserving the caller's complete floating-point state.
struct Coefficient {
  uint32_t constant, linear, quadratic, cubic;
};
inline constexpr Coefficient coefficients[32] = {
    {268435456u, 5814541u, 8060268u, 29u},   {274313427u, 5941862u, 8236765u, 29u},
    {280320109u, 6071972u, 8417127u, 30u},   {286458319u, 6204931u, 8601438u, 31u},
    {292730941u, 6340801u, 8789785u, 31u},   {299140913u, 6479647u, 8982256u, 32u},
    {305691246u, 6621533u, 9178942u, 33u},   {312385013u, 6766525u, 9379934u, 33u},
    {319225354u, 6914693u, 9585328u, 34u},   {326215479u, 7066105u, 9795219u, 35u},
    {333358667u, 7220833u, 10009707u, 36u},  {340658273u, 7378948u, 10228891u, 36u},
    {348117717u, 7540526u, 10452874u, 37u},  {355740503u, 7705642u, 10681762u, 38u},
    {363530205u, 7874374u, 10915663u, 39u},  {371490480u, 8046800u, 11154685u, 40u},
    {379625062u, 8223002u, 11398940u, 41u},  {387937769u, 8403062u, 11648545u, 42u},
    {396432501u, 8587066u, 11903614u, 42u},  {405113242u, 8775098u, 12164270u, 43u},
    {413984066u, 8967247u, 12430633u, 44u},  {423049137u, 9163605u, 12702828u, 45u},
    {432312707u, 9364261u, 12980984u, 46u},  {441779122u, 9569312u, 13265230u, 47u},
    {451452826u, 9778853u, 13555701u, 48u},  {461338356u, 9992982u, 13852532u, 49u},
    {471440350u, 10211799u, 14155863u, 50u}, {481763549u, 10435409u, 14465836u, 52u},
    {492312797u, 10663914u, 14782597u, 53u}, {503093043u, 10897424u, 15106293u, 54u},
    {514109347u, 11136046u, 15437078u, 55u}, {525366876u, 11379894u, 15775106u, 56u}};
inline uint64_t round_even(uint64_t value, unsigned shift) {
  const uint64_t whole = value >> shift, rest = value & ((uint64_t{1} << shift) - 1),
                 midpoint = uint64_t{1} << (shift - 1);
  return whole + (rest > midpoint || (rest == midpoint && (whole & 1)));
}
inline uint32_t evaluate(uint32_t input, bool quiet_snan = true) {
  const uint32_t mag = input & 0x7fffffffu;
  if (mag >= 0x7f800000u) {
    if (mag > 0x7f800000u)
      return input | (quiet_snan ? 0x00400000u : 0u);
    return input >> 31 ? 0u : 0x7f800000u;
  }
  // The small-input interval returns one, including negative inputs.
  if (mag < 0x33800000u)
    return 0x3f800000u;
  const bool negative = input >> 31;
  if (!negative && mag >= 0x43000000u)
    return 0x7f800000u;
  if (negative && mag > 0x42fc0000u)
    return 0;
  // Retain 29 fractional bits. Negative arguments use a ones-complement
  // reduction, so exact negative integers approach the previous interval.
  const int exponent = static_cast<int>(mag >> 23) - 127;
  const uint64_t mantissa = (mag & 0x7fffffu) | 0x800000u;
  const int shift = exponent + 6;
  const uint64_t absolute = shift >= 0 ? mantissa << shift : mantissa >> -shift;
  const int64_t phase =
      negative ? -static_cast<int64_t>(absolute) - 1 : static_cast<int64_t>(absolute);
  const uint32_t fraction = static_cast<uint32_t>(phase) & 0xffffffu;
  const auto c = coefficients[(static_cast<uint64_t>(phase) >> 24) & 31];
  // Round the linear product to Q29 (capped at Q28 for large products),
  // then discard its low guard bit before accumulation.
  const uint64_t lp = uint64_t{c.linear} * fraction;
  const uint64_t linear = lp >= (uint64_t{1} << 47) ? round_even(lp, 24) : round_even(lp, 23) >> 1;
  // The quadratic stage retains 18 coordinate bits and a Q24 square.
  // Round the inner coefficient to Q35 before multiplying by that square.
  const uint32_t coordinate = fraction >> 6;
  const uint64_t square = (uint64_t{coordinate} * coordinate) >> 12;
  const uint64_t inner =
      round_even((uint64_t{c.quadratic} << 7) + uint64_t{c.cubic} * coordinate, 7);
  const uint64_t product = inner * square;
  // Round the product to Q36, capped at 24 significant bits.
  const uint64_t quad =
      product >= (uint64_t{1} << 47) ? round_even(product, 24) * 2 : round_even(product, 23);
  // Combine the terms in Q36 and round once to the FP32 significand.
  const uint64_t sum = ((uint64_t{c.constant} + linear) << 8) + quad;
  return static_cast<uint32_t>(((phase >> 29) + 126) * 0x800000 +
                               static_cast<int64_t>(round_even(sum, 13)));
}
} // namespace util::detail::exp

namespace util {
/// @brief Base-2 exponential matching the captured RDNA3/4 instruction mapping.
inline float amdgpu_exp_f32(float value, bool quiet_snan = true) {
  return std::bit_cast<float>(detail::exp::evaluate(std::bit_cast<uint32_t>(value), quiet_snan));
}
} // namespace util
