// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// @brief Shared single-precision square-root mapping for AMDGPU instruction execution.

#include <bit>
#include <cstdint>

namespace util::detail {

// Staged fixed-point cubic fitted to the exhaustive normalized SQRT captures of
// physical gfx1100 and gfx1201 (byte-identical). The index combines exponent
// parity with the 23-bit input mantissa; 32 segments of 2^19 fractions f each.
// Per segment the result mantissa offset, in units of 2^-24 result ULP, is
//   constant * 2^22                          (constant in quarter ULP)
// + L * 2^19, L = floor((linear * f + b) / 2^17), b = 2^15, or 2^16 when
//                                            linear * f >= 2^40
// - Q * 2^12, Q = round(inner * square / 2^23), square = (g * g) >> 14,
//             g = f & ~3, inner = quadratic * 2^8 - ceil'(cubic * f / 2^10)
// where ceil' negates, adds a quarter of the 2^10 step, and floors, and the
// product rounds at bit 24 instead of bit 23 once it reaches 2^47. The sum
// rounds half-to-even to a mantissa. Quadratic uses eighth ULP; cubic uses
// quarter ULP. This is an empirical mapping, not a claim about the hardware
// implementation. Sparse one-ULP residuals complete the captured mapping.
namespace sqrt {
struct Coefficient {
  uint32_t constant;
  uint32_t linear;
  uint16_t quadratic;
  uint16_t cubic;
};
inline constexpr Coefficient coefficients[] = {
    {0u, 2097143u, 32722u, 474u},        {1032684u, 2034529u, 29882u, 409u},
    {2035417u, 1977208u, 27431u, 356u},  {3010662u, 1924474u, 25297u, 312u},
    {3960563u, 1875745u, 23425u, 275u},  {4886999u, 1830541u, 21774u, 244u},
    {5791627u, 1788454u, 20308u, 218u},  {6675918u, 1749143u, 18999u, 195u},
    {7541186u, 1712315u, 17825u, 176u},  {8388608u, 1677719u, 16767u, 159u},
    {9219243u, 1645139u, 15810u, 145u},  {10034053u, 1614387u, 14940u, 132u},
    {10833909u, 1585296u, 14147u, 120u}, {11619604u, 1557724u, 13422u, 110u},
    {12391866u, 1531542u, 12757u, 102u}, {13151360u, 1506637u, 12145u, 94u},
    {13898700u, 2965809u, 46276u, 670u}, {15359137u, 2877259u, 42260u, 578u},
    {16777216u, 2796194u, 38793u, 503u}, {18156420u, 2721617u, 35775u, 441u},
    {19499783u, 2652705u, 33129u, 389u}, {20809961u, 2588776u, 30793u, 346u},
    {22089298u, 2529256u, 28720u, 308u}, {23339875u, 2473662u, 26869u, 277u},
    {24563549u, 2421579u, 25208u, 249u}, {25761984u, 2372654u, 23712u, 225u},
    {26936680u, 2326579u, 22358u, 205u}, {28088995u, 2283088u, 21129u, 186u},
    {29220162u, 2241948u, 20008u, 171u}, {30331303u, 2202954u, 18982u, 156u},
    {31423446u, 2165927u, 18042u, 144u}, {32497533u, 2130707u, 17176u, 133u},
};

// Sorted residual keys: (index << 1) | 1 selects +1 ULP, otherwise -1 ULP.
inline constexpr uint32_t corrections[] = {
    0x0067350u, 0x00a791au, 0x00c1d1au, 0x00d7428u, 0x00efacau, 0x0180002u, 0x01ca73eu, 0x01eb1a6u,
    0x01ec160u, 0x0241f44u, 0x02a30b4u, 0x02abb44u, 0x02f1d10u, 0x02fd82au, 0x03871a6u, 0x03b8802u,
    0x03b9f4au, 0x03d10e8u, 0x03f1530u, 0x04a9836u, 0x04c3abeu, 0x04c3ea4u, 0x04cec24u, 0x04e9f2au,
    0x0553ac4u, 0x05ab3c8u, 0x05cfba6u, 0x05dd09au, 0x05f089au, 0x05f2246u, 0x05f6e24u, 0x065f474u,
    0x06775c6u, 0x068debau, 0x06a81d0u, 0x06c84c8u, 0x06d4bd4u, 0x06de130u, 0x06e6624u, 0x06f16fcu,
    0x06f94eeu, 0x06fbf06u, 0x06fddaau, 0x06fee94u, 0x0763c3eu, 0x0778aceu, 0x0783472u, 0x079041eu,
    0x07918b6u, 0x07b36f2u, 0x07b95e2u, 0x07c3e74u, 0x07c6838u, 0x07c8ef2u, 0x07d1b22u, 0x07e27b0u,
    0x07e5882u, 0x07e9ee8u, 0x07ec934u, 0x07ee12au, 0x07ee510u, 0x07eefb0u, 0x07f24a6u, 0x07f573cu,
    0x07fd694u, 0x0842278u, 0x088cfbeu, 0x08d66cau, 0x08e77b2u, 0x08ec484u, 0x08f3aecu, 0x08f54d6u,
    0x08f6e26u, 0x08f9b56u, 0x094af02u, 0x094e5a6u, 0x09618e0u, 0x096ea6eu, 0x0995500u, 0x09a165cu,
    0x09ab5f4u, 0x09b55a6u, 0x09c8038u, 0x09cf812u, 0x09dfd00u, 0x09e1c0au, 0x09ec02cu, 0x09fc82cu,
    0x09fdd1au, 0x0a6fabau, 0x0a933e2u, 0x0a98158u, 0x0aa0166u, 0x0ab2166u, 0x0ab576au, 0x0ac2d5eu,
    0x0ac5264u, 0x0acba1eu, 0x0ad63b6u, 0x0adbceeu, 0x0ae2256u, 0x0ae275cu, 0x0ae8c6eu, 0x0aeb076u,
    0x0aebb8cu, 0x0aeea64u, 0x0af21c8u, 0x0af27beu, 0x0af2e50u, 0x0af2f78u, 0x0afe82eu, 0x0aff786u,
    0x0b5497au, 0x0b67d8au, 0x0bac3b8u, 0x0bb5ed0u, 0x0bbf5c6u, 0x0bc4df6u, 0x0bc7e42u, 0x0bd03d6u,
    0x0bd6e34u, 0x0bdc3c6u, 0x0be492cu, 0x0be9b1cu, 0x0bedc16u, 0x0bfafc8u, 0x0bfffb8u, 0x0c191f4u,
    0x0c340f6u, 0x0c45fb0u, 0x0c5ece4u, 0x0c7f2b0u, 0x0c83fb0u, 0x0c87a4au, 0x0c896a0u, 0x0c9df9eu,
    0x0ca50d2u, 0x0cb4b3au, 0x0cd1406u, 0x0cd5bb0u, 0x0ce1bd2u, 0x0ce61b2u, 0x0ce6890u, 0x0cf13a0u,
    0x0cf6216u, 0x0cf92d2u, 0x0cf95f4u, 0x0cfc2e4u, 0x0cfc5a0u, 0x0cff416u, 0x0d569c4u, 0x0d6042au,
    0x0d7eea4u, 0x0d7eecau, 0x0d8e43eu, 0x0da9db4u, 0x0db6864u, 0x0dbacc0u, 0x0dd19c4u, 0x0de56b6u,
    0x0df2864u, 0x0df2eb8u, 0x0df8d1eu, 0x0dfc25au, 0x0e70dfcu, 0x0e8e366u, 0x0e95ba2u, 0x0ea332cu,
    0x0eaaba2u, 0x0ec054au, 0x0ec64bau, 0x0edc480u, 0x0edfaeeu, 0x0ee10f6u, 0x0ee38e2u, 0x0ee41eau,
    0x0eeeeecu, 0x0eefcceu, 0x0ef25d4u, 0x0ef5688u, 0x0efe948u, 0x0f4a7c6u, 0x0f65538u, 0x0f824ccu,
    0x0f8293au, 0x0f8e590u, 0x0f9f04au, 0x0f9fd78u, 0x0faa22au, 0x0fb93c8u, 0x0fbcc60u, 0x0fc88e2u,
    0x0fcd5e8u, 0x0fd64a0u, 0x0fe190cu, 0x0fe1e2au, 0x0feb4e0u, 0x0ff0fdcu, 0x0ff871au, 0x0ffbff0u,
    0x0fffcb8u, 0x10cb3e3u, 0x10d3f75u, 0x10f4b75u, 0x10f5acdu, 0x116ccb3u, 0x11b9a27u, 0x11e2dabu,
    0x139bed8u, 0x13b1320u, 0x13c0b12u, 0x144823au, 0x145ba10u, 0x147709au, 0x1489138u, 0x14c3b76u,
    0x14cd4dcu, 0x14ce90eu, 0x14e310eu, 0x14e8a74u, 0x1562b56u, 0x1580004u, 0x15a0484u, 0x15ae878u,
    0x15e3b62u, 0x166802au, 0x169499eu, 0x169ea94u, 0x16a5abcu, 0x16bd62au, 0x16d779eu, 0x16dac16u,
    0x17aa020u, 0x17af760u, 0x17b6174u, 0x17c1be8u, 0x17de544u, 0x17deba6u, 0x17ed4ceu, 0x17eebbcu,
    0x17f649au, 0x17fc41cu, 0x184ca7cu, 0x18ae6cau, 0x18cfb18u, 0x18d207eu, 0x18d25dcu, 0x18d4766u,
    0x18dca7cu, 0x18e2102u, 0x18eb8c0u, 0x18ec94cu, 0x18f9646u, 0x18fc1a6u, 0x1972f7au, 0x199164eu,
    0x1998574u, 0x199f938u, 0x19a3842u, 0x19ab1dcu, 0x19ad3acu, 0x19b2292u, 0x19b2e2au, 0x19bf558u,
    0x19c8292u, 0x19dd2e4u, 0x19e0ad2u, 0x19e567cu, 0x19e778cu, 0x19e98deu, 0x19f0e18u, 0x19f2ee0u,
    0x19f74c6u, 0x1a50d5cu, 0x1a703a4u, 0x1a7a3b8u, 0x1a97e06u, 0x1ab2278u, 0x1ab3c30u, 0x1acbfdcu,
    0x1ad1b72u, 0x1adb78cu, 0x1adb7fau, 0x1aec3feu, 0x1aedfe6u, 0x1af3da2u, 0x1b4431cu, 0x1b7e7d8u,
    0x1ba06f0u, 0x1ba9424u, 0x1bb16f0u, 0x1bbc774u, 0x1bca2c4u, 0x1bd0538u, 0x1bdabccu, 0x1bdeaaeu,
    0x1bdf3f8u, 0x1beefa0u, 0x1bf24d4u, 0x1bf5c92u, 0x1bfcbacu, 0x1bfd0beu, 0x1bfe3eeu, 0x1c4cbb6u,
    0x1c583e6u, 0x1c5ad72u, 0x1c7c40au, 0x1c8e93cu, 0x1c8f33eu, 0x1c92aa2u, 0x1ca4b26u, 0x1cae5aeu,
    0x1cb3fd4u, 0x1cc4f8cu, 0x1cc7c5eu, 0x1cc90a0u, 0x1cccc6au, 0x1cce930u, 0x1cdfa42u, 0x1ce4c9au,
    0x1ce7e60u, 0x1cf1abau, 0x1cf7e48u, 0x1cfdaf6u, 0x1cfe8b8u, 0x1cff362u, 0x1d49c6eu, 0x1d78e3au,
    0x1d9838eu, 0x1d9b6aeu, 0x1d9f82cu, 0x1da04e4u, 0x1da4dc2u, 0x1da8ebcu, 0x1da91c2u, 0x1db7140u,
    0x1dc1e60u, 0x1dc5cb0u, 0x1dc934cu, 0x1dd64d6u, 0x1dd88b0u, 0x1dd9ba8u, 0x1dde3f8u, 0x1df750au,
    0x1dfbf5au, 0x1e54068u, 0x1e9ee84u, 0x1ea6776u, 0x1eaa586u, 0x1eb2a92u, 0x1eb4094u, 0x1ebc14cu,
    0x1ebdd5au, 0x1ec0412u, 0x1ed5522u, 0x1ed5a22u, 0x1ee173eu, 0x1ee47ccu, 0x1efafa2u, 0x1f83a7eu,
    0x1f8d648u, 0x1f98e28u, 0x1fb593au, 0x1fcbf8au, 0x1fd1a20u, 0x1fd3be0u, 0x1fd87e6u, 0x1fe3880u,
    0x1fe3cd6u, 0x1fe7c68u, 0x1ffab36u, 0x1ffab46u, 0x1ffc73cu,
};
} // namespace sqrt

inline uint32_t amdgpu_sqrt_correct(uint32_t index, uint32_t result) {
  const uint32_t key = index << 1;
  uint32_t first = 0;
  uint32_t last = sizeof(sqrt::corrections) / sizeof(sqrt::corrections[0]);
  const uint32_t end = last;
  while (first < last) {
    const uint32_t middle = first + (last - first) / 2;
    if ((sqrt::corrections[middle] & ~1u) < key)
      first = middle + 1;
    else
      last = middle;
  }
  if (first != end && (sqrt::corrections[first] & ~1u) == key)
    result = (sqrt::corrections[first] & 1u) ? result + 1 : result - 1;
  return result;
}

inline uint32_t amdgpu_sqrt_normalized_bits(uint32_t index) {
  const auto coefficient = sqrt::coefficients[index >> 19];
  const int64_t fraction = index & 0x7ffffu;
  const uint64_t product = uint64_t{coefficient.linear} * static_cast<uint64_t>(fraction);
  const uint64_t bias = product >= (uint64_t{1} << 40) ? uint64_t{1} << 16 : uint64_t{1} << 15;
  const int64_t linear = static_cast<int64_t>((product + bias) >> 17);
  const int64_t square_input = fraction & ~int64_t{3};
  const int64_t square = (square_input * square_input) >> 14;
  const int64_t inner =
      (int64_t{coefficient.quadratic} << 8) + ((256 - int64_t{coefficient.cubic} * fraction) >> 10);
  const int64_t quadratic = inner * square;
  const int64_t negated = quadratic >= (int64_t{1} << 47)
                              ? (((int64_t{1} << 23) - quadratic) >> 24) * 2
                              : ((int64_t{1} << 22) - quadratic) >> 23;
  const int64_t sum = (int64_t{coefficient.constant} << 22) + (linear << 19) + (negated * 4096);
  const int64_t quotient = sum >> 24;
  const int64_t remainder = sum & ((int64_t{1} << 24) - 1);
  const int64_t midpoint = int64_t{1} << 23;
  const int64_t mantissa =
      quotient + (remainder > midpoint || (remainder == midpoint && (quotient & 1)));
  const uint32_t result = 0x3f800000u + static_cast<uint32_t>(mantissa);
  // Exhaustive normalized captures bound every residual to within 1/2048
  // ULP of a rounding midpoint. Other inputs need no correction-table lookup.
  if (remainder < midpoint - 8192 || remainder > midpoint + 8192)
    return result;
  return amdgpu_sqrt_correct(index, result);
}

inline uint32_t amdgpu_sqrt_bits(uint32_t input, bool quiet_snan = true) {
  const uint32_t magnitude = input & 0x7fffffffu;
  const uint32_t sign = input & 0x80000000u;
  if (magnitude > 0x7f800000u)
    return input | (quiet_snan ? 0x00400000u : 0u);
  if (magnitude < 0x00800000u)
    return sign;
  if (sign != 0)
    return 0xffc00000u;
  if (magnitude == 0x7f800000u)
    return input;

  const int exponent = static_cast<int>(magnitude >> 23) - 127;
  const unsigned parity = exponent & 1;
  const uint32_t normalized =
      amdgpu_sqrt_normalized_bits((parity << 23) | (magnitude & 0x007fffffu));
  const int result_exponent =
      static_cast<int>(normalized >> 23) + (exponent - static_cast<int>(parity)) / 2;
  return (static_cast<uint32_t>(result_exponent) << 23) | (normalized & 0x007fffffu);
}
} // namespace util::detail

namespace util {

/// @brief AMD single-precision square root matching captured RDNA3/4 outputs.
/// @details The empirical mapping satisfies the ISA one-ULP bound and flushes input
/// subnormals to signed zero independently of MODE. Signed zero is preserved; negative
/// normal inputs and negative infinity produce 0xffc00000; positive infinity is preserved.
/// NaN payloads keep their sign and are quieted unless @p quiet_snan is false.
/// Results are independent of MODE rounding and host rounding.
inline float amdgpu_sqrt_f32(float value, bool quiet_snan = true) {
  return std::bit_cast<float>(detail::amdgpu_sqrt_bits(std::bit_cast<uint32_t>(value), quiet_snan));
}

} // namespace util
