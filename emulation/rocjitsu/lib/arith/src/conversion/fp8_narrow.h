// SPDX-License-Identifier: MIT

// FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Rounding logic adapted from rocjitsu util/data_types.h, with RDNA4 NaN,
// infinity and subnormal stochastic behavior established by GPU captures.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Convert FP32 bits to an OCP FP8/BF8 byte using nearest-even or the supplied
// stochastic seed. Saturation applies only to finite inputs; NaNs are canonical.
// Uses integer operations and preserves host floating-point state.
template <bool Bf8, bool Stochastic>
inline uint32_t narrow_fp8(uint32_t raw, uint32_t seed, bool saturate) {
  const unsigned fraction_bits = Bf8 ? 2 : 3;
  const unsigned terminal = Bf8 ? 124 : 127;

  // Keep sign separate while rounding magnitude; E4M3FN and E5M2 have different terminal
  // encodings.
  uint32_t magnitude = raw & 0x7fffffff, sign = (raw >> 24) & 128;
  if (magnitude > 0x7f800000) {
    return Bf8 ? 0xfe : 0xff;
  }
  unsigned limit = terminal - (saturate && magnitude != 0x7f800000);
  int exponent = int(magnitude >> 23) - (Bf8 ? 112 : 120);
  if (exponent >= (Bf8 ? 31 : 16)) {
    return sign | limit;
  }
  uint32_t significand = (magnitude & 0x7fffff) | 0x800000, result;
  if constexpr (Stochastic) {
    // Align to the minimum normal exponent before adding the high seed bits.
    // Bits discarded by this alignment do not contribute a sticky bit.
    unsigned alignment = exponent < 1 ? unsigned(1 - exponent) : 0;
    significand = alignment < 32 ? significand >> alignment : 0;
    result = (significand + (seed >> (9 + fraction_bits))) >> (23 - fraction_bits);
  } else {
    unsigned shift = 23 - fraction_bits + (exponent < 1 ? unsigned(1 - exponent) : 0);
    if (shift > 24) {
      return sign;
    }
    result = (significand + (1u << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
  }
  if (exponent > 0) {
    result += unsigned(exponent - 1) << fraction_bits;
  }
  return sign | (result < limit ? result : limit);
}

// Narrow FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Stochastic>
void fp8_narrow_x86_64_v3(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b);

// Narrow FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Stochastic>
void fp8_narrow_x86_64_v4(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b);

} // namespace rocjitsu::arith
