// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local compact extraction and RTZ FP64 encoding shared by AVX2 and AVX512.
// Vector loads, gathers and masked publication are supplied by each ISA translation unit.
#include "common/compiler.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Read zero-padded chunks without allowing an out-of-bounds gather.
U chunk(const uint32_t *table, U index) {
  return gather_chunks(table, index < 51 ? index : U{}) & U(index < 51);
}

// Scale positive finite pieces before flushing tiny results, matching legacy CDNA OMOD.
// TRIG_PREOP has no sign/NaN/overflow cases; only halving can discard a significant bit.
U finish(U value, int omod, unsigned rounding, bool clamp) {
  constexpr uint64_t hidden = 1ULL << 52;
  if (omod == 3) {
    U increment{};
    if (rounding == 0) {
      increment = U((value & 3) == 3) & 1;
    } else if (rounding == 1) {
      increment = value & 1;
    }
    value = value < 2 * hidden ? (value >> 1) + increment : value - hidden;
  } else if (omod) {
    U tiny = value << omod;
    if (omod == 2) {
      tiny = tiny >= 2 * hidden ? (tiny >> 1) + hidden : tiny;
    }
    value = value < hidden ? tiny : value + (uint64_t(omod) << 52);
  }
  if (omod) {
    value &= U(value >= hidden);
  }
  if (clamp) {
    constexpr uint64_t one = 0x3ff0000000000000ULL;
    value = value > one ? U{} + one : value;
  }
  return value;
}

// Stage every source-dependent result before masked publication to permit arbitrary aliases.
void evaluate(int lanes, const uint32_t *a_hi, const uint32_t *b, const uint32_t *table, int omod,
              unsigned rounding, bool clamp, uint64_t exec_mask, uint32_t *const *d) {
  uint64_t result[64];
  for (int lane = 0; lane < lanes; lane += sizeof(U) / sizeof(uint64_t)) {
    const U exponent = (load_words(a_hi + lane) >> 20) & 2047;
    const U shift = 53 * (load_words(b + lane) & 31) + (exponent > 1077 ? exponent - 1077 : U{});
    // All gather indexes are bounded even for selectors beyond the table tail.
    const U bounded = shift < 1201 ? shift : U{};
    const U index = divide24(bounded), offset = bounded - index * 24;
    const U c0 = chunk(table, index), c1 = chunk(table, index + 1);
    const U c2 = chunk(table, index + 2), c3 = chunk(table, index + 3);
    const U high = (c0 << 40) | (c1 << 16) | (c2 >> 8);
    const U low = ((c2 & 255) << 56) | (c3 << 32);
    U value = ((high << offset) | ((low >> ((64 - offset) & 63)) & U(offset != 0))) >> 11;
    const U tail = bounded > 1148 ? bounded - 1148 : U{};
    value &= (~U{} << tail) & U(shift < 1201);

    // Binary-search the leading bit using vector shifts, without a scalar CLZ per lane.
    U reduced = value, leading{};
    RJ_ARITH_UNROLL(6)
    for (int step = 32; step; step /= 2) {
      const U advance = U((reduced >> step) != 0) & uint64_t(step);
      reduced >>= advance;
      leading += advance;
    }
    const U bank = U(exponent >= 1968) & 128;
    // Biased exponent: leading - 53 - shift + bank + 1023.
    const U positive = leading + bank + 970;
    const U normal = ((positive - shift) << 52) | ((value << (52 - leading)) & 0xfffffffffffffULL);
    const U left = 1021 + bank;
    const U tiny = shift <= left ? value << ((left - shift) & 63)
                                 : (value >> ((shift - left) & 63)) & U(shift - left < 64);
    const U bits =
        finish((shift < positive ? normal : tiny) & U(value != 0), omod, rounding, clamp);
    __builtin_memcpy(result + lane, &bits, sizeof(bits));
  }
  for (int lane = 0; lane < lanes; lane += sizeof(U) / sizeof(uint64_t)) {
    U bits;
    __builtin_memcpy(&bits, result + lane, sizeof(bits));
    publish(d, lane, exec_mask >> lane, bits);
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
