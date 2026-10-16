// SPDX-License-Identifier: MIT

// FP16 FMA/FMAC, packed selectors, and guest FP-status checks.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <stdint.h>

namespace rocjitsu::arith::test::half_fma_reference {

// Return whether the FP16 encoding is a NaN.
inline bool nan(uint16_t bits) { return (bits & 0x7fff) > 0x7c00; }

// Independent exact integer oracle: magnitude * 2^exponent. All finite half
// products and aligned half addends fit in uint64_t, including their sum.
struct Number {
  uint64_t magnitude;
  int exponent;
  bool negative;
};

// Decode finite FP16 bits into an integer significand, binary scale, and sign.
inline Number decode(uint16_t bits) {
  int e = (bits >> 10) & 31;
  return {uint64_t((bits & 1023) + (e ? 1024 : 0)), (e ? e : 1) - 25, bool(bits & 0x8000)};
}

// Scale an unsigned significand by a binary shift, rounding right shifts to nearest-even.
inline uint64_t rounded_shift(uint64_t bits, int shift) {
  if (shift <= 0) {
    return bits << -shift;
  }
  uint64_t tail = bits & ((1ULL << shift) - 1);
  uint64_t midpoint = 1ULL << (shift - 1);
  return (bits >> shift) + (tail > midpoint || (tail == midpoint && ((bits >> shift) & 1)));
}

// Round a signed integer-grid value to FP16, optionally saturating finite overflow.
inline uint16_t pack(Number value, bool saturate) {
  uint16_t sign = value.negative ? 0x8000 : 0;
  if (!value.magnitude) {
    return sign;
  }
  int top = 0;
  for (uint64_t bits = value.magnitude; bits >>= 1;) {
    ++top;
  }
  int exponent = top + value.exponent;
  if (exponent < -14) {
    return sign | uint16_t(rounded_shift(value.magnitude, -24 - value.exponent));
  }
  uint64_t sig = rounded_shift(value.magnitude, top - 10);
  if (sig == 2048) {
    sig = 1024;
    ++exponent;
  }
  if (exponent > 15) {
    return sign | (saturate ? 0x7bff : 0x7c00);
  }
  return sign | uint16_t(((exponent + 15) << 10) + sig - 1024);
}

// Compute the test-side FP16 FMA result with modifiers and guest FP rules.
inline uint16_t evaluate(uint32_t a, uint32_t b, uint32_t c, uint32_t mode, bool saturate,
                         uint64_t fp_flags = 0, bool rdna3 = false) {
  if (rdna3 &&
      (!(fp_flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (fp_flags & RJ_ARITH_FP_IEEE_MODE))) {
    mode &= ~RJ_ARITH_ALU_OMOD_HALF;
  }
  uint32_t words[] = {a, b, c};
  uint16_t h[3];
  for (int i = 0; i < 3; ++i) {
    h[i] = uint16_t(words[i] >> (mode & (RJ_ARITH_ALU_HIGH_A << i) ? 16 : 0));
    if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
      h[i] &= 0x7fff;
    }
    if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
      h[i] ^= 0x8000;
    }
    if ((fp_flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (h[i] & 0x7fff) < 0x400) {
      h[i] &= 0x8000;
    }
  }
  unsigned ma = h[0] & 0x7fff, mb = h[1] & 0x7fff, mc = h[2] & 0x7fff;
  uint16_t result;
  if ((!ma && mb == 0x7c00) || (!mb && ma == 0x7c00)) {
    result = 0xfe00;
  } else if (nan(h[0]) || nan(h[1]) || nan(h[2])) {
    result = (nan(h[0])   ? h[0]
              : nan(h[1]) ? h[1]
                          : h[2]) |
             ((!rdna3 || (fp_flags & RJ_ARITH_FP_IEEE_MODE)) ? 0x200 : 0);
  } else if (ma == 0x7c00 || mb == 0x7c00) {
    result = ((h[0] ^ h[1]) & 0x8000) | 0x7c00;
    if (mc == 0x7c00 && ((result ^ h[2]) & 0x8000)) {
      result = 0xfe00;
    }
  } else if (mc == 0x7c00) {
    result = h[2];
  } else {
    auto x = decode(h[0]), y = decode(h[1]), z = decode(h[2]);
    Number p = {x.magnitude * y.magnitude, x.exponent + y.exponent, x.negative != y.negative};
    int grid = std::min(p.exponent, z.exponent);
    uint64_t pm = p.magnitude << (p.exponent - grid), zm = z.magnitude << (z.exponent - grid);
    Number sum = {0, grid, false};
    if (p.negative == z.negative) {
      sum.magnitude = pm + zm;
      sum.negative = p.negative;
    } else {
      sum.magnitude = pm > zm ? pm - zm : zm - pm;
      sum.negative = pm > zm ? p.negative : zm > pm ? z.negative : false;
    }
    result = pack(sum, saturate);
    unsigned omod = (mode >> 6) & 3;
    if (!omod && (fp_flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      bool tiny = grid < -26 ? sum.magnitude < (4095ULL << (-26 - grid))
                             : (sum.magnitude << (grid + 26)) < 4095;
      if (tiny) {
        result &= 0x8000;
      }
    }
    if (omod) {
      // Active OMOD flushes before packing at 2^-14 - 2^-26.
      bool tiny = grid < -26 ? sum.magnitude < (4095ULL << (-26 - grid))
                             : (sum.magnitude << (grid + 26)) < 4095;
      if (tiny) {
        result = 0;
      } else if (omod == 3 && (result & 0x7fff) < 0x0800) {
        result &= 0x8000;
      } else if ((result & 0x7fff) < 0x7c00) {
        auto scaled = decode(result);
        scaled.exponent += omod == 3 ? -1 : int(omod);
        result = pack(scaled, saturate);
      }
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    result = nan(result) || (result & 0x8000) ? 0 : std::min<uint16_t>(result, 0x3c00);
  }
  return result;
}

} // namespace rocjitsu::arith::test::half_fma_reference
