// SPDX-License-Identifier: MIT

// Guest FP32 multiplication exception classification from operands and result.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "fp/fma_integer.h"

#include <stdint.h>

namespace rocjitsu::arith {

// rocjitsu classify_mul_f32 under nearest-even host arithmetic, expressed with
// integer intermediates. Classification precedes guest flushing and CLAMP.
// This inherited classifier reports no INVALID cause, including for 0 * infinity.
inline uint32_t mul_exceptions_f32(uint32_t a, uint32_t b, unsigned omod, bool rdna3) {
  a &= 0x7fffffffU;
  b &= 0x7fffffffU;
  uint32_t causes = ((a && a < 0x00800000U) || (b && b < 0x00800000U)) ? 2U : 0U;
  if (a >= 0x7f800000U || b >= 0x7f800000U) {
    return causes;
  }

  // A finite FP32 product has at most 48 significant bits and is normal in
  // FP64. Thus the existing round-to-odd helper represents it exactly.
  uint64_t exact = fma_finite_round_odd(a, b, 0);
  uint32_t rounded = fma_narrow_f32(exact, false);
  if (omod) {
    if (exact) {
      const int shift = omod == 3 ? -1 : int(omod);
      exact = uint64_t(int64_t(exact) + int64_t(shift) * (1LL << 52));
    }
    if (rounded < 0x7f800000U) {
      const uint32_t scale = omod == 3 ? 0x3f000000U : 0x3f800000U + (omod << 23);
      rounded = fma_narrow_f32(fma_finite_round_odd(rounded, scale, 0), false);
    }
  }
  if (rounded == 0x7f800000U) {
    causes |= 8U;
  }
  if (exact && rounded < 0x00800000U) {
    causes |= 16U;
  }
  if (rounded == 0x7f800000U || fma_finite_round_odd(rounded, 0x3f800000U, 0) != exact) {
    causes |= 32U;
  }
  if (omod && !rdna3) {
    causes &= ~(16U | 32U);
  }
  return causes;
}

} // namespace rocjitsu::arith
