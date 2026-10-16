// SPDX-License-Identifier: MIT

// Scalar exact FP16 unary evaluation and optional exception classification.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include "common/float_formats.h"
#include "common/round_integral.h"
#include "fp/half_fma_scalar.h"
#include "math/half_unary.h"
#include "math/rcp_model.h"
#include "math/rsq_model.h"
#include "math/sqrt_model.h"
#include "math/unary.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate a raw half operand with the borrowed rocjitsu mapping and modifiers.
// The caller selects the source half and supplies the effective architecture OMOD.
// Keep this calculation inline in vector lane loops.
template <Unary Op, bool Rdna3 = false, bool PreserveClampNan = false>
[[gnu::always_inline]] inline uint16_t half_unary_exact_value(uint64_t flags, uint32_t mode,
                                                              uint16_t raw) {
  uint32_t input = raw;
  unsigned omod = (mode >> 6) & 3;
  if (mode & RJ_ARITH_ALU_ABS_A) {
    input &= 0x7fff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    input ^= 0x8000;
  }
  uint32_t magnitude = input & 0x7fff;

  uint32_t value;
  if constexpr (Op == Unary::Exp || Op == Unary::Log) {
    // Borrowed from rocjitsu transcendental::detail::log_exp_f16_nearest.
    // Classify specials before libm; finite FP64 intermediates are normal and
    // bounded. Integer narrowing avoids a second rounding through FP32 and
    // requires no host environment guard.
    if (magnitude > 0x7c00) {
      value = input | ((!Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE)) ? 0x200 : 0);
    } else if (magnitude == 0x7c00) {
      if constexpr (Op == Unary::Log) {
        value = input & 0x8000 ? 0xfe00 : input;
      } else {
        value = input & 0x8000 ? 0 : input;
      }
    } else {
      if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && magnitude < 0x400) {
        input &= 0x8000;
        magnitude = 0;
      }
      if constexpr (Op == Unary::Log) {
        if (!magnitude) {
          value = flags & RJ_ARITH_FP16_OVFL ? 0xfbff : 0xfc00;
        } else if (input & 0x8000) {
          value = 0xfe00;
        } else {
          value = half_fma_narrow(std::log2(double(f16_to_float(uint16_t(input)))),
                                  flags & RJ_ARITH_FP16_OVFL);
        }
      } else {
        double x = double(f16_to_float(uint16_t(input)));
        value = half_fma_narrow(std::exp2(std::clamp(x, -32.0, 32.0)), flags & RJ_ARITH_FP16_OVFL);
      }
      if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (value & 0x7fff) < 0x400) {
        value &= 0x8000;
      }
    }
  } else if constexpr (rocjitsu::arith::half_unary_sfu<Op>) {
    // rocjitsu transcendental::map_f16: apply the half input policy, evaluate
    // the borrowed integer FP32 model, then narrow before output modifiers.
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && magnitude < 0x400) {
      input &= 0x8000;
    }
    uint32_t wide_input = rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(uint16_t(input)));
    bool quiet_snan = !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE);
    uint32_t wide_result;
    if constexpr (Op == rocjitsu::arith::Unary::Rcp) {
      wide_result = rocjitsu::arith::rcp_f32_bits(wide_input, quiet_snan);
    } else if constexpr (Op == rocjitsu::arith::Unary::Rsq) {
      wide_result = rocjitsu::arith::rsq_f32_bits(wide_input, quiet_snan);
    } else {
      wide_result = rocjitsu::arith::sqrt_f32_bits(wide_input, quiet_snan);
    }
    value = rocjitsu::arith::float_to_f16(rocjitsu::arith::as_float(wide_result),
                                          flags & RJ_ARITH_FP16_OVFL);
    if ((flags & RJ_ARITH_FP16_OVFL) && (wide_result & 0x7fffffff) == 0x7f800000 &&
        magnitude < 0x7c00) {
      value = (value & 0x8000) | 0x7bff;
    }

    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (value & 0x7fff) < 0x400) {
      value &= 0x8000;
    }
  } else if constexpr (Op == rocjitsu::arith::Unary::FrexpMant) {
    // rocjitsu widens half inputs before FREXP, so half subnormals survive
    // every guest denormal policy. Normalize directly in half-width fields.
    if (magnitude >= 0x7c00) {
      value = input | (magnitude > 0x7c00 ? 0x200 : 0);
    } else if (magnitude >= 0x400) {
      value = (input & 0x83ff) | 0x3800;
    } else if (magnitude) {
      unsigned shift = rocjitsu::arith::bit_count_zero<false>(magnitude) - 21;
      value = (input & 0x8000) | 0x3800 | ((magnitude << shift) & 0x3ff);
    } else {
      value = input;
    }
  } else {
    constexpr auto rounding =
        Op == rocjitsu::arith::Unary::Trunc   ? rocjitsu::arith::IntegralRound::Trunc
        : Op == rocjitsu::arith::Unary::Ceil  ? rocjitsu::arith::IntegralRound::Ceil
        : Op == rocjitsu::arith::Unary::Floor ? rocjitsu::arith::IntegralRound::Floor
                                              : rocjitsu::arith::IntegralRound::Nearest;
    value = rocjitsu::arith::round_integral_bits<rounding, true>(
        input, flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  }
  // OMOD flushes tiny inputs to +0; newly tiny div:2 results retain their sign.
  magnitude = value & 0x7fff;
  if (omod && magnitude < 0x7c00) {
    if (!magnitude ||
        ((Op == rocjitsu::arith::Unary::Rcp || Op == Unary::Exp || Op == Unary::Log) &&
         magnitude < 0x400)) {
      value = 0;
    } else if (omod == 3 &&
               (Op == rocjitsu::arith::Unary::Rcp || Op == Unary::Exp || Op == Unary::Log) &&
               magnitude < 0x800) {
      value &= 0x8000;
    } else if (omod == 3) {
      value -= 0x400;
    } else if ((magnitude >> 10) + omod >= 31) {
      value = (value & 0x8000) | (flags & RJ_ARITH_FP16_OVFL ? 0x7bff : 0x7c00);
    } else {
      value += omod << 10;
    }
  }
  if ((mode & RJ_ARITH_ALU_CLAMP) && (!PreserveClampNan || (value & 0x7fff) <= 0x7c00)) {
    value = (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : value > 0x3c00 ? 0x3c00 : value;
  }
  return uint16_t(value);
}

} // namespace rocjitsu::arith
