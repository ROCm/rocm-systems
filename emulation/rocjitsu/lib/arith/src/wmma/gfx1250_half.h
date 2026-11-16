// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared gfx1250 half-input WMMA layout, accumulator decoding and final
// packing.
#pragma once
#include "common/float_formats.h"
#include "common/internal.h"
#include "mfma/nan_fma.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/gfx1250_backends.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Locate a packed input half in the flattened eight-VGPR, Wave32 operand.
inline int gfx1250_half_index(int index, int k) {
  int local = (k / 16) * 8 + k % 8;
  return (local / 2 * 32 + index + 16 * (k / 8 % 2)) * 2 + local % 2;
}
// Apply the instruction's absolute-value then negate controls to C's encoding.
inline float gfx1250_accumulator(uint32_t word, uint64_t controls) {
  if (controls & RJ_ARITH_WMMA_ABS_C) {
    word &= 0x7fffffffU;
  }
  if (controls & RJ_ARITH_WMMA_NEG_C) {
    word ^= 0x80000000U;
  }
  return as_float(word);
}
// Read C and apply signs; dispatch widens inline C according to the selected
// semantics.
template <Wmma1250Form Form>
inline float gfx1250_read_c(const uint32_t *const *c, int row, int col, uint64_t controls) {
  constexpr bool packed = Form == Wmma1250Form::F16 || Form == Wmma1250Form::Bf16;
  int lane = col + 16 * (row / 8);
  if constexpr (packed) {
    if (controls & RJ_ARITH_WMMA_INLINE_C) {
      return gfx1250_accumulator(c[0][0], controls);
    }
    uint16_t half = uint16_t(c[row % 8 / 2][lane] >> (16 * (row % 2)));
    float value = Form == Wmma1250Form::F16 ? f16_to_float(half) : bf16_to_float(half);
    return gfx1250_accumulator(as_bits(value), controls);
  } else {
    return gfx1250_accumulator(c[row % 8][lane], controls);
  }
}

// Repair only NaN mixed-result elements, before destination stores can
// overwrite their inputs.
inline float gfx1250_repair_nan(float value, const float *left, const float *right, int row,
                                int col, float initial) {
  if ((as_bits(value) & 0x7fffffffU) <= 0x7f800000U) {
    return value;
  }
  float dot = 0.0f;
  for (int k = 0; k < 32; ++k) {
    dot = matrix_nan_fma(left[gfx1250_half_index(row, k)], right[gfx1250_half_index(col, k)], dot);
  }
  return matrix_nan_fma(dot, 1.0f, initial);
}

// Pack once with RNE in Loose; Compatibility retains legacy BF16 truncation.
// Every source read, including NaN repair, must precede this publication step.
template <Wmma1250Form Form>
inline void gfx1250_write_result(uint64_t flags, uint32_t *const *d, const float result[16][16],
                                 bool saturate_f16 = false) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  constexpr bool packed = Form != Wmma1250Form::F32F16 && Form != Wmma1250Form::F32Bf16;
  for (int reg = 0; reg < (packed ? 4 : 8); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      int row = (packed ? 2 * reg : reg) + 8 * (lane / 16), col = lane % 16;
      if constexpr (packed) {
        auto narrow = [&](float x) {
          if constexpr (Form == Wmma1250Form::F16) {
            // FP8 output retains its legacy saturation policy in Compatibility.
            const bool saturate = saturate_f16 || (!compatibility && (flags & RJ_ARITH_FP16_OVFL));
            uint16_t value = float_to_f16(x, saturate);
            // The WMMA saturation rule also applies to an infinite accumulator.
            if (saturate && (value & 0x7fffU) == 0x7c00U) {
              value = uint16_t((value & 0x8000U) | 0x7bffU);
            }
            return value;
          } else {
            if (compatibility) {
              return uint16_t(as_bits(x) >> 16);
            }
            uint16_t value = float_to_bf16(x);
            // Saturate both accumulation overflow and BF16 rounding overflow.
            if ((flags & RJ_ARITH_FP16_OVFL) && (value & 0x7fffU) == 0x7f80U) {
              value = uint16_t((value & 0x8000U) | 0x7f7fU);
            }
            return value;
          }
        };
        d[reg][lane] =
            uint32_t(narrow(result[row][col])) | (uint32_t(narrow(result[row + 1][col])) << 16);
      } else {
        d[reg][lane] = as_bits(result[row][col]);
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
