// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for signed and unsigned normalized 16-bit conversions.

#include "common/floating_point.h"
#include "conversion/helpers.h"
#include "conversion/normalized_reference.h"
#include "rj_arith/rj_arith.h"

#include <bit>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class Normalized {
  SignedPackedFloat,
  UnsignedPackedFloat,
  SignedPackedHalf,
  UnsignedPackedHalf,
  SignedHalf,
  UnsignedHalf
};

// Check normalized conversion results and destination preservation for the supplied entry point.
template <int Lanes, int Operands, typename Call>
void check_normalized(Call call, Normalized operation) {
  int op = int(operation);
  auto input = conversion_inputs<Lanes>(op < 2 ? 32 : 16);
  // Retain edge classes, but spend the other lanes inside the nonsaturated
  // interval. Fractional products exercise nearest-even rounding.
  std::minstd_rand random;
  for (int reg : {0, 2}) {
    for (int lane = 16; lane < Lanes; ++lane) {
      if (op < 2) {
        input.v[reg][lane] =
            std::bit_cast<uint32_t>(float(int(random() >> 7) - 0x800000) * 0x1p-23f);
      } else {
        input.v[reg][lane] = random_f16_pair(random);
      }
    }
  }

  if (op == 2 || op == 3) {
    // Largest FP16 subnormals produce nonzero normalized integers unless
    // input flushing is enabled.
    input.v[0][16] = 0x83ff03ffU;
    input.v[2][17] = 0x03ff83ffU;
  }

  uint64_t modifiers =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP |
      (op >= 2 ? RJ_ARITH_ALU_HIGH_A : 0) |
      (op >= 4 ? RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_OMOD_HALF
               : RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B | (op >= 2 ? RJ_ARITH_ALU_HIGH_B : 0));
  check_conversion<Lanes, Operands, 1, 1, true>(
      call, input,
      [=](uint64_t a, uint32_t b, uint32_t, uint32_t d, uint64_t flags, uint32_t mode) {
        if (op >= 2 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
          auto flush = [](uint32_t word) {
            for (int shift : {0, 16}) {
              if (((word >> shift) & 0x7fff) < 0x400) {
                word &= ~(0x7fffU << shift);
              }
            }
            return word;
          };
          a = flush(uint32_t(a));
          b = flush(b);
        }
        return normalized_reference(op, uint32_t(a), b, d, mode);
      },
      conversion_bits, modifiers, false,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F32W32, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_i16_f32_w32, Normalized::SignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F32W32, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_u16_f32_w32, Normalized::UnsignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W32Gfx11, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w32, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W32Gfx12, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w32_gfx12, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W32Gfx11, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w32, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W32Gfx12, Arithmetic) {
  check_normalized<32, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w32_gfx12, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormI16F16W32, Arithmetic) {
  check_normalized<32, 1>(rj_arith_v_cvt_norm_i16_f16_w32, Normalized::SignedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormU16F16W32, Arithmetic) {
  check_normalized<32, 1>(rj_arith_v_cvt_norm_u16_f16_w32, Normalized::UnsignedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F32W64, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_i16_f32_w64, Normalized::SignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F32W64, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_u16_f32_w64, Normalized::UnsignedPackedFloat);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W64Gfx11, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w64, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormI16F16W64Gfx12, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_i16_f16_w64_gfx12, Normalized::SignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W64Gfx11, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w64, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtPkNormU16F16W64Gfx12, Arithmetic) {
  check_normalized<64, 2>(rj_arith_v_cvt_pk_norm_u16_f16_w64_gfx12, Normalized::UnsignedPackedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormI16F16W64, Arithmetic) {
  check_normalized<64, 1>(rj_arith_v_cvt_norm_i16_f16_w64, Normalized::SignedHalf);
}

// Check normalized conversion results and destination preservation.
TEST(ConversionCvtNormU16F16W64, Arithmetic) {
  check_normalized<64, 1>(rj_arith_v_cvt_norm_u16_f16_w64, Normalized::UnsignedHalf);
}
