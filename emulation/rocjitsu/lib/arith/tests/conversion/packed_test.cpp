// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for pairwise FP32 conversion into packed RTZ FP16 or saturating I16/U16 results.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "conversion/helpers.h"
#include "conversion/packed_conversion_reference.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <bit>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class PackedConversion { HalfRtz, Signed, Unsigned };

// Check paired 16-bit conversions and packed result ordering for the supplied entry point.
// Exact is false for entry points that stay loose-only until their captures are reviewed.
template <int Lanes, bool Exact = true, typename Call>
void check_packed(Call call, PackedConversion operation, bool sdwa = false) {
  int op = int(operation);
  uint64_t modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_B |
                       RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP |
                       (op == 0 ? RJ_ARITH_ALU_OMOD_HALF : 0);
  check_conversion<Lanes, 2, 1, 1, true>(
      call, conversion_inputs<Lanes>(32),
      [=](uint64_t a, uint32_t b, uint32_t, uint32_t, uint64_t flags, uint32_t mode) {
        uint32_t result = packed_conversion_reference(op, uint32_t(a), b, mode);
        if (op == 0 && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
          for (int shift : {0, 16}) {
            if (((result >> shift) & 0x7fffU) < 0x400U) {
              result &= ~(0x7fffU << shift);
            }
          }
        }
        return result;
      },
      [=](uint64_t actual, uint64_t expected, uint64_t flags, uint32_t mode) {
        if (op) {
          conversion_bits(actual, expected, flags, mode);
        } else {
          for (int shift : {0, 16}) {
            float a = f16_value(uint16_t(actual >> shift)),
                  e = f16_value(uint16_t(expected >> shift));
            conversion_float(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(e), 32);
          }
        }
      },
      modifiers, Exact,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
      sdwa ? RJ_ARITH_CVT_SDWA : 0);
}

} // namespace

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W32Gfx11, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_rtz_f16_f32_w32, PackedConversion::HalfRtz, /*sdwa=*/true);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W32Gfx12, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx12, PackedConversion::HalfRtz);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W32Gfx11, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_i16_f32_w32, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W32Gfx12, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_i16_f32_w32_gfx12, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W32Gfx11, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_u16_f32_w32, PackedConversion::Unsigned);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W32Gfx12, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_u16_f32_w32_gfx12, PackedConversion::Unsigned);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W64Gfx11, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_rtz_f16_f32_w64, PackedConversion::HalfRtz, /*sdwa=*/true);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W64Gfx12, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx12, PackedConversion::HalfRtz);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W64Gfx11, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_i16_f32_w64, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W64Gfx12, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_i16_f32_w64_gfx12, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W64Gfx11, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_u16_f32_w64, PackedConversion::Unsigned);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W64Gfx12, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_u16_f32_w64_gfx12, PackedConversion::Unsigned);
}

// gfx1201 capture: RTZ packing quiets a signaling NaN.
TEST(ConversionPackedGfx12, ExactMatchesHardware) {
  expect_captured_conversion<2, 1>(rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx12, 0, 0x7f800001, 0, 0, 0,
                                   0x7e00);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkI16F32W32Gfx1250, Arithmetic) {
  check_packed<32, false>(rj_arith_v_cvt_pk_i16_f32_w32_gfx1250, PackedConversion::Signed);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkI16F32W64Gfx1250, Arithmetic) {
  check_packed<64, false>(rj_arith_v_cvt_pk_i16_f32_w64_gfx1250, PackedConversion::Signed);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkU16F32W32Gfx1250, Arithmetic) {
  check_packed<32, false>(rj_arith_v_cvt_pk_u16_f32_w32_gfx1250, PackedConversion::Unsigned);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkU16F32W64Gfx1250, Arithmetic) {
  check_packed<64, false>(rj_arith_v_cvt_pk_u16_f32_w64_gfx1250, PackedConversion::Unsigned);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkRtzF16F32W32Gfx1250, Arithmetic) {
  check_packed<32, false>(rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx1250, PackedConversion::HalfRtz);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkRtzF16F32W64Gfx1250, Arithmetic) {
  check_packed<64, false>(rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx1250, PackedConversion::HalfRtz);
}

// gfx942 shares the RDNA3 model and matches MI300X captures.
TEST(ConversionCvtPkRtzF16F32W32Gfx942, Arithmetic) {
  check_packed<32, true>(rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx942, PackedConversion::HalfRtz);
}

TEST(ConversionCvtPkRtzF16F32W64Gfx942, Arithmetic) {
  check_packed<64, true>(rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx942, PackedConversion::HalfRtz);
}

// gfx950 shares the RDNA3/gfx942 model and matches MI350X captures.
TEST(ConversionCvtPkRtzF16F32W32Gfx950, Arithmetic) {
  check_packed<32, true>(rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx950, PackedConversion::HalfRtz);
}

TEST(ConversionCvtPkRtzF16F32W64Gfx950, Arithmetic) {
  check_packed<64, true>(rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx950, PackedConversion::HalfRtz);
}

// gfx10/gfx115x/gfx908/gfx90a share the RDNA3/gfx942 model via loose-only _gfx10 wrappers.
TEST(ConversionCvtPkRtzF16F32W32Gfx10, Arithmetic) {
  check_packed<32, false>(rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx10, PackedConversion::HalfRtz);
}

TEST(ConversionCvtPkRtzF16F32W64Gfx10, Arithmetic) {
  check_packed<64, false>(rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx10, PackedConversion::HalfRtz);
}

// gfx1100 (W7900, mkm2) hw1100-noieee lanes (2026-09-25, MODE.IEEE clear): RTZ packing quiets a
// signaling NaN even with IEEE clear and flushes a tiny result with output denormals flushed;
// integer packing saturates and clamps negatives to zero.
TEST(ConversionPackedGfx11, ExactMatchesHardware) {
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 16>(
      rj_arith_v_cvt_pk_rtz_f16_f32_w32, 0x1f0, 0, 0x7f800001, 0, 0, 0, 0x7e00U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 16>(
      rj_arith_v_cvt_pk_rtz_f16_f32_w32, 0x100, 0, 0x33800000, 0, 0, 0, 0);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 16>(
      rj_arith_v_cvt_pk_rtz_f16_f32_w32, 0x1f0, 0, 0x3f800000, 0x3f800000, 0, 0, 0x3c003c00U);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 32>(
      rj_arith_v_cvt_pk_i16_f32_w32, 0x1f0, 0, 0x7f7fffff, 0, 0, 0, 0x7fffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 32>(
      rj_arith_v_cvt_pk_i16_f32_w32, 0x1f0, 0, 0x7f800000, 0, 0, 0, 0x7fffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<2, 1, 32>(
      rj_arith_v_cvt_pk_u16_f32_w32, 0x1f0, 0, 0xbf800000, 0x00800000, 0, 0, 0);
}

// Legacy SDWA finalizes packed halves without applying a second multiplier.
TEST(ConversionCvtPkRtzF16F32W64Gfx11, SdwaOutput) {
  using namespace rocjitsu::arith::test;
  const uint32_t a[] = {0x3f800000U, 0x80000000U, 0x33800000U, 0x7fc02000U};
  const uint32_t b[] = {0x40000000U, 0x33800000U, 0xb3800000U, 0x7f800000U};
  const uint32_t golden[] = {0x40003c00U, 0, 0, 0x7c007e01U};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      Registers<3, 64> regs;
      for (int lane = 0; lane < 64; ++lane) {
        regs.v[0][lane] = a[lane % 4];
        regs.v[1][lane] = b[lane % 4];
      }
      const auto before = regs;
      uint32_t status = 0x100U;
      const int error = rj_arith_v_cvt_pk_rtz_f16_f32_w64(
          cpu | semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, ~0ULL,
          RJ_ARITH_CVT_SDWA | RJ_ARITH_ALU_OMOD_HALF, regs.v + 2, regs.v, regs.v + 1, &status);
      if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
        EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
        regs.expect_preserved(before, 0, 0);
      } else if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
        EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
        regs.expect_preserved(before, 0, 0);
      } else {
        ASSERT_EQ(error, RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          EXPECT_EQ(regs.v[2][lane], golden[lane % 4]) << cpu << ':' << semantics << ':' << lane;
        }
      }
      EXPECT_EQ(status, 0x100U);
    }
  }
  EXPECT_EQ(rj_arith_v_cvt_pk_rtz_f16_f32_w64(RJ_ARITH_SEMANTICS_LOOSE, 0,
                                              RJ_ARITH_CVT_SDWA | RJ_ARITH_DPP16, nullptr, nullptr,
                                              nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
  EXPECT_EQ(rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx12(RJ_ARITH_SEMANTICS_LOOSE, 0, RJ_ARITH_CVT_SDWA,
                                                    nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
