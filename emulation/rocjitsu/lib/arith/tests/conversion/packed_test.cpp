// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for pairwise FP32 conversion into packed FP16/BF16 results.

#include "common/floating_point.h"
#include "conversion/helpers.h"
#include "conversion/packed_conversion_reference.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

enum class PackedConversion { HalfRtz, Signed, Unsigned };

// Check paired 16-bit conversions and packed result ordering for the supplied entry point.
template <int Lanes, class Call> void check_packed(Call call, PackedConversion operation) {
  int op = int(operation);
  uint64_t modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_B |
                       RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP |
                       (op == 0 ? RJ_ARITH_ALU_OMOD_HALF : 0);
  check_conversion<Lanes, 2, 1, 1, true>(
      call, conversion_inputs<Lanes>(32),
      [=](uint64_t a, uint32_t b, uint32_t, uint32_t, uint64_t, uint32_t mode) {
        return packed_conversion_reference(op, uint32_t(a), b, mode);
      },
      [=](uint64_t actual, uint64_t expected, uint64_t flags, uint32_t mode) {
        if (op) {
          conversion_bits(actual, expected, flags, mode);
        } else {
          for (int shift : {0, 16}) {
            float a = f16_value(uint16_t(actual >> shift)),
                  e = f16_value(uint16_t(expected >> shift));
            conversion_float(bit_cast<uint32_t>(a), bit_cast<uint32_t>(e), 32);
          }
        }
      },
      modifiers, false,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W32, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_rtz_f16_f32_w32, PackedConversion::HalfRtz);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W32Gfx12, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_rtz_f16_f32_w32_gfx12, PackedConversion::HalfRtz);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W32, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_i16_f32_w32, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W32Gfx12, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_i16_f32_w32_gfx12, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W32, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_u16_f32_w32, PackedConversion::Unsigned);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W32Gfx12, Arithmetic) {
  check_packed<32>(rj_arith_v_cvt_pk_u16_f32_w32_gfx12, PackedConversion::Unsigned);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W64, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_rtz_f16_f32_w64, PackedConversion::HalfRtz);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkRtzF16F32W64Gfx12, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_rtz_f16_f32_w64_gfx12, PackedConversion::HalfRtz);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W64, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_i16_f32_w64, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkI16F32W64Gfx12, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_i16_f32_w64_gfx12, PackedConversion::Signed);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W64, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_u16_f32_w64, PackedConversion::Unsigned);
}

// Check paired 16-bit conversions and packed result ordering.
TEST(ConversionCvtPkU16F32W64Gfx12, Arithmetic) {
  check_packed<64>(rj_arith_v_cvt_pk_u16_f32_w64_gfx12, PackedConversion::Unsigned);
}
