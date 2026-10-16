// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for byte/nibble widening and selected-byte packing.

#include "conversion/byte_conversion_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Check selected-byte conversion and its gfx-specific modifiers for the supplied entry point.
template <int Lanes, int Byte, bool Gfx12, class Call> void check_byte(Call call) {
  check_conversion<Lanes, 1, 1, 1, true>(
      call, conversion_inputs<Lanes>(32),
      [](uint64_t a, uint32_t, uint32_t, uint32_t, uint64_t flags, uint32_t mode) {
        mode = conversion_effective_mode(flags, mode, Gfx12);
        if constexpr (Byte < 0) {
          return nibble_offset_reference(uint32_t(a), mode);
        } else {
          return byte_conversion_reference(Byte, uint32_t(a), mode);
        }
      },
      conversion_bits, RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP, true,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

// Check selected-byte packing and preservation of untouched bytes for the supplied entry point.
template <int Lanes, class Call> void check_byte_pack(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  // Ties, saturation around 255, and all four insertion positions.
  const uint32_t edges[] = {0x3f000000, 0x3fc00000, 0x40200000, 0x437e8000, 0x437f0000, 0x437f8000};
  for (int lane = 0; lane < 6; ++lane) {
    input.v[0][lane] = edges[lane];
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[2][lane] = uint32_t(lane);
  }
  check_conversion<Lanes, 3, 1, 1, true>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t c, uint32_t, uint64_t, uint32_t mode) {
        return byte_pack_reference(uint32_t(a), b, c, mode);
      },
      conversion_bits, RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP, false,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W32, Arithmetic) {
  check_byte<32, -1, false>(rj_arith_v_cvt_off_f32_i4_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W32Gfx12, Arithmetic) {
  check_byte<32, -1, true>(rj_arith_v_cvt_off_f32_i4_w32_gfx12);
}

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W32, Arithmetic) { check_byte_pack<32>(rj_arith_v_cvt_pk_u8_f32_w32); }

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W32Gfx12, Arithmetic) {
  check_byte_pack<32>(rj_arith_v_cvt_pk_u8_f32_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W32, Arithmetic) {
  check_byte<32, 0, false>(rj_arith_v_cvt_f32_ubyte0_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W32Gfx12, Arithmetic) {
  check_byte<32, 0, true>(rj_arith_v_cvt_f32_ubyte0_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W32, Arithmetic) {
  check_byte<32, 1, false>(rj_arith_v_cvt_f32_ubyte1_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W32Gfx12, Arithmetic) {
  check_byte<32, 1, true>(rj_arith_v_cvt_f32_ubyte1_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W32, Arithmetic) {
  check_byte<32, 2, false>(rj_arith_v_cvt_f32_ubyte2_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W32Gfx12, Arithmetic) {
  check_byte<32, 2, true>(rj_arith_v_cvt_f32_ubyte2_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W32, Arithmetic) {
  check_byte<32, 3, false>(rj_arith_v_cvt_f32_ubyte3_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W32Gfx12, Arithmetic) {
  check_byte<32, 3, true>(rj_arith_v_cvt_f32_ubyte3_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W64, Arithmetic) {
  check_byte<64, 0, false>(rj_arith_v_cvt_f32_ubyte0_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W64Gfx12, Arithmetic) {
  check_byte<64, 0, true>(rj_arith_v_cvt_f32_ubyte0_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W64, Arithmetic) {
  check_byte<64, 1, false>(rj_arith_v_cvt_f32_ubyte1_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W64Gfx12, Arithmetic) {
  check_byte<64, 1, true>(rj_arith_v_cvt_f32_ubyte1_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W64, Arithmetic) {
  check_byte<64, 2, false>(rj_arith_v_cvt_f32_ubyte2_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W64Gfx12, Arithmetic) {
  check_byte<64, 2, true>(rj_arith_v_cvt_f32_ubyte2_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W64, Arithmetic) {
  check_byte<64, 3, false>(rj_arith_v_cvt_f32_ubyte3_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W64Gfx12, Arithmetic) {
  check_byte<64, 3, true>(rj_arith_v_cvt_f32_ubyte3_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W64, Arithmetic) {
  check_byte<64, -1, false>(rj_arith_v_cvt_off_f32_i4_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W64Gfx12, Arithmetic) {
  check_byte<64, -1, true>(rj_arith_v_cvt_off_f32_i4_w64_gfx12);
}

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W64, Arithmetic) { check_byte_pack<64>(rj_arith_v_cvt_pk_u8_f32_w64); }

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W64Gfx12, Arithmetic) {
  check_byte_pack<64>(rj_arith_v_cvt_pk_u8_f32_w64_gfx12);
}
