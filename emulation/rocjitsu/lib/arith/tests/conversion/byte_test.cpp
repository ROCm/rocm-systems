// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for byte/nibble widening and selected-byte packing.

#include "conversion/byte_conversion_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <bit>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Check selected-byte conversion and its gfx-specific modifiers for the supplied entry point.
template <int Lanes, int Byte, bool Gfx12, typename Call> void check_byte(Call call) {
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
// Exact is false for PKACCUM and for entry points that stay loose-only until their captures
// are reviewed.
template <int Lanes, bool Accumulate = false, bool Exact = !Accumulate, typename Call>
void check_byte_pack(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  // Ties, saturation around 255, and all four insertion positions.
  const uint32_t edges[] = {0x3f000000, 0x3fc00000, 0x40200000, 0x437e8000, 0x437f0000, 0x437f8000};
  for (int lane = 0; lane < 6; ++lane) {
    input.v[0][lane] = edges[lane];
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[2][lane] = uint32_t(lane);
  }
  check_conversion<Lanes, 3, 1, 1, !Accumulate>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t c, uint32_t, uint64_t, uint32_t mode) {
        if constexpr (Accumulate) {
          const float value = std::bit_cast<float>(uint32_t(a));
          const uint32_t byte = !(value > 0) ? 0 : value >= 255 ? 255 : uint32_t(value);
          const int shift = (b & 3) * 8;
          return (c & ~(0xffU << shift)) | (byte << shift);
        } else {
          return byte_pack_reference(uint32_t(a), b, c, mode);
        }
      },
      conversion_bits,
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP |
          (Accumulate ? RJ_ARITH_ALU_OMOD_MASK : 0),
      Exact,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W32Gfx11, Arithmetic) {
  check_byte<32, -1, false>(rj_arith_v_cvt_off_f32_i4_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W32Gfx12, Arithmetic) {
  check_byte<32, -1, true>(rj_arith_v_cvt_off_f32_i4_w32_gfx12);
}

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W32Gfx11, Arithmetic) {
  check_byte_pack<32>(rj_arith_v_cvt_pk_u8_f32_w32);
}

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W32Gfx12, Arithmetic) {
  check_byte_pack<32>(rj_arith_v_cvt_pk_u8_f32_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W32Gfx11, Arithmetic) {
  check_byte<32, 0, false>(rj_arith_v_cvt_f32_ubyte0_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W32Gfx12, Arithmetic) {
  check_byte<32, 0, true>(rj_arith_v_cvt_f32_ubyte0_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W32Gfx11, Arithmetic) {
  check_byte<32, 1, false>(rj_arith_v_cvt_f32_ubyte1_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W32Gfx12, Arithmetic) {
  check_byte<32, 1, true>(rj_arith_v_cvt_f32_ubyte1_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W32Gfx11, Arithmetic) {
  check_byte<32, 2, false>(rj_arith_v_cvt_f32_ubyte2_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W32Gfx12, Arithmetic) {
  check_byte<32, 2, true>(rj_arith_v_cvt_f32_ubyte2_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W32Gfx11, Arithmetic) {
  check_byte<32, 3, false>(rj_arith_v_cvt_f32_ubyte3_w32);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W32Gfx12, Arithmetic) {
  check_byte<32, 3, true>(rj_arith_v_cvt_f32_ubyte3_w32_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W64Gfx11, Arithmetic) {
  check_byte<64, 0, false>(rj_arith_v_cvt_f32_ubyte0_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte0W64Gfx12, Arithmetic) {
  check_byte<64, 0, true>(rj_arith_v_cvt_f32_ubyte0_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W64Gfx11, Arithmetic) {
  check_byte<64, 1, false>(rj_arith_v_cvt_f32_ubyte1_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte1W64Gfx12, Arithmetic) {
  check_byte<64, 1, true>(rj_arith_v_cvt_f32_ubyte1_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W64Gfx11, Arithmetic) {
  check_byte<64, 2, false>(rj_arith_v_cvt_f32_ubyte2_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte2W64Gfx12, Arithmetic) {
  check_byte<64, 2, true>(rj_arith_v_cvt_f32_ubyte2_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W64Gfx11, Arithmetic) {
  check_byte<64, 3, false>(rj_arith_v_cvt_f32_ubyte3_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtF32Ubyte3W64Gfx12, Arithmetic) {
  check_byte<64, 3, true>(rj_arith_v_cvt_f32_ubyte3_w64_gfx12);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W64Gfx11, Arithmetic) {
  check_byte<64, -1, false>(rj_arith_v_cvt_off_f32_i4_w64);
}

// Check selected-byte conversion and its gfx-specific modifiers.
TEST(ConversionCvtOffF32I4W64Gfx12, Arithmetic) {
  check_byte<64, -1, true>(rj_arith_v_cvt_off_f32_i4_w64_gfx12);
}

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W64Gfx11, Arithmetic) {
  check_byte_pack<64>(rj_arith_v_cvt_pk_u8_f32_w64);
}

// Check selected-byte packing and preservation of untouched bytes.
TEST(ConversionCvtPkU8F32W64Gfx12, Arithmetic) {
  check_byte_pack<64>(rj_arith_v_cvt_pk_u8_f32_w64_gfx12);
}

TEST(ConversionCvtPkaccumU8F32W32, Arithmetic) {
  check_byte_pack<32, true>(rj_arith_v_cvt_pkaccum_u8_f32_w32);
}
TEST(ConversionCvtPkaccumU8F32W64, Arithmetic) {
  check_byte_pack<64, true>(rj_arith_v_cvt_pkaccum_u8_f32_w64);
}

// gfx1201 captures: byte packing rounds in the guest FP32 mode before inserting at B's byte.
TEST(ConversionBytePackGfx12, ExactMatchesHardware) {
  expect_captured_conversion<3, 1>(rj_arith_v_cvt_pk_u8_f32_w32_gfx12, 0, 0x3fc00000, 0, 0, 0, 2);
  expect_captured_conversion<3, 1>(rj_arith_v_cvt_pk_u8_f32_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO,
                                   0x3fc00000, 0, 0, 0, 1);
  expect_captured_conversion<3, 1>(rj_arith_v_cvt_pk_u8_f32_w32_gfx12, 0, 0x3fc00000, 1, 0, 0,
                                   0x200);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkU8F32W32Gfx1250, Arithmetic) {
  check_byte_pack<32, false, false>(rj_arith_v_cvt_pk_u8_f32_w32_gfx1250);
}

// gfx1250 inherits the gfx12 model but stays loose-only pending gfx1250 captures.
TEST(ConversionCvtPkU8F32W64Gfx1250, Arithmetic) {
  check_byte_pack<64, false, false>(rj_arith_v_cvt_pk_u8_f32_w64_gfx1250);
}

// gfx942 shares the RDNA3 numeric model and matches MI300X captures.
TEST(ConversionCvtPkU8F32W32Gfx942, Arithmetic) {
  check_byte_pack<32, false, true>(rj_arith_v_cvt_pk_u8_f32_w32_gfx942);
}

TEST(ConversionCvtPkU8F32W64Gfx942, Arithmetic) {
  check_byte_pack<64, false, true>(rj_arith_v_cvt_pk_u8_f32_w64_gfx942);
}

// gfx950 inherits the RDNA3/gfx942 model but stays loose-only until gfx950 captures are reviewed.
TEST(ConversionCvtPkU8F32W32Gfx950, Arithmetic) {
  check_byte_pack<32, false, false>(rj_arith_v_cvt_pk_u8_f32_w32_gfx950);
}

TEST(ConversionCvtPkU8F32W64Gfx950, Arithmetic) {
  check_byte_pack<64, false, false>(rj_arith_v_cvt_pk_u8_f32_w64_gfx950);
}

// MI300X (2026-11-24) captures: unmodified V_CVT_PKACCUM_U8_F32 honors guest FP32 rounding and
// input-denormal flushing and accepts mode 2; VOP3 modifiers remain loose-only.
TEST(ConversionCvtPkaccumU8F32Gfx942, ExactAndModifiers) {
  expect_captured_conversion<3, 1>(rj_arith_v_cvt_pkaccum_u8_f32_w32_gfx942, 0, 0x3fc00000U, 0,
                                   0xaabbccddU, 0, 0xaabbcc02U);
  expect_captured_conversion<3, 1>(rj_arith_v_cvt_pkaccum_u8_f32_w32_gfx942, RJ_ARITH_FP_ROUND_ZERO,
                                   0x3fc00000U, 0, 0xaabbccddU, 0, 0xaabbcc01U);
  expect_captured_conversion<3, 1>(rj_arith_v_cvt_pkaccum_u8_f32_w32_gfx942, RJ_ARITH_FP_ROUND_UP,
                                   0x00000001U, 2, 0xaabbccddU, 0, 0xaa01ccddU);
  expect_captured_conversion<3, 1>(rj_arith_v_cvt_pkaccum_u8_f32_w32_gfx942,
                                   RJ_ARITH_FP_ROUND_UP | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS,
                                   0x00000001U, 2, 0xaabbccddU, 0, 0xaa00ccddU);

  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool wave64 : {false, true}) {
      uint32_t av[64]{}, bv[64]{}, cv[64]{}, dv[64]{};
      av[0] = 0x3fc00000U;
      bv[0] = 1;
      cv[0] = 0x11223344U;
      const uint32_t *a = av, *b = bv, *c = cv;
      uint32_t *d = dv;
      uint32_t status = 0x55;
      auto invoke = [&](uint64_t flags, uint64_t modifiers) {
        return wave64 ? rj_arith_v_cvt_pkaccum_u8_f32_w64_gfx942(flags, 1ULL, modifiers, &d, &a, &b,
                                                                 &c, &status)
                      : rj_arith_v_cvt_pkaccum_u8_f32_w32_gfx942(flags, 1U, modifiers, &d, &a, &b,
                                                                 &c, &status);
      };
      dv[0] = 0xdeadbeefU;
      EXPECT_EQ(invoke(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0),
                RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
      EXPECT_EQ(dv[0], 0xdeadbeefU);
      EXPECT_EQ(invoke(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(dv[0], 0x11220244U);

      dv[0] = 0xdeadbeefU;
      EXPECT_EQ(
          invoke(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_ALU_CLAMP),
          RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(dv[0], 0xdeadbeefU);
      EXPECT_EQ(invoke(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_ALU_CLAMP),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(dv[0], 0x11220244U);

      dv[0] = 0xdeadbeefU;
      EXPECT_EQ(invoke(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_ALU_ABS_B),
                RJ_ARITH_ERROR_INVALID_FLAGS);
      EXPECT_EQ(dv[0], 0xdeadbeefU);
      EXPECT_EQ(status, 0x55U);
    }
  }
}

// gfx1100 (W7900, mkm2) hw1100-noieee lanes (2026-09-25, MODE.IEEE clear): byte packing
// saturates to 255 and rounds 1.5 to 2 under the FP32 guest rounding of these words (nearest).
TEST(ConversionBytePackGfx11, ExactMatchesHardware) {
  rocjitsu::arith::test::expect_captured_conversion_gfx11<3, 1, 32>(
      rj_arith_v_cvt_pk_u8_f32_w32, 0x1f0, 0, 0x7f7fffff, 0, 0, 0, 0xffU);
  rocjitsu::arith::test::expect_captured_conversion_gfx11<3, 1, 32>(
      rj_arith_v_cvt_pk_u8_f32_w32, 0x150, 0, 0x3fc00000, 0, 0, 0, 2);
}
