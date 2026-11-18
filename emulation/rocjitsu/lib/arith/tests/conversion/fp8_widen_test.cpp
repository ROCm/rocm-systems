// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP8/BF8 widening, including packed source selections.

#include "common/fp8_conversion_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Decode the inherited unsigned E5M3 format without production conversion helpers.
uint32_t e5m3_reference(uint8_t code) {
  if (code == 255) {
    return 0x7fc00000U;
  }
  int exponent = code >> 3;
  unsigned significand = (code & 7) + (exponent ? 8 : 0);
  return uint32_t(conversion_encode(false, significand, (exponent ? exponent : 1) - 18, 23, 127));
}

// Check FP8/BF8 widening and packed source selection for the supplied entry point.
template <int Lanes, bool Packed, bool Bf8, int Arch = 12, bool Half = false, typename Call>
void check_fp8_widen(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  // Keep every class in both selected bytes, including NaNs and signed zeros.
  const uint32_t edge[] = {0x807f7c00U, 0xfffe7e01U, 0x01807fffU, 0x7c007e80U};
  for (int lane = 0; lane < 4; ++lane) {
    input.v[0][lane] = edge[lane];
  }
  check_conversion<Lanes, 1, 1, Packed && !Half ? 2 : 1, !Packed>(
      call, input,
      [](uint64_t a, uint32_t, uint32_t, uint32_t old, uint64_t flags, uint32_t mode) {
        int shift = Packed ? (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0) : 8 * ((mode >> 16) & 3);
        auto decode = [&](uint8_t code) {
          uint32_t raw =
              Arch == 1250 && (mode & RJ_ARITH_ALU_CLAMP)
                  ? e5m3_reference(code)
                  : fp8_conversion_reference(Bf8, code, Arch == 942, Arch == 950 || Arch == 1250);
          if constexpr (Half) {
            uint32_t result = uint32_t(conversion_reencode(raw, 23, 127, 10, 15));
            if ((flags & RJ_ARITH_FP16_OVFL) && (raw & 0x7fffffff) < 0x7f800000 &&
                (result & 0x7fff) == 0x7c00) {
              --result;
            }
            return result;
          }
          return raw;
        };
        uint64_t result = decode(uint8_t(a >> shift));
        if constexpr (Packed) {
          result |= uint64_t(decode(uint8_t(a >> (shift + 8)))) << (Half ? 16 : 32);
        } else if constexpr (Half) {
          int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
          result = (old & ~(0xffffU << destination_shift)) | (result << destination_shift);
        }
        return result;
      },
      conversion_bits,
      (Packed ? RJ_ARITH_ALU_HIGH_A : RJ_ARITH_CVT_BYTE_3) |
          (Arch == 1250 ? RJ_ARITH_ALU_CLAMP : 0) |
          (Half ? RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK |
                      RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP
                : 0),
      true);
}

} // namespace

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Fp8W32, Arithmetic) {
  check_fp8_widen<32, false, false>(rj_arith_v_cvt_f32_fp8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Bf8W32, Arithmetic) {
  check_fp8_widen<32, false, true>(rj_arith_v_cvt_f32_bf8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Fp8W32, Arithmetic) {
  check_fp8_widen<32, true, false>(rj_arith_v_cvt_pk_f32_fp8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Bf8W32, Arithmetic) {
  check_fp8_widen<32, true, true>(rj_arith_v_cvt_pk_f32_bf8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Fp8W64, Arithmetic) {
  check_fp8_widen<64, false, false>(rj_arith_v_cvt_f32_fp8_w64);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Bf8W64, Arithmetic) {
  check_fp8_widen<64, false, true>(rj_arith_v_cvt_f32_bf8_w64);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Fp8W64, Arithmetic) {
  check_fp8_widen<64, true, false>(rj_arith_v_cvt_pk_f32_fp8_w64);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Bf8W64, Arithmetic) {
  check_fp8_widen<64, true, true>(rj_arith_v_cvt_pk_f32_bf8_w64);
}

TEST(ConversionCvtF32Fp8W32Gfx942, Arithmetic) {
  check_fp8_widen<32, false, false, 942>(rj_arith_v_cvt_f32_fp8_w32_gfx942);
}

TEST(ConversionCvtF32Fp8W64Gfx942, Arithmetic) {
  check_fp8_widen<64, false, false, 942>(rj_arith_v_cvt_f32_fp8_w64_gfx942);
}

TEST(ConversionCvtF32Bf8W32Gfx942, Arithmetic) {
  check_fp8_widen<32, false, true, 942>(rj_arith_v_cvt_f32_bf8_w32_gfx942);
}

TEST(ConversionCvtF32Bf8W64Gfx942, Arithmetic) {
  check_fp8_widen<64, false, true, 942>(rj_arith_v_cvt_f32_bf8_w64_gfx942);
}

TEST(ConversionCvtPkF32Fp8W32Gfx942, Arithmetic) {
  check_fp8_widen<32, true, false, 942>(rj_arith_v_cvt_pk_f32_fp8_w32_gfx942);
}

TEST(ConversionCvtPkF32Fp8W64Gfx942, Arithmetic) {
  check_fp8_widen<64, true, false, 942>(rj_arith_v_cvt_pk_f32_fp8_w64_gfx942);
}

TEST(ConversionCvtPkF32Bf8W32Gfx942, Arithmetic) {
  check_fp8_widen<32, true, true, 942>(rj_arith_v_cvt_pk_f32_bf8_w32_gfx942);
}

TEST(ConversionCvtPkF32Bf8W64Gfx942, Arithmetic) {
  check_fp8_widen<64, true, true, 942>(rj_arith_v_cvt_pk_f32_bf8_w64_gfx942);
}

TEST(ConversionCvtF32Fp8W32Gfx950, Arithmetic) {
  check_fp8_widen<32, false, false, 950>(rj_arith_v_cvt_f32_fp8_w32_gfx950);
}

TEST(ConversionCvtF32Fp8W64Gfx950, Arithmetic) {
  check_fp8_widen<64, false, false, 950>(rj_arith_v_cvt_f32_fp8_w64_gfx950);
}

TEST(ConversionCvtF32Bf8W32Gfx950, Arithmetic) {
  check_fp8_widen<32, false, true, 950>(rj_arith_v_cvt_f32_bf8_w32_gfx950);
}

TEST(ConversionCvtF32Bf8W64Gfx950, Arithmetic) {
  check_fp8_widen<64, false, true, 950>(rj_arith_v_cvt_f32_bf8_w64_gfx950);
}

TEST(ConversionCvtPkF32Fp8W32Gfx950, Arithmetic) {
  check_fp8_widen<32, true, false, 950>(rj_arith_v_cvt_pk_f32_fp8_w32_gfx950);
}

TEST(ConversionCvtPkF32Fp8W64Gfx950, Arithmetic) {
  check_fp8_widen<64, true, false, 950>(rj_arith_v_cvt_pk_f32_fp8_w64_gfx950);
}

TEST(ConversionCvtPkF32Bf8W32Gfx950, Arithmetic) {
  check_fp8_widen<32, true, true, 950>(rj_arith_v_cvt_pk_f32_bf8_w32_gfx950);
}

TEST(ConversionCvtPkF32Bf8W64Gfx950, Arithmetic) {
  check_fp8_widen<64, true, true, 950>(rj_arith_v_cvt_pk_f32_bf8_w64_gfx950);
}

// The tiny format domain also checks the inherited model for every byte encoding.
TEST(ConversionCdnaFp8, AllCodes) {
  using Call = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *, const uint32_t *const *);
  const Call calls[] = {rj_arith_v_cvt_f32_fp8_w64_gfx942, rj_arith_v_cvt_f32_bf8_w64_gfx942,
                        rj_arith_v_cvt_f32_fp8_w64_gfx950, rj_arith_v_cvt_f32_bf8_w64_gfx950,
                        rj_arith_v_cvt_f32_fp8_w64_gfx1250};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int operation = 0; operation < 5; ++operation) {
      for (int start = 0; start < 256; start += 64) {
        uint32_t source[64], result[64];
        for (int lane = 0; lane < 64; ++lane) {
          source[lane] = start + lane;
        }
        uint32_t *d = result;
        const uint32_t *a = source;
        uint64_t mode = operation == 4 ? RJ_ARITH_ALU_CLAMP : 0;
        ASSERT_EQ(calls[operation](cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, ~0ULL, mode, &d, &a),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          uint8_t code = uint8_t(start + lane);
          uint32_t expected =
              operation == 4
                  ? e5m3_reference(code)
                  : fp8_conversion_reference(operation & 1, code, operation < 2, operation >= 2);
          EXPECT_EQ(result[lane], expected);
        }
      }
    }
  }
}

TEST(ConversionCvtF32Fp8W32Gfx1250, Arithmetic) {
  check_fp8_widen<32, false, false, 1250>(rj_arith_v_cvt_f32_fp8_w32_gfx1250);
}

TEST(ConversionCvtF32Fp8W64Gfx1250, Arithmetic) {
  check_fp8_widen<64, false, false, 1250>(rj_arith_v_cvt_f32_fp8_w64_gfx1250);
}

TEST(ConversionCvtF16Fp8W32, Arithmetic) {
  check_fp8_widen<32, false, false, 1250, true>(rj_arith_v_cvt_f16_fp8_w32);
}

TEST(ConversionCvtF16Fp8W64, Arithmetic) {
  check_fp8_widen<64, false, false, 1250, true>(rj_arith_v_cvt_f16_fp8_w64);
}

TEST(ConversionCvtPkF16Fp8W32, Arithmetic) {
  check_fp8_widen<32, true, false, 950, true>(rj_arith_v_cvt_pk_f16_fp8_w32);
}

TEST(ConversionCvtPkF16Fp8W64, Arithmetic) {
  check_fp8_widen<64, true, false, 950, true>(rj_arith_v_cvt_pk_f16_fp8_w64);
}

TEST(ConversionCvtF16Bf8W32, Arithmetic) {
  check_fp8_widen<32, false, true, 950, true>(rj_arith_v_cvt_f16_bf8_w32);
}

TEST(ConversionCvtF16Bf8W64, Arithmetic) {
  check_fp8_widen<64, false, true, 950, true>(rj_arith_v_cvt_f16_bf8_w64);
}

TEST(ConversionCvtPkF16Bf8W32, Arithmetic) {
  check_fp8_widen<32, true, true, 950, true>(rj_arith_v_cvt_pk_f16_bf8_w32);
}

TEST(ConversionCvtPkF16Bf8W64, Arithmetic) {
  check_fp8_widen<64, true, true, 950, true>(rj_arith_v_cvt_pk_f16_bf8_w64);
}
