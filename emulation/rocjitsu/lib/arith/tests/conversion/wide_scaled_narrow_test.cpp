// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide low-precision narrowing: interleaved/packed sources, seeded rounding,
// fields crossing registers, EXEC, special scales and source/destination overlap.
#include "common/contracts.h"
#include "common/conversion64_reference.h"
#include "common/cpu.h"
#include "common/finite_narrow_reference.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "conversion/fp8_narrow_reference.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {
using namespace rocjitsu::arith::test;

// Interpret source encodings independently of production conversion helpers.
float decode(int source, uint32_t raw) {
  uint32_t bits = source == 0   ? raw
                  : source == 1 ? uint32_t(conversion_reencode(uint16_t(raw), 10, 15, 23, 127))
                                : raw << 16;
  return std::bit_cast<float>(bits);
}

// Bind operand shapes once, then share contracts, arithmetic and aliases across
// source formats and rounding modes without multiplying per-instruction fixtures.
template <int Lanes, int Src, int Format, bool Stochastic, bool FullScale = false, typename Call>
void check(Call call) {
  constexpr bool Bf6 = Format == 1;
  constexpr int bits = Format == 2 ? 4 : Format >= 3 ? 8 : 6;
  using Mask = WaveMask<Lanes>;
  using Sources = const uint32_t *const *;
  auto invoke = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, Sources a,
                    Sources b, Sources c) {
    if constexpr (Stochastic || (!FullScale && Src == 0)) {
      return call(flags, exec_mask, mode, d, a, b, c);
    } else {
      return call(flags, exec_mask, mode, d, a, c);
    }
  };
  constexpr int elements = FullScale ? (bits == 6 ? 16 : 8) : 32;
  constexpr int words = elements * bits / 32;
  Registers<56, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 56; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  const int exponents[] = {127, 0, 254, 255, 126, 128, 129, 125};
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[48][lane] = (uint32_t(exponents[lane % 8]) << 23) | (input.v[48][lane] & 0x807fffffU);
    if constexpr (Stochastic) {
      const uint32_t seeds[] = {0, ~0U, 1, 0x80000000U};
      input.v[32][lane] = seeds[lane % 4];
    }
  }
  if constexpr (FullScale) {
    const uint32_t scales[] = {0x3f800000U, 0, 0x80000000U, 0x7f800000U,
                               0x7fc12345U, 1, 0x7f7fffffU, 0xc0200000U};
    for (int lane = 0; lane < Lanes; ++lane) {
      if (lane % 2 == 0) {
        input.v[48][lane] = scales[(lane / 2) % 8];
      }
    }
  }
  // Keep the inherited SR tiny-value cutoff and NaN/Inf policy visible even
  // when arbitrary random values would rarely hit these exact encodings.
  constexpr uint32_t cutoff = Src == 0   ? (Bf6 ? 0x3d000000U : 0x3d800000U)
                              : Src == 1 ? (Bf6 ? 0x2800U : 0x2c00U)
                                         : (Bf6 ? 0x3d00U : 0x3d80U);
  constexpr uint32_t fp4_cutoff = Src == 0 ? 0x3e800000U : Src == 1 ? 0x3400U : 0x3e80U;
  input.v[0][0] = (Format == 2 ? fp4_cutoff : cutoff) - 1;
  input.v[1][0] = Format == 2 ? fp4_cutoff : cutoff;
  input.v[2][0] = Src == 0 ? 0x7fc12345U : Src == 1 ? 0x7e017c00U : 0x7fc17f80U;
  input.v[3][0] = Src == 0 ? 0xff800000U : Src == 1 ? 0xfc007c00U : 0xff807f80U;
  if constexpr (Src == 0 && !FullScale) {
    // Scaling by 2^-127 makes input-denormal handling observable after narrowing.
    input.v[0][1] = 0x00400000U;
    input.v[32][1] = 0x80400000U;
  }
  if constexpr (Stochastic) {
    input.v[32][0] = ~0U;
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        return invoke(flags, ~Mask(0), mode, nullptr, nullptr, nullptr, nullptr);
      },
      0);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 5; ++scenario) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        const int offsets[] = {50, 0, 27, 32, 48};
        int destination = offsets[scenario];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario);
        uint64_t flags =
            cpu | semantics |
            (scenario % 2 ? RJ_ARITH_FP16_OVFL | (Src == 0 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0)
                          : 0);
        auto actual = input, expected = input;
        int error = invoke(flags, exec_mask, 0, actual.v + destination, actual.v, actual.v + 32,
                           actual.v + 48);
        bool exact = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                     semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        ASSERT_EQ(error, exact ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
        if (!exact) {
          for (int lane = 0; lane < Lanes; ++lane) {
            int exponent = (input.v[48][lane] >> 23) & 255;
            if (!((exec_mask >> lane) & 1) || (!FullScale && exponent == 255)) {
              continue;
            }
            for (int word = 0; word < words; ++word) {
              expected.v[destination + word][lane] = 0;
            }
            uint32_t seed = Stochastic ? input.v[32][lane] : 0;
            for (int element = 0; element < elements; ++element) {
              uint32_t raw = Src == 0 ? ((Stochastic || FullScale)
                                             ? input.v[element][lane]
                                             : input.v[(element % 2 ? 32 : 0) + element / 2][lane])
                                      : input.v[element / 2][lane] >> ((element % 2) * 16);
              if (Src == 0 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
                  (raw & 0x7f800000U) == 0) {
                raw &= 0x80000000U;
              }
              float value = FullScale
                                ? decode(Src, raw) / std::bit_cast<float>(input.v[48][lane])
                                : float(double(decode(Src, raw)) / std::ldexp(1.0, exponent - 127));
              uint32_t code;
              if constexpr (Format >= 3) {
                code = fp8_narrow_reference(Format == 4, Stochastic, std::bit_cast<uint32_t>(value),
                                            seed, flags & RJ_ARITH_FP16_OVFL, 950);
              } else {
                code = quantize_finite<(Bf6 ? 3 : 2), (Format == 2 ? 1
                                                       : Bf6       ? 2
                                                                   : 3)>(value, Stochastic, seed);
              }
              for (int bit = 0; bit < bits; ++bit) {
                int offset = element * bits + bit;
                expected.v[destination + offset / 32][lane] |= ((code >> bit) & 1) << (offset % 32);
              }
              seed = (seed << 1) ^ ((seed >> 31) ? 197U : 0U);
            }
          }
        }
        for (int word = 0; word < words; ++word) {
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(actual.v[destination + word][lane], expected.v[destination + word][lane])
                << "word=" << word << ", lane=" << lane;
          }
        }
        actual.expect_preserved(input, destination, exact ? 0 : words);
      }
    }
  }
}
} // namespace

TEST(ScaledNarrowWideF32FP6RNEW32, Arithmetic) {
  check<32, 0, false, false>(rj_arith_v_cvt_scalef32_2xpk16_fp6_f32_w32_gfx950);
}

TEST(ScaledNarrowWideF32FP6RNEW64, Arithmetic) {
  check<64, 0, false, false>(rj_arith_v_cvt_scalef32_2xpk16_fp6_f32_w64_gfx950);
}

TEST(ScaledNarrowWideF32FP6SRW32, Arithmetic) {
  check<32, 0, false, true>(rj_arith_v_cvt_scalef32_sr_pk32_fp6_f32_w32_gfx950);
}

TEST(ScaledNarrowWideF32FP6SRW64, Arithmetic) {
  check<64, 0, false, true>(rj_arith_v_cvt_scalef32_sr_pk32_fp6_f32_w64_gfx950);
}

TEST(ScaledNarrowWideF16FP6RNEW32, Arithmetic) {
  check<32, 1, false, false>(rj_arith_v_cvt_scalef32_pk32_fp6_f16_w32_gfx950);
}

TEST(ScaledNarrowWideF16FP6RNEW64, Arithmetic) {
  check<64, 1, false, false>(rj_arith_v_cvt_scalef32_pk32_fp6_f16_w64_gfx950);
}

TEST(ScaledNarrowWideF16FP6SRW32, Arithmetic) {
  check<32, 1, false, true>(rj_arith_v_cvt_scalef32_sr_pk32_fp6_f16_w32_gfx950);
}

TEST(ScaledNarrowWideF16FP6SRW64, Arithmetic) {
  check<64, 1, false, true>(rj_arith_v_cvt_scalef32_sr_pk32_fp6_f16_w64_gfx950);
}

TEST(ScaledNarrowWideBF16FP6RNEW32, Arithmetic) {
  check<32, 2, false, false>(rj_arith_v_cvt_scalef32_pk32_fp6_bf16_w32_gfx950);
}

TEST(ScaledNarrowWideBF16FP6RNEW64, Arithmetic) {
  check<64, 2, false, false>(rj_arith_v_cvt_scalef32_pk32_fp6_bf16_w64_gfx950);
}

TEST(ScaledNarrowWideBF16FP6SRW32, Arithmetic) {
  check<32, 2, false, true>(rj_arith_v_cvt_scalef32_sr_pk32_fp6_bf16_w32_gfx950);
}

TEST(ScaledNarrowWideBF16FP6SRW64, Arithmetic) {
  check<64, 2, false, true>(rj_arith_v_cvt_scalef32_sr_pk32_fp6_bf16_w64_gfx950);
}

TEST(ScaledNarrowWideF32BF6RNEW32, Arithmetic) {
  check<32, 0, true, false>(rj_arith_v_cvt_scalef32_2xpk16_bf6_f32_w32_gfx950);
}

TEST(ScaledNarrowWideF32BF6RNEW64, Arithmetic) {
  check<64, 0, true, false>(rj_arith_v_cvt_scalef32_2xpk16_bf6_f32_w64_gfx950);
}

TEST(ScaledNarrowWideF32BF6SRW32, Arithmetic) {
  check<32, 0, true, true>(rj_arith_v_cvt_scalef32_sr_pk32_bf6_f32_w32_gfx950);
}

TEST(ScaledNarrowWideF32BF6SRW64, Arithmetic) {
  check<64, 0, true, true>(rj_arith_v_cvt_scalef32_sr_pk32_bf6_f32_w64_gfx950);
}

TEST(ScaledNarrowWideF16BF6RNEW32, Arithmetic) {
  check<32, 1, true, false>(rj_arith_v_cvt_scalef32_pk32_bf6_f16_w32_gfx950);
}

TEST(ScaledNarrowWideF16BF6RNEW64, Arithmetic) {
  check<64, 1, true, false>(rj_arith_v_cvt_scalef32_pk32_bf6_f16_w64_gfx950);
}

TEST(ScaledNarrowWideF16BF6SRW32, Arithmetic) {
  check<32, 1, true, true>(rj_arith_v_cvt_scalef32_sr_pk32_bf6_f16_w32_gfx950);
}

TEST(ScaledNarrowWideF16BF6SRW64, Arithmetic) {
  check<64, 1, true, true>(rj_arith_v_cvt_scalef32_sr_pk32_bf6_f16_w64_gfx950);
}

TEST(ScaledNarrowWideBF16BF6RNEW32, Arithmetic) {
  check<32, 2, true, false>(rj_arith_v_cvt_scalef32_pk32_bf6_bf16_w32_gfx950);
}

TEST(ScaledNarrowWideBF16BF6RNEW64, Arithmetic) {
  check<64, 2, true, false>(rj_arith_v_cvt_scalef32_pk32_bf6_bf16_w64_gfx950);
}

TEST(ScaledNarrowWideBF16BF6SRW32, Arithmetic) {
  check<32, 2, true, true>(rj_arith_v_cvt_scalef32_sr_pk32_bf6_bf16_w32_gfx950);
}

TEST(ScaledNarrowWideBF16BF6SRW64, Arithmetic) {
  check<64, 2, true, true>(rj_arith_v_cvt_scalef32_sr_pk32_bf6_bf16_w64_gfx950);
}

TEST(ScaledNarrowWideF32FP6RNEGfx1250, Arithmetic) {
  check<32, 0, false, false, true>(rj_arith_v_cvt_scalef32_pk16_fp6_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF32FP6SRGfx1250, Arithmetic) {
  check<32, 0, false, true, true>(rj_arith_v_cvt_scalef32_sr_pk16_fp6_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF16FP6RNEGfx1250, Arithmetic) {
  check<32, 1, false, false, true>(rj_arith_v_cvt_scalef32_pk16_fp6_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideF16FP6SRGfx1250, Arithmetic) {
  check<32, 1, false, true, true>(rj_arith_v_cvt_scalef32_sr_pk16_fp6_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16FP6RNEGfx1250, Arithmetic) {
  check<32, 2, false, false, true>(rj_arith_v_cvt_scalef32_pk16_fp6_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16FP6SRGfx1250, Arithmetic) {
  check<32, 2, false, true, true>(rj_arith_v_cvt_scalef32_sr_pk16_fp6_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideF32BF6RNEGfx1250, Arithmetic) {
  check<32, 0, true, false, true>(rj_arith_v_cvt_scalef32_pk16_bf6_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF32BF6SRGfx1250, Arithmetic) {
  check<32, 0, true, true, true>(rj_arith_v_cvt_scalef32_sr_pk16_bf6_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF16BF6RNEGfx1250, Arithmetic) {
  check<32, 1, true, false, true>(rj_arith_v_cvt_scalef32_pk16_bf6_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideF16BF6SRGfx1250, Arithmetic) {
  check<32, 1, true, true, true>(rj_arith_v_cvt_scalef32_sr_pk16_bf6_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16BF6RNEGfx1250, Arithmetic) {
  check<32, 2, true, false, true>(rj_arith_v_cvt_scalef32_pk16_bf6_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16BF6SRGfx1250, Arithmetic) {
  check<32, 2, true, true, true>(rj_arith_v_cvt_scalef32_sr_pk16_bf6_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideF32FP4RNEGfx1250, Arithmetic) {
  check<32, 0, 2, false, true>(rj_arith_v_cvt_scalef32_pk8_fp4_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF32FP4SRGfx1250, Arithmetic) {
  check<32, 0, 2, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_fp4_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF16FP4RNEGfx1250, Arithmetic) {
  check<32, 1, 2, false, true>(rj_arith_v_cvt_scalef32_pk8_fp4_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideF16FP4SRGfx1250, Arithmetic) {
  check<32, 1, 2, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_fp4_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16FP4RNEGfx1250, Arithmetic) {
  check<32, 2, 2, false, true>(rj_arith_v_cvt_scalef32_pk8_fp4_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16FP4SRGfx1250, Arithmetic) {
  check<32, 2, 2, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_fp4_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideF32FP8RNEGfx1250, Arithmetic) {
  check<32, 0, 3, false, true>(rj_arith_v_cvt_scalef32_pk8_fp8_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF32FP8SRGfx1250, Arithmetic) {
  check<32, 0, 3, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_fp8_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF16FP8RNEGfx1250, Arithmetic) {
  check<32, 1, 3, false, true>(rj_arith_v_cvt_scalef32_pk8_fp8_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideF16FP8SRGfx1250, Arithmetic) {
  check<32, 1, 3, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_fp8_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16FP8RNEGfx1250, Arithmetic) {
  check<32, 2, 3, false, true>(rj_arith_v_cvt_scalef32_pk8_fp8_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16FP8SRGfx1250, Arithmetic) {
  check<32, 2, 3, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_fp8_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideF32BF8RNEGfx1250, Arithmetic) {
  check<32, 0, 4, false, true>(rj_arith_v_cvt_scalef32_pk8_bf8_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF32BF8SRGfx1250, Arithmetic) {
  check<32, 0, 4, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_bf8_f32_w32_gfx1250);
}

TEST(ScaledNarrowWideF16BF8RNEGfx1250, Arithmetic) {
  check<32, 1, 4, false, true>(rj_arith_v_cvt_scalef32_pk8_bf8_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideF16BF8SRGfx1250, Arithmetic) {
  check<32, 1, 4, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_bf8_f16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16BF8RNEGfx1250, Arithmetic) {
  check<32, 2, 4, false, true>(rj_arith_v_cvt_scalef32_pk8_bf8_bf16_w32_gfx1250);
}

TEST(ScaledNarrowWideBF16BF8SRGfx1250, Arithmetic) {
  check<32, 2, 4, true, true>(rj_arith_v_cvt_scalef32_sr_pk8_bf8_bf16_w32_gfx1250);
}
