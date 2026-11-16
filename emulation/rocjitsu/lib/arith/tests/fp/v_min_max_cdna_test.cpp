// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA legacy MIN/MAX: inherited two-NaN and IEEE signaling priority, plus
// ordinary selection, aliased VGPRs and inactive lanes.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/selection_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace {

// Check all three precisions and both wave sizes against selected encodings, not host NaN
// arithmetic.
template <int Width, int Lanes, bool Packed = false, typename Call>
void check(Call call, bool maximum) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr int Words = (Width + 31) / 32;
  constexpr uint64_t Sign = Width == 16   ? 0x8000ULL
                            : Width == 32 ? 0x80000000ULL
                                          : 0x8000000000000000ULL;
  constexpr uint64_t Infinity = Width == 16   ? 0x7c00ULL
                                : Width == 32 ? 0x7f800000ULL
                                              : 0x7ff0000000000000ULL;
  constexpr uint64_t Quiet = Width == 16   ? 0x0200ULL
                             : Width == 32 ? 0x00400000ULL
                                           : 0x0008000000000000ULL;
  if constexpr (Width == 16) {
    check_selection<16, Lanes, 2, Packed>(
        call, [=](double x, double y) { return select_fp(x, y, maximum, false); });
  }
  if constexpr (Lanes == 32 && !Packed) {
    check_selection_omod<Width, 2>(call);
    check_selection_omod_nan<Width, 2>(call);
  }
  Registers<6, Lanes> input;
  uint64_t a[Lanes], b[Lanes];
  std::minstd_rand random;
  for (int lane = 0; lane < Lanes; ++lane) {
    // High engine bits give positive finite encodings whose integer ordering is numeric ordering.
    if constexpr (Width == 16) {
      a[lane] = 0x3800U | (random() >> 21);
      b[lane] = 0x3800U | (random() >> 21);
    } else {
      a[lane] = (Width == 32 ? 0x3f000000ULL : 0x3fe0000000000000ULL) | (random() >> 8);
      b[lane] = (Width == 32 ? 0x3f000000ULL : 0x3fe0000000000000ULL) | (random() >> 8);
    }
  }
  a[0] = Infinity | Quiet | 1;
  b[0] = Infinity | Quiet | 2;
  a[1] = Infinity | 1;
  b[1] = Infinity | Quiet | 2;
  a[2] = Infinity | Quiet | 1;
  b[2] = Infinity | 2;
  a[3] = Infinity | 1;
  b[3] = Infinity | 2;
  a[4] = Sign;
  b[4] = 0;
  a[5] = Infinity;
  b[5] = Infinity | Sign;
  for (int lane = 0; lane < Lanes; ++lane) {
    for (int word = 0; word < Words; ++word) {
      input.v[word][lane] =
          Width == 16 ? uint32_t(a[lane]) * 0x10001U : uint32_t(a[lane] >> (32 * word));
      input.v[Words + word][lane] =
          Width == 16 ? uint32_t(b[lane]) * 0x10001U : uint32_t(b[lane] >> (32 * word));
    }
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (bool ieee : {false, true}) {
        for (int aliasing_offset : {0, 1, 4}) {
          for (Mask exec_mask : {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL)}) {
            auto r = input;
            uint32_t status = 0x80000000U;
            const uint64_t flags = cpu | semantics | (ieee ? RJ_ARITH_FP_IEEE_MODE : 0);
            int error = call(flags, exec_mask, 0, r.v + aliasing_offset, r.v, r.v + Words, &status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              continue;
            }
            ASSERT_EQ(error, RJ_ARITH_SUCCESS);
            EXPECT_EQ(status, 0x80000000U);
            r.expect_preserved(input, aliasing_offset, Words);
            for (int lane = 0; lane < Lanes; ++lane) {
              uint64_t expected;
              if (lane < 4) {
                // IEEE prioritizes signaling A, then signaling B. Otherwise take B unchanged.
                expected = ieee && !(a[lane] & Quiet)   ? a[lane] | Quiet
                           : ieee && !(b[lane] & Quiet) ? b[lane] | Quiet
                                                        : b[lane];
              } else if (lane == 4) {
                expected = maximum ? 0 : Sign;
              } else if (lane == 5) {
                expected = maximum ? Infinity : Infinity | Sign;
              } else {
                expected = (maximum ? a[lane] > b[lane] : a[lane] < b[lane]) ? a[lane] : b[lane];
              }
              if constexpr (Width == 16) {
                expected |= Packed ? expected << 16 : input.v[aliasing_offset][lane] & 0xffff0000U;
              }
              for (int word = 0; word < Words; ++word) {
                EXPECT_EQ(r.v[aliasing_offset + word][lane],
                          (exec_mask >> lane) & 1 ? uint32_t(expected >> (32 * word))
                                                  : input.v[aliasing_offset + word][lane]);
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

TEST(FpCdnaSelection, F32Wave32) {
  check<32, 32>(rj_arith_v_min_f32_w32_gfx942, false);
  check<32, 32>(rj_arith_v_max_f32_w32_gfx942, true);
}
TEST(FpCdnaSelection, F32Wave64) {
  check<32, 64>(rj_arith_v_min_f32_w64_gfx942, false);
  check<32, 64>(rj_arith_v_max_f32_w64_gfx942, true);
}
TEST(FpCdnaSelection, F64Wave32) {
  check<64, 32>(rj_arith_v_min_f64_w32_gfx942, false);
  check<64, 32>(rj_arith_v_max_f64_w32_gfx942, true);
}
TEST(FpCdnaSelection, F64Wave64) {
  check<64, 64>(rj_arith_v_min_f64_w64_gfx942, false);
  check<64, 64>(rj_arith_v_max_f64_w64_gfx942, true);
}

TEST(FpCdnaSelection, F16Wave32) {
  check<16, 32, false>(rj_arith_v_min_f16_w32_gfx942, false);
  check<16, 32, false>(rj_arith_v_max_f16_w32_gfx942, true);
}

TEST(FpCdnaSelection, F16Wave64) {
  check<16, 64, false>(rj_arith_v_min_f16_w64_gfx942, false);
  check<16, 64, false>(rj_arith_v_max_f16_w64_gfx942, true);
}

TEST(FpCdnaSelection, PackedF16Wave32) {
  check<16, 32, true>(rj_arith_v_pk_min_f16_w32_gfx942, false);
  check<16, 32, true>(rj_arith_v_pk_max_f16_w32_gfx942, true);
}

TEST(FpCdnaSelection, PackedF16Wave64) {
  check<16, 64, true>(rj_arith_v_pk_min_f16_w64_gfx942, false);
  check<16, 64, true>(rj_arith_v_pk_max_f16_w64_gfx942, true);
}
