// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// CDNA TRIG_PREOP checks compact extraction against a bit-at-a-time legacy reference.
#include "common/cpu.h"
#include "common/registers.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace {
using namespace rocjitsu::arith::test;
// Reconstruct the inherited 1201-bit window in wider arithmetic, then explicitly round.
uint64_t reference(int exponent, unsigned selector, int omod, unsigned rounding, bool clamp) {
  constexpr char digits[] = "a2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebb"
                            "c561b7246e3a424dd2e006492eea09d1921cfe1deb1cb129a73ee88235f5"
                            "2ebb4484e99c7026b45f7e413991d639835339f49c845f8bbdf9283b1ff8"
                            "97ffde05980fef2f118b5a0a6d1f6d367ecf27cb09b74f463f669e5fea2d"
                            "7527bac7ebe5f17b3d0739f78a5292ea6bfb5fb11f8d5d0856033046fc7b6babf0";
  const int shift = 53 * int(selector & 31) + std::max(exponent - 1077, 0);
  uint64_t significand = 0;
  for (int bit = 0; bit < 53; ++bit) {
    const int index = shift + bit;
    unsigned nibble = 0;
    if (index < 1201) {
      const char c = digits[index / 4];
      nibble = c <= '9' ? c - '0' : c - 'a' + 10;
    }
    significand = (significand << 1) | ((nibble >> (3 - index % 4)) & 1);
  }
  const long double exact =
      std::ldexp(static_cast<long double>(significand), -53 - shift + (exponent >= 1968 ? 128 : 0));
  double value = double(exact);
  if (static_cast<long double>(value) > exact) {
    value = std::nextafter(value, 0.0);
  }
  if (omod) {
    const long double scaled = std::ldexp(static_cast<long double>(value), omod == 3 ? -1 : omod);
    value = double(scaled);
    if (rounding == 1 && value < scaled) {
      value = std::nextafter(value, std::numeric_limits<double>::infinity());
    }
    if (rounding >= 2 && value > scaled) {
      value = std::nextafter(value, 0.0);
    }
    if (std::fpclassify(value) == FP_SUBNORMAL) {
      value = 0;
    }
  }
  if (clamp) {
    value = std::min(value, 1.0);
  }
  return std::bit_cast<uint64_t>(value);
}

// Cover the tail, exponent-bank boundary, guest scaling, masks, aliases and status preservation.
template <int Lanes, bool Gfx1250, typename Call> void check(Call call) {
  using Mask = std::conditional_t<Lanes == 64, uint64_t, uint32_t>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int exponent : {0, 1077, 1100, 1967, 1968, 1969, 2000, 2047}) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        for (int omod = 0; omod < 4; ++omod) {
          for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE}) {
            for (int destination : {0, 3}) {
              Registers<5, Lanes> input;
              for (int lane = 0; lane < Lanes; ++lane) {
                input.v[0][lane] = 0xdeadbeefU;
                input.v[1][lane] = uint32_t(exponent) << 20;
                input.v[2][lane] = uint32_t(lane);
              }
              auto actual = input;
              constexpr Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL) | 1;
              uint32_t state = 0x55;
              const uint64_t flags =
                  cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | fp | (uint64_t(rounding) << 24);
              const uint64_t mode = uint64_t(omod) * RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_ABS_A |
                                    RJ_ARITH_ALU_NEG_A |
                                    (destination == 3 ? RJ_ARITH_ALU_CLAMP : 0);
              ASSERT_EQ(call(flags, exec_mask, mode, actual.v + destination, actual.v, actual.v + 2,
                             &state),
                        RJ_ARITH_SUCCESS);
              EXPECT_EQ(state, 0x55U);
              actual.expect_preserved(input, destination, 2);
              const int effective = Gfx1250 || ((fp & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
                                                !(fp & RJ_ARITH_FP_IEEE_MODE))
                                        ? omod
                                        : 0;
              for (int lane = 0; lane < Lanes; ++lane) {
                if (!(exec_mask & (Mask(1) << lane))) {
                  EXPECT_EQ(actual.v[destination][lane], input.v[destination][lane]);
                  EXPECT_EQ(actual.v[destination + 1][lane], input.v[destination + 1][lane]);
                  continue;
                }
                const uint64_t got =
                    actual.v[destination][lane] | (uint64_t(actual.v[destination + 1][lane]) << 32);
                EXPECT_EQ(got, reference(exponent, lane, effective, rounding, destination == 3))
                    << exponent << ':' << lane << ':' << effective;
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(
      call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, Mask(0), 0, nullptr, nullptr, nullptr, nullptr),
      RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  EXPECT_EQ(call(0, Mask(0), 0, nullptr, nullptr, nullptr, nullptr), RJ_ARITH_SUCCESS);
  EXPECT_EQ(call(0, Mask(0), 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
} // namespace

TEST(CdnaTrigPreop, Gfx942Wave32) { check<32, false>(rj_arith_v_trig_preop_f64_w32_gfx942); }
TEST(CdnaTrigPreop, Gfx942Wave64) { check<64, false>(rj_arith_v_trig_preop_f64_w64_gfx942); }
TEST(CdnaTrigPreop, Gfx1250) { check<32, true>(rj_arith_v_trig_preop_f64_w32_gfx1250); }

// Exhaust all exponent/selector pairs against the scalar encoding; vary scaling and rounding.
TEST(CdnaTrigPreop, SimdMatchesScalarWindows) {
  for (uint64_t cpu : supported_cpu_levels()) {
    if (cpu == RJ_ARITH_CPU_BASELINE) {
      continue;
    }
    for (int exponent = 0; exponent < 2048; ++exponent) {
      Registers<5, 32> input;
      for (int lane = 0; lane < 32; ++lane) {
        input.v[1][lane] = uint32_t(exponent) << 20;
        input.v[2][lane] = uint32_t(lane);
      }
      for (int omod = 0; omod < 4; ++omod) {
        auto expected = input, actual = input;
        const uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(exponent % 4) << 24);
        const uint64_t mode = uint64_t(omod) * RJ_ARITH_ALU_OMOD_2;
        ASSERT_EQ(rj_arith_v_trig_preop_f64_w32_gfx1250(flags, ~0U, mode, expected.v + 3,
                                                        expected.v, expected.v + 2, nullptr),
                  0);
        ASSERT_EQ(rj_arith_v_trig_preop_f64_w32_gfx1250(flags | cpu, ~0U, mode, actual.v + 3,
                                                        actual.v, actual.v + 2, nullptr),
                  0);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(actual.v[3][lane], expected.v[3][lane]) << exponent << ':' << lane;
          EXPECT_EQ(actual.v[4][lane], expected.v[4][lane]) << exponent << ':' << lane;
        }
      }
    }
  }
}
