// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for exponent-selected pieces of 2/pi for trigonometric range reduction.

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/detail/rj_arith_math.h"
#include <algorithm>
#include <bit>
#include <cmath>

namespace {
using namespace rocjitsu::arith::test;

// Inherited independent PREOP reference: extract a 53-bit window of 2/pi,
// then use wider arithmetic for construction and scaling. The host uses
// nearest-even rounding with denormals preserved.
uint64_t reference(int exponent, unsigned selector, unsigned mode, unsigned guest,
                   bool gfx12 = false) {
  constexpr char digits[] = "a2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebb"
                            "c561b7246e3a424dd2e006492eea09d1921cfe1deb1cb129a73ee88235f5"
                            "2ebb4484e99c7026b45f7e413991d639835339f49c845f8bbdf9283b1ff8"
                            "97ffde05980fef2f118b5a0a6d1f6d367ecf27cb09b74f463f669e5fea2d"
                            "7527bac7ebe5f17b3d0739f78a5292ea6bfb5fb11f8d5d0856033046fc7b6babf0";
  int shift = 53 * int(selector & 31) + (exponent > 1077 ? exponent - 1077 : 0);
  uint64_t segment = 0;
  for (int i = 0; i < 53; ++i) {
    int bit = shift + i;
    unsigned nibble = 0;
    if (bit < 1184) {
      char c = digits[bit / 4];
      nibble = c <= '9' ? c - '0' : c - 'a' + 10;
    }
    segment = (segment << 1) | ((nibble >> (3 - bit % 4)) & 1);
  }
  long double unrounded =
      std::ldexp(static_cast<long double>(segment), -53 - shift + (exponent >= 1968 ? 128 : 0));
  double value = static_cast<double>(unrounded);
  if (static_cast<long double>(value) > unrounded) {
    value = std::nextafter(value, 0.0);
  }
  unsigned omod = (mode >> 6) & 3;
  if (!gfx12 && (!(guest & 2) || (guest & 4))) {
    omod = 0;
  }
  if (omod) {
    if (std::fpclassify(value) == FP_SUBNORMAL) {
      value = 0;
    }
    value = static_cast<double>(
        std::ldexp(static_cast<long double>(value), omod == 3 ? -1 : int(omod)));
    if (std::fpclassify(value) == FP_SUBNORMAL) {
      value = 0;
    }
  }
  if ((guest & 2) && std::fpclassify(value) == FP_SUBNORMAL) {
    value = 0;
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    value = std::min(value, 1.0);
  }
  return std::bit_cast<uint64_t>(value);
}

// Check TRIG_PREOP results across selectors, gfx policies, and aliases for the supplied entry
// point.
template <int Lanes, bool Gfx12, typename Call> void check_preop(Call call) {
  Registers<5, Lanes> input;
  std::minstd_rand random;
  for (int lane = 0; lane < Lanes; ++lane) {
    int sample = lane % 32;
    input.v[0][lane] = random_word(random);
    input.v[1][lane] =
        (uint32_t((1023 + sample * 33) % 2048) << 20) | (random_word(random) & 0x800fffffU);
    input.v[2][lane] = uint32_t(sample % 7) | 0xdeadbee0U;
  }
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 32; ++scenario) {
        for (int d : {0, 1, 2, 3}) {
          auto actual = input;
          auto exec_mask = exec_mask_for_case<Lanes>(scenario);
          uint64_t mode =
              (scenario & 1 ? RJ_ARITH_ALU_ABS_A : 0) | (scenario & 2 ? RJ_ARITH_ALU_NEG_A : 0) |
              (uint64_t((scenario >> 2) & 3) << 6) | (scenario & 16 ? RJ_ARITH_ALU_CLAMP : 0);
          uint32_t status = 0x80000000U;
          int wanted = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                           ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                           : RJ_ARITH_SUCCESS;
          ASSERT_EQ(
              call(cpu | semantics, exec_mask, mode, actual.v + d, actual.v, actual.v + 2, &status),
              wanted);
          EXPECT_EQ(status, 0x80000000U);
          if (wanted) {
            actual.expect_preserved(input, 0, 0);
            continue;
          }
          // Source signs and fractions are irrelevant. The inherited table
          // oracle accounts for architecture-specific precision and OMOD.
          for (int lane = 0; lane < Lanes; ++lane) {
            for (int word = 0; word < 2; ++word) {
              uint64_t expected = reference((input.v[1][lane] >> 20) & 2047, input.v[2][lane],
                                            uint32_t(mode), 0, Gfx12);
              EXPECT_EQ(actual.v[d + word][lane], ((exec_mask >> lane) & 1)
                                                      ? uint32_t(expected >> (32 * word))
                                                      : input.v[d + word][lane]);
            }
          }
          actual.expect_preserved(input, d, 2);
        }
      }
    }
  }
  auto actual = input;
  expect_invalid_flags(
      [&](auto f, auto m) {
        uint32_t status = 123;
        int error = call(f, WaveMask<Lanes>(~WaveMask<Lanes>(0)), m, actual.v + 3, actual.v,
                         actual.v + 2, &status);
        actual.expect_preserved(input, 0, 0);
        EXPECT_EQ(status, 123U);
        return error;
      },
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP);
}
} // namespace

// Check TRIG_PREOP results across selectors, gfx policies, and aliases.
TEST(MathTrigPreopF64W32Gfx11, Arithmetic) {
  check_preop<32, false>(rj_arith_v_trig_preop_f64_w32);
}

// Check TRIG_PREOP results across selectors, gfx policies, and aliases.
TEST(MathTrigPreopF64W32Gfx12, Arithmetic) {
  check_preop<32, true>(rj_arith_v_trig_preop_f64_w32_gfx12);
}

// Check TRIG_PREOP results across selectors, gfx policies, and aliases.
TEST(MathTrigPreopF64W64Gfx11, Arithmetic) {
  check_preop<64, false>(rj_arith_v_trig_preop_f64_w64);
}

// Check TRIG_PREOP results across selectors, gfx policies, and aliases.
TEST(MathTrigPreopF64W64Gfx12, Arithmetic) {
  check_preop<64, true>(rj_arith_v_trig_preop_f64_w64_gfx12);
}

// gfx1100/gfx1201 captures pin the table tail and output flushing without OMOD.
TEST(TrigPreop, TableTailAndOutputFlushing) {
  for (auto cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_trig_preop_f64_w32, rj_arith_v_trig_preop_f64_w32_gfx12}) {
      uint32_t lo[32]{}, hi[32]{}, selector[32]{}, dlo[32]{}, dhi[32]{};
      const uint32_t *a[] = {lo, hi}, *b[] = {selector};
      uint32_t *d[] = {dlo, dhi};
      uint32_t status = 0;
      hi[0] = 0x7b012345;
      selector[0] = 5;
      const auto flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
      ASSERT_EQ(call(flags, 1, 0, d, a, b, &status), 0);
      EXPECT_EQ((uint64_t(dhi[0]) << 32) | dlo[0], 0x0000180cc1180000ULL);
      hi[0] = 0x00012345;
      selector[0] = 0x14;
      ASSERT_EQ(call(flags, 1, 0, d, a, b, &status), 0);
      EXPECT_EQ((uint64_t(dhi[0]) << 32) | dlo[0], 0x294aULL);
      ASSERT_EQ(call(flags | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1, 0, d, a, b, &status), 0);
      EXPECT_EQ((uint64_t(dhi[0]) << 32) | dlo[0], 0ULL);
      // OMOD also flushes this captured table-tail value, independently of MODE.
      for (auto mode : {RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF}) {
        ASSERT_EQ(call(flags | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1, mode, d, a, b, &status), 0);
        EXPECT_EQ((uint64_t(dhi[0]) << 32) | dlo[0], 0ULL);
      }
    }
  }
}

// gfx1100 IEEE=0 capture (hw1100-noieee, MODE 0x00/0x50/0x800000): mul:2/mul:4 flush the
// subnormal table tail before scaling, as gfx1201 does; scaling it first gives 0x19e839cfbc5294.
TEST(TrigPreop, OmodFlushesSubnormalTableValueBeforeScaling) {
  const struct {
    uint64_t a;
    uint32_t selector;
    uint64_t omod;
  } cases[] = {{0x68b6bd57cfd4edafULL, 0x2c808, RJ_ARITH_ALU_OMOD_2},
               {0xdb611d249dc9626bULL, 0x864c654c, RJ_ARITH_ALU_OMOD_2},
               {0x4790569f07e532c1ULL, 0xdc49732, RJ_ARITH_ALU_OMOD_4},
               {0x5ec2448f9d586cc9ULL, 0x1e648b, RJ_ARITH_ALU_OMOD_4}};
  for (auto cpu : supported_cpu_levels()) {
    for (uint64_t input_flush : {uint64_t(0), RJ_ARITH_FP_FLUSH_INPUT_DENORMALS}) {
      for (const auto &sample : cases) {
        Registers<3, 64> r;
        for (int lane = 0; lane < 64; ++lane) {
          r.v[0][lane] = uint32_t(sample.a);
          r.v[1][lane] = uint32_t(sample.a >> 32);
          r.v[2][lane] = sample.selector;
        }
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                               RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | input_flush;
        for (int call = 0; call < 3; ++call) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << input_flush << '/' << call << '/' << std::hex << sample.a);
          uint32_t lo[64], hi[64];
          uint32_t *d[] = {lo, hi};
          const uint32_t *a[] = {r.v[0], r.v[1]}, *b[] = {r.v[2]};
          ASSERT_EQ(
              call == 0 ? rj_arith_v_trig_preop_f64_w32(flags, ~0U, sample.omod, d, a, b, nullptr)
              : call == 1
                  ? rj_arith_v_trig_preop_f64_w64(flags, ~0ULL, sample.omod, d, a, b, nullptr)
                  : rj_arith_v_trig_preop_f64_w32_gfx12(flags, ~0U, sample.omod, d, a, b, nullptr),
              RJ_ARITH_SUCCESS);
          for (int lane = 0; lane < (call == 1 ? 64 : 32); ++lane) {
            EXPECT_EQ((uint64_t(hi[lane]) << 32) | lo[lane], 0ULL) << lane;
          }
        }
      }
    }
  }
}
