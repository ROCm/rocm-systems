// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx12 FP16 FMA: deterministic fractional inputs, exceptional values,
// EXEC, whole-register aliases, preserved halfwords, and errors.

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using RegBank = rocjitsu::arith::test::Registers<5, 32>;
} // namespace

// Check low-half arithmetic and that all storage outside active results is preserved.
TEST(FpFmaF16W32Gfx12, Arithmetic) {
  using namespace rocjitsu::arith::test;
  RegBank input;
  std::minstd_rand random;
  for (int reg = 0; reg < 5; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = (random_f16_pair(random) & 0xffff0000U) | random_f16_bits(random);
    }
  }
  for (int reg = 0; reg < 3; ++reg) {
    input.v[reg][reg * 2] = kF16QuietNaNBits;
    input.v[reg][reg * 2 + 1] = kF16PositiveInfinityBits;
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        return rj_arith_v_fma_f16_w32_gfx12(flags, 0xffffffffU, mode, nullptr, nullptr, nullptr,
                                            nullptr, nullptr);
      },
      0);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint32_t exec_mask : {0U, 0xffffffffU, 0xaaaaaaaaU, 0x80000000U}) {
        for (int aliasing_offset : {0, 1, 2, 3}) {
          RegBank r = input;
          uint32_t status = 0x80000000U;
          ASSERT_EQ(rj_arith_v_fma_f16_w32_gfx12(cpu | semantics, exec_mask, 0,
                                                 r.v + aliasing_offset, r.v, r.v + 1, r.v + 2,
                                                 &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x80000000U);
          r.expect_preserved(input, aliasing_offset, 1);
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t actual = r.v[aliasing_offset][lane];
            if (!((exec_mask >> lane) & 1)) {
              EXPECT_EQ(actual, input.v[aliasing_offset][lane]);
              continue;
            }
            EXPECT_EQ(actual >> 16, input.v[aliasing_offset][lane] >> 16);
            // A computed reference is sufficient for loose arithmetic.
            double value = std::fma(half_value(uint16_t(input.v[0][lane])),
                                    half_value(uint16_t(input.v[1][lane])),
                                    half_value(uint16_t(input.v[2][lane])));
            float expected = float(half_value(half_bits(value)));
            expect_f32(std::bit_cast<uint32_t>(float(half_value(uint16_t(actual)))), expected,
                       semantics,
                       /*atol=*/1e-5, /*rtol=*/5e-4);
          }
        }
      }
    }
  }
  EXPECT_EQ(rj_arith_v_fma_f16_w32_gfx12(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, 0,
                                         nullptr, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
}

// Guard ordinary midpoints and the finite/infinite boundary against double rounding.
TEST(FpFmaF16W32Gfx12, RoundingBoundaries) {
  using namespace rocjitsu::arith::test;
  struct Case {
    uint16_t a, b, c, expected, saturated;
  };
  const Case cases[] = {
      {0x88e4, 0x0a8b, 0x81ff, 0x81ff, 0x81ff}, // Subnormal midpoint correction.
      {0x32f9, 0xb0b7, 0x3780, 0x36fd, 0x36fd}, {0x4300, 0x7492, 0x9400, 0x7bff, 0x7bff},
      {0xc300, 0x7492, 0x1400, 0xfbff, 0xfbff}, {0x7bff, 0x4000, 0, 0x7c00, 0x7bff},
      {0xfbff, 0x4000, 0, 0xfc00, 0xfbff},      {0x7c00, 0x3c00, 0, 0x7c00, 0x7c00},
      {0xfc00, 0x3c00, 0, 0xfc00, 0xfc00},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE}) {
      for (uint64_t fp_flags : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
        for (const auto &sample : cases) {
          SCOPED_TRACE(testing::Message() << "cpu=" << cpu << " semantics=" << semantics
                                          << " fp_flags=" << fp_flags << " a=" << sample.a);
          Registers<4, 32> r;
          r.v[0][0] = sample.a;
          r.v[1][0] = sample.b;
          r.v[2][0] = sample.c;
          ASSERT_EQ(rj_arith_v_fma_f16_w32_gfx12(cpu | semantics | fp_flags, 1, 0, r.v + 3, r.v,
                                                 r.v + 1, r.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(uint16_t(r.v[3][0]), fp_flags ? sample.saturated : sample.expected);
        }
      }
    }
  }
}

// Reject controls not implemented by this slice, before accessing operands.
TEST(FpFmaF16W32Gfx12, UnsupportedGuestControls) {
  for (uint64_t flag :
       {RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
        RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_FP_ROUND_UP,
        RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16,
        RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16}) {
    EXPECT_EQ(rj_arith_v_fma_f16_w32_gfx12(flag, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}
