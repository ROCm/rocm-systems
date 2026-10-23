// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx12 FP16 FMA: deterministic fractional inputs, exceptional values,
// EXEC, whole-register aliases, preserved halfwords, FP-status reporting, and errors.

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/half_fma_reference.h"
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
          if (semantics != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL || !exec_mask) {
            EXPECT_EQ(status, 0x80000000U);
          } else {
            EXPECT_EQ(status & 0x80000000U, 0x80000000U);
          }
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
            uint16_t expected_bits =
                semantics == RJ_ARITH_SEMANTICS_LOOSE
                    ? half_bits(value)
                    : half_fma_reference::evaluate(uint16_t(input.v[0][lane]),
                                                   uint16_t(input.v[1][lane]),
                                                   uint16_t(input.v[2][lane]), false);
            float expected = float(half_value(expected_bits));
            expect_f32(std::bit_cast<uint32_t>(float(half_value(uint16_t(actual)))), expected,
                       semantics,
                       /*atol=*/1e-5, /*rtol=*/5e-4);
          }
        }
      }
    }
  }
}

// Guard ordinary midpoints and the finite/infinite boundary against double rounding.
TEST(FpFmaF16W32Gfx12, RoundingBoundaries) {
  using namespace rocjitsu::arith::test;
  struct Case {
    uint16_t a, b, c, expected, saturated;
    uint32_t expected_status;
  };
  const Case cases[] = {
      // Overflow is classified after rounding, not by comparison with 65504.
      {0x4300, 0x7492, 0, 0x7c00, 0x7bff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x88e4, 0x0a8b, 0x81ff, 0x81ff, 0x81ff,
       RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x32f9, 0xb0b7, 0x3780, 0x36fd, 0x36fd, RJ_ARITH_EXCEPTION_INEXACT},
      {0x4300, 0x7492, 0x9400, 0x7bff, 0x7bff, RJ_ARITH_EXCEPTION_INEXACT},
      {0xc300, 0x7492, 0x1400, 0xfbff, 0xfbff, RJ_ARITH_EXCEPTION_INEXACT},
      {0x7bff, 0x4000, 0, 0x7c00, 0x7bff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0xfbff, 0x4000, 0, 0xfc00, 0xfbff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x7c00, 0x3c00, 0, 0x7c00, 0x7c00, 0},
      {0xfc00, 0x3c00, 0, 0xfc00, 0xfc00, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint64_t fp_flags : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
        for (const auto &sample : cases) {
          SCOPED_TRACE(testing::Message()
                       << "cpu=" << cpu << " semantics=" << semantics << " fp_flags=" << fp_flags
                       << " a=" << sample.a << " b=" << sample.b << " c=" << sample.c);
          Registers<4, 32> r;
          r.v[0][0] = sample.a;
          r.v[1][0] = sample.b;
          r.v[2][0] = sample.c;
          uint32_t status = 0;
          ASSERT_EQ(rj_arith_v_fma_f16_w32_gfx12(cpu | semantics | fp_flags, 1, 0, r.v + 3, r.v,
                                                 r.v + 1, r.v + 2, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(uint16_t(r.v[3][0]), fp_flags ? sample.saturated : sample.expected);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            EXPECT_EQ(status, sample.expected_status);
          }
        }
      }
    }
  }
}

// Reject controls not implemented by this slice, before accessing operands.
TEST(FpFmaF16W32Gfx12, UnsupportedGuestControls) {
  using namespace rocjitsu::arith::test;
  const Registers<4, 32> input;
  constexpr uint32_t initial_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    for (uint64_t flag :
         {RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
          RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_FP_ROUND_UP,
          RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16,
          RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16}) {
      SCOPED_TRACE(testing::Message() << "semantics=" << semantics << " flag=" << flag);
      auto r = input;
      uint32_t status = initial_status;
      EXPECT_EQ(rj_arith_v_fma_f16_w32_gfx12(semantics | flag, 1, 0, r.v + 3, r.v, r.v + 1, r.v + 2,
                                             &status),
                RJ_ARITH_ERROR_INVALID_FLAGS);
      r.expect_preserved(input, 0, 0);
      EXPECT_EQ(status, initial_status);
      EXPECT_EQ(rj_arith_v_fma_f16_w32_gfx12(semantics | flag, 1, 0, nullptr, nullptr, nullptr,
                                             nullptr, &status),
                RJ_ARITH_ERROR_INVALID_FLAGS);
      EXPECT_EQ(status, initial_status);
    }
  }
}

// Raw encodings from rocjitsu's gfx12 FP16 FMA model: rounding and NaN precedence.
TEST(FpFmaF16W32Gfx12, ExactEncodings) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint16_t a, b, c, expected;
    uint64_t flags;
  } cases[] = {
      {0x7e01, 0x7c03, 0, 0x7e01, 0},      // Earlier qNaN beats later sNaN.
      {0x3c00, 0x7e05, 0x7d07, 0x7e05, 0}, // B NaN beats C sNaN.
      {0x7c00, 0x3c00, 0xfc00, 0xfe00, 0}, // Opposite infinities.
      {0x8000, 0x3c00, 0x8000, 0x8000, 0}, // Matching negative zeros.
      {0x8001, 0x3800, 0, 0x8000, 0},      // Negative halfway underflow.
      {0x0001, 0x3800, 0, 0x0000, 0},      // Tie rounds to even zero.
      {0x0003, 0x3800, 0, 0x0002, 0},      // Tie rounds up to even subnormal.

      {0x4782, 0xda00, 0x03e1, 0xe5a1, 0},             // Rounding witness.
      {0, 0x7c00, 0x7e01, 0xfe00, 0},                  // Invalid product precedes C NaN.
      {0x7c03, 0x7e01, 0, 0x7e03, 0},                  // Quiet the first NaN.
      {0x7bff, 0x7bff, 0, 0x7c00, 0},                  // Finite overflow.
      {0x7bff, 0x7bff, 0, 0x7bff, RJ_ARITH_FP16_OVFL}, // Saturate finite overflow.
      {0x3c00, 0x3c00, 0xbc00, 0, 0},                  // Exact cancellation.
      {1, 0x3c00, 0, 1, 0},                            // Exact subnormal result.
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (const auto &sample : cases) {
        SCOPED_TRACE(testing::Message()
                     << "cpu=" << cpu << " semantics=" << semantics << " a=" << sample.a
                     << " b=" << sample.b << " c=" << sample.c << " flags=" << sample.flags);
        Registers<4, 32> r;
        r.v[0][0] = sample.a;
        r.v[1][0] = sample.b;
        r.v[2][0] = sample.c;
        uint32_t status = 0;
        ASSERT_EQ(rj_arith_v_fma_f16_w32_gfx12(cpu | semantics | sample.flags, 1, 0, r.v + 3, r.v,
                                               r.v + 1, r.v + 2, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(uint16_t(r.v[3][0]), sample.expected);
      }
    }
  }
}

// gfx1201 WAVE_EXCP_FLAG_USER observations under MODE=0xf0 (RNE, preserve denormals).
// FP16_OVFL rows additionally set MODE bit 23 (MODE=0x008000f0).
// These cases distinguish tininess, NaN suppression, and finite saturation.
TEST(FpFmaF16W32Gfx12, StatusBoundariesAndExec) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint16_t a, b, c, expected;
    uint32_t causes;
    uint64_t flags;
  } cases[] = {
      {0x3c00, 0x3c00, 0, 0x3c00, 0, 0},
      {0x3c01, 0x3c01, 0, 0x3c02, RJ_ARITH_EXCEPTION_INEXACT, 0},
      {1, 0x3800, 0, 0,
       RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT,
       0},
      // gfx1201 VOP3 lane 31, EXEC=0x80000000, MODE=0x010000f0.
      {0x3c00, 0x3c00, 0x7c01, 0x7e01, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x3c00, 0x7c00, 0xfc00, 0xfe00, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0, 0x7c00, 1, 0xfe00, RJ_ARITH_EXCEPTION_INVALID, 0},

      // Also observed by the reviewer on gfx1201 (R9700), with and without FP16_OVFL.
      {1, 0xb800, 0x0400, 0x0400,
       RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT,
       0},
      {1, 0xb400, 0x0400, 0x0400, RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_INEXACT, 0},
      {0x0400, 0x3bff, 0, 0x0400, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT, 0},
      {1, 0x3c00, 0, 1, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0},
      {1, 0x3c00, 0x7e00, 0x7e00, 0, 0},
      {1, 0x7c00, 0, 0x7c00, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0},
      {0x7c01, 0x3c00, 0, 0x7e01, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x7e02, 0x7c01, 0, 0x7e02, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x7c00, 0x3c00, 0xfc00, 0xfe00, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x7c00, 0x3c00, 0x7c00, 0x7c00, 0, 0},
      {1, 0x7c00, 0x7c00, 0x7c00, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0},
      {0x7bff, 0x4000, 0, 0x7bff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT,
       RJ_ARITH_FP16_OVFL},
  };
  constexpr uint32_t preserved = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      SCOPED_TRACE(testing::Message() << "cpu=" << cpu << " a=" << sample.a << " b=" << sample.b
                                      << " c=" << sample.c << " flags=" << sample.flags);
      Registers<4, 32> r;
      r.v[0][31] = sample.a;
      r.v[1][31] = sample.b;
      r.v[2][31] = sample.c;
      uint32_t status = preserved;
      ASSERT_EQ(
          rj_arith_v_fma_f16_w32_gfx12(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | sample.flags,
                                       0x80000000U, 0, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(uint16_t(r.v[3][31]), sample.expected);
      EXPECT_EQ(status, preserved | sample.causes);
    }
    // Each exceptional lane contributes independently, and masked lanes contribute nothing.
    for (uint32_t exec_mask : {0U, 1U, 0x80000000U, 0x80000001U}) {
      SCOPED_TRACE(testing::Message() << "cpu=" << cpu << " exec_mask=" << exec_mask);
      Registers<4, 32> r;
      r.v[0][0] = 0;
      r.v[1][0] = 0x7c00;
      r.v[2][0] = 0;
      r.v[0][31] = 0x7bff;
      r.v[1][31] = 0x4000;
      r.v[2][31] = 0;
      uint32_t status = preserved;
      ASSERT_EQ(rj_arith_v_fma_f16_w32_gfx12(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL,
                                             exec_mask, 0, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
                RJ_ARITH_SUCCESS);
      uint32_t expected = preserved;
      if (exec_mask & 1) {
        expected |= RJ_ARITH_EXCEPTION_INVALID;
      }
      if (exec_mask & 0x80000000U) {
        expected |= RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
      }
      EXPECT_EQ(status, expected);
    }
  }
}
