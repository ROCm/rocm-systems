// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Preserve the imported gfx10 rounding model and exercise wave-level dispatch/aliasing.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/integer.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#define RJ_ARITH_IMPL_NAMESPACE interp_test
#include "graphics/rounding.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith;
// Wave64 uses both halves of EXEC and retains the shared half-source selector.
TEST(Gfx10Interpolation, Wave64) {
  using Call = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                       const uint32_t *const *, const uint32_t *const *);
  const Call calls[] = {rj_arith_v_interp_mov_f32_w64_gfx10,  rj_arith_v_interp_p1_f32_w64_gfx10,
                        rj_arith_v_interp_p2_f32_w64_gfx10,   rj_arith_v_interp_p1ll_f16_w64_gfx10,
                        rj_arith_v_interp_p1lv_f16_w64_gfx10, rj_arith_v_interp_p2_f16_w64_gfx10};
  for (int op = 0; op < 6; ++op) {
    for (uint64_t cpu : test::supported_cpu_levels()) {
      for (bool negative : {false, true}) {
        uint32_t a[64], b[64], c[64];
        const uint32_t original = op >= 3 ? 0x3c004000 : 0x3f800000;
        std::fill_n(a, 64, original);
        std::fill_n(b, 64, 0x40000000);
        std::fill_n(c, 64, (op == 3 || op == 4) ? 0x3c004200 : 0x40400000);
        uint32_t *d = a;
        const uint32_t *ap = a, *bp = b, *cp = c;
        const uint64_t exec_mask = (1ULL << 63) | 2;
        const uint64_t mode = (op >= 3 ? RJ_ARITH_ALU_HIGH_A : 0) |
                              (op == 5 ? RJ_ARITH_ALU_HIGH_D : 0) |
                              (negative ? RJ_ARITH_ALU_NEG_B : 0);
        ASSERT_EQ(
            calls[op](cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, exec_mask, mode, &d, &ap, &bp, &cp),
            RJ_ARITH_SUCCESS);
        uint32_t expected = op == 0                ? 0x3f800000
                            : (op == 3 || op == 4) ? (negative ? 0xbf800000 : 0x40400000)
                                                   : (negative ? 0x3f800000 : 0x40a00000);
        if (op == 5) {
          expected = (negative ? 0x3c000000 : 0x45000000) | 0x4000;
        }
        for (int lane = 0; lane < 64; ++lane) {
          EXPECT_EQ(a[lane], ((exec_mask >> lane) & 1) ? expected : original);
        }
      }
    }
  }
}

// Native nonfinite FMA payload selection is inherited, not newly qualified.
// All CPU tiers must reuse the baseline choice, including when both factors are NaNs.
TEST(Gfx10Interpolation, HalfNonfinitePolicy) {
  uint32_t a[32], b[32], c[32], reference[32]{}, actual[32]{};
  std::fill_n(a, 32, 0x7dc2);
  std::fill_n(b, 32, 0x7fd8edf8);
  std::fill_n(c, 32, 0xb8efc9dc);
  const uint32_t *ap = a, *bp = b, *cp = c;
  uint32_t *rp = reference, *dp = actual;
  const uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_ROUND_DOWN;
  const uint64_t mode = RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_C;
  ASSERT_EQ(rj_arith_v_interp_p2_f16_w32_gfx10(flags, ~0U, mode, &rp, &ap, &bp, &cp),
            RJ_ARITH_SUCCESS);
  for (uint64_t cpu : test::supported_cpu_levels()) {
    ASSERT_EQ(rj_arith_v_interp_p2_f16_w32_gfx10(flags | cpu, ~0U, mode, &dp, &ap, &bp, &cp),
              RJ_ARITH_SUCCESS);
    for (int lane = 0; lane < 32; ++lane) {
      EXPECT_EQ(actual[lane], reference[lane]);
    }
  }
}

TEST(Gfx10InterpolationModel, RoundingAndSignedZero) {
  for (uint32_t mode = 0; mode < 4; ++mode) {
    const uint32_t expected = mode == 1 ? 0x3f800001u : 0x3f800000u;
    EXPECT_EQ(
        std::bit_cast<uint32_t>(graphics_rounding::fma_f32(1.0f, 1.0f, 0x1p-24f, mode, 3, false)),
        expected);
    EXPECT_EQ(
        std::bit_cast<uint32_t>(graphics_rounding::fma_f32(1.0f, 1.0f, -1.0f, mode, 3, false)),
        mode == 2 ? 0x80000000u : 0u);
    EXPECT_EQ(
        std::bit_cast<uint32_t>(graphics_rounding::fma_f32(-0.0f, 1.0f, -0.0f, mode, 3, false)),
        0x80000000u);
  }
}

TEST(Gfx10InterpolationModel, MixedF16FmaExactZeroSigns) {
  struct Case {
    uint32_t a, b, c;
    std::array<uint16_t, 4> expected;
  };
  constexpr std::array<Case, 8> cases{{
      {0, 0x3f800000, 0, {0, 0, 0, 0}},
      {0, 0x3f800000, 0x80000000, {0, 0, 0x8000, 0}},
      {0x80000000, 0x3f800000, 0, {0, 0, 0x8000, 0}},
      {0x80000000, 0x3f800000, 0x80000000, {0x8000, 0x8000, 0x8000, 0x8000}},
      {0, 0xbf800000, 0, {0, 0, 0x8000, 0}},
      {0, 0xbf800000, 0x80000000, {0x8000, 0x8000, 0x8000, 0x8000}},
      {0x80000000, 0xbf800000, 0, {0, 0, 0, 0}},
      {0x80000000, 0xbf800000, 0x80000000, {0, 0, 0x8000, 0}},
  }};
  for (const auto &test : cases) {
    for (bool zero_second : {false, true}) {
      for (bool swap : {false, true}) {
        for (bool clamp : {false, true}) {
          for (uint32_t mode = 0; mode < 4; ++mode) {
            uint32_t a = test.a, b = zero_second ? test.b & 0x80000000u : test.b;
            if (swap) {
              std::swap(a, b);
            }
            const uint16_t actual = graphics_rounding::fma_f32_to_f16_nearest_environment(
                std::bit_cast<float>(a), std::bit_cast<float>(b), std::bit_cast<float>(test.c),
                mode, clamp, false, false);
            EXPECT_EQ(actual, clamp ? 0u : test.expected[mode]);
          }
        }
      }
    }
  }
}

TEST(Gfx10InterpolationModel, MixedF16FmaCancellationAcrossF32Range) {
  constexpr std::array<std::array<uint32_t, 3>, 5> cases{{
      {0x7f7fffff, 0x3f800000, 0xff7fffff},
      {1, 0x3f800000, 0x80000001},
      {0x00800000, 0x3f000000, 0x80400000},
      {0x5f800000, 0x1f800000, 0xbf800000},
      {0x00800000, 0x34000000, 0x80000001},
  }};
  for (const auto &test : cases) {
    for (uint32_t mode = 0; mode < 4; ++mode) {
      EXPECT_EQ(graphics_rounding::fma_f32_to_f16_nearest_environment(
                    std::bit_cast<float>(test[0]), std::bit_cast<float>(test[1]),
                    std::bit_cast<float>(test[2]), mode, false, false, false),
                mode == 2 ? 0x8000u : 0u);
    }
  }
  // These products are far below F16 range, but are nonzero F64 values. A
  // sign-only repair must not turn directed underflow into exact cancellation.
  for (uint32_t mode = 0; mode < 4; ++mode) {
    EXPECT_EQ(graphics_rounding::fma_f32_to_f16_nearest_environment(std::bit_cast<float>(1u),
                                                                    std::bit_cast<float>(1u), 0.0f,
                                                                    mode, false, false, false),
              mode == 1 ? 1u : 0u);
    EXPECT_EQ(graphics_rounding::fma_f32_to_f16_nearest_environment(
                  std::bit_cast<float>(0x80000001u), std::bit_cast<float>(1u), 0.0f, mode, false,
                  false, false),
              mode == 2 ? 0x8001u : 0x8000u);
  }
}

TEST(Gfx10InterpolationModel, FusedResultAndOutputFlush) {
  EXPECT_EQ(std::bit_cast<uint32_t>(
                graphics_rounding::fma_f32(0x1.000002p0f, 0x1.fffffcp-1f, -1.0f, 0, 3, false)),
            0xa8800000u);
  EXPECT_EQ(
      std::bit_cast<uint32_t>(graphics_rounding::fma_f32(-0x1p-126f, 0.5f, -0.0f, 0, 1, false)),
      0x80000000u);
}

TEST(Gfx10Interpolation, Arithmetic) {
  using Call = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                       const uint32_t *const *, const uint32_t *const *);
  const Call calls[] = {rj_arith_v_interp_mov_f32_w32_gfx10,  rj_arith_v_interp_p1_f32_w32_gfx10,
                        rj_arith_v_interp_p2_f32_w32_gfx10,   rj_arith_v_interp_p1ll_f16_w32_gfx10,
                        rj_arith_v_interp_p1lv_f16_w32_gfx10, rj_arith_v_interp_p2_f16_w32_gfx10};
  std::minstd_rand random;
  for (int op = 0; op < 6; ++op) {
    alignas(64) uint32_t a[32], b[32], c[32], reference[32], output[32];
    for (int lane = 0; lane < 32; ++lane) {
      a[lane] = test::random_word(random);
      b[lane] = test::random_word(random);
      c[lane] = test::random_word(random);
    }
    a[3] = 0x7c007f80;
    b[5] = 0x7fc12345;
    c[7] = 0x80000000;
    const uint32_t *ap = a, *bp = b, *cp = c;
    for (uint64_t semantics : test::semantics_modes()) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        for (uint64_t cpu : test::supported_cpu_levels()) {
          SCOPED_TRACE(testing::Message()
                       << op << ":" << semantics << ":" << rounding << ":" << cpu);
          const uint64_t flags = semantics | (uint64_t(rounding) << 24);
          uint32_t *ref = reference, *out = output;
          std::fill_n(reference, 32, 0xdeadbeef);
          std::fill_n(output, 32, 0xdeadbeef);
          const int expected =
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                  ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                  : RJ_ARITH_SUCCESS;
          ASSERT_EQ(calls[op](flags, 0xaaaaaaaa, 0, &ref, &ap, &bp, &cp), expected);
          ASSERT_EQ(calls[op](flags | cpu, 0xaaaaaaaa, 0, &out, &ap, &bp, &cp), expected);
          // Loose may differ in intermediate rounding and NaN payloads across CPU paths.
          const auto expect_result = [&](uint32_t actual, uint32_t value, bool written) {
            if (written && op != 0 && semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
              if (op == 5) {
                EXPECT_EQ(actual & 0xffff0000U, value & 0xffff0000U);
                test::expect_f32(std::bit_cast<uint32_t>(test::f16_value(uint16_t(actual))),
                                 test::f16_value(uint16_t(value)), RJ_ARITH_SEMANTICS_LOOSE,
                                 /*atol=*/0x1p-24f, /*rtol=*/1e-3f);
              } else {
                test::expect_f32(actual, std::bit_cast<float>(value), RJ_ARITH_SEMANTICS_LOOSE,
                                 /*atol=*/1e-44f, /*rtol=*/2e-7f);
              }
            } else {
              EXPECT_EQ(actual, value);
            }
          };
          for (int lane = 0; lane < 32; ++lane) {
            expect_result(output[lane], reference[lane],
                          expected == RJ_ARITH_SUCCESS && ((0xaaaaaaaaU >> lane) & 1));
          }
          std::copy_n(a, 32, output);
          const uint32_t *alias = output;
          ASSERT_EQ(calls[op](flags | cpu, 0xaaaaaaaa, 0, &out, &alias, &bp, &cp), expected);
          for (int lane = 0; lane < 32; ++lane) {
            const bool written = expected == RJ_ARITH_SUCCESS && ((0xaaaaaaaaU >> lane) & 1);
            uint32_t value = reference[lane];
            if (op == 5) {
              value = (a[lane] & 0xffff0000U) | (value & 0xffffU);
            }
            expect_result(output[lane], written ? value : a[lane], written);
          }
        }
      }
    }
  }
}
// Guest rounding must not become host rounding, including the atomic exact-zero path.
TEST(Gfx10InterpolationModel, ExplicitRoundingPreservesHostControls) {
  ASSERT_EQ(std::fegetround(), FE_TONEAREST);
#if defined(__x86_64__)
  const uint32_t controls = _mm_getcsr() & ~uint32_t(_MM_EXCEPT_MASK);
#endif
  for (uint32_t mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    EXPECT_EQ(
        std::bit_cast<uint32_t>(graphics_rounding::fma_f32(1.0f, 1.0f, -1.0f, mode, 3, false)),
        mode == 2 ? 0x80000000u : 0u);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
#if defined(__x86_64__)
  // Ordinary arithmetic is allowed to raise sticky flags; compare controls only.
  EXPECT_EQ(_mm_getcsr() & ~uint32_t(_MM_EXCEPT_MASK), controls);
#endif
}

} // namespace

// Directed guest rounding remains observable even when Loose uses native arithmetic.
TEST(Gfx10Interpolation, LooseDirectedHalfOverflow) {
  for (auto cpu : test::supported_cpu_levels()) {
    uint32_t a[32]{0x3c00}, b[32]{std::bit_cast<uint32_t>(65520.f)}, c[32]{}, d[32]{};
    const uint32_t *ap = a, *bp = b, *cp = c;
    uint32_t *dp = d;
    d[0] = 0xdeadbeefU;
    ASSERT_EQ(
        rj_arith_v_interp_p2_f16_w32_gfx10(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_ROUND_ZERO,
                                           1, 0, &dp, &ap, &bp, &cp),
        RJ_ARITH_SUCCESS);
    EXPECT_EQ(d[0], 0xdead7bffU);
    EXPECT_EQ(d[1], 0U);
  }
}
