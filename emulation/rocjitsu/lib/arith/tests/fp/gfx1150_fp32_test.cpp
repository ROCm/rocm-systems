// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1150 ordinary FP32 arithmetic shares numerical kernels, with its inherited
// pre-CLAMP Compatibility causes and DX10_CLAMP reporting gate.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/binary_arithmetic_test.h"
#include "fp/fma_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace {

// Run selected inherited cause witnesses on both wave halves and aliased outputs.
// The ordinary arithmetic runners below supply deterministic fractional coverage.
template <int Lanes, typename Call> void check_causes(Call call, bool fused, bool multiply) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const struct {
    uint32_t a, b, c, causes;
    uint64_t fp;
    uint32_t modifiers;
  } cases[] = {
      {0x3f800000, 0x3f800000, 0, 0, 0, 0},
      {0x7f800001, 0x3f800000, 0, RJ_ARITH_EXCEPTION_INVALID, 0, 0},
      {0x00000001, 0x7fc00000, 0, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0, 0},
      {0x00000001, 0x7fc00000, 0, 0, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0},
      {0x7f7fffff, 0x40000000, 0,
       fused || multiply ? RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT
                         : RJ_ARITH_EXCEPTION_INEXACT,
       0, RJ_ARITH_ALU_CLAMP},
      {0x3f800001, fused || multiply ? 0x3f800001U : 0x33800000U, 0, RJ_ARITH_EXCEPTION_INEXACT, 0,
       0},
      {0x7f7fffff, 0x3f800000, 0, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
      {0x00800000, 0x3f000001, 0,
       fused || multiply ? RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT
                         : RJ_ARITH_EXCEPTION_INEXACT,
       0, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      for (uint64_t semantics : semantics_modes()) {
        for (bool dx10_clamp : {false, true}) {
          for (int aliasing_offset : {0, 1, 2, 3}) {
            SCOPED_TRACE(testing::Message() << cpu << '/' << sample.a << '/' << semantics << '/'
                                            << dx10_clamp << '/' << aliasing_offset);
            Registers<4, Lanes> input;
            for (int lane = 0; lane < Lanes; ++lane) {
              input.v[0][lane] = sample.a;
              input.v[1][lane] = sample.b;
              input.v[2][lane] = sample.c;
              input.v[3][lane] = 0xdeadbeef;
            }
            auto result = input;
            const Mask exec_mask = Mask(1) | (Mask(1) << (Lanes - 1));
            const uint64_t flags =
                cpu | semantics | sample.fp | (dx10_clamp ? 0 : RJ_ARITH_FP_DX10_CLAMP_OFF);
            uint32_t status = 0x80000000;
            const int error = call(flags, exec_mask, sample.modifiers, result.v + aliasing_offset,
                                   result.v, result.v + 1, result.v + 2, &status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              result.expect_preserved(input, 0, 0);
              EXPECT_EQ(status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(error, RJ_ARITH_SUCCESS);
            EXPECT_EQ(status,
                      0x80000000U | (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY && !dx10_clamp
                                         ? sample.causes
                                         : 0));
            result.expect_preserved(input, aliasing_offset, 1);
            for (int lane = 1; lane < Lanes - 1; ++lane) {
              EXPECT_EQ(result.v[aliasing_offset][lane], input.v[aliasing_offset][lane]);
            }
            // Numeric-only calls must not observe the status pointer. An empty
            // reporting call observes neither operand arrays nor status storage.
            EXPECT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_DX10_CLAMP_OFF,
                           Mask(0), 0, nullptr, nullptr, nullptr, nullptr, nullptr),
                      RJ_ARITH_SUCCESS);
            if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY || dx10_clamp) {
              EXPECT_EQ(call(flags, exec_mask, sample.modifiers, result.v + 3, input.v, input.v + 1,
                             input.v + 2, nullptr),
                        RJ_ARITH_SUCCESS);
            }
          }
        }
      }
    }
    Registers<4, Lanes> input;
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[0][lane] = 0x7f800123;
      input.v[1][lane] = 0x3f800000;
      input.v[2][lane] = 0;
      input.v[3][lane] = 0xdeadbeef;
    }
    for (bool enabled_row : {false, true}) {
      auto result = input;
      uint32_t status = 0x80000000;
      const uint64_t dpp = RJ_ARITH_DPP16 | (0xe4ULL << RJ_ARITH_DPP_CTRL_SHIFT) |
                           RJ_ARITH_DPP_BANK_MASK |
                           (enabled_row ? 1ULL << (RJ_ARITH_DPP_ROW_SHIFT + Lanes / 16 - 1) : 0);
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_DX10_CLAMP_OFF, ~Mask(0),
                     dpp, result.v + 3, input.v, input.v + 1, input.v + 2, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(status, 0x80000000U | (enabled_row ? RJ_ARITH_EXCEPTION_INVALID : 0));
      for (int lane = 0; lane < Lanes - 16; ++lane) {
        EXPECT_EQ(result.v[3][lane], input.v[3][lane]);
      }
      if (enabled_row) {
        EXPECT_EQ(result.v[3][Lanes - 1] & 0x7fffffffU, 0x7f800123U);
      } else {
        EXPECT_EQ(result.v[3][Lanes - 1], 0xdeadbeefU);
      }
    }
  }
}

} // namespace

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpAddF32W32Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32_gfx1150, [](double x, double y) { return x + y; }, /*gfx12=*/false);
  check_causes<32>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_add_f32_w32_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/false);
}

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpAddF32W64Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64_gfx1150, [](double x, double y) { return x + y; }, /*gfx12=*/false);
  check_causes<64>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_add_f32_w64_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/false);
}

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpSubF32W32Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32_gfx1150, [](double x, double y) { return x - y; }, /*gfx12=*/false);
  check_causes<32>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_sub_f32_w32_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/false);
}

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpSubF32W64Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64_gfx1150, [](double x, double y) { return x - y; }, /*gfx12=*/false);
  check_causes<64>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_sub_f32_w64_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/false);
}

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpSubrevF32W32Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32_gfx1150, [](double x, double y) { return y - x; }, /*gfx12=*/false);
  check_causes<32>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_subrev_f32_w32_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/false);
}

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpSubrevF32W64Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64_gfx1150, [](double x, double y) { return y - x; }, /*gfx12=*/false);
  check_causes<64>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_subrev_f32_w64_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/false);
}

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpMulF32W32Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_mul_f32_w32_gfx1150, [](double x, double y) { return x * y; }, /*gfx12=*/false);
  check_causes<32>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_mul_f32_w32_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/true);
}

// Check the numeric API contract and inherited optional Compatibility causes.
TEST(FpMulF32W64Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64_gfx1150, [](double x, double y) { return x * y; }, /*gfx12=*/false);
  check_causes<64>(
      [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
         const uint32_t *const *, auto status) {
        return rj_arith_v_mul_f32_w64_gfx1150(flags, exec_mask, modifiers, d, a, b, status);
      },
      /*fused=*/false, /*multiply=*/true);
}

// Check fused arithmetic and its gfx1150 Compatibility reporting policy.
TEST(FpFmaF32W32Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 32>(rj_arith_v_fma_f32_w32_gfx1150, /*gfx12=*/false);
  check_causes<32>(rj_arith_v_fma_f32_w32_gfx1150, /*fused=*/true, /*multiply=*/false);
}

// Check fused arithmetic and its gfx1150 Compatibility reporting policy.
TEST(FpFmaF32W64Gfx1150, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 64>(rj_arith_v_fma_f32_w64_gfx1150, /*gfx12=*/false);
  check_causes<64>(rj_arith_v_fma_f32_w64_gfx1150, /*fused=*/true, /*multiply=*/false);
}
