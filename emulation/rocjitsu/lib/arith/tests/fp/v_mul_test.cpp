// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point multiplication.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/binary_arithmetic_test.h"
#include "fp/elementwise.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

namespace {

// Selected gfx1201 default-mode FP exception contributions.
// Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
struct StatusCase {
  uint32_t a, b;
  uint32_t causes;
};

const StatusCase status_cases_f32[] = {
    {0x3f800000U, 0x40000000U, 0},
    {0x00ffffffU, 0x3f000000U, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f118e00U, 0x00e12000U, RJ_ARITH_EXCEPTION_INEXACT},
    {0x00000001U, 0x7fc00000U, 0},
    {0x00000001U, 0x7f800001U, RJ_ARITH_EXCEPTION_INVALID},
    {0x0U, 0x7f800000U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7f800001U, 0x3f800000U, RJ_ARITH_EXCEPTION_INVALID},
    {0x1U, 0x3f800000U, RJ_ARITH_EXCEPTION_INPUT_DENORM},
    {0x1U, 0x3f000000U,
     RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x7f7fffffU, 0x40000000U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f800001U, 0x3f800001U, RJ_ARITH_EXCEPTION_INEXACT},
};

// Check inherited MUL exception causes and accumulation, including empty EXEC.
template <int Lanes, typename Call> void check_status(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (const auto &test : status_cases_f32) {
      for (uint64_t instruction_flags : {uint64_t(0)}) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << instruction_flags << '/' << test.a << '/' << test.b);
        Registers r = input;
        const int lane = Lanes - 1;
        rocjitsu::arith::test::write_fp(r.v, lane, 32, test.a);
        rocjitsu::arith::test::write_fp(r.v + 2, lane, 32, test.b);
        Mask exec_mask = Mask(1) << lane;
        uint32_t fp_status = 0x80000000U;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, instruction_flags,
                       r.v + 6, r.v, r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        uint32_t causes = test.causes;
        EXPECT_EQ(fp_status, 0x80000000U | causes);
        // Preserve previously accumulated causes as well as unrelated bits.
        fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, instruction_flags,
                       r.v + 6, r.v, r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
        Registers before = r;
        exec_mask = 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, instruction_flags,
                       r.v + 6, r.v, r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        r.expect_preserved(before, 0, 0);
        EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
      }
    }
  }
}

} // namespace

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_mul_f32_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true,
      /*fp_status_supported=*/true);
  check_status<32>(rj_arith_v_mul_f32_w32_gfx12);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_mul_f32_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true,
      /*fp_status_supported=*/true);
  check_status<64>(rj_arith_v_mul_f32_w64_gfx12);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_mul_f64_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_mul_f64_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_mul_f64_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_mul_f64_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Reject unqualified status domains before touching outputs; Compatibility
// leaves status untouched even when the arithmetic overflows.
TEST(FpMulF32, ReportingDomain) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_mul_f32_w32, rj_arith_v_mul_f32_w32_gfx12}) {
      for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS,
                          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
        for (uint32_t instruction_flags : {0U, RJ_ARITH_ALU_CLAMP, RJ_ARITH_ALU_OMOD_2}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x7f7fffffU;
          r.v[1][0] = 0x40000000U;
          r.v[2][0] = 0xdeadbeefU;
          uint32_t status = 0x80000000U;
          if (call != rj_arith_v_mul_f32_w32_gfx12 || fp || instruction_flags) {
            EXPECT_EQ(call(cpu | fp | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 1, instruction_flags,
                           nullptr, nullptr, nullptr, &status),
                      RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
          }
          ASSERT_EQ(call(cpu | fp | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, instruction_flags, r.v + 2,
                         r.v, r.v + 1, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x80000000U);
        }
      }
    }
  }
}

namespace {

// Inexact positive/negative products distinguish directed rounding without a
// computed FP reference. All lanes are finite so SIMD stays eligible.
template <int Lanes, typename Call> void check_mul_rounding(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (uint64_t rounding :
           {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
        for (bool negative : {false, true}) {
          SCOPED_TRACE(testing::Message()
                       << cpu << '/' << semantics << '/' << rounding << '/' << negative);
          Registers<3, Lanes> r;
          const uint32_t sign = negative ? 0x80000000U : 0;
          for (int lane = 0; lane < Lanes; ++lane) {
            r.v[0][lane] = sign | 0x3f800001U;
            r.v[1][lane] = 0x3f800001U;
          }
          ASSERT_EQ(call(cpu | semantics | rounding, ~Mask(0), 0, r.v + 2, r.v, r.v + 1, nullptr),
                    RJ_ARITH_SUCCESS);
          const bool away = rounding == (negative ? RJ_ARITH_FP_ROUND_DOWN : RJ_ARITH_FP_ROUND_UP);
          const uint32_t expected = sign | (away ? 0x3f800003U : 0x3f800002U);
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(r.v[2][lane], expected);
          }
        }
      }
    }
  }
}

} // namespace

TEST(FpMulF32, DirectedRounding) {
  check_mul_rounding<32>(rj_arith_v_mul_f32_w32);
  check_mul_rounding<32>(rj_arith_v_mul_f32_w32_gfx12);
  check_mul_rounding<64>(rj_arith_v_mul_f32_w64);
  check_mul_rounding<64>(rj_arith_v_mul_f32_w64_gfx12);
}

// Compatibility preserves intermediate overflow; Loose can keep the finite wide product.
TEST(FpMulF16, CompatibilityOmodRounding) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_mul_f16_w32, rj_arith_v_mul_f16_w32_gfx12}) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
        for (uint64_t rounding :
             {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x7bffU;
          r.v[1][0] = 0x4000U;
          uint32_t status = 0;
          ASSERT_EQ(call(cpu | semantics | rounding | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1,
                         RJ_ARITH_ALU_OMOD_HALF, r.v + 2, r.v, r.v + 1, &status),
                    RJ_ARITH_SUCCESS);
          uint32_t expected =
              semantics == RJ_ARITH_SEMANTICS_LOOSE                                      ? 0x7bffU
              : rounding == RJ_ARITH_FP_ROUND_DOWN || rounding == RJ_ARITH_FP_ROUND_ZERO ? 0x77ffU
                                                                                         : 0x7c00U;
          EXPECT_EQ(r.v[2][0] & 0xffffU, expected);
          EXPECT_EQ(status, 0U);
        }
      }
    }
  }
}

TEST(FpMulF64, GuestRounding) {
  using rocjitsu::arith::test::check_binary64_rounding;
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32, 0x3ff0000000000001ULL, 0x3ff0000000000001ULL,
                              0x3ff0000000000002ULL);
  check_binary64_rounding<64>(rj_arith_v_mul_f64_w64, 0x3ff0000000000001ULL, 0x3ff0000000000001ULL,
                              0x3ff0000000000002ULL);
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, 0x3ff0000000000001ULL,
                              0x3ff0000000000001ULL, 0x3ff0000000000002ULL);
  check_binary64_rounding<64>(rj_arith_v_mul_f64_w64_gfx12, 0x3ff0000000000001ULL,
                              0x3ff0000000000001ULL, 0x3ff0000000000002ULL);
}
