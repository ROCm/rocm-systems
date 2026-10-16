// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point multiplication.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/binary_arithmetic_test.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
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
TEST(FpMulF16W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

namespace {

// Literal exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
struct StatusCase {
  uint64_t a, b, c;
  uint32_t causes;
};

const StatusCase status_cases_f32[] = {
    {0x3f800000U, 0x40000000U, 0x0U, 0},
    {0x0U, 0x7f800000U, 0x0U, 0},
    {0x1U, 0x3f800000U, 0x0U, RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW},
    {0x1U, 0x3f000000U, 0x0U,
     RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x7f7fffffU, 0x40000000U, 0x0U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f800001U, 0x3f800001U, 0x0U, RJ_ARITH_EXCEPTION_INEXACT},
};

// Check inherited MUL exception causes and accumulation, including empty EXEC.
template <int Lanes, class Call> void check_status(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (const auto &test : status_cases_f32) {
      for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_CLAMP)}) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << mode << '/' << test.a << '/' << test.b << '/' << test.c);
        Registers r = input;
        const int lane = Lanes - 1;
        for (int part = 0; part < 1; ++part) {
          rocjitsu::arith::test::write_fp(r.v, lane, 32, test.a, part);
          rocjitsu::arith::test::write_fp(r.v + 2, lane, 32, test.b, part);
          rocjitsu::arith::test::write_fp(r.v + 4, lane, 32, test.c, part);
        }
        Mask exec_mask = Mask(1) << lane;
        uint32_t fp_status = 0x80000000U;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6, r.v,
                       r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        uint32_t causes = test.causes;
        EXPECT_EQ(fp_status, 0x80000000U | causes);
        // Preserve previously accumulated causes as well as unrelated bits.
        fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
        for (int part = 0; part < 1; ++part) {
          rocjitsu::arith::test::write_fp(r.v + 4, lane, 32, test.c, part);
        }
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6, r.v,
                       r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
        Registers before = r;
        exec_mask = 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6, r.v,
                       r.v + 2, &fp_status),
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
TEST(FpMulF32W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_mul_f32_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/true);
  check_status<32>(rj_arith_v_mul_f32_w32);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true,
      /*fp_status_supported=*/true);
  check_status<64>(rj_arith_v_mul_f32_w64_gfx12);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/true);
  check_status<64>(rj_arith_v_mul_f32_w64);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_mul_f64_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_mul_f64_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_mul_f64_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_mul_f64_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}
