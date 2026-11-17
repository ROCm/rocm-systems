// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point multiplication.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/binary_arithmetic_test.h"
#include "fp/captured_gfx11_test.h"
#include "fp/elementwise.h"
#include "rj_arith/detail/rj_arith_fp.h"

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

// Tininess is judged after rounding. (1 - 2^-11) * 2^-14 is exact at FP16 precision, so it is
// tiny in every rounding mode although subnormal rounding can reach the smallest normal;
// (1 - 2^-20) * 2^-14 is not tiny unless rounded down. Flushing outputs and OMOD zero tiny
// results, OMOD even when MODE keeps denormals. Halving gives a signed zero only when the
// rounded result is below 2^-13. Expected results are gfx1201 captures in nearest, up, down
// and zero rounding (~0U: not captured); gfx1100 flushes the same products.
TEST(FpMulF16, TinyAfterRounding) {
  using namespace rocjitsu::arith::test;
  constexpr uint32_t not_captured = ~0U;
  const struct {
    uint32_t a, b;
    uint64_t fp_flags;
    uint32_t instruction_flags;
    uint32_t expected[4];
  } cases[] = {
      {0x3bff, 0x0400, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, {0, 0, 0, 0}},
      {0x3bff, 0x8400, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, {0x8000, 0x8000, 0x8000, 0x8000}},
      {0x3bff, 0x0400, 0, 0, {0x0400, 0x0400, 0x03ff, 0x03ff}},
      {0x3bfe, 0x0401, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, {0x0400, 0x0400, 0, 0}},
      {0x3bff, 0x0400, 0, RJ_ARITH_ALU_OMOD_2, {0, 0, 0, 0}},
      {0x3bfe, 0x0401, 0, RJ_ARITH_ALU_OMOD_2, {0x0800, 0x0800, 0, 0}},
      {0x3800, 0x87ff, 0, RJ_ARITH_ALU_OMOD_HALF, {0, 0, 0, 0}},
      {0x9873, 0x2b31, 0, RJ_ARITH_ALU_OMOD_HALF, {0x8400, not_captured, not_captured, 0x8000}},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &test : cases) {
        for (unsigned rounding = 0; rounding < 4; ++rounding) {
          if (test.expected[rounding] == not_captured) {
            continue;
          }
          const uint64_t flags = cpu | semantics | test.fp_flags | (uint64_t(rounding) << 24);
          // gfx11 applies OMOD only when output denormals are flushed.
          for (auto call : {rj_arith_v_mul_f16_w32, rj_arith_v_mul_f16_w32_gfx12}) {
            if (call == rj_arith_v_mul_f16_w32 && test.instruction_flags) {
              continue;
            }
            Registers<3, 32> r;
            r.v[0][0] = test.a;
            r.v[1][0] = test.b;
            r.v[2][0] = 0xa5a50000U;
            ASSERT_EQ(call(flags, 1, test.instruction_flags, r.v + 2, r.v, r.v + 1, nullptr),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(r.v[2][0], 0xa5a50000U | test.expected[rounding])
                << std::hex << test.a << " * " << test.b << " rounding " << rounding;
          }
        }
      }
    }
  }
}

TEST(FpMulF64, GuestRounding) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const struct {
    uint64_t a, b, expected[4];
  } cases[] = {{0x3ff0000000000001ULL,
                0x3ff0000000000001ULL,
                {0x3ff0000000000002ULL, 0x3ff0000000000003ULL, 0x3ff0000000000002ULL,
                 0x3ff0000000000002ULL}},
               {0xbff0000000000001ULL,
                0x3ff0000000000001ULL,
                {0xbff0000000000002ULL, 0xbff0000000000002ULL, 0xbff0000000000003ULL,
                 0xbff0000000000002ULL}},
               {0x8000000000000000ULL,
                0x3ff0000000000000ULL,
                {0x8000000000000000ULL, 0x8000000000000000ULL, 0x8000000000000000ULL,
                 0x8000000000000000ULL}},
               {0x7fefffffffffffffULL,
                0x4000000000000000ULL,
                {0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7fefffffffffffffULL,
                 0x7fefffffffffffffULL}}};
  for (const auto &sample : cases) {
    for (auto call : {rj_arith_v_mul_f64_w32, rj_arith_v_mul_f64_w32_gfx12}) {
      check_binary64_rounding<32>(call, sample.a, sample.b, sample.expected);
    }
    for (auto call : {rj_arith_v_mul_f64_w64, rj_arith_v_mul_f64_w64_gfx12}) {
      check_binary64_rounding<64>(call, sample.a, sample.b, sample.expected);
    }
  }
}

TEST(FpMulF64, OutputScalingAndFlushing) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const uint64_t overflow[] = {0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7fefffffffffffffULL,
                               0x7fefffffffffffffULL};
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, 0x7fefffffffffffffULL,
                              0x3ff0000000000000ULL, overflow, RJ_ARITH_ALU_OMOD_2);
}

// Tininess is judged after rounding. (1 - 2^-53) * 2^-1022 is exact at full precision, so it
// is tiny in every rounding mode although subnormal rounding can reach the smallest normal;
// (1 - 2^-52) * (2^-1022 + 2^-1074) is not tiny unless rounded down. Flushing outputs and OMOD
// zero tiny results, OMOD even when MODE keeps denormals. Expected results are gfx1201
// captures in nearest, up, down and zero rounding; gfx1100 flushes the same products.
TEST(FpMulF64, TinyAfterRounding) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const uint64_t tiny = 0x3fefffffffffffffULL, not_tiny = 0x3feffffffffffffeULL;
  const uint64_t zero[] = {0, 0, 0, 0};
  const uint64_t kept[] = {0x0010000000000000ULL, 0x0010000000000000ULL, 0x000fffffffffffffULL,
                           0x000fffffffffffffULL};
  const uint64_t not_tiny_flushed[] = {0x0010000000000000ULL, 0x0010000000000000ULL, 0, 0};
  const uint64_t not_tiny_doubled[] = {0x0020000000000000ULL, 0x0020000000000000ULL, 0, 0};
  for (auto call : {rj_arith_v_mul_f64_w32, rj_arith_v_mul_f64_w32_gfx12}) {
    check_binary64_rounding<32>(call, tiny, 0x0010000000000000ULL, zero, 0,
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
    check_binary64_rounding<32>(call, not_tiny, 0x0010000000000001ULL, not_tiny_flushed, 0,
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
  }
  for (auto call : {rj_arith_v_mul_f64_w64, rj_arith_v_mul_f64_w64_gfx12}) {
    check_binary64_rounding<64>(call, tiny, 0x0010000000000000ULL, zero, 0,
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
  }
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, tiny, 0x0010000000000000ULL, kept);
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, tiny, 0x0010000000000000ULL, zero,
                              RJ_ARITH_ALU_OMOD_2);
  check_binary64_rounding<64>(rj_arith_v_mul_f64_w64_gfx12, not_tiny, 0x0010000000000001ULL,
                              not_tiny_doubled, RJ_ARITH_ALU_OMOD_2);
}

// gfx1201 captures: FP64 MUL returns the first NaN source, quieted, in operand order under every
// guest rounding mode.
TEST(FpMulF64, NanPriority) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const uint64_t cases[][3] = {
      {0x7ff0000000000001ULL, 0x7ff8000000000000ULL, 0x7ff8000000000001ULL},
      {0xfff8000000000000ULL, 0x7ff8000000000000ULL, 0xfff8000000000000ULL},
      {0x7ff4000012345678ULL, 0x7ff8000000000000ULL, 0x7ffc000012345678ULL},
      {0x7ff8000000000000ULL, 0x7ff0000000000001ULL, 0x7ff8000000000000ULL},
      {0xfff8000000000000ULL, 0x7ff0000000000001ULL, 0xfff8000000000000ULL},
  };
  for (const auto &sample : cases) {
    const uint64_t expected[] = {sample[2], sample[2], sample[2], sample[2]};
    check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, sample[0], sample[1], expected);
    check_binary64_rounding<64>(rj_arith_v_mul_f64_w64_gfx12, sample[0], sample[1], expected);
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 lanes from the MODE.IEEE-clear corpus (hw1100-noieee): a signaling NaN source is
// returned unquieted, also under directed rounding and OMOD.
const CapturedGfx11 mul_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x6b007c01, 0x4f160000, 0, 0xa5a57c01},
    {"E32RoundDown", 0, 0x15a, 0x323e0000, 0xd51d7c01, 0, 0xa5a57c01},
    {"NegB", RJ_ARITH_ALU_NEG_B, 0x1f0, 0xde7e0000, 0xa4cc7c01, 0, 0xa5a5fc01},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x32a07c01, 0xab200000, 0, 0xa5a57c01},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0x1f0,
     0xfdc50400, 0x5e2a0000, 0, 0xfdc5a5a5},
};

// gfx1100 (W7900) CLAMP lanes with a signaling-NaN source, also under OMOD (active with output
// flushing). DX10_CLAMP clear lets the NaN through CLAMP, unquieted with IEEE clear and quieted
// with IEEE set; DX10_CLAMP set clamps it to 0.
const CapturedGfx11 mul_f16_clamp_lanes[] = {
    {"Dx10ClampOff", RJ_ARITH_ALU_CLAMP, 0x0f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a57c01},
    {"Dx10ClampOn", RJ_ARITH_ALU_CLAMP, 0x1f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a50000},
    {"IeeeSet", RJ_ARITH_ALU_CLAMP, 0x2f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a57e01},
    {"Mul2Dx10ClampOff", RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_2, 0x000, 0xa5a57c01, 0xa5a53c00, 0,
     0xa5a57c01},
    {"Mul2Dx10ClampOn", RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_2, 0x100, 0xa5a57c01, 0xa5a53c00, 0,
     0xa5a50000},
    {"Mul2IeeeSet", RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_2, 0x200, 0xa5a57c01, 0xa5a53c00, 0,
     0xa5a57e01},
};

// gfx1100 MUL_F64 returns the first NaN source with its modifiers applied, quieted only when
// MODE.IEEE is set. IEEE-set lanes are from the gfx1100 v2 capture (hw2-gfx1100).
const CapturedGfx11 mul_f64_lanes[] = {
    {"SignalingFirst", 0, 0x1f0, 0x7ff0000000000001, 0x7ff8000000000000, 0, 0x7ff0000000000001},
    {"NegativeQuietFirst", 0, 0x1f0, 0xfff8000000000000, 0x7ff8000000000000, 0, 0xfff8000000000000},
    {"SignalingSecondRoundZero", 0, 0x1ff, 0x0, 0x7ff0000000000001, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff8000000000000, 0x7ff8000000000000, 0,
     0xfff8000000000000},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x7ff0000000000001, 0x0, 0, 0x7ff0000000000001},
    {"IeeeSetSignalingFirst", 0, 0x3f0, 0x7ff0000000000001, 0x7ff8000000000000, 0,
     0x7ff8000000000001},
    {"IeeeSetNegativeQuietFirst", 0, 0x3f0, 0xfff8000000000000, 0x7ff8000000000000, 0,
     0xfff8000000000000},
};

} // namespace

// Compatibility reproduces captured gfx1100 FP16 MUL lanes.
TEST(FpMulF16Gfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 16>(rj_arith_v_mul_f16_w32, mul_f16_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_mul_f16_w64, mul_f16_lanes);
}

// Compatibility reproduces captured gfx1100 clamped FP16 MUL NaN lanes.
TEST(FpMulF16Gfx11, CapturedClampedNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 16>(rj_arith_v_mul_f16_w32, mul_f16_clamp_lanes);
  check_captured_gfx11<64, 16>(rj_arith_v_mul_f16_w64, mul_f16_clamp_lanes);
}

// Compatibility reproduces captured gfx1100 FP64 MUL NaN lanes.
TEST(FpMulF64Gfx11, CapturedNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 64>(rj_arith_v_mul_f64_w32, mul_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_mul_f64_w64, mul_f64_lanes);
}

// Compatibility retains a normal FP16 product recovered by narrowing before OMOD.
TEST(FpMulF16Cdna, RoundedMinimumNormal) {
  using namespace rocjitsu::arith::test;
  for (auto cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_mul_f16_w32_gfx942, rj_arith_v_mul_f16_w32_gfx1250}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x3bff;
      r.v[1][0] = 0x0400;
      const uint64_t flags =
          cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
      ASSERT_EQ(call(flags, 1, RJ_ARITH_ALU_OMOD_2, r.v + 2, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0] & 0xffffU, 0x0800U);
    }
  }
}
