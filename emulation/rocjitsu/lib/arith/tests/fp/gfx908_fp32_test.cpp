// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx908 ADD/MUL reuse guest numerical kernels but retain their distinct
// Compatibility IEEE causes, raw-denormal observation and explicit output flush.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/binary_arithmetic_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace {

// Literal IEEE witnesses distinguish gfx908 from the later vector FMA classifier.
// Numerical fractional/mask/modifier coverage comes from the common binary runner.
template <bool Multiply, int Lanes, typename Call> void check_causes(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const struct {
    uint32_t a, b, causes;
    uint64_t fp;
    uint32_t modifiers;
  } cases[] = {
      {0x3f800000, 0x3f800000, 0, 0, 0},
      {0, 0x7f800000, Multiply ? RJ_ARITH_EXCEPTION_INVALID : 0U, 0, 0},
      {0x7f800000, 0xff800000, Multiply ? 0U : RJ_ARITH_EXCEPTION_INVALID, 0, 0},
      {0x7f800123, 0x3f800000, RJ_ARITH_EXCEPTION_INVALID, 0, 0},
      {1, 0x7fc00000, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0, 0},
      {1, 0x7fc00000, RJ_ARITH_EXCEPTION_INPUT_DENORM, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0},
      {1, 0x3f800000, RJ_ARITH_EXCEPTION_INPUT_DENORM, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0},
      {0x3f800001, Multiply ? 0x3f800001U : 0x33800000U, RJ_ARITH_EXCEPTION_INEXACT, 0, 0},
      {0x7f7fffff, Multiply ? 0x40000000U : 0x7f7fffffU,
       RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT, 0, RJ_ARITH_ALU_CLAMP},
      {0x00800000, Multiply ? 0x3f000000U : 0x807fffffU,
       Multiply ? 0U : RJ_ARITH_EXCEPTION_INPUT_DENORM, 0, 0},
      {0x00800000, Multiply ? 0x3f000000U : 0x807fffffU,
       RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT |
           (Multiply ? 0U : RJ_ARITH_EXCEPTION_INPUT_DENORM),
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {0x7f7fffff, Multiply ? 0x3f800000U : 0U,
       RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
       RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
      {Multiply ? 0x3f7fffffU : 0x3f800000U, Multiply ? 0x00800000U : 0x3f800000U,
       Multiply ? RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT : 0U, 0, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      for (uint64_t rounding :
           {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
        for (uint64_t semantics : semantics_modes()) {
          for (uint64_t dx10 : {uint64_t(0), RJ_ARITH_FP_DX10_CLAMP_OFF}) {
            for (int aliasing_offset : {0, 1, 2}) {
              SCOPED_TRACE(testing::Message() << cpu << '/' << sample.a << '/' << semantics << '/'
                                              << dx10 << '/' << aliasing_offset);
              Registers<3, Lanes> input;
              for (int lane = 0; lane < Lanes; ++lane) {
                input.v[0][lane] = sample.a;
                input.v[1][lane] = sample.b;
                input.v[2][lane] = 0xdeadbeef;
              }
              auto result = input;
              const Mask exec_mask = Mask(1) | (Mask(1) << (Lanes - 1));
              const uint64_t flags = cpu | semantics | sample.fp | dx10 | rounding;
              uint32_t status = 0x80000000;
              const int error = call(flags, exec_mask, sample.modifiers, result.v + aliasing_offset,
                                     result.v, result.v + 1, &status);
              if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
                EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
                result.expect_preserved(input, 0, 0);
                EXPECT_EQ(status, 0x80000000U);
                continue;
              }
              ASSERT_EQ(error, RJ_ARITH_SUCCESS);
              EXPECT_EQ(status,
                        0x80000000U |
                            (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY ? sample.causes : 0));
              result.expect_preserved(input, aliasing_offset, 1);
              for (int lane = 1; lane < Lanes - 1; ++lane) {
                EXPECT_EQ(result.v[aliasing_offset][lane], input.v[aliasing_offset][lane]);
              }
              EXPECT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, Mask(0), 0, nullptr, nullptr,
                             nullptr, nullptr),
                        RJ_ARITH_SUCCESS);
              if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
                EXPECT_EQ(call(flags, exec_mask, sample.modifiers, result.v + 2, input.v,
                               input.v + 1, nullptr),
                          RJ_ARITH_SUCCESS);
              }
            }
          }
        }
      }
    }
    Registers<3, Lanes> input;
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[0][lane] = 0x7f800123;
      input.v[1][lane] = 0x3f800000;
      input.v[2][lane] = 0xdeadbeef;
    }
    for (bool enabled_row : {false, true}) {
      auto result = input;
      uint32_t status = 0x80000000;
      const uint64_t dpp = RJ_ARITH_DPP16 | (0xe4ULL << RJ_ARITH_DPP_CTRL_SHIFT) |
                           RJ_ARITH_DPP_BANK_MASK |
                           (enabled_row ? 1ULL << (RJ_ARITH_DPP_ROW_SHIFT + Lanes / 16 - 1) : 0);
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, ~Mask(0), dpp, result.v + 2, input.v,
                     input.v + 1, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(status, 0x80000000U | (enabled_row ? RJ_ARITH_EXCEPTION_INVALID : 0));
      for (int lane = 0; lane < Lanes - 16; ++lane) {
        EXPECT_EQ(result.v[2][lane], input.v[2][lane]);
      }
      EXPECT_EQ(result.v[2][Lanes - 1], enabled_row ? 0x7f800123U : 0xdeadbeefU);
    }
  }
}

} // namespace

// Check numerical contracts, masks/aliases and inherited Compatibility causes.
TEST(FpAddF32W32Gfx908, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32_gfx908, [](double x, double y) { return x + y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false, {.compatibility_reports_status = true});
  check_causes<false, 32>(rj_arith_v_add_f32_w32_gfx908);
}

// Check numerical contracts, masks/aliases and inherited Compatibility causes.
TEST(FpAddF32W64Gfx908, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64_gfx908, [](double x, double y) { return x + y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false, {.compatibility_reports_status = true});
  check_causes<false, 64>(rj_arith_v_add_f32_w64_gfx908);
}

// Check numerical contracts, masks/aliases and inherited Compatibility causes.
TEST(FpMulF32W32Gfx908, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_mul_f32_w32_gfx908, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false, {.compatibility_reports_status = true});
  check_causes<true, 32>(rj_arith_v_mul_f32_w32_gfx908);
}

// Check numerical contracts, masks/aliases and inherited Compatibility causes.
TEST(FpMulF32W64Gfx908, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64_gfx908, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false, {.compatibility_reports_status = true});
  check_causes<true, 64>(rj_arith_v_mul_f32_w64_gfx908);
}
