// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP32 multiplication with DX9 zero-product rules.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace {

// Fill active destination elements from the independent arithmetic reference.
template <int Lanes>
void compute_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input, uint64_t exec_mask,
                      uint64_t mode, int aliasing_offset,
                      rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = mode & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, source_lane, 32, 0);
    if ((mode & RJ_ARITH_DPP8) && !(mode & RJ_ARITH_DPP_FI) && !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, 32, 0);

    double x = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(a, 32), mode, 0);
    double y = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(b, 32), mode, 1);
    double value = x * y;
    if (x == 0 || y == 0) {
      value = 0;
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    }
    uint64_t expected = rocjitsu::arith::test::fp_bits(value, 32);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, 32, expected, 0);
  }
}

// Exercise this instruction contract for either wave size.
template <int Lanes, class Call> void check_instruction(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U;

    int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, &fp_status), result);
    return result;
  });

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A), uint64_t(RJ_ARITH_ALU_CLAMP),
                            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT)),
                            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) |
                                     RJ_ARITH_DPP_FI | RJ_ARITH_ALU_NEG_A)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2}) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result =
                call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            compute_expected(input, exec_mask, mode, aliasing_offset, expected_registers);

            rocjitsu::arith::test::expect_fp_result<32>(input, r, expected_registers, exec_mask,
                                                        aliasing_offset, 0, semantics,
                                                        /*atol=*/1e-5, /*rtol=*/1e-6);
          }
        }
      }
    }
  }
}

} // namespace

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32Gfx12, Arithmetic) {
  check_instruction<32>(rj_arith_v_mul_dx9_zero_f32_w32_gfx12);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32, Arithmetic) { check_instruction<32>(rj_arith_v_mul_dx9_zero_f32_w32); }

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W64Gfx12, Arithmetic) {
  check_instruction<64>(rj_arith_v_mul_dx9_zero_f32_w64_gfx12);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W64, Arithmetic) { check_instruction<64>(rj_arith_v_mul_dx9_zero_f32_w64); }
