// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP32 multiplication with DX9 zero-product rules.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <algorithm>
#include <bit>
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
                      uint64_t instruction_flags, int aliasing_offset,
                      rocjitsu::arith::test::Registers<10, Lanes> &expected_registers) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    // DPP8 reverses each group; FI controls reads from inactive sources.
    int source_lane = instruction_flags & RJ_ARITH_DPP8 ? (lane & ~7) + 7 - (lane & 7) : lane;
    uint64_t a = rocjitsu::arith::test::read_fp(input.v, source_lane, 32, 0);
    if ((instruction_flags & RJ_ARITH_DPP8) && !(instruction_flags & RJ_ARITH_DPP_FI) &&
        !((exec_mask >> source_lane) & 1)) {
      a = 0;
    }
    uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, 32, 0);

    double x = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(a, 32),
                                                  instruction_flags, 0);
    double y = rocjitsu::arith::test::modified_fp(rocjitsu::arith::test::fp_value(b, 32),
                                                  instruction_flags, 1);
    double value = x * y;
    if (x == 0 || y == 0) {
      value = 0;
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    }
    uint64_t expected = rocjitsu::arith::test::fp_bits(value, 32);
    rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, 32, expected, 0);
  }
}

// Exercise this instruction contract for either wave size.
template <int Lanes, typename Call> void check_instruction(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();
  // DX9 zero products suppress Inf rather than producing IEEE's invalid NaN.
  input.v[0][8] = 0;
  input.v[2][8] = 0x7f800000U;
  input.v[0][9] = 0x7f800000U;
  input.v[2][9] = 0x80000000U;

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags,
                                                        uint64_t instruction_flags) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U;

    int result =
        call(flags, exec_mask, instruction_flags, r.v + aliasing_offset, r.v, r.v + 2, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U);
    EXPECT_EQ(call(flags, exec_mask, instruction_flags, nullptr, nullptr, nullptr, &fp_status),
              result);
    return result;
  });

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t instruction_flags :
           {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A), uint64_t(RJ_ARITH_ALU_CLAMP),
            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT)),
            uint64_t(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                     RJ_ARITH_ALU_NEG_A)}) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2}) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << instruction_flags
                                              << '/' << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, instruction_flags, r.v + aliasing_offset, r.v,
                              r.v + 2, &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            EXPECT_EQ(fp_status, 0x80000000U);
            compute_expected(input, exec_mask, instruction_flags, aliasing_offset,
                             expected_registers);

            rocjitsu::arith::test::expect_fp_result<32>(input, r, expected_registers, exec_mask,
                                                        aliasing_offset, 0, semantics,
                                                        /*atol=*/1e-5, /*rtol=*/1e-6);
          }
        }
      }
    }
  }
}

// Zero lanes in an otherwise normal wave: the special inputs elsewhere in make_fp_inputs keep
// waves off the SIMD exact path, so this wave reaches it. A zero, or a denormal under guest
// input flushing, gives +0 against Inf, NaN and negative operands.
template <typename Call> void check_zero_lanes(Call call) {
  const uint32_t others[] = {0x7f800000U, 0xff800000U, 0x7fc00000U, 0x7fa00000U, 0xc0400000U};
  uint32_t a[32], b[32];
  for (int lane = 0; lane < 32; ++lane) {
    a[lane] = 0x3fc00000U + uint32_t(lane) * 0x00012345U;
    b[lane] = (lane & 1 ? 0xc0000000U : 0x40000000U) + uint32_t(lane) * 0x00054321U;
  }
  for (int lane = 3; lane < 32; lane += 4) {
    a[lane] = lane & 4 ? 0x80000000U : 0x00000000U;
    b[lane] = others[lane % 5];
  }
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t flush : {uint64_t(0), uint64_t(RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)}) {
      for (uint64_t instruction_flags : {uint64_t(0), uint64_t(RJ_ARITH_ALU_NEG_A),
                                         uint64_t(RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP)}) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << flush << '/' << instruction_flags);
        uint32_t x[32], y[32], d[32];
        std::copy(a, a + 32, x);
        std::copy(b, b + 32, y);
        if (flush) {
          x[3] = 0x80000001U;
          y[7] = 0x007fffffU;
          x[7] = 0x7f800000U;
        }
        const uint32_t *ap[] = {x}, *bp[] = {y};
        uint32_t *dp[] = {d};
        uint32_t fp_status = 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | flush, ~0U,
                       instruction_flags, dp, ap, bp, &fp_status),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          if (lane % 4 == 3) {
            EXPECT_EQ(d[lane], 0U) << lane;
            continue;
          }
          double value = double(std::bit_cast<float>(x[lane])) * std::bit_cast<float>(y[lane]);
          if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
            value = -value;
          }
          if (instruction_flags & RJ_ARITH_ALU_NEG_B) {
            value = -value;
          }
          if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
            value = !(value > 0) ? 0 : std::min(value, 1.0);
          }
          EXPECT_EQ(d[lane], std::bit_cast<uint32_t>(float(value))) << lane;
        }
      }
    }
  }
}

// One subnormal lane in an otherwise normal wave: only guest input flushing makes it a DX9
// zero. Without flushing, the largest subnormal times 2 is the exact product 0x00fffffe.
template <typename Call> void check_subnormal_lane(Call call) {
  constexpr int subnormal_lane = 5;
  uint32_t a[32], b[32];
  for (int lane = 0; lane < 32; ++lane) {
    a[lane] = 0x3fc00000U + uint32_t(lane) * 0x00012345U;
    b[lane] = (lane & 1 ? 0xc0000000U : 0x40000000U) + uint32_t(lane) * 0x00054321U;
  }
  a[subnormal_lane] = 0x007fffffU;
  b[subnormal_lane] = 0x40000000U;
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t flush : {uint64_t(0), uint64_t(RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)}) {
      SCOPED_TRACE(::testing::Message() << cpu << '/' << flush);
      uint32_t d[32];
      const uint32_t *ap[] = {a}, *bp[] = {b};
      uint32_t *dp[] = {d};
      uint32_t fp_status = 0;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | flush, ~0U, 0, dp,
                     ap, bp, &fp_status),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        uint32_t expected =
            lane != subnormal_lane
                ? std::bit_cast<uint32_t>(
                      float(double(std::bit_cast<float>(a[lane])) * std::bit_cast<float>(b[lane])))
            : flush ? 0U
                    : 0x00fffffeU;
        EXPECT_EQ(d[lane], expected) << lane;
      }
    }
  }
}

} // namespace

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32Gfx12, SubnormalLaneFollowsInputFlushing) {
  check_subnormal_lane(rj_arith_v_mul_dx9_zero_f32_w32_gfx12);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32Gfx11, SubnormalLaneFollowsInputFlushing) {
  check_subnormal_lane(rj_arith_v_mul_dx9_zero_f32_w32);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32Gfx12, ZeroLanesInNormalWave) {
  check_zero_lanes(rj_arith_v_mul_dx9_zero_f32_w32_gfx12);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32Gfx11, ZeroLanesInNormalWave) {
  check_zero_lanes(rj_arith_v_mul_dx9_zero_f32_w32);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32Gfx12, Arithmetic) {
  check_instruction<32>(rj_arith_v_mul_dx9_zero_f32_w32_gfx12);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W32Gfx11, Arithmetic) {
  check_instruction<32>(rj_arith_v_mul_dx9_zero_f32_w32);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W64Gfx12, Arithmetic) {
  check_instruction<64>(rj_arith_v_mul_dx9_zero_f32_w64_gfx12);
}

// Check FP32 multiplication with DX9 zero-product rules.
TEST(FpMulDx9ZeroF32W64Gfx11, Arithmetic) {
  check_instruction<64>(rj_arith_v_mul_dx9_zero_f32_w64);
}
