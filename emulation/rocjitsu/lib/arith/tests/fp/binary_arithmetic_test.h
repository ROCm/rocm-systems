// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared FP add/subtract/multiply test scenarios and result checks.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace rocjitsu::arith::test {

// Test ADD/SUB/SUBREV/MUL, using the independent scalar reference at the call site.
// FP32 has exact numeric support on these inputs; FP16/FP64 and packed arithmetic
// are loose-only. Required reporting is supported only by FP32 MUL, whose
// named tests also exercise its literal exception cases separately.
template <int Width, int Lanes, bool Packed = false, class Call, class Reference>
void check_binary_arithmetic(Call call, Reference reference, bool gfx12,
                             bool fp_status_supported = false) {
  static_assert(Width == 16 || Width == 32 || Width == 64);
  static_assert(Lanes == 32 || Lanes == 64);
  static_assert(!Packed || Width == 16);
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Bank = Registers<10, Lanes>;
  Bank input = make_fp_inputs<Width, Lanes>();

  expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Bank result = input;
    uint32_t fp_status = 0x80000000U;
    int error = call(flags, ~Mask(0), mode, result.v + 6, result.v, result.v + 2, &fp_status);
    result.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U);
    EXPECT_EQ(call(flags, ~Mask(0), mode, nullptr, nullptr, nullptr, &fp_status), error);
    return error;
  });

  std::vector<uint64_t> modes = {0, RJ_ARITH_ALU_NEG_A,
                                 Packed ? RJ_ARITH_PK_CLAMP : RJ_ARITH_ALU_CLAMP};
  if constexpr (Width == 16 && !Packed) {
    modes.push_back(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
  }
  if constexpr (Width != 64 && !Packed) {
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT));
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                    RJ_ARITH_ALU_NEG_A);
  }
  if constexpr (Width == 32) {
    modes.push_back(RJ_ARITH_ALU_OMOD_2);
    modes.push_back(RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A);
  }
  std::vector<int> aliasing_offsets = {6, 0, 2};
  if constexpr (Width == 64) {
    aliasing_offsets.push_back(1);
  }

  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : aliasing_offsets) {
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Bank result = input, expected = input;
            uint32_t fp_status = 0x80000000U;
            int error = call(cpu | semantics, exec_mask, mode, result.v + aliasing_offset, result.v,
                             result.v + 2, &fp_status);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL && !fp_status_supported) {
              EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              result.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            if (Width != 32 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
              EXPECT_EQ(error, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              result.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(error, RJ_ARITH_SUCCESS);
            bool reporting =
                fp_status_supported && (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                                        semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY);
            if (!reporting || !exec_mask) {
              EXPECT_EQ(fp_status, 0x80000000U);
            } else {
              EXPECT_EQ(fp_status & 0x80000000U, 0x80000000U);
            }
            for (int lane = 0; lane < Lanes; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                continue;
              }
              for (int half = 0; half < (Packed ? 2 : 1); ++half) {
                int a_half = Packed ? half : Width == 16 && (mode & RJ_ARITH_ALU_HIGH_A);
                int b_half = Packed ? half : Width == 16 && (mode & RJ_ARITH_ALU_HIGH_B);
                int d_half = Packed ? half : Width == 16 && (mode & RJ_ARITH_ALU_HIGH_D);
                // DPP8 reverses groups of eight; inactive sources become zero unless FI is set.
                int source_lane =
                    !Packed && (mode & RJ_ARITH_DPP8) ? (lane & ~7) + 7 - (lane & 7) : lane;
                uint64_t a = read_fp(input.v, source_lane, Width, a_half);
                if (!Packed && (mode & RJ_ARITH_DPP8) && !(mode & RJ_ARITH_DPP_FI) &&
                    !((exec_mask >> source_lane) & 1)) {
                  a = 0;
                }
                uint64_t b = read_fp(input.v + 2, lane, Width, b_half);
                uint64_t source_mode = Packed ? (mode >> (half ? 3 : 0)) & 7 : mode;
                double x = modified_fp(fp_value(a, Width), source_mode, 0);
                double y = modified_fp(fp_value(b, Width), source_mode, 1);
                double value = reference(x, y);
                // gfx11 ignores OMOD under the default guest FP mode.
                if (Width == 32 && gfx12 && (mode & RJ_ARITH_ALU_OMOD_2)) {
                  value = double(float(value)) * 2;
                }
                if (mode & (Packed ? RJ_ARITH_PK_CLAMP : RJ_ARITH_ALU_CLAMP)) {
                  value = !(value > 0) ? 0 : std::min(value, 1.0);
                }
                write_fp(expected.v + aliasing_offset, lane, Width, fp_bits(value, Width), d_half);
              }
            }
            expect_fp_result<Width, Packed>(input, result, expected, exec_mask, aliasing_offset,
                                            mode,
                                            Width == 32 ? semantics : RJ_ARITH_SEMANTICS_LOOSE,
                                            /*atol=*/Width == 64 ? 1e-14 : 1e-5,
                                            /*rtol=*/Width == 16 ? 5e-4
                                            : Width == 32        ? 1e-6
                                                                 : 1e-14);
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
