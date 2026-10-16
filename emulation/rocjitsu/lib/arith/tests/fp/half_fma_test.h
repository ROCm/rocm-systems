// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 FMA/FMAC, packed selectors, and guest FP-status checks.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/fma_reference.h"
#include "fp/fma_simd_test.h"
#include "fp/half_fma_reference.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace rocjitsu::arith::test {

// Fill active destination elements from the independent arithmetic reference.
template <int Lanes, bool Accumulate>
void compute_packed_fma_expected(const rocjitsu::arith::test::Registers<10, Lanes> &input,
                                 uint64_t exec_mask, uint64_t mode, int aliasing_offset,
                                 rocjitsu::arith::test::Registers<10, Lanes> &expected_registers,
                                 bool gfx12) {
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    for (int half = 0; half < 2; ++half) {
      // Low selectors choose the upper source half; high selectors choose the lower.
      const int a_half =
          Accumulate ? half
                     : half ^ bool(mode & (half ? RJ_ARITH_PK_HI_A_LOW : RJ_ARITH_PK_LO_A_HIGH));
      const int b_half =
          Accumulate ? half
                     : half ^ bool(mode & (half ? RJ_ARITH_PK_HI_B_LOW : RJ_ARITH_PK_LO_B_HIGH));
      const int c_half =
          Accumulate ? half
                     : half ^ bool(mode & (half ? RJ_ARITH_PK_HI_C_LOW : RJ_ARITH_PK_LO_C_HIGH));
      const int d_half = half;
      uint64_t a = rocjitsu::arith::test::read_fp(input.v, lane, 16, a_half);
      uint64_t b = rocjitsu::arith::test::read_fp(input.v + 2, lane, 16, b_half);
      uint64_t c = rocjitsu::arith::test::read_fp(input.v + (Accumulate ? aliasing_offset : 4),
                                                  lane, 16, c_half);
      // Existing independent integer FMA oracle, validated by the old
      // suite against the inherited rocjitsu model and captured cases.
      uint32_t reference_mode = uint32_t(mode) & 0x1ffU;
      if constexpr (!Accumulate) {
        reference_mode = uint32_t((mode >> (half ? 3 : 0)) & 7) |
                         (mode & RJ_ARITH_PK_CLAMP ? RJ_ARITH_ALU_CLAMP : 0);
      }
      uint64_t expected = rocjitsu::arith::test::half_fma_reference::evaluate(
          uint32_t(a), uint32_t(b), uint32_t(c), reference_mode, false, 0, !gfx12);
      rocjitsu::arith::test::write_fp(expected_registers.v + aliasing_offset, lane, 16, expected,
                                      d_half);
    }
  }
}

// Exercise this instruction contract for either wave size.
template <int Lanes, bool Packed, bool Accumulate, typename Call, typename Cases>
void check_half_fma(Call entry_point, bool gfx12, const Cases &status_cases) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  static_assert(Lanes == 32 || Lanes == 64);
  const uint64_t clamp = Packed && !Accumulate ? RJ_ARITH_PK_CLAMP : RJ_ARITH_ALU_CLAMP;
  auto call = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                  uint32_t *fp_status) {
    if constexpr (Accumulate) {
      return entry_point(flags, exec_mask, mode, d, a, b, fp_status);
    } else {
      return entry_point(flags, exec_mask, mode, d, a, b, c, fp_status);
    }
  };
  if constexpr (!Packed && !Accumulate) {
    check_half_fma_directed_flush<Lanes>(call);
  }
  std::vector<uint64_t> simd_modes = {0};
  if constexpr (!(Packed && Accumulate)) {
    simd_modes.insert(simd_modes.end(), {RJ_ARITH_ALU_NEG_A, clamp});
  }
  if constexpr (Packed && !Accumulate) {
    simd_modes.insert(simd_modes.end(),
                      {RJ_ARITH_PK_LO_A_HIGH, RJ_ARITH_PK_LO_B_HIGH, RJ_ARITH_PK_LO_C_HIGH,
                       RJ_ARITH_PK_HI_A_LOW, RJ_ARITH_PK_HI_B_LOW, RJ_ARITH_PK_HI_C_LOW});
  }
  if constexpr (!Packed) {
    simd_modes.insert(simd_modes.end(),
                      {RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF,
                       RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D});
  }
  check_fma_simd<16, Lanes, Packed>(call, simd_modes);
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<16, Lanes>();

  rocjitsu::arith::test::expect_invalid_fp_requests([&](uint64_t flags, uint64_t mode) {
    Registers r = input;
    int aliasing_offset = 6;
    Mask exec_mask = ~Mask(0);
    uint32_t fp_status = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;

    int result =
        call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4, &fp_status);
    r.expect_preserved(input, 0, 0);
    EXPECT_EQ(fp_status, 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0);
    EXPECT_EQ(call(flags, exec_mask, mode, nullptr, nullptr, nullptr, nullptr, &fp_status), result);
    return result;
  });

  std::vector<uint64_t> modes = {0};
  if constexpr (!(Packed && Accumulate)) {
    modes.insert(modes.end(), {RJ_ARITH_ALU_NEG_A, clamp});
  }
  if constexpr (Packed && !Accumulate) {
    modes.insert(modes.end(), {RJ_ARITH_PK_LO_A_HIGH, RJ_ARITH_PK_LO_B_HIGH, RJ_ARITH_PK_LO_C_HIGH,
                               RJ_ARITH_PK_HI_A_LOW, RJ_ARITH_PK_HI_B_LOW, RJ_ARITH_PK_HI_C_LOW});
  }
  if constexpr (Packed && Accumulate) {
    for (uint64_t invalid :
         {RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_ABS_A, RJ_ARITH_ALU_CLAMP, RJ_ARITH_ALU_OMOD_2}) {
      EXPECT_EQ(call(RJ_ARITH_SEMANTICS_LOOSE, Mask(1), invalid, nullptr, nullptr, nullptr, nullptr,
                     nullptr),
                RJ_ARITH_ERROR_INVALID_FLAGS);
    }
  }
  if constexpr (!Packed) {
    modes.push_back(RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D);
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT));
    modes.push_back(RJ_ARITH_DPP8 | (0x53977ULL << RJ_ARITH_DPP8_SELECT_SHIFT) | RJ_ARITH_DPP_FI |
                    RJ_ARITH_ALU_NEG_A);
  }

  // Default guest rounding and normal fractional inputs: the existing rocjitsu
  // arithmetic model reduces to IEEE arithmetic here. Special values are checked
  // by classification. No production helper or rj-arith result is a numeric oracle.
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  if constexpr (Packed && !Accumulate) {
    // A=(1,2), B=(3,4), C=(5,6): selecting A.high for the low result gives (11,14).
    for (uint64_t cpu : cpu_levels) {
      for (int destination : {0, 2, 4, 6}) {
        Registers r = input;
        r.v[0][0] = 0x40003c00U;
        r.v[2][0] = 0x44004200U;
        r.v[4][0] = 0x46004500U;
        uint32_t status = 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, Mask(1),
                       RJ_ARITH_PK_LO_A_HIGH, r.v + destination, r.v, r.v + 2, r.v + 4, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[destination][0], 0x4b004980U);
      }
    }
  }
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (uint64_t mode : modes) {
        for (Mask exec_mask :
             {Mask(0), ~Mask(0), Mask(0xaaaaaaaaaaaaaaaaULL), Mask(1) << (Lanes - 1)}) {
          for (int aliasing_offset : {6, 0, 2, 4}) {
            if (Accumulate && aliasing_offset == 4) {
              continue;
            }
            SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << mode << '/'
                                              << exec_mask << '/' << aliasing_offset);
            Registers r = input, expected_registers = input;
            uint64_t flags = cpu | semantics;
            uint32_t fp_status = 0x80000000U;

            int result = call(flags, exec_mask, mode, r.v + aliasing_offset, r.v, r.v + 2, r.v + 4,
                              &fp_status);
            if (!gfx12 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
              r.expect_preserved(input, 0, 0);
              EXPECT_EQ(fp_status, 0x80000000U);
              continue;
            }
            ASSERT_EQ(result, RJ_ARITH_SUCCESS);
            bool reporting = gfx12 && (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
            if (!reporting || !exec_mask) {
              EXPECT_EQ(fp_status, 0x80000000U);
            } else {
              EXPECT_EQ(fp_status & 0x80000000U, 0x80000000U);
            }
            if constexpr (Packed) {
              compute_packed_fma_expected<Lanes, Accumulate>(
                  input, exec_mask, mode, aliasing_offset, expected_registers, gfx12);
            } else {
              compute_fma_expected<16>(input, exec_mask, mode, aliasing_offset, expected_registers,
                                       gfx12, Accumulate);
            }
            // Check the half oracle's exact encodings, including NaN sign/payload.
            if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
              for (int part = 0; part < (Packed ? 2 : 1); ++part) {
                int half = Packed ? part : bool(mode & RJ_ARITH_ALU_HIGH_D);
                for (int lane = 0; lane < Lanes; ++lane) {
                  if ((exec_mask >> lane) & 1) {
                    EXPECT_EQ(read_fp(r.v + aliasing_offset, lane, 16, half),
                              read_fp(expected_registers.v + aliasing_offset, lane, 16, half));
                  }
                }
              }
            }
            expect_fp_result<16, Packed>(input, r, expected_registers, exec_mask, aliasing_offset,
                                         mode, semantics, /*atol=*/1e-5, /*rtol=*/5e-4);
          }
        }
      }
    }
  }
  if (gfx12) {
    std::vector<uint64_t> status_modes = {0};
    if constexpr (!(Packed && Accumulate)) {
      status_modes.push_back(clamp);
    }
    for (uint64_t cpu : cpu_levels) {
      for (const auto &test : status_cases) {
        for (uint64_t mode : status_modes) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << mode << '/' << test.a << '/' << test.b << '/' << test.c);
          Registers r = input;
          const int lane = Lanes - 1;
          for (int part = 0; part < (Packed ? 2 : 1); ++part) {
            rocjitsu::arith::test::write_fp(r.v, lane, 16, test.a, part);
            rocjitsu::arith::test::write_fp(r.v + 2, lane, 16, test.b, part);
            rocjitsu::arith::test::write_fp(r.v + (Accumulate ? 6 : 4), lane, 16, test.c, part);
          }
          Mask exec_mask = Mask(1) << lane;
          uint32_t fp_status = 0x80000000U;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, &fp_status),
                    RJ_ARITH_SUCCESS);
          uint32_t causes = mode ? 0 : test.causes;
          EXPECT_EQ(fp_status, 0x80000000U | causes);
          // Preserve previously accumulated causes as well as unrelated bits.
          fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
          for (int part = 0; part < (Packed ? 2 : 1); ++part) {
            rocjitsu::arith::test::write_fp(r.v + (Accumulate ? 6 : 4), lane, 16, test.c, part);
          }
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, &fp_status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
          Registers before = r;
          exec_mask = 0;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, mode, r.v + 6,
                         r.v, r.v + 2, r.v + 4, &fp_status),
                    RJ_ARITH_SUCCESS);
          r.expect_preserved(before, 0, 0);
          EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
