// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared DPP16 routing witnesses for ordinary FMA and mixed FP32-output FMA.
// Exactly representable operands isolate row selection, source access, and masked writes.

#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"

#include <bit>
#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// A row-mirror DPP16 witness with only the last destination row enabled.
// Wave64 must select rows from the upper half without modifying lower rows.
template <int Lanes, bool Half = false, typename Call> void check_fma_dpp16(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  Registers<4, Lanes> input;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = Half ? (0x3c00U + lane) : std::bit_cast<uint32_t>(float(lane + 1));
    input.v[1][lane] = Half ? 0x3c00U : std::bit_cast<uint32_t>(1.0f);
    input.v[2][lane] = 0;
  }
  uint64_t instruction_flags =
      RJ_ARITH_DPP16 | RJ_ARITH_DPP_FI | (0x140ULL << RJ_ARITH_DPP_CTRL_SHIFT) |
      ((1ULL << (Lanes / 16 - 1)) << RJ_ARITH_DPP_ROW_SHIFT) | (15ULL << RJ_ARITH_DPP_BANK_SHIFT);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      auto r = input;
      ASSERT_EQ(call(cpu | semantics, ~Mask(0), instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2,
                     nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        uint32_t expected = input.v[0][lane ^ 15];
        if constexpr (Half) {
          expected |= input.v[3][lane] & 0xffff0000U;
        }
        EXPECT_EQ(r.v[3][lane], lane < Lanes - 16 ? input.v[3][lane] : expected);
      }
      r.expect_preserved(input, 3, 1);
    }
  }
  // Distinguish row rotation, half-row mirror, sharing, XOR and bank-0 masking.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int control : {0x121, 0x141, 0x153, 0x165, 0xe4}) {
      auto r = input;
      const uint64_t banks = control == 0xe4 ? 14 : 15;
      uint64_t instruction_flags = RJ_ARITH_DPP16 | (uint64_t(control) << RJ_ARITH_DPP_CTRL_SHIFT) |
                                   (15ULL << RJ_ARITH_DPP_ROW_SHIFT) |
                                   (banks << RJ_ARITH_DPP_BANK_SHIFT);
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, ~Mask(0),
                     instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        int source = control == 0x121   ? (lane & ~15) | ((lane + 15) & 15)
                     : control == 0x141 ? lane ^ 7
                     : control == 0x153 ? (lane & ~15) | 3
                     : control == 0x165 ? lane ^ 5
                                        : lane;
        uint32_t expected = input.v[0][source];
        if constexpr (Half) {
          expected |= input.v[3][lane] & 0xffff0000U;
        }
        if (control == 0xe4 && lane % 16 < 4) {
          expected = input.v[3][lane];
        }
        EXPECT_EQ(r.v[3][lane], expected) << control << '/' << lane;
      }
    }
  }
  // Representative routing branches, with sparse EXEC and partial row/bank masks.
  // The scalar router is independent of the SIMD permutation implementation.
  const struct {
    uint32_t control;
    bool fi, bc;
  } routes[] = {{0x1b, false, false}, {0x109, false, true}, {0x11a, true, false},
                {0x12c, true, true},  {0x141, false, true}, {0x15b, true, false},
                {0x16d, false, false}};
  for (const auto &route : routes) {
    const Mask exec_mask = Mask(0xb5b5b5b5b5b5b5b5ULL);
    const uint64_t instruction_flags =
        RJ_ARITH_DPP16 | (uint64_t(route.control) << RJ_ARITH_DPP_CTRL_SHIFT) |
        (0xaULL << RJ_ARITH_DPP_ROW_SHIFT) | (0xbULL << RJ_ARITH_DPP_BANK_SHIFT) |
        (route.fi ? RJ_ARITH_DPP_FI : 0) | (route.bc ? RJ_ARITH_DPP_BOUND_CTRL : 0);
    auto expected = input;
    ASSERT_EQ(call(RJ_ARITH_SEMANTICS_LOOSE, exec_mask, instruction_flags, expected.v + 3,
                   expected.v, expected.v + 1, expected.v + 2, nullptr),
              RJ_ARITH_SUCCESS);
    for (uint64_t cpu : supported_cpu_levels()) {
      auto actual = input;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE, exec_mask, instruction_flags, actual.v + 3,
                     actual.v, actual.v + 1, actual.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(actual.v[3][lane], expected.v[3][lane])
            << route.control << '/' << cpu << '/' << lane;
      }
      actual.expect_preserved(input, 3, 1);
    }
  }
  // gfx1100/gfx1201 Wave32 observations: FI reads inactive sources, not out-of-range
  // sources (unlike the FI=1 rows of RDNA4 ISA Table 39). BC supplies zero for either.
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = Half ? half_bits(float(lane + 1)) : std::bit_cast<uint32_t>(float(lane + 1));
    input.v[2][lane] = Half ? 0x3800U : 0x3f000000U;
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int source_delta : {1, -1}) {
      for (int controls = 0; controls < 4; ++controls) {
        const bool bc = controls & 1, fi = controls & 2;
        for (unsigned banks : {15U, 1U}) {
          auto r = input;
          const Mask exec_mask = ~Mask(8);
          uint64_t instruction_flags =
              RJ_ARITH_DPP16 |
              (uint64_t(source_delta > 0 ? 0x101 : 0x111) << RJ_ARITH_DPP_CTRL_SHIFT) |
              (15ULL << RJ_ARITH_DPP_ROW_SHIFT) | (uint64_t(banks) << RJ_ARITH_DPP_BANK_SHIFT) |
              (bc ? RJ_ARITH_DPP_BOUND_CTRL : 0) | (fi ? RJ_ARITH_DPP_FI : 0);
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, exec_mask,
                         instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          for (int lane = 0; lane < Lanes; ++lane) {
            const int source = lane + source_delta;
            const bool out_of_range = lane % 16 == (source_delta > 0 ? 15 : 0);
            const bool inactive_source = source == 3;
            const bool readable = !out_of_range && (!inactive_source || fi);
            uint32_t expected = input.v[3][lane];
            if (lane != 3 && (banks & (1U << ((lane % 16) / 4))) && (readable || bc)) {
              float value = readable ? float(source + 1) + 0.5f : 0.5f;
              expected = Half ? (input.v[3][lane] & 0xffff0000U) | half_bits(value)
                              : std::bit_cast<uint32_t>(value);
            }
            EXPECT_EQ(r.v[3][lane], expected)
                << lane << '/' << source_delta << '/' << controls << '/' << banks;
          }
          r.expect_preserved(input, 3, 1);
        }
      }
    }
  }
}
} // namespace rocjitsu::arith::test
