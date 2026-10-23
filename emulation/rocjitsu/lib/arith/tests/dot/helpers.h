// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dot-product input generation, call-shape adapters, and shared contract checks.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Enumerate supported dot modifiers, including combined and DPP cases.
inline std::vector<uint64_t> dot_modes(uint32_t modifiers) {
  std::vector<uint64_t> modes{0};
  for (int bit = 0; bit < 32; ++bit) {
    if (modifiers & (1U << bit)) {
      modes.push_back(1U << bit);
    }
  }
  if (modifiers) {
    modes.push_back(modifiers);
  }
  for (uint64_t dpp : dpp_modes) {
    modes.push_back(dpp | modifiers);
  }
  return modes;
}

// Packed factors and accumulators bounded by four, with different halves.
// Inject a NaN and infinity in each source without overwhelming finite lanes.
template <int Lanes, bool Bf16, bool HalfOutput> Registers<4, Lanes> dot2_inputs() {
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      if (reg < 2 || HalfOutput) {
        uint16_t lo = Bf16 ? random_bf16_bits(random) : random_f16_bits(random);
        uint16_t hi = Bf16 ? random_bf16_bits(random) : random_f16_bits(random);
        input.v[reg][lane] = lo | (uint32_t(hi) << 16);
      } else {
        input.v[reg][lane] =
            std::bit_cast<uint32_t>(float(int(random() >> 7) - 0x800000) * 0x1p-21f);
      }
    }
  }
  for (int reg = 0; reg < 3; ++reg) {
    int lane = 2 + reg * 4;
    if (reg < 2 || HalfOutput) {
      input.v[reg][lane] =
          (input.v[reg][lane] & 0xffff0000U) | (Bf16 ? kBF16QuietNaNBits : kF16QuietNaNBits);
      input.v[reg][lane + 1] =
          (input.v[reg][lane + 1] & 0xffffU) |
          (uint32_t(Bf16 ? kBF16PositiveInfinityBits : kF16PositiveInfinityBits) << 16);
    } else {
      input.v[reg][lane] = kF32QuietNaNBits;
      input.v[reg][lane + 1] = kF32PositiveInfinityBits;
    }
  }
  return input;
}

// Invoke the selected dot API shape, accounting for accumulator and FP-status parameters.
template <bool Accumulate, bool HasStatus, typename Call, typename Mask>
int invoke_dot(Call call, uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *status) {
  if constexpr (Accumulate) {
    return call(flags, exec_mask, mode, d, a, b, status);
  } else if constexpr (HasStatus) {
    return call(flags, exec_mask, mode, d, a, b, c, status);
  } else {
    return call(flags, exec_mask, mode, d, a, b, c);
  }
}

// Only loose arithmetic uses the mathematical reference. Exact DOT2 uses its
// architecture's captured goldens separately, including fallback/compatibility.
template <int Lanes, bool Accumulate, bool HasStatus, typename Call, typename Check>
void check_loose_dot(Call call, const Registers<4, Lanes> &input, Check check, uint32_t modifiers,
                     bool has_exact, uint64_t guest_flags = 0) {
  using Mask = WaveMask<Lanes>;
  expect_alu_invalid_flags<true>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        uint32_t status = 0x80000055U;
        int error = invoke_dot<Accumulate, HasStatus>(call, flags, ~Mask(0), mode, r.v + 3, r.v,
                                                      r.v + 1, r.v + 2, &status);
        r.expect_preserved(input, 0, 0);
        EXPECT_EQ(status, 0x80000055U);
        EXPECT_EQ((invoke_dot<Accumulate, HasStatus>(call, flags, ~Mask(0), mode, nullptr, nullptr,
                                                     nullptr, nullptr, &status)),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return error;
      },
      modifiers);
  auto modes = dot_modes(modifiers);
  auto cpu_levels = supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      if (has_exact && semantics != RJ_ARITH_SEMANTICS_LOOSE) {
        continue;
      }
      for (int scenario = 0; scenario < 5 + int(modes.size()); ++scenario) {
        uint64_t mode = scenario < 5 ? 0 : modes[scenario - 5];
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : scenario % 2 ? 0 : 3);
        uint64_t flags = cpu | semantics | (scenario % 2 ? RJ_ARITH_FP16_OVFL : 0) |
                         (scenario % 3 ? guest_flags : 0);
        int error = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL && HasStatus
                        ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                    : semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                            semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                        ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                        : RJ_ARITH_SUCCESS;
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        for (int aliasing_offset : {3, 0, 1, 2}) {
          auto r = input;
          // Keep a sentinel in unused FP-status storage to detect unintended updates.
          uint32_t status = 0x80000055U;
          uint32_t *status_ptr = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL || scenario % 2
                                     ? &status
                                     : nullptr;
          ASSERT_EQ((invoke_dot<Accumulate, HasStatus>(
                        call, flags, exec_mask, mode, exec_mask ? r.v + aliasing_offset : nullptr,
                        exec_mask ? r.v : nullptr, exec_mask ? r.v + 1 : nullptr,
                        exec_mask ? r.v + 2 : nullptr, status_ptr)),
                    error);
          EXPECT_EQ(status, 0x80000055U);
          if (error) {
            r.expect_preserved(input, 0, 0);
            continue;
          }
          r.expect_preserved(input, aliasing_offset, 1);
          for (int lane = 0; lane < Lanes; ++lane) {
            int source = lane;
            bool write = mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)
                             ? dpp_source(mode, exec_mask, lane, source)
                             : bool((exec_mask >> lane) & 1);
            if (!write) {
              EXPECT_EQ(r.v[aliasing_offset][lane], input.v[aliasing_offset][lane]);
              continue;
            }
            uint32_t a = source < 0 ? 0 : input.v[0][source], b = input.v[1][lane];
            uint32_t c = input.v[Accumulate ? aliasing_offset : 2][lane];
            check(r.v[aliasing_offset][lane], a, b, c, input.v[aliasing_offset][lane], flags,
                  uint32_t(mode));
          }
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
