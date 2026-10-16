// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Comparison operand storage, mask checks, and routing scenarios.

#pragma once

#include "common/contracts.h"
#include "common/dpp_reference.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/wave.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Raw operand access: 16-bit halves, 32-bit words or separate low/high VGPRs.
inline uint64_t compare_operand(const uint32_t *const *v, int lane, int width, bool high) {
  if (lane < 0) {
    return 0; // DPP supplied zero rather than an input lane.
  }
  if (width == 16) {
    return uint16_t(v[0][lane] >> (high ? 16 : 0));
  }
  return v[0][lane] | (width == 64 ? uint64_t(v[1][lane]) << 32 : 0);
}

// Write raw operand bits using the selected width and FP16 halfword placement.
inline void write_compare_operand(uint32_t *const *v, int lane, int width, uint64_t bits,
                                  bool high = false) {
  if (width == 16) {
    int shift = high ? 16 : 0;
    v[0][lane] = (v[0][lane] & ~(0xffffU << shift)) | (uint32_t(bits) << shift);
  } else {
    v[0][lane] = uint32_t(bits);
    if (width == 64) {
      v[1][lane] = uint32_t(bits >> 32);
    }
  }
}

// Predicates produce a scalar mask: inactive bits become zero, not preserved.
// DPP permutes A only; source -1 means zero. The copied oracle is inherited
// from tests/register and covers row/bank enables, FI and boundary behavior.
template <int Lanes, class Predicate>
WaveMask<Lanes> comparison_mask(uint64_t mode, WaveMask<Lanes> exec_mask, Predicate predicate) {
  WaveMask<Lanes> result = 0;
  for (int lane = 0; lane < Lanes; ++lane) {
    int source = lane;
    bool active = mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)
                      ? dpp_source(mode, exec_mask, lane, source)
                      : bool((exec_mask >> lane) & 1);
    if (active && predicate(source, lane)) {
      result |= WaveMask<Lanes>(1) << lane;
    }
  }
  return result;
}

// Check one call and every byte of source/guard storage. aliasing_register=-1
// uses an independent mask; 0/2 overlap A/B, near the upper/lower wave boundary.
// Wave64 APIs support byte-copy stores, so read aliased masks with memcpy too.
template <int Lanes, class Call>
void expect_comparison(const Registers<4, Lanes> &input, WaveMask<Lanes> expected,
                       int expected_error, int aliasing_register, Call call) {
  using Mask = WaveMask<Lanes>;
  Registers<4, Lanes> r = input, want = input;
  Mask outside = ~Mask(0), actual;
  void *storage = aliasing_register < 0 ? static_cast<void *>(&outside)
                                        : static_cast<void *>(r.v[aliasing_register] + 31);
  ASSERT_EQ(call(r, static_cast<Mask *>(storage)), expected_error);
  if (expected_error == RJ_ARITH_SUCCESS) {
    std::memcpy(&actual, storage, sizeof actual);
    EXPECT_EQ(actual, expected);
    if (aliasing_register >= 0) {
      std::memcpy(want.v[aliasing_register] + 31, &expected, sizeof expected);
    }
  } else if (aliasing_register < 0) {
    EXPECT_EQ(outside, ~Mask(0));
  }
  EXPECT_EQ(std::memcmp(r.words, want.words, sizeof r.words), 0);
}

} // namespace rocjitsu::arith::test
