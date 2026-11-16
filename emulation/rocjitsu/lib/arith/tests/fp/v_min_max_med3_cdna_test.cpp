// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA three-source selection: shared arithmetic/API checks plus the inherited
// second-NaN priority and MED3's numerical equality of opposite-signed zeros.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// Exercise modifiers, EXEC, aliases and selected legacy encoding differences.
template <int Width, int Lanes, class Call> void check(Call call, int operation) {
  using namespace rocjitsu::arith::test;
  check_selection<Width, Lanes, 3>(call, [=](double x, double y, double z) {
    auto pick = [](double a, double b, bool maximum) { return select_fp(a, b, maximum, false); };
    if (operation != 2) {
      return pick(pick(x, y, operation == 1), z, operation == 1);
    }
    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
      return pick(pick(x, y, false), z, false);
    }
    double largest = pick(pick(x, y, true), z, true);
    return largest == x ? pick(y, z, true) : largest == y ? pick(x, z, true) : pick(x, y, true);
  });
  constexpr uint64_t sign = 1ULL << (Width - 1);
  constexpr uint64_t infinity = Width == 16 ? 0x7c00ULL : 0x7f800000ULL;
  constexpr uint64_t quiet = Width == 16 ? 0x0200ULL : 0x00400000ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool ieee : {false, true}) {
      Registers<8, Lanes> r;
      // The first lane distinguishes CDNA's median zero tie from RDNA's.
      write_fp(r.v, 0, Width, sign);
      write_fp(r.v + 2, 0, Width, 0);
      write_fp(r.v + 4, 0, Width, sign);
      for (int source = 0; source < 3; ++source) {
        write_fp(r.v + source * 2, 1, Width, infinity | quiet | (source + 1));
        write_fp(r.v + source * 2, 2, Width, infinity | (source + 1));
      }
      uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                       (ieee ? RJ_ARITH_FP_IEEE_MODE : 0);
      ASSERT_EQ(call(flags, 7, 0, r.v + 6, r.v, r.v + 2, r.v + 4, nullptr), RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 6, 0, Width), operation == 0 ? sign : 0);
      EXPECT_EQ(read_fp(r.v + 6, 1, Width), infinity | quiet | 3);
      EXPECT_EQ(read_fp(r.v + 6, 2, Width), infinity | (ieee ? quiet : 0) | 3);
    }
  }
  if constexpr (Lanes == 32) {
    check_selection_omod<Width, 3>(call);
    check_selection_omod_nan<Width, 3>(call);
  }
}

} // namespace

TEST(FpCdnaTernarySelection, F16Wave32) {
  check<16, 32>(rj_arith_v_min3_f16_w32_gfx942, 0);
  check<16, 32>(rj_arith_v_max3_f16_w32_gfx942, 1);
  check<16, 32>(rj_arith_v_med3_f16_w32_gfx942, 2);
}

TEST(FpCdnaTernarySelection, F16Wave64) {
  check<16, 64>(rj_arith_v_min3_f16_w64_gfx942, 0);
  check<16, 64>(rj_arith_v_max3_f16_w64_gfx942, 1);
  check<16, 64>(rj_arith_v_med3_f16_w64_gfx942, 2);
}

TEST(FpCdnaTernarySelection, F32Wave32) {
  check<32, 32>(rj_arith_v_min3_f32_w32_gfx942, 0);
  check<32, 32>(rj_arith_v_max3_f32_w32_gfx942, 1);
  check<32, 32>(rj_arith_v_med3_f32_w32_gfx942, 2);
}

TEST(FpCdnaTernarySelection, F32Wave64) {
  check<32, 64>(rj_arith_v_min3_f32_w64_gfx942, 0);
  check<32, 64>(rj_arith_v_max3_f32_w64_gfx942, 1);
  check<32, 64>(rj_arith_v_med3_f32_w64_gfx942, 2);
}

namespace {

// Propagating selection keeps the first NaN; output controls retain CDNA policy.
template <int Lanes, class Call> void check_propagating(Call call, bool maximum) {
  using namespace rocjitsu::arith::test;
  check_selection<32, Lanes, 3>(call, [=](double x, double y, double z) {
    return select_fp(select_fp(x, y, maximum, true), z, maximum, true);
  });
  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool ieee : {false, true}) {
      for (bool flush : {false, true}) {
        for (bool preserve_nan : {false, true}) {
          Registers<4, Lanes> r;
          for (int source = 0; source < 3; ++source) {
            r.v[source][0] = 0x3e800000U; // 0.25: OMOD is observable before CLAMP.
            r.v[source][1] = 0x7f800001U + source;
          }
          uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                           (ieee ? RJ_ARITH_FP_IEEE_MODE : 0) |
                           (flush ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                           (preserve_nan ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
          ASSERT_EQ(call(flags, 3, RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP, r.v + 3, r.v, r.v + 1,
                         r.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[3][0], flush && !ieee ? 0x3f000000U : 0x3e800000U);
          EXPECT_EQ(r.v[3][1], preserve_nan ? 0x7fc00001U : 0U);
        }
      }
    }
  }
  // Forced IEEE's interaction with OMOD is unqualified: strict numeric mode
  // rejects effective scaling, while Compatibility retains the inherited policy.
  for (uint64_t semantics : semantics_modes()) {
    Registers<4, Lanes> r;
    r.v[0][0] = r.v[1][0] = r.v[2][0] = 0x3f800000U;
    r.v[3][0] = 0xdeadbeefU;
    const int status = call(semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1, RJ_ARITH_ALU_OMOD_2,
                            r.v + 3, r.v, r.v + 1, r.v + 2, nullptr);
    if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
      EXPECT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(r.v[3][0], 0xdeadbeefU);
    } else if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
      EXPECT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
    } else {
      EXPECT_EQ(status, RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], 0x40000000U);
    }
  }
}

} // namespace

TEST(FpCdnaTernarySelection, PropagatingF32Wave32) {
  check_propagating<32>(rj_arith_v_minimum3_f32_w32_gfx950, false);
  check_propagating<32>(rj_arith_v_maximum3_f32_w32_gfx950, true);
}

TEST(FpCdnaTernarySelection, PropagatingF32Wave64) {
  check_propagating<64>(rj_arith_v_minimum3_f32_w64_gfx950, false);
  check_propagating<64>(rj_arith_v_maximum3_f32_w64_gfx950, true);
}
