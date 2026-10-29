// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for legacy FP min/max and the shared selection policies.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <iterator>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(rj_arith_v_min_f16_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(rj_arith_v_min_f16_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(rj_arith_v_min_f32_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(rj_arith_v_min_f32_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(rj_arith_v_min_f64_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(rj_arith_v_min_f64_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(rj_arith_v_max_f16_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(rj_arith_v_max_f16_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(rj_arith_v_max_f32_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(rj_arith_v_max_f32_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(rj_arith_v_max_f64_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(rj_arith_v_max_f64_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Guest policy affects numeric-vs-NaN selection, clamping and input flushing,
// even though loose semantics does not fix NaN payloads or signed-zero ties.
TEST(FpSelection, GuestControls) {
  using namespace rocjitsu::arith::test;
  struct Case {
    decltype(&rj_arith_v_min_f32_w32) call;
    int width;
    uint64_t a, b, fp, mode, expected;
    bool nan;
  };
  const Case cases[] = {
      // Inherited selection policy preserves subnormal results under output-only flushing.
      {rj_arith_v_min_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, 1, false},
      {rj_arith_v_min_num_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, 1,
       false},
      {rj_arith_v_minimum_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, 1,
       false},
      {rj_arith_v_min_f64_w32, 64, 1, 0x3ff0000000000000ULL, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0,
       1, false},
      {rj_arith_v_min_f32_w32, 32, 0x7f800001, 0x3f800000, 0, 0, 0x3f800000, false},
      {rj_arith_v_min_f32_w32, 32, 0x7f800001, 0x3f800000, RJ_ARITH_FP_IEEE_MODE, 0, 0, true},
      {rj_arith_v_min_f32_w32, 32, 0x7fc00001, 0x7fc00002, RJ_ARITH_FP_DX10_CLAMP_OFF,
       RJ_ARITH_ALU_CLAMP, 0, true},
      {rj_arith_v_min_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0, 0, false},
      {rj_arith_v_min_f16_w32, 16, 1, 0x3c00, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0, 0, false},
      {rj_arith_v_min_f64_w32, 64, 1, 0x3ff0000000000000ULL, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0,
       0, false},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &test : cases) {
      SCOPED_TRACE(::testing::Message() << cpu << '/' << test.width << '/' << test.fp);
      Registers<6, 32> r;
      write_fp(r.v, 0, test.width, test.a);
      write_fp(r.v + 2, 0, test.width, test.b);
      ASSERT_EQ(test.call(cpu | RJ_ARITH_SEMANTICS_LOOSE | test.fp, 1, test.mode, r.v + 4, r.v,
                          r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      auto actual = read_fp(r.v + 4, 0, test.width);
      if (test.nan) {
        EXPECT_TRUE(std::isnan(fp_value(actual, test.width)));
      } else {
        EXPECT_EQ(actual, test.expected);
      }
    }
  }
}

TEST(FpSelection, CompatibilityOmodRounding) {
  using namespace rocjitsu::arith::test;
  check_selection_omod<16, 2>(rj_arith_v_min_f16_w32);
  check_selection_omod<16, 2>(rj_arith_v_min_num_f16_w32);
  check_selection_omod<32, 2>(rj_arith_v_min_f32_w32);
  check_selection_omod<32, 2>(rj_arith_v_min_num_f32_w32);
  check_selection_omod<64, 2>(rj_arith_v_min_f64_w32);
  check_selection_omod<64, 2>(rj_arith_v_min_num_f64_w32);
  check_selection_omod<16, 3>(rj_arith_v_min3_f16_w32);
  check_selection_omod<16, 3>(rj_arith_v_min3_num_f16_w32);
  check_selection_omod<32, 3>(rj_arith_v_min3_f32_w32);
  check_selection_omod<32, 3>(rj_arith_v_min3_num_f32_w32);
}

// F16C must not quiet a legacy sNaN before IEEE-sensitive selection.
TEST(FpSelection, HalfSignalingNan) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      Registers<4, 32> r;
      r.v[0][0] = 0x7c017c01;
      r.v[1][0] = 0x3c003c00;
      r.v[2][0] = 0x40004000;
      uint64_t flags = cpu | semantics | RJ_ARITH_FP_IEEE_MODE;
      ASSERT_EQ(rj_arith_v_min_f16_w32(flags, 1, 0, r.v + 3, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_TRUE(std::isnan(f16_value(uint16_t(r.v[3][0]))));
      ASSERT_EQ(rj_arith_v_min3_f16_w32(flags, 1, 0, r.v + 3, r.v + 1, r.v + 2, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_TRUE(std::isnan(f16_value(uint16_t(r.v[3][0]))));
      ASSERT_EQ(rj_arith_v_pk_min_f16_w32(flags, 1, 0, r.v + 3, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      for (int shift : {0, 16}) {
        EXPECT_TRUE(std::isnan(f16_value(uint16_t(r.v[3][0] >> shift))));
      }
      ASSERT_EQ(rj_arith_v_pk_min_num_f16_w32(flags, 1, 0, r.v + 3, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], 0x3c003c00U);
    }
  }
}

// Selected signaling NaNs survive OMOD in all affected scalar and SIMD families.
TEST(FpSelection, CompatibilityOmodNan) {
  using namespace rocjitsu::arith::test;
  check_selection_omod_nan<16, 2>(rj_arith_v_min_f16_w32);
  check_selection_omod_nan<16, 3>(rj_arith_v_min3_f16_w32);
  check_selection_omod_nan<32, 3>(rj_arith_v_min3_f32_w32);
  check_selection_omod_nan<64, 2>(rj_arith_v_min_f64_w32);
}

// Check native SIMD comparisons/min/max with bitwise NaN selection and quieting
// against the scalar model. NaN encoding checks enforce the Compatibility
// contract and exceed the Loose guarantees.
TEST(FpSelection, NativeNanSelection) {
  using namespace rocjitsu::arith::test;
  using Call = decltype(&rj_arith_v_min_num_f32_w32);
  struct Family {
    int width;
    uint64_t infinity, quiet, sign;
    Call calls[4];
  };
  const Family families[] = {
      {16,
       0x7c00,
       0x0200,
       0x8000,
       {rj_arith_v_min_num_f16_w32, rj_arith_v_max_num_f16_w32, rj_arith_v_minimum_f16_w32,
        rj_arith_v_maximum_f16_w32}},
      {32,
       0x7f800000,
       0x00400000,
       0x80000000,
       {rj_arith_v_min_num_f32_w32, rj_arith_v_max_num_f32_w32, rj_arith_v_minimum_f32_w32,
        rj_arith_v_maximum_f32_w32}},
      {64,
       0x7ff0000000000000ULL,
       0x0008000000000000ULL,
       0x8000000000000000ULL,
       {rj_arith_v_min_num_f64_w32, rj_arith_v_max_num_f64_w32, rj_arith_v_minimum_f64_w32,
        rj_arith_v_maximum_f64_w32}},
  };
  const auto levels = supported_cpu_levels();
  for (const auto &family : families) {
    // Pairs: signed zeros, smallest subnormals, largest finite values,
    // infinities, signaling NaNs, and quiet NaNs.
    const uint64_t values[] = {0,
                               family.sign,
                               1,
                               family.sign | 1,
                               family.infinity - 1,
                               family.sign | (family.infinity - 1),
                               family.infinity,
                               family.sign | family.infinity,
                               family.infinity | 1,
                               family.sign | family.infinity | 2,
                               family.infinity | family.quiet | 3,
                               family.sign | family.infinity | family.quiet | 4};
    constexpr int value_count = std::size(values);
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      SCOPED_TRACE(semantics);
      for (Call call : family.calls) {
        for (uint64_t a : values) {
          Registers<8, 32> r;
          for (int lane = 0; lane < 32; ++lane) {
            write_fp(r.v, lane, family.width, a);
            write_fp(r.v + 2, lane, family.width, values[lane % value_count]);
          }
          ASSERT_EQ(call(semantics, 0xffffffffU, 0, r.v + 4, r.v, r.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          for (uint64_t cpu : levels) {
            if (cpu == RJ_ARITH_CPU_BASELINE) {
              continue;
            }
            ASSERT_EQ(call(cpu | semantics, 0xffffffffU, 0, r.v + 6, r.v, r.v + 2, nullptr),
                      RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              EXPECT_EQ(read_fp(r.v + 6, lane, family.width), read_fp(r.v + 4, lane, family.width))
                  << family.width << '/' << cpu << '/' << a << '/' << values[lane % value_count];
            }
          }
        }
      }
    }
  }
}
