// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for legacy FP min/max.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

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
