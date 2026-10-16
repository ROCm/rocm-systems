// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP64 rounding to integral values.

#include "fp/loose_unary_f64.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

// Check FP64 rounding to integral values.
TEST(FpCeilF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_ceil_f64_w32_gfx12, [](double x) {
    double value = std::ceil(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpCeilF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_ceil_f64_w32, [](double x) {
    double value = std::ceil(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpCeilF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_ceil_f64_w64_gfx12, [](double x) {
    double value = std::ceil(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpCeilF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_ceil_f64_w64, [](double x) {
    double value = std::ceil(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpFloorF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_floor_f64_w32_gfx12, [](double x) {
    double value = std::floor(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpFloorF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_floor_f64_w32, [](double x) {
    double value = std::floor(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpFloorF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_floor_f64_w64_gfx12, [](double x) {
    double value = std::floor(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpFloorF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_floor_f64_w64, [](double x) {
    double value = std::floor(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpTruncF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_trunc_f64_w32_gfx12, [](double x) {
    double value = std::trunc(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpTruncF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_trunc_f64_w32, [](double x) {
    double value = std::trunc(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpTruncF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_trunc_f64_w64_gfx12, [](double x) {
    double value = std::trunc(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpTruncF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_trunc_f64_w64, [](double x) {
    double value = std::trunc(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpRndneF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rndne_f64_w32_gfx12, [](double x) {
    double value = std::nearbyint(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpRndneF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rndne_f64_w32, [](double x) {
    double value = std::nearbyint(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpRndneF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rndne_f64_w64_gfx12, [](double x) {
    double value = std::nearbyint(x);
    return value;
  });
}

// Check FP64 rounding to integral values.
TEST(FpRndneF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rndne_f64_w64, [](double x) {
    double value = std::nearbyint(x);
    return value;
  });
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_fract_f64_w32_gfx12, [](double x) {
    double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
    return value;
  });
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_fract_f64_w32, [](double x) {
    double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
    return value;
  });
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_fract_f64_w64_gfx12, [](double x) {
    double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
    return value;
  });
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_fract_f64_w64, [](double x) {
    double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
    return value;
  });
}
