// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for division operand rescaling and the resulting condition mask.

#include "fp/div_scale_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <type_traits>

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 32>(rj_arith_v_div_scale_f32_w32_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 32>(rj_arith_v_div_scale_f32_w32);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 64>(rj_arith_v_div_scale_f32_w64_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 64>(rj_arith_v_div_scale_f32_w64);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 32>(rj_arith_v_div_scale_f64_w32_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 32>(rj_arith_v_div_scale_f64_w32);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 64>(rj_arith_v_div_scale_f64_w64_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 64>(rj_arith_v_div_scale_f64_w64);
}

// gfx1100/W7900 captures supplied in PR #13952: selected sNaNs stay signaling
// regardless of IEEE; A equals the NaN denominator or numerator.
template <int Width, int Lanes, typename Call> static void check_rdna3_nan(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr uint64_t one = Width == 32 ? 0x3f800000ULL : 0x3ff0000000000000ULL;
  constexpr uint64_t snan = Width == 32 ? 0x7f800001ULL : 0x7ff0000000000001ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        for (int operand : {1, 2}) {
          Registers<8, Lanes> r;
          for (int i = 0; i < 3; ++i) {
            write_fp(r.v + 2 * i, Lanes - 1, Width, i == 0 || i == operand ? snan : one);
          }
          Mask condition = ~Mask(0);
          ASSERT_EQ(call(cpu | semantics | ieee, Mask(1) << (Lanes - 1), 0, r.v + 6, &condition,
                         r.v, r.v + 2, r.v + 4),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(condition & ~(Mask(1) << (Lanes - 1)), Mask(0));
          EXPECT_EQ(read_fp(r.v + 6, Lanes - 1, Width), snan);
        }
      }
    }
  }
}

TEST(FpDivScaleGfx11, SignalingNan) {
  check_rdna3_nan<32, 32>(rj_arith_v_div_scale_f32_w32);
  check_rdna3_nan<32, 64>(rj_arith_v_div_scale_f32_w64);
  check_rdna3_nan<64, 32>(rj_arith_v_div_scale_f64_w32);
  check_rdna3_nan<64, 64>(rj_arith_v_div_scale_f64_w64);
}

namespace {

// One gfx1201 capture (gfx1100 is bit-identical): MODE, A, B, C, result and condition.
struct DivScaleCapture {
  uint32_t mode;
  uint64_t a, b, c, d;
  bool condition;
};

// Scaling A down by 2^-64 (FP32) or 2^-128 (FP64) below the normal range: the overflow value,
// FP32 truncation and flushed zeros, across flushing and directed MODEs. Develop returned
// gradually underflowed values for every case.
const DivScaleCapture scale_down_f32[] = {
    {0xf0, 0x1, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xf0, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xf0, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x400001, 0},
    {0xf5, 0x1, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xf5, 0x80000001, 0x7f7fffff, 0x7f7fffff, 0xff7fffff, 0},
    {0xf5, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xf5, 0x80800000, 0x7f7fffff, 0x7f7fffff, 0xff7fffff, 0},
    {0xf5, 0x20000001, 0x7f7fffff, 0x7f7fffff, 0x400000, 0},
    {0xfa, 0x1, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0},
    {0xfa, 0x80000001, 0x7f7fffff, 0x7f7fffff, 0xff800000, 0},
    {0xfa, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0},
    {0xfa, 0x80800000, 0x7f7fffff, 0x7f7fffff, 0xff800000, 0},
    {0xfa, 0xa0000001, 0x7f7fffff, 0x7f7fffff, 0x80400000, 0},
    {0xff, 0x1, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0},
    {0xff, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0},
    {0x00, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x00, 0x207fffff, 0x7f7fffff, 0x7f7fffff, 0x0, 0},
    {0x50, 0x1, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x50, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x50, 0x207fffff, 0x7f7fffff, 0x7f7fffff, 0x0, 0},
    {0xa0, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xa0, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x400001, 0},
    {0x10, 0x1, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x10, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x10, 0x207fffff, 0x7f7fffff, 0x7f7fffff, 0x0, 0},
    {0x20, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x20, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x400001, 0},
    {0x30, 0x1, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x30, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x30, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x400001, 0},
    {0x40, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x40, 0x207fffff, 0x7f7fffff, 0x7f7fffff, 0x0, 0},
    {0x80, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0x80, 0x207fffff, 0x7f7fffff, 0x7f7fffff, 0x0, 0},
    {0xc0, 0x800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xc0, 0x207fffff, 0x7f7fffff, 0x7f7fffff, 0x0, 0},
};

const DivScaleCapture scale_down_f64[] = {
    {0xf0, 0x1, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xf0, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xf5, 0x1, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xf5, 0x8000000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xffefffffffffffff, 0},
    {0xf5, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xf5, 0x8010000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xffefffffffffffff, 0},
    {0xfa, 0x1, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0},
    {0xfa, 0x8000000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfff0000000000000, 0},
    {0xfa, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0},
    {0xfa, 0x8010000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfff0000000000000, 0},
    {0xff, 0x1, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0},
    {0xff, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0},
    {0x00, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x00, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0x50, 0x1, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x50, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x50, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0xa0, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x10, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x10, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0x20, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x20, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0x30, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x30, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0x40, 0x1, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x40, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0x40, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0x80, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xc0, 0x1, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xc0, 0x10000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
};

// Run every lane with the same capture, so SIMD lanes repair together.
template <int Width, int Lanes, size_t N, typename Call>
void check_scale_down(Call call, const DivScaleCapture (&captures)[N]) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (const auto &test : captures) {
        const uint32_t round = Width == 32 ? test.mode & 3 : (test.mode >> 2) & 3;
        const uint32_t denorm = Width == 32 ? (test.mode >> 4) & 3 : (test.mode >> 6) & 3;
        const uint64_t flags = cpu | semantics | (uint64_t(round) << 24) |
                               (denorm & 1 ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) |
                               (denorm & 2 ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
        Registers<8, Lanes> r;
        for (int lane = 0; lane < Lanes; ++lane) {
          write_fp(r.v, lane, Width, test.a);
          write_fp(r.v + 2, lane, Width, test.b);
          write_fp(r.v + 4, lane, Width, test.c);
        }
        Mask condition = 0;
        ASSERT_EQ(call(flags, ~Mask(0), 0, r.v + 6, &condition, r.v, r.v + 2, r.v + 4),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(condition, test.condition ? ~Mask(0) : Mask(0));
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(read_fp(r.v + 6, lane, Width), test.d)
              << std::hex << cpu << '/' << semantics << " mode 0x" << test.mode << " a 0x" << test.a
              << " lane " << std::dec << lane;
        }
      }
    }
  }
}

} // namespace

TEST(FpDivScale, ScaleDownLeavesNormalRange) {
  check_scale_down<32, 32>(rj_arith_v_div_scale_f32_w32_gfx12, scale_down_f32);
  check_scale_down<32, 64>(rj_arith_v_div_scale_f32_w64_gfx12, scale_down_f32);
  check_scale_down<32, 32>(rj_arith_v_div_scale_f32_w32, scale_down_f32);
  check_scale_down<64, 32>(rj_arith_v_div_scale_f64_w32_gfx12, scale_down_f64);
  check_scale_down<64, 64>(rj_arith_v_div_scale_f64_w64_gfx12, scale_down_f64);
  check_scale_down<64, 32>(rj_arith_v_div_scale_f64_w32, scale_down_f64);
}
