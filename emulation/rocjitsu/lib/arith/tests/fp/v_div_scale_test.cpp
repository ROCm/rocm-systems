// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for division operand rescaling and the resulting condition mask.

#include "fp/div_omod_gfx11_test.h"
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
  uint32_t instruction_flags = 0;
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
    // Edge of the FP32 window: adjusted exponent -1 overflows, 0 keeps a truncated denormal.
    {0xf0, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xf5, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xfa, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0},
    {0xff, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0},
    {0x00, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0},
    {0xf0, 0x9f800000, 0x7f7fffff, 0x7f7fffff, 0xff800000, 0},
    {0xf5, 0x9f800000, 0x7f7fffff, 0x7f7fffff, 0xff7fffff, 0},
    {0xfa, 0x9f800000, 0x7f7fffff, 0x7f7fffff, 0xff800000, 0},
    {0xff, 0x9f800000, 0x7f7fffff, 0x7f7fffff, 0xff7fffff, 0},
    {0x00, 0x9f800000, 0x7f7fffff, 0x7f7fffff, 0xff800000, 0},
    {0xf0, 0x20000000, 0x7f7fffff, 0x7f7fffff, 0x400000, 0},
    {0xf5, 0x20000000, 0x7f7fffff, 0x7f7fffff, 0x400000, 0},
    {0xfa, 0x20000000, 0x7f7fffff, 0x7f7fffff, 0x400000, 0},
    {0xff, 0x20000000, 0x7f7fffff, 0x7f7fffff, 0x400000, 0},
    {0x00, 0x20000000, 0x7f7fffff, 0x7f7fffff, 0x0, 0},
    {0xf0, 0xa0000001, 0x7f7fffff, 0x7f7fffff, 0x80400000, 0},
    {0xf5, 0xa0000001, 0x7f7fffff, 0x7f7fffff, 0x80400000, 0},
    {0xfa, 0xa0000001, 0x7f7fffff, 0x7f7fffff, 0x80400000, 0},
    {0xff, 0xa0000001, 0x7f7fffff, 0x7f7fffff, 0x80400000, 0},
    {0x00, 0xa0000001, 0x7f7fffff, 0x7f7fffff, 0x80000000, 0},
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
    // Boundary of the retained window: adjusted exponent -12 overflows, -11 and -1 keep a
    // guest-rounded denormal, in all four rounding modes and with outputs flushed.
    {0xf0, 0x740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xf5, 0x740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xfa, 0x740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0},
    {0xff, 0x740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0},
    {0x00, 0x740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000, 0},
    {0xf0, 0x8740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfff0000000000000, 0},
    {0xf5, 0x8740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xffefffffffffffff, 0},
    {0xfa, 0x8740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfff0000000000000, 0},
    {0xff, 0x8740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xffefffffffffffff, 0},
    {0x00, 0x8740000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfff0000000000000, 0},
    {0xf0, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0},
    {0xf5, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000001, 0},
    {0xfa, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0},
    {0xff, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0},
    {0x00, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0xf0, 0x8750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8000010000000000, 0},
    {0xf5, 0x8750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8000010000000000, 0},
    {0xfa, 0x8750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8000010000000001, 0},
    {0xff, 0x8750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8000010000000000, 0},
    {0x00, 0x8750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8000000000000000, 0},
    {0xf0, 0x7f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x4000000000000, 0},
    {0xf5, 0x7f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x4000000000001, 0},
    {0xfa, 0x7f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x4000000000000, 0},
    {0xff, 0x7f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x4000000000000, 0},
    {0x00, 0x7f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    {0xf0, 0x87f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8004000000000000, 0},
    {0xf5, 0x87f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8004000000000000, 0},
    {0xfa, 0x87f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8004000000000001, 0},
    {0xff, 0x87f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8004000000000000, 0},
    {0x00, 0x87f0000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8000000000000000, 0},
    {0xf0, 0x750000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0},
    {0xf5, 0x750000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0},
    {0xfa, 0x750000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0},
    {0xff, 0x750000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0},
    {0x00, 0x750000000000000, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0},
    // Guest rounding away from zero carries the largest denormal to the smallest normal.
    {0xf0, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000000, 0},
    {0xf5, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000000, 0},
    {0xfa, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfffffffffffff, 0},
    {0xff, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfffffffffffff, 0},
    {0xf0, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8010000000000000, 0},
    {0xf5, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x800fffffffffffff, 0},
    {0xfa, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x8010000000000000, 0},
    {0xff, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x800fffffffffffff, 0},
};

// gfx1201 output modifiers on window results (RDNA4 entry points; RDNA3 applies OMOD only with
// IEEE clear and outputs flushed). OMOD turns a denormal into +0, including FP64 values whose guest
// rounding would reach the smallest normal (A = 0x080fffffffffffff); CLAMP keeps that rounding.
const DivScaleCapture scale_down_omod_f32[] = {
    {0xf0, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xfa, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0x80, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xf0, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x0, 0, RJ_ARITH_ALU_OMOD_HALF},
    {0xfa, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x0, 0, RJ_ARITH_ALU_OMOD_HALF},
    {0x80, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x0, 0, RJ_ARITH_ALU_OMOD_HALF},
    {0xf0, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x400001, 0, RJ_ARITH_ALU_CLAMP},
    {0xfa, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x400001, 0, RJ_ARITH_ALU_CLAMP},
    {0x80, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x0, 0, RJ_ARITH_ALU_CLAMP},
    {0xf0, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0, RJ_ARITH_ALU_OMOD_2},
    {0xfa, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f7fffff, 0, RJ_ARITH_ALU_OMOD_2},
    {0x80, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0, RJ_ARITH_ALU_OMOD_2},
    {0xf0, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0, RJ_ARITH_ALU_OMOD_HALF},
    {0xfa, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7effffff, 0, RJ_ARITH_ALU_OMOD_HALF},
    {0x80, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000, 0, RJ_ARITH_ALU_OMOD_HALF},
    {0xf0, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x3f800000, 0, RJ_ARITH_ALU_CLAMP},
    {0xfa, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x3f800000, 0, RJ_ARITH_ALU_CLAMP},
    {0x80, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x3f800000, 0, RJ_ARITH_ALU_CLAMP},
};

const DivScaleCapture scale_down_omod_f64[] = {
    {0xf0, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xfa, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0x80, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xf0, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0xfa, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0x80, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0xf0, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000000, 0,
     RJ_ARITH_ALU_CLAMP},
    {0xfa, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0xfffffffffffff, 0,
     RJ_ARITH_ALU_CLAMP},
    {0x80, 0x80fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000000, 0,
     RJ_ARITH_ALU_CLAMP},
    {0xf0, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xfa, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0x80, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xf0, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0xfa, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0x80, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0xf0, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_CLAMP},
    {0xfa, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_CLAMP},
    {0x80, 0x880fffffffffffff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_CLAMP},
    {0xf0, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xfa, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0x80, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0, RJ_ARITH_ALU_OMOD_2},
    {0xf0, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0xfa, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0x80, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x0, 0,
     RJ_ARITH_ALU_OMOD_HALF},
    {0xf0, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0,
     RJ_ARITH_ALU_CLAMP},
    {0xfa, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0,
     RJ_ARITH_ALU_CLAMP},
    {0x80, 0x750000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff, 0x10000000000, 0,
     RJ_ARITH_ALU_CLAMP},
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
        ASSERT_EQ(call(flags, ~Mask(0), test.instruction_flags, r.v + 6, &condition, r.v, r.v + 2,
                       r.v + 4),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(condition, test.condition ? ~Mask(0) : Mask(0));
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(read_fp(r.v + 6, lane, Width), test.d)
              << std::hex << cpu << '/' << semantics << " mode 0x" << test.mode << " a 0x" << test.a
              << " flags 0x" << test.instruction_flags << " lane " << std::dec << lane;
        }
      }
    }
  }
}

// gfx1201 wave32 capture (gfx1100 identical): lane 19 scales below the normal range while every
// other lane stays normal, so the repaired lane sits in a later SIMD chunk. Wave64 repeats it in
// lane 51. Each lane must keep its own operands.
template <int Width, int Lanes, typename Call> void check_repaired_lane(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr uint64_t b = Width == 32 ? 0x7f7fffffULL : 0x7fefffffffffffffULL;
  constexpr uint64_t normal = Width == 32 ? 0x3f800000ULL : 0x3ff0000000000000ULL;
  constexpr uint64_t normal_d = Width == 32 ? 0x1f800000ULL : 0x37f0000000000000ULL;
  constexpr uint64_t tiny = Width == 32 ? 0x20000003ULL : 0x0750000000000001ULL;
  constexpr uint64_t tiny_d = Width == 32 ? 0x00400001ULL : 0x0000010000000000ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      Registers<8, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        write_fp(r.v, lane, Width, lane % 32 == 19 ? tiny : normal);
        write_fp(r.v + 2, lane, Width, b);
        write_fp(r.v + 4, lane, Width, b);
      }
      Mask condition = 0;
      ASSERT_EQ(call(cpu | semantics, ~Mask(0), 0, r.v + 6, &condition, r.v, r.v + 2, r.v + 4),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(read_fp(r.v + 6, lane, Width), lane % 32 == 19 ? tiny_d : normal_d)
            << std::hex << cpu << '/' << semantics << " lane " << std::dec << lane;
      }
    }
  }
}

} // namespace

TEST(FpDivScale, ScaleDownLeavesNormalRange) {
  check_scale_down<32, 32>(rj_arith_v_div_scale_f32_w32_gfx12, scale_down_f32);
  check_scale_down<32, 64>(rj_arith_v_div_scale_f32_w64_gfx12, scale_down_f32);
  check_scale_down<32, 32>(rj_arith_v_div_scale_f32_w32, scale_down_f32);
  check_scale_down<32, 64>(rj_arith_v_div_scale_f32_w64, scale_down_f32);
  check_scale_down<64, 32>(rj_arith_v_div_scale_f64_w32_gfx12, scale_down_f64);
  check_scale_down<64, 64>(rj_arith_v_div_scale_f64_w64_gfx12, scale_down_f64);
  check_scale_down<64, 32>(rj_arith_v_div_scale_f64_w32, scale_down_f64);
  check_scale_down<64, 64>(rj_arith_v_div_scale_f64_w64, scale_down_f64);
}

TEST(FpDivScale, ScaleDownOutputModifiers) {
  check_scale_down<32, 32>(rj_arith_v_div_scale_f32_w32_gfx12, scale_down_omod_f32);
  check_scale_down<32, 64>(rj_arith_v_div_scale_f32_w64_gfx12, scale_down_omod_f32);
  check_scale_down<64, 32>(rj_arith_v_div_scale_f64_w32_gfx12, scale_down_omod_f64);
  check_scale_down<64, 64>(rj_arith_v_div_scale_f64_w64_gfx12, scale_down_omod_f64);
}

TEST(FpDivScale, RepairedLaneKeepsItsOperands) {
  check_repaired_lane<32, 32>(rj_arith_v_div_scale_f32_w32_gfx12);
  check_repaired_lane<32, 64>(rj_arith_v_div_scale_f32_w64_gfx12);
  check_repaired_lane<32, 32>(rj_arith_v_div_scale_f32_w32);
  check_repaired_lane<32, 64>(rj_arith_v_div_scale_f32_w64);
  check_repaired_lane<64, 32>(rj_arith_v_div_scale_f64_w32_gfx12);
  check_repaired_lane<64, 64>(rj_arith_v_div_scale_f64_w64_gfx12);
  check_repaired_lane<64, 32>(rj_arith_v_div_scale_f64_w32);
  check_repaired_lane<64, 64>(rj_arith_v_div_scale_f64_w64);
}

// CDNA ignores output modifiers and retains its inherited gradual-underflow rule.
template <int Width, int Lanes, typename Call>
static void check_cdna_scale(Call call, Call inherited) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  auto input = make_div_scale_inputs<Width, Lanes>();
  constexpr uint64_t snan = Width == 32 ? 0x7f800001ULL : 0x7ff0000000000001ULL;
  constexpr uint64_t quiet = Width == 32 ? 0x400000ULL : 0x8000000000000ULL;
  write_fp(input.v, 0, Width, snan);
  write_fp(input.v + 2, 0, Width, snan);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int sample = 0; sample < 64; ++sample) {
      const uint64_t fp = (uint64_t(sample & 3) << 24) |
                          (sample & 4 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                          (sample & 8 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                          (sample & 16 ? RJ_ARITH_FP_IEEE_MODE : 0);
      const uint64_t signs = sample & 32 ? RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B : 0;
      const uint64_t ignored = uint64_t(sample & 3) * RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP;
      const Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL) | 1;
      const int destination = sample & 1 ? 0 : 6;
      auto expected = input;
      Mask expected_condition = 0;
      ASSERT_EQ(inherited(fp | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, exec_mask,
                          signs, expected.v + destination, &expected_condition, expected.v,
                          expected.v + 2, expected.v + 4),
                0);
      for (uint64_t semantics :
           {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
            RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
        auto actual = input;
        Mask condition = ~Mask(0);
        ASSERT_EQ(call(cpu | fp | semantics, exec_mask, signs | ignored, actual.v + destination,
                       &condition, actual.v, actual.v + 2, actual.v + 4),
                  0);
        EXPECT_EQ(condition, expected_condition);
        actual.expect_preserved(input, destination, Width / 32);
        for (int reg = 0; reg < Width / 32; ++reg) {
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(actual.v[destination + reg][lane], expected.v[destination + reg][lane]);
          }
        }
        EXPECT_EQ(read_fp(actual.v + destination, 0, Width) & ~(1ULL << (Width - 1)), snan | quiet);
      }
    }
    // The RDNA below-window overflow rule must not replace inherited CDNA underflow.
    auto tiny = input;
    for (int lane = 0; lane < Lanes; ++lane) {
      write_fp(tiny.v, lane, Width, Width == 32 ? 0x00800000ULL : 0x0010000000000000ULL);
      write_fp(tiny.v + 2, lane, Width, Width == 32 ? 0x7f000000ULL : 0x7fe0000000000000ULL);
      write_fp(tiny.v + 4, lane, Width, Width == 32 ? 0x64000000ULL : 0x6000000000000000ULL);
    }
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      auto actual = tiny;
      for (int lane = 0; lane < Lanes; ++lane) {
        write_fp(actual.v + 6, lane, Width, 0xdeadbeefULL);
      }
      Mask condition = ~Mask(0);
      ASSERT_EQ(call(cpu | semantics, ~Mask(0), 0, actual.v + 6, &condition, actual.v, actual.v + 2,
                     actual.v + 4),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(condition, Mask(0));
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(read_fp(actual.v + 6, lane, Width), 0U);
      }
    }
  }
  Mask condition = 7;
  EXPECT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, Mask(0), 0, nullptr, &condition, nullptr,
                 nullptr, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  EXPECT_EQ(condition, Mask(7));
  EXPECT_EQ(call(0, Mask(0), 0, nullptr, &condition, nullptr, nullptr, nullptr), 0);
  EXPECT_EQ(condition, Mask(0));
}

TEST(CdnaDivScale, Arithmetic) {
  check_cdna_scale<32, 32>(rj_arith_v_div_scale_f32_w32_gfx942, rj_arith_v_div_scale_f32_w32_gfx12);
  check_cdna_scale<32, 64>(rj_arith_v_div_scale_f32_w64_gfx942, rj_arith_v_div_scale_f32_w64_gfx12);
  check_cdna_scale<64, 32>(rj_arith_v_div_scale_f64_w32_gfx942, rj_arith_v_div_scale_f64_w32_gfx12);
  check_cdna_scale<64, 64>(rj_arith_v_div_scale_f64_w64_gfx942, rj_arith_v_div_scale_f64_w64_gfx12);
}

namespace {

using rocjitsu::arith::test::DivOmodGfx11;

// W7900 (gfx1100, ROCm 7.2) wave32 lanes with output modifiers, the hardware MODE word written per
// lane (rounding, denormals, DX10_CLAMP bit 8, IEEE bit 9): the 2026-11-12 probe3 scale-down
// sweep with IEEE clear (hw3-1100-noieee) and the 2026-11-18 omod3 probe. OMOD is active only
// with output denormals flushed and IEEE clear; it scales the rounded value, overflowing per the
// guest rounding, and turns a denormal into +0 while the smallest normal halved keeps its sign.
// With output denormals preserved the window result and the overflow value stay unscaled.
const DivOmodGfx11 omod_lanes_f32[] = {
    {"mul2 overflow nearest", 0x100, 0x7f000000, 0x3f800000, 0x3f800000, 0x7f800000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round up", 0x101, 0x7f000000, 0x3f800000, 0x3f800000, 0x7f800000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round down", 0x102, 0x7f000000, 0x3f800000, 0x3f800000, 0x7f7fffff,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow toward zero", 0x103, 0x7f000000, 0x3f800000, 0x3f800000, 0x7f7fffff,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 with input denormals allowed", 0x110, 0x7f000000, 0x3f800000, 0x3f800000, 0x7f800000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with output denormals", 0x120, 0x7f000000, 0x3f800000, 0x3f800000, 0x7f000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with IEEE", 0x300, 0x7f000000, 0x3f800000, 0x3f800000, 0x7f000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul4 negative overflow nearest", 0x100, 0xfe800000, 0x3f800000, 0x3f800000, 0xff800000,
     RJ_ARITH_ALU_OMOD_4},
    {"mul4 negative overflow round up", 0x101, 0xfe800000, 0x3f800000, 0x3f800000, 0xff7fffff,
     RJ_ARITH_ALU_OMOD_4},
    {"div2 flushes the smallest normal", 0x100, 0x00800000, 0x3f800000, 0x3f800000, 0x00000000,
     RJ_ARITH_ALU_OMOD_HALF},
    {"div2 keeps the flushed sign", 0x100, 0x80800000, 0x3f800000, 0x3f800000, 0x80000000,
     RJ_ARITH_ALU_OMOD_HALF},
    {"mul2 of a negative denormal is +0", 0x110, 0x80000001, 0x3f800000, 0x3f800000, 0x00000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 after scaling a denormal denominator up", 0x110, 0x00000001, 0x00000001, 0x3f800000,
     0x15800000, RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 after scaling the numerator up", 0x110, 0x00400000, 0x00400000, 0x7f000000, 0x20800000,
     RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 ignored after scaling the numerator up", 0x130, 0x00400000, 0x00400000, 0x7f000000,
     0x20000000, RJ_ARITH_ALU_OMOD_2, true},
    {"flushed denominator gives NaN under mul2", 0x100, 0x00400000, 0x00400000, 0x7f000000,
     0xffc00000, RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 keeps a signaling NaN", 0x100, 0x7f800001, 0x3f800000, 0x3f800000, 0x7f800001,
     RJ_ARITH_ALU_OMOD_2},
    {"zero denominator gives NaN under mul2", 0x100, 0x3f800000, 0x00000000, 0x3f800000, 0xffc00000,
     RJ_ARITH_ALU_OMOD_2, true},
    {"clamp of a NaN with DX10_CLAMP", 0x100, 0x7f800001, 0x3f800000, 0x3f800000, 0x00000000,
     RJ_ARITH_ALU_CLAMP},
    {"clamp keeps a NaN without DX10_CLAMP", 0x000, 0x7f800001, 0x3f800000, 0x3f800000, 0x7f800001,
     RJ_ARITH_ALU_CLAMP},
    // Scale-down window (probe3): with outputs flushed the window result is zero either way; with
    // output denormals preserved OMOD is ignored, so the truncated denormal and the overflow value
    // survive where RDNA4 would scale them.
    {"mul2 flushed window result", 0x180, 0x20000003, 0x7f7fffff, 0x7f7fffff, 0x00000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored keeps the truncated denormal", 0x1f0, 0x20000003, 0x7f7fffff, 0x7f7fffff,
     0x00400001, RJ_ARITH_ALU_OMOD_2},
    {"div2 ignored keeps the truncated denormal round down", 0x1fa, 0x20000003, 0x7f7fffff,
     0x7f7fffff, 0x00400001, RJ_ARITH_ALU_OMOD_HALF},
    {"div2 ignored keeps the overflow value round down", 0x1fa, 0x1f800000, 0x7f7fffff, 0x7f7fffff,
     0x7f7fffff, RJ_ARITH_ALU_OMOD_HALF},
    {"mul2 of the overflow value nearest", 0x100, 0x1f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 of the negative overflow value", 0x180, 0x9f800000, 0x7f7fffff, 0x7f7fffff, 0xff800000,
     RJ_ARITH_ALU_OMOD_2},
};

const DivOmodGfx11 omod_lanes_f64[] = {
    {"mul2 overflow nearest", 0x100, 0x7fe0000000000000, 0x3ff0000000000000, 0x3ff0000000000000,
     0x7ff0000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round up", 0x104, 0x7fe0000000000000, 0x3ff0000000000000, 0x3ff0000000000000,
     0x7ff0000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round down", 0x108, 0x7fe0000000000000, 0x3ff0000000000000, 0x3ff0000000000000,
     0x7fefffffffffffff, RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow toward zero", 0x10c, 0x7fe0000000000000, 0x3ff0000000000000, 0x3ff0000000000000,
     0x7fefffffffffffff, RJ_ARITH_ALU_OMOD_2},
    {"mul2 with input denormals allowed", 0x140, 0x7fe0000000000000, 0x3ff0000000000000,
     0x3ff0000000000000, 0x7ff0000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with output denormals", 0x180, 0x7fe0000000000000, 0x3ff0000000000000,
     0x3ff0000000000000, 0x7fe0000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with all denormals", 0x1c0, 0x7fe0000000000000, 0x3ff0000000000000,
     0x3ff0000000000000, 0x7fe0000000000000, RJ_ARITH_ALU_OMOD_2},
    {"div2 flushes the smallest normal", 0x100, 0x0010000000000000, 0x3ff0000000000000,
     0x3ff0000000000000, 0x0000000000000000, RJ_ARITH_ALU_OMOD_HALF},
    {"div2 keeps the flushed sign", 0x100, 0x8010000000000000, 0x3ff0000000000000,
     0x3ff0000000000000, 0x8000000000000000, RJ_ARITH_ALU_OMOD_HALF},
    {"mul2 of a negative denormal is +0", 0x140, 0x8000000000000001, 0x3ff0000000000000,
     0x3ff0000000000000, 0x0000000000000000, RJ_ARITH_ALU_OMOD_2},
    {"flushed denominator gives NaN under mul2", 0x100, 0x7fe0000000000000, 0x0008000000000000,
     0x7fe0000000000000, 0xfff8000000000000, RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 keeps a signaling NaN", 0x100, 0x7ff0000000000001, 0x3ff0000000000000,
     0x3ff0000000000000, 0x7ff0000000000001, RJ_ARITH_ALU_OMOD_2},
    // Scale-down window (probe3): the carry to the smallest normal is kept only where OMOD is
    // ignored; active OMOD on a flushed output gives zero.
    {"mul2 ignored keeps the carry to the smallest normal", 0x1f0, 0x080fffffffffffff,
     0x7fefffffffffffff, 0x7fefffffffffffff, 0x0010000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored keeps the largest denormal round down", 0x1fa, 0x080fffffffffffff,
     0x7fefffffffffffff, 0x7fefffffffffffff, 0x000fffffffffffff, RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with output denormals keeps the carry", 0x180, 0x080fffffffffffff,
     0x7fefffffffffffff, 0x7fefffffffffffff, 0x0010000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 flushed window result", 0x100, 0x080fffffffffffff, 0x7fefffffffffffff,
     0x7fefffffffffffff, 0x0000000000000000, RJ_ARITH_ALU_OMOD_2},
    {"div2 flushed window result with input denormals", 0x150, 0x080fffffffffffff,
     0x7fefffffffffffff, 0x7fefffffffffffff, 0x0000000000000000, RJ_ARITH_ALU_OMOD_HALF},
    {"mul2 ignored keeps the truncated denormal", 0x1f0, 0x0750000000000001, 0x7fefffffffffffff,
     0x7fefffffffffffff, 0x0000010000000000, RJ_ARITH_ALU_OMOD_2},
};

} // namespace

// Active RDNA3 OMOD is accepted in strict exact and reproduces the W7900 lanes at every CPU level.
TEST(FpDivScaleGfx11, CapturedOutputModifiers) {
  using rocjitsu::arith::test::check_div_scale_omod_gfx11;
  check_div_scale_omod_gfx11<32, 32>(rj_arith_v_div_scale_f32_w32, omod_lanes_f32);
  check_div_scale_omod_gfx11<64, 32>(rj_arith_v_div_scale_f32_w64, omod_lanes_f32);
  check_div_scale_omod_gfx11<32, 64>(rj_arith_v_div_scale_f64_w32, omod_lanes_f64);
  check_div_scale_omod_gfx11<64, 64>(rj_arith_v_div_scale_f64_w64, omod_lanes_f64);
}
