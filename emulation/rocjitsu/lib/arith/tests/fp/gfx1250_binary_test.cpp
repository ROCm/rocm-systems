// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 binary numerics, inherited Compatibility causes and strict qualification.

#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>

// Result and status witnesses distinguish gfx1250 from the gfx12 status policy.
TEST(FpBinaryF32Gfx1250, Arithmetic) {
  decltype(&rj_arith_v_add_f32_w32_gfx1250) calls32[] = {
      rj_arith_v_add_f32_w32_gfx1250, rj_arith_v_sub_f32_w32_gfx1250,
      rj_arith_v_subrev_f32_w32_gfx1250, rj_arith_v_mul_f32_w32_gfx1250};
  decltype(&rj_arith_v_add_f32_w64_gfx1250) calls64[] = {
      rj_arith_v_add_f32_w64_gfx1250, rj_arith_v_sub_f32_w64_gfx1250,
      rj_arith_v_subrev_f32_w64_gfx1250, rj_arith_v_mul_f32_w64_gfx1250};
  struct Case {
    int op;
    uint32_t a, b, result, causes;
    uint64_t modifiers = 0;
  };
  const Case cases[] = {{0, 0x3f800000, 0x40000000, 0x40400000, 0},
                        {0, 0x7f800000, 0xff800000, 0xffc00000, 0},
                        {1, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0},
                        {2, 0xff7fffff, 0x7f7fffff, 0x7f800000, 0},
                        {3, 0, 0x7f800000, 0xffc00000, 0},
                        {3, 0x7f7fffff, 0x40000000, 0x7f800000, 0x28},
                        {3, 0x00800000, 0x00800000, 0, 0x30},
                        {3, 1, 0x3f800000, 1, 0x12},
                        {3, 0x00800000, 0x3f000000, 0, 0, RJ_ARITH_ALU_OMOD_HALF}};
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (bool wave64 : {false, true}) {
        for (const auto &c : cases) {
          uint32_t a[64], b[64];
          for (int lane = 0; lane < 64; ++lane) {
            a[lane] = b[lane] = 0x3f800000;
          }
          a[0] = c.a;
          b[0] = c.b;
          const uint32_t *x = a, *y = b;
          uint32_t *d = a; // Classification must read the inputs before an aliased write.
          uint32_t status = 0x80;
          // ADD/SUB do not report status, so their output pointer remains unused.
          uint32_t *status_output = c.op == 3 ? &status : nullptr;
          const int error =
              wave64 ? calls64[c.op](cpu | semantics, 1, c.modifiers, &d, &x, &y, status_output)
                     : calls32[c.op](cpu | semantics, 1, c.modifiers, &d, &x, &y, status_output);
          const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
          EXPECT_EQ(error, strict ? (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                                         ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                                         : RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS)
                                  : RJ_ARITH_SUCCESS);
          const bool compatible = semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY;
          if (!strict && !compatible && (c.result & 0x7fffffffU) > 0x7f800000U) {
            EXPECT_GT(a[0] & 0x7fffffffU, 0x7f800000U);
          } else {
            EXPECT_EQ(a[0], strict ? c.a : c.result);
          }
          EXPECT_EQ(a[1], 0x3f800000U);
          EXPECT_EQ(status, 0x80U | (compatible ? c.causes : 0U));
        }
      }
    }
  }
}
