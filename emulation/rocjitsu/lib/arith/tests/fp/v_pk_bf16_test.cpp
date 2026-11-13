// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed BF16 arithmetic shares inputs across CPU paths, including aliases and overflow.
#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <gtest/gtest.h>
#include <stdint.h>

namespace {
// Check exact representable results, masked lanes and finite-overflow saturation.
void check(int op) {
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      uint32_t values[32];
      for (auto &value : values) {
        value = 0x40003f80U;
      }
      uint32_t *d = values;
      const uint32_t *a = values;
      uint32_t status = 123;
      const uint64_t flags = cpu | semantics;
      int error = op == 0   ? rj_arith_v_pk_add_bf16_w32(flags, 1, 0, &d, &a, &a, &status)
                  : op == 1 ? rj_arith_v_pk_mul_bf16_w32(flags, 1, 0, &d, &a, &a, &status)
                            : rj_arith_v_pk_fma_bf16_w32(flags, 1, 0, &d, &a, &a, &a, &status);
      ASSERT_EQ(error, RJ_ARITH_SUCCESS);
      EXPECT_EQ(values[0], op == 0 ? 0x40804000U : op == 1 ? 0x40803f80U : 0x40c04000U);
      EXPECT_EQ(values[1], 0x40003f80U);
      EXPECT_EQ(status, 123U);
      values[0] = 0x7f7f7f7fU;
      error =
          op == 0
              ? rj_arith_v_pk_add_bf16_w32(flags | RJ_ARITH_FP16_OVFL, 1, 0, &d, &a, &a, nullptr)
          : op == 1
              ? rj_arith_v_pk_mul_bf16_w32(flags | RJ_ARITH_FP16_OVFL, 1, 0, &d, &a, &a, nullptr)
              : rj_arith_v_pk_fma_bf16_w32(flags | RJ_ARITH_FP16_OVFL, 1, 0, &d, &a, &a, &a,
                                           nullptr);
      ASSERT_EQ(error, RJ_ARITH_SUCCESS);
      EXPECT_EQ(values[0], 0x7f7f7f7fU);
    }
  }
}
} // namespace
TEST(PackedBf16, Add) { check(0); }
TEST(PackedBf16, Mul) { check(1); }
TEST(PackedBf16, Fma) { check(2); }

// A compact inherited-model table covers zero ordering, denormals and NaN kinds.
TEST(PackedBf16, NumericSelection) {
  struct Case {
    uint32_t a, b, minimum, maximum;
  };
  const Case cases[] = {{0x00008000U, 0x80000000U, 0x80008000U, 0},
                        {0x80010001U, 0x00000000U, 0x80010000U, 0x00000001U},
                        {0x7fc13f80U, 0x3f807fc1U, 0x3f803f80U, 0x3f803f80U},
                        {0x7f813f80U, 0x3f80ff82U, 0x3f803f80U, 0x3f803f80U},
                        {0x7fc17f81U, 0xff827fc2U, 0x7fc17fc1U, 0x7fc17fc1U}};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (auto function : {rj_arith_v_pk_min_num_bf16_w32, rj_arith_v_pk_max_num_bf16_w32}) {
        for (const auto &test : cases) {
          uint32_t a[32] = {test.a, 123}, b[32] = {test.b};
          uint32_t *d = a;
          const uint32_t *pa = a, *pb = b;
          ASSERT_EQ(function(cpu | semantics, 1, 0, &d, &pa, &pb, nullptr), RJ_ARITH_SUCCESS);
          EXPECT_EQ(a[0], function == rj_arith_v_pk_min_num_bf16_w32 ? test.minimum : test.maximum);
          EXPECT_EQ(a[1], 123U);
        }
      }
    }
  }
}
