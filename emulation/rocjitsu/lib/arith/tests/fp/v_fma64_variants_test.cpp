// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA accumulating and literal FP64 FMA forms share directed-rounding goldens.
#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace {

// Check the low/high VGPR mapping, masked stores and inherited semantics support.
template <int Lanes, typename Call> void check_variant(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics : rocjitsu::arith::test::semantics_modes()) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        uint32_t av[2][Lanes]{}, bv[2][Lanes]{}, dv[2][Lanes]{};
        for (int lane = 0; lane < Lanes; ++lane) {
          av[1][lane] = 0x3ff00000U; // 1
          bv[1][lane] = 0x3ca00000U; // 2^-53, halfway to the next number above 1.
          dv[0][lane] = 0x12345678U;
          dv[1][lane] = 0x9abcdef0U;
        }
        const uint32_t *a[] = {av[0], av[1]}, *b[] = {bv[0], bv[1]};
        uint32_t *d[] = {dv[0], dv[1]};
        uint32_t status = 0x1234;
        const int result = call(cpu | semantics | (uint64_t(rounding) << 24),
                                Mask(1) << (Lanes - 1), d, a, b, &status);
        if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
            semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
          EXPECT_NE(result, RJ_ARITH_SUCCESS);
          continue;
        }
        ASSERT_EQ(result, RJ_ARITH_SUCCESS);
        EXPECT_EQ(dv[0][Lanes - 1], rounding == 1 ? 1U : 0U);
        EXPECT_EQ(dv[1][Lanes - 1], 0x3ff00000U);
        EXPECT_EQ(status, 0x1234U);
        for (int lane = 0; lane < Lanes - 1; ++lane) {
          EXPECT_EQ(dv[0][lane], 0x12345678U);
          EXPECT_EQ(dv[1][lane], 0x9abcdef0U);
        }
      }
    }
  }
}

// Seed the implicit accumulator only in active lanes, retaining inactive sentinels.
template <int Lanes, typename Call> auto accumulating(Call call) {
  return [call](uint64_t flags, auto mask, uint32_t *const *d, const uint32_t *const *a,
                const uint32_t *const *b, uint32_t *status) {
    for (int lane = 0; lane < Lanes; ++lane) {
      if ((mask >> lane) & 1) {
        d[0][lane] = b[0][lane];
        d[1][lane] = b[1][lane];
      }
    }
    return call(flags, mask, 0, d, a, a, status);
  };
}
// D may be a multiplicand; rejected accumulator modifiers leave it untouched.
template <int Lanes, typename Call> void check_accumulator_alias(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    uint32_t words[2][Lanes]{};
    for (int lane = 0; lane < Lanes; ++lane) {
      words[1][lane] = 0x3ff00000U;
    }
    uint32_t *d[] = {words[0], words[1]};
    const uint32_t *a[] = {words[0], words[1]};
    uint32_t state = 0;
    const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY;
    ASSERT_EQ(call(flags, Mask(1), RJ_ARITH_ALU_ABS_C, d, a, a, &state),
              RJ_ARITH_ERROR_INVALID_FLAGS);
    EXPECT_EQ(words[1][0], 0x3ff00000U);
    ASSERT_EQ(call(flags, Mask(1), 0, d, a, a, &state), RJ_ARITH_SUCCESS);
    EXPECT_EQ(words[0][0], 0U);
    EXPECT_EQ(words[1][0], 0x40000000U);
    EXPECT_EQ(words[1][1], 0x3ff00000U);
  }
}
// The inherited CDNA FMAC rounds before flushing: rounding can rescue the
// smallest normal from a product just below it, unlike the gfx12 tiny-product rule.
template <int Lanes, typename Call> void check_min_normal(Call call) {
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      uint32_t av[2][Lanes]{}, bv[2][Lanes]{}, dv[2][Lanes]{};
      av[1][0] = 0x00100000U;
      bv[0][0] = 0xffffffffU;
      bv[1][0] = 0x3fefffffU;
      const uint32_t *a[] = {av[0], av[1]}, *b[] = {bv[0], bv[1]};
      uint32_t *d[] = {dv[0], dv[1]};
      uint32_t status = 0;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                         (uint64_t(rounding) << 24),
                     1, 0, d, a, b, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(dv[0][0], 0U);
      EXPECT_EQ(dv[1][0], rounding < 2 ? 0x00100000U : 0U);
    }
  }
}
} // namespace

TEST(FpFmacF64W32Gfx942, Arithmetic) {
  check_variant<32>(accumulating<32>(rj_arith_v_fmac_f64_w32_gfx942));
  check_accumulator_alias<32>(rj_arith_v_fmac_f64_w32_gfx942);
  check_min_normal<32>(rj_arith_v_fmac_f64_w32_gfx942);
}
TEST(FpFmacF64W64Gfx942, Arithmetic) {
  check_variant<64>(accumulating<64>(rj_arith_v_fmac_f64_w64_gfx942));
  check_accumulator_alias<64>(rj_arith_v_fmac_f64_w64_gfx942);
  check_min_normal<64>(rj_arith_v_fmac_f64_w64_gfx942);
}
TEST(FpFmacF64W32Gfx1250, Arithmetic) {
  check_variant<32>(accumulating<32>(rj_arith_v_fmac_f64_w32_gfx1250));
  check_accumulator_alias<32>(rj_arith_v_fmac_f64_w32_gfx1250);
  check_min_normal<32>(rj_arith_v_fmac_f64_w32_gfx1250);
}
TEST(FpFmamkF64W32Gfx1250, Arithmetic) {
  check_variant<32>([](auto flags, auto mask, auto d, auto a, auto b, auto status) {
    return rj_arith_v_fmamk_f64_w32_gfx1250(flags, mask, 0, d, a, 0x3ff0000000000000ULL, b, status);
  });
}
TEST(FpFmaakF64W32Gfx1250, Arithmetic) {
  check_variant<32>([](auto flags, auto mask, auto d, auto a, auto, auto status) {
    return rj_arith_v_fmaak_f64_w32_gfx1250(flags, mask, 0, d, a, a, 0x3ca0000000000000ULL, status);
  });
}

// All CDNA product forms share round-then-flush at the normal/subnormal boundary.
TEST(FpF64Cdna, ProductRoundThenFlush) {
  check_min_normal<32>(rj_arith_v_mul_f64_w32_gfx942);
  check_min_normal<32>([](auto flags, auto mask, auto mode, auto d, auto a, auto b, auto status) {
    const uint32_t zeros[32]{};
    const uint32_t *c[] = {zeros, zeros};
    return rj_arith_v_fma_f64_w32_gfx942(flags, mask, mode, d, a, b, c, status);
  });
  check_min_normal<32>(rj_arith_v_mul_f64_w32_gfx1250);
  check_min_normal<32>([](auto flags, auto mask, auto mode, auto d, auto a, auto b, auto status) {
    const uint32_t zeros[32]{};
    const uint32_t *c[] = {zeros, zeros};
    return rj_arith_v_fma_f64_w32_gfx1250(flags, mask, mode, d, a, b, c, status);
  });
  check_min_normal<64>(rj_arith_v_mul_f64_w64_gfx942);
  check_min_normal<64>([](auto flags, auto mask, auto mode, auto d, auto a, auto b, auto status) {
    const uint32_t zeros[64]{};
    const uint32_t *c[] = {zeros, zeros};
    return rj_arith_v_fma_f64_w64_gfx942(flags, mask, mode, d, a, b, c, status);
  });
  check_min_normal<64>(rj_arith_v_mul_f64_w64_gfx1250);
  check_min_normal<64>([](auto flags, auto mask, auto mode, auto d, auto a, auto b, auto status) {
    const uint32_t zeros[64]{};
    const uint32_t *c[] = {zeros, zeros};
    return rj_arith_v_fma_f64_w64_gfx1250(flags, mask, mode, d, a, b, c, status);
  });
  check_min_normal<32>([](auto flags, auto mask, auto mode, auto d, auto a, auto, auto status) {
    const uint32_t zeros[32]{};
    const uint32_t *c[] = {zeros, zeros};
    return rj_arith_v_fmamk_f64_w32_gfx1250(flags, mask, mode, d, a, 0x3fefffffffffffffULL, c,
                                            status);
  });
  check_min_normal<32>([](auto flags, auto mask, auto mode, auto d, auto a, auto b, auto status) {
    return rj_arith_v_fmaak_f64_w32_gfx1250(flags, mask, mode, d, a, b, 0, status);
  });
}
