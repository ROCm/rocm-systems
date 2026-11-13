// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 fused multiply-accumulate.

#include "fp/fma_reference.h"
#include "fp/half_fma_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// FP exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
using rocjitsu::arith::test::StatusCase;

const StatusCase status_cases[] = {
    {0x3c00U, 0x3c00U, 0x0U, 0},
    {0x0U, 0x7c00U, 0x0U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7bffU, 0x4000U, 0x0U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3c01U, 0x3c01U, 0x0U, RJ_ARITH_EXCEPTION_INEXACT},
    {0x1U, 0x3800U, 0x0U,
     RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
};

} // namespace

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmacF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<32, /*Packed=*/true, /*Accumulate=*/true>(
      rj_arith_v_pk_fmac_f16_w32_gfx12,
      /*gfx12=*/true, status_cases);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmacF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<32, /*Packed=*/true, /*Accumulate=*/true>(
      rj_arith_v_pk_fmac_f16_w32,
      /*gfx12=*/false, status_cases);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmacF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/true, /*Accumulate=*/true>(
      rj_arith_v_pk_fmac_f16_w64_gfx12,
      /*gfx12=*/true, status_cases);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmacF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/true, /*Accumulate=*/true>(
      rj_arith_v_pk_fmac_f16_w64,
      /*gfx12=*/false, status_cases);
}

namespace {

// CDNA packed accumulation reuses the inherited half model, with MODE gating OMOD.
template <int Lanes, typename Call> void check_cdna_fmac(Call entry) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  auto call = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *,
                  uint32_t *status) { return entry(flags, exec_mask, mode, d, a, b, status); };
  check_fma_simd<16, Lanes, true>(
      call, {0, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF,
             RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP});
  const auto cpus = supported_cpu_levels();
  for (auto cpu : cpus) {
    for (auto semantics : semantics_modes()) {
      for (uint64_t fp : {uint64_t(0), uint64_t(RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS),
                          uint64_t(RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE)}) {
        Registers<4, Lanes> r;
        r.v[0][0] = r.v[1][0] = 0x3c003c00U; // (1,1) * (1,1) + (0,0).
        r.v[2][0] = 0;
        auto before = r;
        uint32_t status = 0x80000000U;
        int result = entry(cpu | semantics | fp, Mask(1), RJ_ARITH_ALU_OMOD_2, r.v + 2, r.v,
                           r.v + 1, &status);
        if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
          EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
          r.expect_preserved(before, 0, 0);
        } else {
          ASSERT_EQ(result, RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[2][0],
                    fp == RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS ? 0x40004000U : 0x3c003c00U);
          r.expect_preserved(before, 2, 1);
        }
        EXPECT_EQ(status, 0x80000000U);
      }
    }
  }
}

} // namespace

TEST(FpPkFmacF16W32Gfx942, Arithmetic) { check_cdna_fmac<32>(rj_arith_v_pk_fmac_f16_w32_gfx942); }

TEST(FpPkFmacF16W64Gfx942, Arithmetic) { check_cdna_fmac<64>(rj_arith_v_pk_fmac_f16_w64_gfx942); }
