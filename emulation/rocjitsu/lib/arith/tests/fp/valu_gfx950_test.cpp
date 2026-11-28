// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MI350X (gfx950) VALU FP arithmetic, packed FP, exponent/preop, and unary math contracts.

#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

constexpr uint64_t kDppQuadPerm = RJ_ARITH_DPP16 | (0xb1ULL << RJ_ARITH_DPP_CTRL_SHIFT) |
                                  RJ_ARITH_DPP_ROW_MASK | RJ_ARITH_DPP_BANK_MASK;

} // namespace

TEST(ValuFpGfx950, HalfBinaryAndPackedContracts) {
  decltype(&rj_arith_v_add_f16_w32_gfx950) bin32[] = {
      rj_arith_v_add_f16_w32_gfx950, rj_arith_v_sub_f16_w32_gfx950,
      rj_arith_v_subrev_f16_w32_gfx950, rj_arith_v_mul_f16_w32_gfx950};
  decltype(&rj_arith_v_add_f16_w64_gfx950) bin64[] = {
      rj_arith_v_add_f16_w64_gfx950, rj_arith_v_sub_f16_w64_gfx950,
      rj_arith_v_subrev_f16_w64_gfx950, rj_arith_v_mul_f16_w64_gfx950};
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (bool wave64 : {false, true}) {
      for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
        const bool mode3 = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        for (int op = 0; op < 4; ++op) {
          uint32_t a[64]{}, b[64]{}, d[64]{};
          a[0] = op == 3 ? 0x0000U : 0x7c00U;
          b[0] = op == 0 ? 0xfc00U : 0x7c00U;
          a[1] = 0x7c01U;
          b[1] = 0x3c00U;
          d[0] = d[1] = 0xdeadbeefU;
          const uint32_t *pa = a, *pb = b;
          uint32_t *pd = d;
          uint32_t trap = 0x40U;
          const int rc = wave64 ? bin64[op](cpu | semantics, 3ULL, 0, &pd, &pa, &pb, &trap)
                                : bin32[op](cpu | semantics, 3U, 0, &pd, &pa, &pb, &trap);
          EXPECT_EQ(rc, mode3 ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS);
          EXPECT_EQ(trap, 0x40U);
          if (!mode3) {
            if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
              // Fresh invalid operations emit negative qNaN (0xfe00); sNaN preserves when IEEE=0
              // (with SUBREV negating source A's sign bit).
              EXPECT_EQ(d[0] & 0xffffU, 0xfe00U);
              EXPECT_EQ(d[1] & 0xffffU, op == 2 ? 0xfc01U : 0x7c01U);
            }
            EXPECT_EQ(d[0] >> 16, 0xdeadU);
          } else {
            EXPECT_EQ(d[0], 0xdeadbeefU);
          }
        }

        // Plain SDWA on v_mul_f16_gfx950 is mode-2-qualified; SDWA with OMOD/CLAMP and DPP are not.
        uint32_t a[64]{0x3c00U}, b[64]{0x4000U}, d[64]{0xdeadbeefU};
        const uint32_t *pa = a, *pb = b;
        uint32_t *pd = d;
        const uint64_t m2 = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
        EXPECT_EQ(
            wave64
                ? rj_arith_v_mul_f16_w64_gfx950(m2, 1ULL, RJ_ARITH_ALU_SDWA, &pd, &pa, &pb, nullptr)
                : rj_arith_v_mul_f16_w32_gfx950(m2, 1U, RJ_ARITH_ALU_SDWA, &pd, &pa, &pb, nullptr),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(d[0] & 0xffffU, 0x4000U);
        EXPECT_EQ(
            wave64 ? rj_arith_v_mul_f16_w64_gfx950(
                         m2, 1ULL, RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_2, &pd, &pa, &pb, nullptr)
                   : rj_arith_v_mul_f16_w32_gfx950(m2, 1U, RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_2,
                                                   &pd, &pa, &pb, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(
            wave64
                ? rj_arith_v_add_f16_w64_gfx950(m2, 1ULL, RJ_ARITH_ALU_SDWA, &pd, &pa, &pb, nullptr)
                : rj_arith_v_add_f16_w32_gfx950(m2, 1U, RJ_ARITH_ALU_SDWA, &pd, &pa, &pb, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(
            wave64 ? rj_arith_v_add_f16_w64_gfx950(m2, 1ULL, kDppQuadPerm, &pd, &pa, &pb, nullptr)
                   : rj_arith_v_add_f16_w32_gfx950(m2, 1U, kDppQuadPerm, &pd, &pa, &pb, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);

        // Packed FP16 add/mul/fma/fmac: -qNaN (0xfe00) on fresh invalid, sNaN preservation at
        // IEEE=0, and unbounded-exponent tininess flushing on MUL when FLUSH_OUTPUT_DENORMALS is
        // set.
        uint32_t pk_a[64]{0x00003bffU}, pk_b[64]{0x7c000400U}, pk_c[64]{0x00007c01U};
        uint32_t pk_d[64]{0xdeadbeefU};
        const uint32_t *ppa = pk_a, *ppb = pk_b, *ppc = pk_c;
        uint32_t *ppd = pk_d;
        const uint64_t flush = cpu | semantics | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                               RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
        const int pk_mul_rc =
            wave64 ? rj_arith_v_pk_mul_f16_w64_gfx950(flush, 1ULL, 0, &ppd, &ppa, &ppb, nullptr)
                   : rj_arith_v_pk_mul_f16_w32_gfx950(flush, 1U, 0, &ppd, &ppa, &ppb, nullptr);
        EXPECT_EQ(pk_mul_rc, mode3 ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS);
        if (!mode3) {
          EXPECT_EQ(pk_d[0] & 0xffffU, 0x0000U);
          if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
            EXPECT_EQ(pk_d[0] >> 16, 0xfe00U);
          }
        }
        const int pk_fma_rc = wave64
                                  ? rj_arith_v_pk_fma_f16_w64_gfx950(cpu | semantics, 1ULL, 0, &ppd,
                                                                     &ppa, &ppb, &ppc, nullptr)
                                  : rj_arith_v_pk_fma_f16_w32_gfx950(cpu | semantics, 1U, 0, &ppd,
                                                                     &ppa, &ppb, &ppc, nullptr);
        EXPECT_EQ(pk_fma_rc, mode3 ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS);
        if (!mode3 && semantics != RJ_ARITH_SEMANTICS_LOOSE) {
          EXPECT_EQ(pk_d[0], 0xfe007c01U);
        }
        pk_d[0] = 0x00007c01U;
        const int pk_fmac_rc = wave64 ? rj_arith_v_pk_fmac_f16_w64_gfx950(cpu | semantics, 1ULL, 0,
                                                                          &ppd, &ppa, &ppb, nullptr)
                                      : rj_arith_v_pk_fmac_f16_w32_gfx950(
                                            cpu | semantics, 1U, 0, &ppd, &ppa, &ppb, nullptr);
        EXPECT_EQ(pk_fmac_rc, mode3 ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS : RJ_ARITH_SUCCESS);
        if (!mode3 && semantics != RJ_ARITH_SEMANTICS_LOOSE) {
          EXPECT_EQ(pk_d[0], 0xfe007c01U);
        }
      }
    }
  }
}

TEST(ValuFpGfx950, MadF16AndPackedF32AndMulLegacyContracts) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (bool wave64 : {false, true}) {
      for (auto semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
        // Two-step separately-rounded V_MAD_F16 vs fused FMA witness:
        // 0x3be0 * 0x3be0 = 0x3bc0 (1 - 2^-5) rounded to FP16 first, minus 0.5 (0xb800) = 0x3780
        // (whereas fused FMA retains +2^-12 and returns 0x3781).
        uint32_t a[64]{0x3be0U}, b[64]{0x3be0U}, c[64]{0xb800U}, d[64]{0xdead0000U};
        const uint32_t *pa = a, *pb = b, *pc = c;
        uint32_t *pd = d;
        const uint64_t flags = cpu | semantics | RJ_ARITH_FP_IEEE_MODE;
        EXPECT_EQ(wave64
                      ? rj_arith_v_mad_f16_w64_gfx950(flags, 1ULL, 0, &pd, &pa, &pb, &pc, nullptr)
                      : rj_arith_v_mad_f16_w32_gfx950(flags, 1U, 0, &pd, &pa, &pb, &pc, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d[0], 0xdead3780U);

        // CDNA V_MAD_LEGACY_F16 shares V_MAD_F16's IEEE invalid rule: 0 * Inf + 1.0 = -qNaN
        // (0xfe00).
        a[0] = 0x8000U;
        b[0] = 0x7c00U;
        c[0] = 0x3c00U;
        EXPECT_EQ(
            wave64
                ? rj_arith_v_mad_legacy_f16_w64_gfx950(flags, 1ULL, 0, &pd, &pa, &pb, &pc, nullptr)
                : rj_arith_v_mad_legacy_f16_w32_gfx950(flags, 1U, 0, &pd, &pa, &pb, &pc, nullptr),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(d[0] & 0xffffU, 0xfe00U);

        // V_MUL_LEGACY_F32: 0 * Inf = +0, sNaN preserved when IEEE=0, DPP rejected in mode 2.
        uint32_t ml_a[64]{0x80000000U, 0x7f800123U}, ml_b[64]{0x7f800000U, 0x3f800000U};
        uint32_t ml_d[64]{0xdeadbeefU, 0xdeadbeefU};
        const uint32_t *pma = ml_a, *pmb = ml_b;
        uint32_t *pmd = ml_d;
        EXPECT_EQ(wave64 ? rj_arith_v_mul_legacy_f32_w64_gfx950(cpu | semantics, 3ULL, 0, &pmd,
                                                                &pma, &pmb, nullptr)
                         : rj_arith_v_mul_legacy_f32_w32_gfx950(cpu | semantics, 3U, 0, &pmd, &pma,
                                                                &pmb, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(ml_d[0], 0x00000000U);
        EXPECT_EQ(ml_d[1], 0x7f800123U);
        if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
          EXPECT_EQ(wave64 ? rj_arith_v_mul_legacy_f32_w64_gfx950(
                                 cpu | semantics, 1ULL, kDppQuadPerm, &pmd, &pma, &pmb, nullptr)
                           : rj_arith_v_mul_legacy_f32_w32_gfx950(cpu | semantics, 1U, kDppQuadPerm,
                                                                  &pmd, &pma, &pmb, nullptr),
                    RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
        }

        // V_PK_{FMA,ADD,MUL}_F32: preserves sNaN at IEEE=0 and emits -qNaN (0xffc00000) on invalid.
        uint32_t pf_a0[64]{0x00000000U}, pf_a1[64]{0x7f800123U};
        uint32_t pf_b0[64]{0x7f800000U}, pf_b1[64]{0x3f800000U};
        uint32_t pf_c0[64]{0x3f800000U}, pf_c1[64]{0x00000000U};
        uint32_t pf_d0[64]{0xdeadbeefU}, pf_d1[64]{0xdeadbeefU};
        const uint32_t *pa64[2] = {pf_a0, pf_a1}, *pb64[2] = {pf_b0, pf_b1},
                       *pc64[2] = {pf_c0, pf_c1};
        uint32_t *pd64[2] = {pf_d0, pf_d1};
        EXPECT_EQ(wave64 ? rj_arith_v_pk_fma_f32_w64_gfx950(cpu | semantics, 1ULL, 0, pd64, pa64,
                                                            pb64, pc64, nullptr)
                         : rj_arith_v_pk_fma_f32_w32_gfx950(cpu | semantics, 1U, 0, pd64, pa64,
                                                            pb64, pc64, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(pf_d0[0], 0xffc00000U);
        EXPECT_EQ(pf_d1[0], 0x7f800123U);
        EXPECT_EQ(wave64 ? rj_arith_v_pk_mul_f32_w64_gfx950(cpu | semantics, 1ULL, 0, pd64, pa64,
                                                            pb64, nullptr)
                         : rj_arith_v_pk_mul_f32_w32_gfx950(cpu | semantics, 1U, 0, pd64, pa64,
                                                            pb64, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(pf_d0[0], 0xffc00000U);
        EXPECT_EQ(pf_d1[0], 0x7f800123U);
      }
    }
  }
}

TEST(ValuFpGfx950, LdexpTrigPreopAndUnaryMathContracts) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (bool wave64 : {false, true}) {
      const uint64_t m2 = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
      const uint64_t m3 = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
      uint32_t trap = 0;

      // LDEXP F16/F32/F64 preserve sNaN when MODE.IEEE=0 and quiet when MODE.IEEE=1.
      uint32_t a16[64]{0x7c01U}, b16[64]{2U}, d16[64]{0xdeadbeefU};
      const uint32_t *pa16 = a16, *pb16 = b16;
      uint32_t *pd16 = d16;
      EXPECT_EQ(wave64 ? rj_arith_v_ldexp_f16_w64_gfx950(m2, 1ULL, 0, &pd16, &pa16, &pb16, &trap)
                       : rj_arith_v_ldexp_f16_w32_gfx950(m2, 1U, 0, &pd16, &pa16, &pb16, &trap),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d16[0] & 0xffffU, 0x7c01U);

      uint32_t a32[64]{0x7f800123U}, b32[64]{2U}, d32[64]{0xdeadbeefU};
      const uint32_t *pa32 = a32, *pb32 = b32;
      uint32_t *pd32 = d32;
      EXPECT_EQ(wave64 ? rj_arith_v_ldexp_f32_w64_gfx950(m2, 1ULL, 0, &pd32, &pa32, &pb32, &trap)
                       : rj_arith_v_ldexp_f32_w32_gfx950(m2, 1U, 0, &pd32, &pa32, &pb32, &trap),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d32[0], 0x7f800123U);
      EXPECT_EQ(wave64 ? rj_arith_v_ldexp_f32_w64_gfx950(m3, 1ULL, 0, &pd32, &pa32, &pb32, &trap)
                       : rj_arith_v_ldexp_f32_w32_gfx950(m3, 1U, 0, &pd32, &pa32, &pb32, &trap),
                RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);

      uint32_t a64_lo[64]{0x12345678U}, a64_hi[64]{0x7ff00000U}, seg[64]{0U};
      uint32_t d64_lo[64]{0U}, d64_hi[64]{0U};
      const uint32_t *pa64[2] = {a64_lo, a64_hi}, *pseg = seg;
      uint32_t *pd64[2] = {d64_lo, d64_hi};
      EXPECT_EQ(wave64 ? rj_arith_v_ldexp_f64_w64_gfx950(m2, 1ULL, 0, pd64, pa64, &pseg, &trap)
                       : rj_arith_v_ldexp_f64_w32_gfx950(m2, 1U, 0, pd64, pa64, &pseg, &trap),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d64_lo[0], 0x12345678U);
      EXPECT_EQ(d64_hi[0], 0x7ff00000U);

      // V_TRIG_PREOP_F64: mode-2-qualified, extracts the first 53-bit slice of 2/pi at +0.0, seg=0.
      a64_lo[0] = 0U;
      a64_hi[0] = 0U;
      EXPECT_EQ(wave64 ? rj_arith_v_trig_preop_f64_w64_gfx950(m2, 1ULL, 0, pd64, pa64, &pseg, &trap)
                       : rj_arith_v_trig_preop_f64_w32_gfx950(m2, 1U, 0, pd64, pa64, &pseg, &trap),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d64_lo[0], 0x6dc9c882U);
      EXPECT_EQ(d64_hi[0], 0x3fe45f30U);
      a64_lo[0] = 0x12345678U;
      a64_hi[0] = 0x7ff00000U;

      // Integral unary, mantissa, and fract F16/F32 preserve sNaN when MODE.IEEE=0 in mode 2.
      decltype(&rj_arith_v_trunc_f16_w32_gfx950) u16_32[] = {
          rj_arith_v_trunc_f16_w32_gfx950,      rj_arith_v_ceil_f16_w32_gfx950,
          rj_arith_v_floor_f16_w32_gfx950,      rj_arith_v_rndne_f16_w32_gfx950,
          rj_arith_v_frexp_mant_f16_w32_gfx950, rj_arith_v_fract_f16_w32_gfx950};
      decltype(&rj_arith_v_trunc_f16_w64_gfx950) u16_64[] = {
          rj_arith_v_trunc_f16_w64_gfx950,      rj_arith_v_ceil_f16_w64_gfx950,
          rj_arith_v_floor_f16_w64_gfx950,      rj_arith_v_rndne_f16_w64_gfx950,
          rj_arith_v_frexp_mant_f16_w64_gfx950, rj_arith_v_fract_f16_w64_gfx950};
      decltype(&rj_arith_v_trunc_f32_w32_gfx950) u32_32[] = {
          rj_arith_v_trunc_f32_w32_gfx950,      rj_arith_v_ceil_f32_w32_gfx950,
          rj_arith_v_floor_f32_w32_gfx950,      rj_arith_v_rndne_f32_w32_gfx950,
          rj_arith_v_frexp_mant_f32_w32_gfx950, rj_arith_v_fract_f32_w32_gfx950};
      decltype(&rj_arith_v_trunc_f32_w64_gfx950) u32_64[] = {
          rj_arith_v_trunc_f32_w64_gfx950,      rj_arith_v_ceil_f32_w64_gfx950,
          rj_arith_v_floor_f32_w64_gfx950,      rj_arith_v_rndne_f32_w64_gfx950,
          rj_arith_v_frexp_mant_f32_w64_gfx950, rj_arith_v_fract_f32_w64_gfx950};
      for (int i = 0; i < 6; ++i) {
        d16[0] = 0xdead0000U;
        d32[0] = 0xdeadbeefU;
        EXPECT_EQ(wave64 ? u16_64[i](m2, 1ULL, 0, &pd16, &pa16, &trap)
                         : u16_32[i](m2, 1U, 0, &pd16, &pa16, &trap),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d16[0], 0xdead7c01U);
        EXPECT_EQ(wave64 ? u32_64[i](m2, 1ULL, 0, &pd32, &pa32, &trap)
                         : u32_32[i](m2, 1U, 0, &pd32, &pa32, &trap),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d32[0], 0x7f800123U);
      }

      // Cross-CPU regression: V_FREXP_MANT_F32 with MODE.IEEE=0, output-denormal flushing, and
      // OMOD x2 preserves signaling NaN (0x7f800123) across baseline, AVX2, AVX-512, and Zen 4
      // in both Compatibility and mode 2 across Wave32 and Wave64.
      for (uint64_t sem : {RJ_ARITH_SEMANTICS_COMPATIBILITY,
                           RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
        const uint64_t f_omod = cpu | sem | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
        d32[0] = 0xdeadbeefU;
        EXPECT_EQ(wave64 ? rj_arith_v_frexp_mant_f32_w64_gfx950(f_omod, 1ULL, RJ_ARITH_ALU_OMOD_2,
                                                                &pd32, &pa32, &trap)
                         : rj_arith_v_frexp_mant_f32_w32_gfx950(f_omod, 1U, RJ_ARITH_ALU_OMOD_2,
                                                                &pd32, &pa32, &trap),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(d32[0], 0x7f800123U);
      }
    }
  }
}
