// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar-result special functions without EXEC masking.

#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_scalar.h"
#include <bit>

namespace {
using namespace rocjitsu::arith::test;

// Check scalar-result special functions and semantics support for the supplied entry point.
// Every RDNA4 entry accepts strict exact; the gfx1250 entries refuse it and run the same model
// in Compatibility. Exact FP16 EXP/LOG are checked bit-exactly on exactly representable
// identities (as the VALU tests do) and loosely elsewhere; the captured gfx1201 rows below pin
// their remaining rules.
template <bool Half, bool Gfx1250 = false, typename Call> void check_pseudo(Call call, int op) {
  const bool exp_log = op == 7 || op == 8;
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      std::minstd_rand random;
      for (int sample = 0; sample < 64; ++sample) {
        for (int scenario = 0; scenario < 8; ++scenario) {
          uint32_t a =
              Half ? uint32_t(random_f16_bits(random)) | (uint32_t(random_f16_bits(random)) << 16)
                   : random_f32_bits(random);
          bool precise = semantics != RJ_ARITH_SEMANTICS_LOOSE &&
                         (!Gfx1250 || semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY);
          if (!Half && precise) {
            a = unary_inputs[sample % 32];
          }
          if (sample == 0) {
            a = Half ? 0x7e01 : 0x7fc00001U;
          }
          if (sample == 1) {
            a = Half ? 0x7c00 : 0x7f800000U;
          }
          if (!Half && precise) {
            a = unary_inputs[sample % 32];
          }
          if (Half && exp_log && precise && sample >= 2) {
            // Exactly representable exp2/log2 identities: 0, 1, 2 and 4.
            const uint16_t identities[] = {0, 0x3c00, 0x4000, 0x4400};
            a = uint32_t(identities[sample % 4]) * 0x10001U;
          }
          uint64_t mode = uint64_t(scenario & 3) << 6;
          if (scenario & 4) {
            mode |= RJ_ARITH_ALU_CLAMP;
          }
          uint32_t d = 123, status = 0x80000000U;
          int wanted = Gfx1250 && (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                                   semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS)
                           ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                       : semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                           ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                           : RJ_ARITH_SUCCESS;
          ASSERT_EQ(call(cpu | semantics, mode, &d, a, &status), wanted);
          EXPECT_EQ(status, 0x80000000U);
          // Rejected modes must leave both the numeric and FP-status outputs untouched.
          if (wanted) {
            EXPECT_EQ(d, 123U);
            continue;
          }
          float expected;
          if constexpr (Half) {
            EXPECT_EQ(d >> 16, 0U);
            expected = f16_value(half_unary_reference(op, a, uint32_t(mode), false));
            expect_f32(std::bit_cast<uint32_t>(f16_value(uint16_t(d))), expected,
                       precise ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               : RJ_ARITH_SEMANTICS_LOOSE,
                       /*atol=*/0x1p-24f, /*rtol=*/1.1e-3f);
          } else {
            expected =
                precise ? omod_f32_reference(std::bit_cast<float>(unary_goldens[op][sample % 32]),
                                             uint32_t(mode))
                        : unary_reference(op, std::bit_cast<float>(a), uint32_t(mode));
            if (mode & RJ_ARITH_ALU_CLAMP) {
              expected = !(expected > 0) ? 0 : std::min(expected, 1.0f);
            }
            expect_f32(d, expected,
                       precise ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               : RJ_ARITH_SEMANTICS_LOOSE,
                       /*atol=*/0.0f, /*rtol=*/2.5e-7f);
          }
          uint32_t alias = a;
          ASSERT_EQ(call(cpu | semantics, mode, &alias, alias, &status), RJ_ARITH_SUCCESS);
          EXPECT_EQ(alias, d);
        }
      }
    }
  }
  uint32_t d = 123, status = 456;
  expect_invalid_flags(
      [&](auto f, auto m) {
        int error = call(f, m, &d, 0, &status);
        EXPECT_EQ(d, 123U);
        EXPECT_EQ(status, 456U);
        return error;
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP);
}
} // namespace

// Check scalar-result special functions and semantics support.
TEST(v_s_exp_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_exp_f32, 7); }

// Check scalar-result special functions and semantics support.
TEST(v_s_log_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_log_f32, 8); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rcp_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_rcp_f32, 5); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rsq_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_rsq_f32, 6); }

// Check scalar-result special functions and semantics support.
TEST(v_s_sqrt_f32, Arithmetic) { check_pseudo<false>(rj_arith_v_s_sqrt_f32, 4); }

// Check scalar-result special functions and semantics support.
TEST(v_s_exp_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_exp_f16, 7); }

// Check scalar-result special functions and semantics support.
TEST(v_s_log_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_log_f16, 8); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rcp_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_rcp_f16, 5); }

// Check scalar-result special functions and semantics support.
TEST(v_s_rsq_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_rsq_f16, 6); }

// Check scalar-result special functions and semantics support.
TEST(v_s_sqrt_f16, Arithmetic) { check_pseudo<true>(rj_arith_v_s_sqrt_f16, 4); }

namespace {

// One captured gfx1201 scalar lane: the MODE word (FP16 denormals in bits 7:6, FP16_OVFL in
// bit 23), the raw SGPR input (the high half is ignored) and the hardware result.
struct CapturedScalar {
  const char *name;
  uint64_t modifiers;
  uint32_t mode, a, expected;
};

uint64_t scalar_guest_flags(uint32_t mode) {
  uint64_t flags = 0;
  if (!(mode & 0x40)) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  }
  if (!(mode & 0x80)) {
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
  if (mode & 0x800000) {
    flags |= RJ_ARITH_FP16_OVFL;
  }
  return flags;
}

// Compatibility and strict exact reproduce each lane at every CPU level.
template <typename Call, size_t N>
void check_captured_scalar(Call call, const CapturedScalar (&lanes)[N]) {
  for (const CapturedScalar &c : lanes) {
    SCOPED_TRACE(c.name);
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : hardware_semantics()) {
        uint32_t d = 0xdeadbeefU, status = 0x80000000U;
        ASSERT_EQ(call(cpu | semantics | scalar_guest_flags(c.mode), c.modifiers, &d, c.a, &status),
                  RJ_ARITH_SUCCESS)
            << cpu << '/' << semantics;
        EXPECT_EQ(d, c.expected) << cpu << '/' << semantics;
        EXPECT_EQ(status, 0x80000000U);
      }
    }
  }
}

// gfx1201 (RX 9070 XT, rj-valu-audit-local hw v_s_exp_f16__*, 2026-11-05; the 0xcd11 rows from
// an R9700 run on 2026-11-19), dispatch MODE.
// FP16_OVFL saturates an overflowing result to 65504; +/-infinity and NaN inputs pass through;
// a tiny result is flushed only with output denormals flushed and otherwise kept as
// a denormal; div:2 flushes a tiny result to +0; CLAMP maps results above 1 to 1.0. A kernel that
// returns infinity under FP16_OVFL or skips the denormal result fails these rows.
//
// The flushing rows use 0xcd11, whose unflushed result is the nonzero subnormal 0x000d: the
// 0xd0e3 and 0xdbac rows below round to +0 before either stage, so they hold whether or not
// output flushing and the pre-scale flush run. Keeping one input across the four cases isolates
// each stage: output flushing alone gives zero, flushing the input instead does not, and with
// denormals kept div:2 still gives zero, where ordinary FP16 halving would give 0x0006.
const CapturedScalar exp_f16_lanes[] = {
    {"OvflSaturatesMaxInput", 0, 0x8000f0, 0x55e27bff, 0x7bff},
    {"OvflSaturates1024", 0, 0x8000f0, 0x73846400, 0x7bff},
    {"PositiveInfinity", 0, 0xf0, 0x213c7c00, 0x7c00},
    {"NegativeInfinityGivesZero", 0, 0xf0, 0x65befc00, 0x0},
    {"QuietNan", 0, 0xf0, 0xda8c7e00, 0x7e00},
    {"FarBelowSubnormalFlushed", 0, 0x0, 0x099cd0e3, 0x0},
    {"TinyResultKept", 0, 0xf0, 0xbd36cd11, 0xd},
    {"TinyResultOutputFlushed", 0, 0x70, 0x0000cd11, 0x0},
    {"TinyResultKeptWithInputFlushed", 0, 0xb0, 0x0000cd11, 0xd},
    {"TinyResultBothFlushed", 0, 0x00, 0x0000cd11, 0x0},
    {"Div2TinyKeptFlushes", RJ_ARITH_ALU_OMOD_HALF, 0xf0, 0x0000cd11, 0x0},
    {"Mul2TinyKeptFlushes", RJ_ARITH_ALU_OMOD_2, 0xf0, 0x0000cd11, 0x0},
    {"TinyResultKeptLarger", 0, 0xf0, 0xba0dcbad, 0x191},
    {"Div2FarBelowSubnormalFlushes", RJ_ARITH_ALU_OMOD_HALF, 0xf0, 0x95fadbac, 0x0},
    {"Div2Halves", RJ_ARITH_ALU_OMOD_HALF, 0xf0, 0x6813ca63, 0x4ae},
    {"ClampAboveOne", RJ_ARITH_ALU_CLAMP, 0xf0, 0xfea73c00, 0x3c00},
    {"ClampNearOne", RJ_ARITH_ALU_CLAMP, 0xf0, 0x89c40400, 0x3c00},
};

// gfx1201 (RX 9070 XT, rj-valu-audit-local hw v_s_log_f16__*, 2026-11-05), dispatch MODE.
// Negative inputs give the canonical NaN; a denormal input flushed to zero gives -infinity,
// kept it gives its logarithm (2^-24 -> -24) or NaN when negative; mul:2 doubles log2(2^-14).
// A kernel that ignores the input denormal mode fails the SmallestDenormal rows.
const CapturedScalar log_f16_lanes[] = {
    {"NegativeOneGivesNan", 0, 0xf0, 0x686ebc00, 0xfe00},
    {"NegativeMinNormalGivesNan", 0, 0xf0, 0xde668400, 0xfe00},
    {"SmallestDenormalFlushedGivesNegInfinity", 0, 0x0, 0xbaf40001, 0xfc00},
    {"LargestDenormalFlushedGivesNegInfinity", 0, 0x0, 0xaf2083ff, 0xfc00},
    {"SmallestDenormalKept", 0, 0xf0, 0xbaf40001, 0xce00},
    {"NegativeDenormalKeptGivesNan", 0, 0xf0, 0xaf2083ff, 0xfe00},
    {"Mul2MinNormal", RJ_ARITH_ALU_OMOD_2, 0xf0, 0xe6ea0400, 0xcf00},
    {"Mul2One", RJ_ARITH_ALU_OMOD_2, 0xf0, 0xddce3c00, 0x0},
};

} // namespace

// Compatibility and strict exact reproduce captured gfx1201 V_S_EXP_F16 and V_S_LOG_F16 lanes.
TEST(v_s_exp_log_f16, CapturedGfx12) {
  check_captured_scalar(rj_arith_v_s_exp_f16, exp_f16_lanes);
  check_captured_scalar(rj_arith_v_s_log_f16, log_f16_lanes);
}

// gfx1250 preserves inherited models but does not advertise new exact support.
TEST(PseudoGfx1250, Arithmetic) {
  check_pseudo<true, true>(rj_arith_v_s_exp_f16_gfx1250, 7);
  check_pseudo<false, true>(rj_arith_v_s_exp_f32_gfx1250, 7);
  check_pseudo<true, true>(rj_arith_v_s_log_f16_gfx1250, 8);
  check_pseudo<false, true>(rj_arith_v_s_log_f32_gfx1250, 8);
  check_pseudo<true, true>(rj_arith_v_s_rcp_f16_gfx1250, 5);
  check_pseudo<false, true>(rj_arith_v_s_rcp_f32_gfx1250, 5);
  check_pseudo<true, true>(rj_arith_v_s_rsq_f16_gfx1250, 6);
  check_pseudo<false, true>(rj_arith_v_s_rsq_f32_gfx1250, 6);
  check_pseudo<true, true>(rj_arith_v_s_sqrt_f16_gfx1250, 4);
  check_pseudo<false, true>(rj_arith_v_s_sqrt_f32_gfx1250, 4);
}
