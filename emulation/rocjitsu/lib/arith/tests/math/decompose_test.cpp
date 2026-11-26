// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fractional-part and mantissa decomposition.

#include "common/cpu.h"
#include "common/wave.h"
#include "fp/captured_gfx11_test.h"
#include "fp/elementwise.h"
#include "fp/reference.h"
#include "math/older_half_unary_test.h"
#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <gtest/gtest.h>
#include <stdint.h>

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_fract_f32_w32, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_frexp_mant_f32_w32, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_fract_f32_w32_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_frexp_mant_f32_w32_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_fract_f32_w64, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_frexp_mant_f32_w64, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_fract_f32_w64_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_frexp_mant_f32_w64_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_fract_f16_w32, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_frexp_mant_f16_w32, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_fract_f16_w32_gfx12, 9);
  rocjitsu::arith::test::check_half_fract_rounding<32>(rj_arith_v_fract_f16_w32_gfx12);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_frexp_mant_f16_w32_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_fract_f16_w64, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_frexp_mant_f16_w64, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_fract_f16_w64_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_frexp_mant_f16_w64_gfx12, 10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF16W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<32, true>(rj_arith_v_frexp_mant_f16_w32_gfx942,
                                                             10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF16W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<64, true>(rj_arith_v_frexp_mant_f16_w64_gfx942,
                                                             10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF32W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<32, false>(rj_arith_v_frexp_mant_f32_w32_gfx942,
                                                              10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF32W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<64, false>(rj_arith_v_frexp_mant_f32_w64_gfx942,
                                                              10);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF16W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_fract_f16_w32_gfx942, 9, false,
                                                      false);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF16W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_fract_f16_w64_gfx942, 9, false,
                                                      false);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF32W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_fract_f32_w32_gfx942, 9, false,
                                                       false);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF32W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_fract_f32_w64_gfx942, 9, false,
                                                       false);
}

namespace {

using UnaryFn = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                        uint32_t *);

// gfx1201 captures with IEEE set: guest rounding applies to the subtraction, a tiny negative
// input caps below one unless input flushing zeroes it, and NaNs are quieted with their payload.
// FP16 forms write the low half and keep the high half.
struct FractCase {
  UnaryFn call;
  uint64_t flags;
  uint32_t input, expected;
};

const FractCase fract_cases[] = {
    {rj_arith_v_fract_f32_w32_gfx12, 0, 0xb5a168adU, 0x3f7fffecU},
    {rj_arith_v_fract_f32_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO, 0xb5a168adU, 0x3f7fffebU},
    {rj_arith_v_fract_f32_w32_gfx12, 0, 0x80000001U, 0x3f7fffffU},
    {rj_arith_v_fract_f32_w32_gfx12, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0x80000001U, 0},
    {rj_arith_v_fract_f32_w32_gfx12, 0, 0x7f800001U, 0x7fc00001U},
    {rj_arith_v_fract_f16_w32_gfx12, 0, 0xa6e1U, 0x3bc9U},
    {rj_arith_v_fract_f16_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO, 0xa6e1U, 0x3bc8U},
    {rj_arith_v_fract_f16_w32_gfx12, 0, 0x8001U, 0x3bffU},
    {rj_arith_v_fract_f16_w32_gfx12, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0x8001U, 0},
    {rj_arith_v_fract_f16_w32_gfx12, 0, 0x7c01U, 0x7e01U},
};

} // namespace

// Pin the captured FRACT results in Compatibility and strict exact mode, so that a faster loose
// path cannot change what default-mode RDNA4 execution returns.
TEST(MathFractGfx12, ExactMatchesHardware) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : hardware_semantics()) {
      for (const FractCase &test : fract_cases) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << semantics << '/' << (&test - fract_cases));
        const bool half = test.call == rj_arith_v_fract_f16_w32_gfx12;
        Registers<2, 32> r;
        r.v[0][0] = half ? 0xa5a50000U | test.input : test.input;
        r.v[1][0] = 0xa5a5a5a5U;
        ASSERT_EQ(test.call(cpu | semantics | RJ_ARITH_FP_IEEE_MODE | test.flags, 1, 0, r.v + 1,
                            r.v, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[1][0], half ? 0xa5a50000U | test.expected : test.expected);
      }
    }
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 (W7900, mkm2) FRACT lanes: hw1100-noieee (2026-09-25, MODE.IEEE clear) for 0x1f0 and
// 0x100, the Track C hw1100-wide corpus (2026-11-18) for the directed-rounding words 0x1f5 and
// 0x1fa, and hw2-gfx1100 (2026-10-06, MODE.IEEE set) for 0x300/0x3f0. RDNA3 applies OMOD only with
// output denormals flushed and IEEE clear: mul:2 doubles 0.5 at 0x100 and leaves it alone at
// 0x1f0 and 0x300, so a kernel that always scales fails two rows. A tiny negative input caps
// below one under every rounding; signaling NaNs pass through with IEEE clear and quiet with it.
const CapturedGfx11 fract_f32_gfx11_lanes[] = {
    {"Mul2FlushScales", RJ_ARITH_ALU_OMOD_2, 0x100, 0x3f000000, 0, 0, 0x3f800000},
    {"Mul2MinNormalFlushScales", RJ_ARITH_ALU_OMOD_2, 0x100, 0x00800000, 0, 0, 0x01000000},
    {"Mul2DenormalsKeptIgnored", RJ_ARITH_ALU_OMOD_2, 0x1f0, 0x3f000000, 0, 0, 0x3f000000},
    {"Mul2IeeeIgnored", RJ_ARITH_ALU_OMOD_2, 0x300, 0x00800000, 0, 0, 0x00800000},
    {"TinyNegativeNearest", 0, 0x1f0, 0x80800000, 0, 0, 0x3f7fffff},
    {"TinyNegativeRoundUp", 0, 0x1f5, 0x80800000, 0, 0, 0x3f7fffff},
    {"TinyNegativeRoundDown", 0, 0x1fa, 0x80800000, 0, 0, 0x3f7fffff},
    {"MinusHalf", 0, 0x1f0, 0xbf000000, 0, 0, 0x3f000000},
    {"SignalingKeptIeeeClear", 0, 0x1f0, 0x7f800001, 0, 0, 0x7f800001},
};
// FP16 forms write the selected destination half and keep the other, which the capture filled
// with 0xa5a5.
const CapturedGfx11 fract_f16_gfx11_lanes[] = {
    {"TinyNegativeCaps", 0, 0x1f0, 0x8ced8400, 0, 0, 0xa5a53bff},
    {"PositiveFraction", 0, 0x1f0, 0x4daab800, 0, 0, 0xa5a53800},
    {"Mul2FlushScalesMinNormal", RJ_ARITH_ALU_OMOD_2, 0x100, 0xb19a0400, 0, 0, 0xa5a50800},
    {"Mul2FlushScalesHalf", RJ_ARITH_ALU_OMOD_2, 0x100, 0x6e8e3800, 0, 0, 0xa5a53c00},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D, 0x1f0, 0xd49c0000, 0, 0, 0x3400a5a5},
    {"HighSourceZero", RJ_ARITH_ALU_HIGH_A, 0x1f0, 0x7b3b8000, 0, 0, 0xa5a50000},
    {"SignalingKeptIeeeClear", 0, 0x1f0, 0xc3957c01, 0, 0, 0xa5a57c01},
    {"SignalingQuietedIeee", 0, 0x3f0, 0xc3957c01, 0, 0, 0xa5a57e01},
};

} // namespace

// Compatibility and strict exact reproduce the captured RDNA3 FRACT rules.
TEST(MathFractGfx11, CapturedOmodRoundingAndNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 32>(rj_arith_v_fract_f32_w32, fract_f32_gfx11_lanes);
  check_captured_gfx11_exact<64, 32>(rj_arith_v_fract_f32_w64, fract_f32_gfx11_lanes);
  check_captured_gfx11_exact<32, 16>(rj_arith_v_fract_f16_w32, fract_f16_gfx11_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_fract_f16_w64, fract_f16_gfx11_lanes);
}

// Inherited mantissa mapping retains payload quieting, overlap and unsupported strict modes.
TEST(MathOlderHalfFrexp, Arithmetic) {
  rocjitsu::arith::test::check_older_half_unary<32>(rj_arith_v_frexp_mant_f16_w32_gfx908, 4);
  rocjitsu::arith::test::check_older_half_unary<64>(rj_arith_v_frexp_mant_f16_w64_gfx908, 4);
}

// Fractional decomposition keeps the inherited special-value and aliasing contract.
TEST(MathOlderHalfFract, Arithmetic) {
  rocjitsu::arith::test::check_older_half_unary<32>(rj_arith_v_fract_f16_w32_gfx908, 5);
  rocjitsu::arith::test::check_half_fract_rounding<32>(rj_arith_v_fract_f16_w32_gfx908);
  rocjitsu::arith::test::check_older_half_unary<64>(rj_arith_v_fract_f16_w64_gfx908, 5);
  rocjitsu::arith::test::check_half_fract_rounding<64>(rj_arith_v_fract_f16_w64_gfx908);
}

namespace {

// Check the inherited frexp model and legacy guest output controls, including NaN quieting.
template <int Width, int Lanes, typename Call> void check_legacy_mantissa(Call call) {
  using namespace rocjitsu::arith::test;
  const uint64_t infinity = Width == 64 ? 0x7ff0000000000000ULL : 0x7f800000ULL;
  const uint64_t normal = Width == 64 ? 0x0010000000000000ULL : 0x00800000ULL;
  const uint64_t sign = 1ULL << (Width - 1),
                 quiet = Width == 64 ? 0x0008000000000000ULL : 0x00400000ULL;
  expect_alu_invalid_flags<Width == 32>(
      [&](uint64_t flags, uint64_t mode) {
        return call(flags, 0, mode, nullptr, nullptr, nullptr);
      },
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
          (Width == 32 ? RJ_ARITH_ALU_SDWA : 0));
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    EXPECT_EQ(call(semantics, 0, 0, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  }
  const uint64_t special[] = {0,
                              sign,
                              1,
                              sign | 1,
                              normal - 1,
                              normal,
                              infinity - 1,
                              infinity,
                              sign | infinity,
                              infinity + 1,
                              sign | infinity | quiet,
                              Width == 64 ? 0x3ff0000000000000ULL : 0x3f800000ULL};
  Registers<6, Lanes> input;
  std::minstd_rand random;
  for (int lane = 0; lane < Lanes; ++lane) {
    uint64_t bits = 0;
    for (int half = 0; half < Width / 16; ++half) {
      bits = (bits << 16) | uint64_t(random() >> 15);
    }
    if (lane < int(std::size(special))) {
      bits = special[lane];
    }
    write_fp(input.v, lane, Width, bits);
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (int scenario = 0; scenario < 16; ++scenario) {
        const uint64_t flags =
            cpu | semantics | (scenario & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
            (scenario & 2 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
            (scenario & 4 ? RJ_ARITH_FP_IEEE_MODE : 0) |
            (scenario & 8 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0) | (uint64_t(scenario % 4) << 24);
        const uint32_t mode =
            uint32_t(scenario % 4) << 6 | (scenario & 2 ? RJ_ARITH_ALU_CLAMP : 0) |
            (scenario & 4 ? RJ_ARITH_ALU_ABS_A : 0) | (scenario & 8 ? RJ_ARITH_ALU_NEG_A : 0);
        const uint64_t exec_mask = Lanes == 64 ? 0x800f0003800fffffULL : 0x800fffffULL;
        for (int aliasing_offset : {0, 1, 3}) {
          SCOPED_TRACE(::testing::Message()
                       << Width << ':' << Lanes << ':' << cpu << ':' << semantics << ':' << scenario
                       << ':' << aliasing_offset);
          auto actual = input;
          uint32_t status = 0x100;
          ASSERT_EQ(call(flags, exec_mask, mode, actual.v + aliasing_offset, actual.v, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x100U);
          actual.expect_preserved(input, aliasing_offset, Width / 32);
          for (int lane = 0; lane < Lanes; ++lane) {
            uint64_t expected = read_fp(input.v + aliasing_offset, lane, Width);
            if (exec_mask >> lane & 1) {
              uint64_t bits = read_fp(input.v, lane, Width);
              if (mode & RJ_ARITH_ALU_ABS_A) {
                bits &= ~sign;
              }
              if (mode & RJ_ARITH_ALU_NEG_A) {
                bits ^= sign;
              }
              uint64_t magnitude = bits & ~sign;
              if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && magnitude < normal) {
                bits &= sign;
                magnitude = 0;
              }
              if (magnitude > infinity) {
                expected = bits | quiet;
              } else {
                double value = fp_value(bits, Width);
                if (magnitude && magnitude < infinity) {
                  int exponent;
                  value = std::frexp(value, &exponent);
                }
                const unsigned omod =
                    (flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)
                        ? 0
                        : unsigned((mode >> 6) & 3);
                if (omod) {
                  value = value == 0 ? 0 : value * (omod == 3 ? 0.5 : double(1U << omod));
                }
                if (mode & RJ_ARITH_ALU_CLAMP) {
                  value = value < 0 ? 0 : value > 1 ? 1 : value;
                  if (value == 0) {
                    value = 0;
                  }
                }
                expected = Width == 64 ? std::bit_cast<uint64_t>(value)
                                       : uint64_t(std::bit_cast<uint32_t>(float(value)));
              }
              if ((mode & RJ_ARITH_ALU_CLAMP) && !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
                  (expected & ~sign) > infinity) {
                expected = 0;
              }
            }
            EXPECT_EQ(read_fp(actual.v + aliasing_offset, lane, Width), expected) << lane;
          }
        }
      }
    }
  }
}

} // namespace

// Exercise the public legacy-target policy on every supported CPU level.
TEST(MathFrexpMantF32W32Gfx908, Arithmetic) {
  check_legacy_mantissa<32, 32>(rj_arith_v_frexp_mant_f32_w32_gfx908);
}
// Check the Wave64 legacy FP32 contract with the same arithmetic scenarios.
TEST(MathFrexpMantF32W64Gfx908, Arithmetic) {
  check_legacy_mantissa<32, 64>(rj_arith_v_frexp_mant_f32_w64_gfx908);
}
// Check split-word legacy FP64 mantissas and cross-word destination overlap.
TEST(MathFrexpMantF64W32Gfx908, Arithmetic) {
  check_legacy_mantissa<64, 32>(rj_arith_v_frexp_mant_f64_w32_gfx908);
}
// Check the Wave64 split-word contract with the same arithmetic scenarios.
TEST(MathFrexpMantF64W64Gfx908, Arithmetic) {
  check_legacy_mantissa<64, 64>(rj_arith_v_frexp_mant_f64_w64_gfx908);
}

// SDWA scales the rounded core result with guest rounding before tiny-result flushing.
TEST(MathMantissaF32Legacy, SdwaOutput) {
  using namespace rocjitsu::arith::test;
  const uint32_t input[] = {0x00ffffffU, 0x3fc00000U, 0x7f812345U, 0x7f800000U};
  const uint32_t golden[] = {0x3effffffU, 0x3ec00000U, 0x7fc12345U, 0x7f800000U};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (uint64_t rounding : {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_ZERO}) {
        Registers<2, 64> regs;
        for (int lane = 0; lane < 64; ++lane) {
          regs.v[0][lane] = input[lane % 4];
        }
        const uint64_t flags = cpu | semantics | rounding | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                               RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
        ASSERT_EQ(rj_arith_v_frexp_mant_f32_w64_gfx908(flags, ~0ULL,
                                                       RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_HALF,
                                                       regs.v + 1, regs.v, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          uint32_t expected = golden[lane % 4];
          EXPECT_EQ(regs.v[1][lane], expected)
              << cpu << ':' << semantics << ':' << rounding << ':' << lane;
        }
      }
    }
  }
  EXPECT_EQ(rj_arith_v_frexp_mant_f32_w64_gfx908(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0,
                                                 RJ_ARITH_ALU_SDWA | RJ_ARITH_DPP16, nullptr,
                                                 nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}

namespace {

// Selected checkpoints generated from Rocjitsu's already-validated integer FRACT model.
constexpr uint64_t input32[] = {0,           0x80000000U, 1,           0x80000001U, 0x007fffffU,
                                0x00800000U, 0x807fffffU, 0xbeffffffU, 0xbf000000U, 0xbf000001U,
                                0xbdcccccdU, 0xb3000001U, 0xb3800001U, 0x3f7fffffU, 0x3fc00000U,
                                0x7f7fffffU, 0x7f800000U, 0xff800000U, 0x7f812345U, 0xffc12345U};
constexpr uint64_t input64[] = {0,
                                0x8000000000000000ULL,
                                1,
                                0x8000000000000001ULL,
                                0x000fffffffffffffULL,
                                0x0010000000000000ULL,
                                0x800fffffffffffffULL,
                                0xbfdfffffffffffffULL,
                                0xbfe0000000000000ULL,
                                0xbfe0000000000001ULL,
                                0xbfb999999999999aULL,
                                0xbc90000000000001ULL,
                                0xbca0000000000001ULL,
                                0x3fefffffffffffffULL,
                                0x3ff8000000000000ULL,
                                0x7fefffffffffffffULL,
                                0x7ff0000000000000ULL,
                                0xfff0000000000000ULL,
                                0x7ff0123456789abcULL,
                                0xfff8abcdef012345ULL};
// Rows follow guest rounding: nearest-even, toward +infinity, toward -infinity, toward zero.
constexpr uint64_t golden32[4][20] = {
    {
        0x0ULL,        0x0ULL,        0x1ULL,        0x3f7fffffULL, 0x7fffffULL,
        0x800000ULL,   0x3f7fffffULL, 0x3f000000ULL, 0x3f000000ULL, 0x3efffffeULL,
        0x3f666666ULL, 0x3f7fffffULL, 0x3f7fffffULL, 0x3f7fffffULL, 0x3f000000ULL,
        0x0ULL,        0xffc00000ULL, 0xffc00000ULL, 0x7fc12345ULL, 0xffc12345ULL,
    },
    {
        0x0ULL,        0x0ULL,        0x1ULL,        0x3f7fffffULL, 0x7fffffULL,
        0x800000ULL,   0x3f7fffffULL, 0x3f000001ULL, 0x3f000000ULL, 0x3efffffeULL,
        0x3f666667ULL, 0x3f7fffffULL, 0x3f7fffffULL, 0x3f7fffffULL, 0x3f000000ULL,
        0x0ULL,        0xffc00000ULL, 0xffc00000ULL, 0x7fc12345ULL, 0xffc12345ULL,
    },
    {
        0x0ULL,        0x0ULL,        0x1ULL,        0x3f7fffffULL, 0x7fffffULL,
        0x800000ULL,   0x3f7fffffULL, 0x3f000000ULL, 0x3f000000ULL, 0x3efffffeULL,
        0x3f666666ULL, 0x3f7fffffULL, 0x3f7ffffeULL, 0x3f7fffffULL, 0x3f000000ULL,
        0x0ULL,        0xffc00000ULL, 0xffc00000ULL, 0x7fc12345ULL, 0xffc12345ULL,
    },
    {
        0x0ULL,        0x0ULL,        0x1ULL,        0x3f7fffffULL, 0x7fffffULL,
        0x800000ULL,   0x3f7fffffULL, 0x3f000000ULL, 0x3f000000ULL, 0x3efffffeULL,
        0x3f666666ULL, 0x3f7fffffULL, 0x3f7ffffeULL, 0x3f7fffffULL, 0x3f000000ULL,
        0x0ULL,        0xffc00000ULL, 0xffc00000ULL, 0x7fc12345ULL, 0xffc12345ULL,
    },
};
constexpr uint64_t golden64[4][20] = {
    {
        0x0ULL,
        0x0ULL,
        0x1ULL,
        0x3fefffffffffffffULL,
        0xfffffffffffffULL,
        0x10000000000000ULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000000ULL,
        0x3fe0000000000000ULL,
        0x3fdffffffffffffeULL,
        0x3feccccccccccccdULL,
        0x3fefffffffffffffULL,
        0x3fefffffffffffffULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000000ULL,
        0x0ULL,
        0xfff8000000000000ULL,
        0xfff8000000000000ULL,
        0x7ff8123456789abcULL,
        0xfff8abcdef012345ULL,
    },
    {
        0x0ULL,
        0x0ULL,
        0x1ULL,
        0x3fefffffffffffffULL,
        0xfffffffffffffULL,
        0x10000000000000ULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000001ULL,
        0x3fe0000000000000ULL,
        0x3fdffffffffffffeULL,
        0x3feccccccccccccdULL,
        0x3fefffffffffffffULL,
        0x3fefffffffffffffULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000000ULL,
        0x0ULL,
        0xfff8000000000000ULL,
        0xfff8000000000000ULL,
        0x7ff8123456789abcULL,
        0xfff8abcdef012345ULL,
    },
    {
        0x0ULL,
        0x0ULL,
        0x1ULL,
        0x3fefffffffffffffULL,
        0xfffffffffffffULL,
        0x10000000000000ULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000000ULL,
        0x3fe0000000000000ULL,
        0x3fdffffffffffffeULL,
        0x3fecccccccccccccULL,
        0x3fefffffffffffffULL,
        0x3feffffffffffffeULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000000ULL,
        0x0ULL,
        0xfff8000000000000ULL,
        0xfff8000000000000ULL,
        0x7ff8123456789abcULL,
        0xfff8abcdef012345ULL,
    },
    {
        0x0ULL,
        0x0ULL,
        0x1ULL,
        0x3fefffffffffffffULL,
        0xfffffffffffffULL,
        0x10000000000000ULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000000ULL,
        0x3fe0000000000000ULL,
        0x3fdffffffffffffeULL,
        0x3fecccccccccccccULL,
        0x3fefffffffffffffULL,
        0x3feffffffffffffeULL,
        0x3fefffffffffffffULL,
        0x3fe0000000000000ULL,
        0x0ULL,
        0xfff8000000000000ULL,
        0xfff8000000000000ULL,
        0x7ff8123456789abcULL,
        0xfff8abcdef012345ULL,
    },
};

// Check all CPU paths against inherited rounding checkpoints and legacy output policy.
template <int Width, int Lanes, typename Call> void check_legacy_fract(Call call) {
  using namespace rocjitsu::arith::test;
  expect_alu_invalid_flags<Width == 32>(
      [&](uint64_t flags, uint64_t mode) {
        return call(flags, 0, mode, nullptr, nullptr, nullptr);
      },
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
          (Width == 32 ? RJ_ARITH_ALU_SDWA : 0));
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    EXPECT_EQ(call(semantics, 0, 0, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  }
  const uint64_t normal = Width == 64 ? 0x0010000000000000ULL : 0x00800000ULL;
  const uint64_t sign = 1ULL << (Width - 1),
                 infinity = Width == 64 ? 0x7ff0000000000000ULL : 0x7f800000ULL;
  const uint64_t one = Width == 64 ? 0x3ff0000000000000ULL : 0x3f800000ULL;
  const uint64_t *inputs = Width == 64 ? input64 : input32;
  const auto &goldens = Width == 64 ? golden64 : golden32;
  Registers<6, Lanes> input;
  for (int lane = 0; lane < Lanes; ++lane) {
    write_fp(input.v, lane, Width, inputs[lane % 20]);
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (int scenario = 0; scenario < 16; ++scenario) {
        const unsigned rounding = scenario % 4;
        const uint64_t flags =
            cpu | semantics | (scenario & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
            (scenario & 2 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
            (scenario & 4 ? RJ_ARITH_FP_IEEE_MODE : 0) |
            (scenario & 8 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0) | (uint64_t(rounding) << 24);
        const uint32_t mode = uint32_t(scenario % 4) << 6 | (scenario & 8 ? RJ_ARITH_ALU_CLAMP : 0);
        const uint64_t exec_mask = Lanes == 64 ? 0x800f0003800fffffULL : 0x800fffffULL;
        for (int aliasing_offset : {0, 1, 3}) {
          SCOPED_TRACE(::testing::Message()
                       << Width << ':' << Lanes << ':' << cpu << ':' << semantics << ':' << scenario
                       << ':' << aliasing_offset);
          auto actual = input;
          uint32_t status = 0x100;
          ASSERT_EQ(call(flags, exec_mask, mode, actual.v + aliasing_offset, actual.v, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x100U);
          actual.expect_preserved(input, aliasing_offset, Width / 32);
          for (int lane = 0; lane < Lanes; ++lane) {
            uint64_t expected = read_fp(input.v + aliasing_offset, lane, Width);
            if (exec_mask >> lane & 1) {
              expected = goldens[rounding][lane % 20];
              if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
                  (inputs[lane % 20] & ~sign) < normal) {
                expected = 0;
              }
              if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (expected & ~sign) < normal) {
                expected = 0;
              }
              const unsigned omod =
                  (flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)
                      ? 0
                      : unsigned((mode >> 6) & 3);
              if (omod && (expected & ~sign) < infinity) {
                if (expected < normal || (omod == 3 && expected < 2 * normal)) {
                  expected = 0;
                } else {
                  expected = omod == 3 ? expected - normal : expected + uint64_t(omod) * normal;
                }
              }
              if ((mode & RJ_ARITH_ALU_CLAMP) &&
                  (!(flags & RJ_ARITH_FP_DX10_CLAMP_OFF) || (expected & ~sign) <= infinity)) {
                expected = expected & sign || (expected & ~sign) > infinity ? 0
                           : expected > one                                 ? one
                                                                            : expected;
              }
            }
            EXPECT_EQ(read_fp(actual.v + aliasing_offset, lane, Width), expected) << lane;
          }
        }
      }
    }
  }
}

} // namespace

// Check legacy FP32 fractional parts and the uniform API contract.
TEST(MathFractF32W32Gfx908, Arithmetic) {
  check_legacy_fract<32, 32>(rj_arith_v_fract_f32_w32_gfx908);
}
// Check Wave64 FP32 fractional parts with the same compact checkpoints.
TEST(MathFractF32W64Gfx908, Arithmetic) {
  check_legacy_fract<32, 64>(rj_arith_v_fract_f32_w64_gfx908);
}
// Check split-word FP64 results and source/destination overlap.
TEST(MathFractF64W32Gfx908, Arithmetic) {
  check_legacy_fract<64, 32>(rj_arith_v_fract_f64_w32_gfx908);
}
// Check Wave64 split-word results with the same compact checkpoints.
TEST(MathFractF64W64Gfx908, Arithmetic) {
  check_legacy_fract<64, 64>(rj_arith_v_fract_f64_w64_gfx908);
}

// SDWA scales the rounded core result with guest rounding before tiny-result flushing.
TEST(MathFractF32Legacy, SdwaOutput) {
  using namespace rocjitsu::arith::test;
  const uint32_t input[] = {0x00ffffffU, 0x3fc00000U, 0x7f812345U, 0x7f800000U};
  const uint32_t golden[] = {0x00800000U, 0x3e800000U, 0x7fc12345U, 0xffc00000U};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (uint64_t rounding : {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_ZERO}) {
        Registers<2, 64> regs;
        for (int lane = 0; lane < 64; ++lane) {
          regs.v[0][lane] = input[lane % 4];
        }
        const uint64_t flags = cpu | semantics | rounding | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                               RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
        ASSERT_EQ(rj_arith_v_fract_f32_w64_gfx908(flags, ~0ULL,
                                                  RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_HALF,
                                                  regs.v + 1, regs.v, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          uint32_t expected = golden[lane % 4];
          if (lane % 4 == 0 && rounding == RJ_ARITH_FP_ROUND_ZERO) {
            expected = 0;
          }
          EXPECT_EQ(regs.v[1][lane], expected)
              << cpu << ':' << semantics << ':' << rounding << ':' << lane;
        }
      }
    }
  }
  EXPECT_EQ(rj_arith_v_fract_f32_w64_gfx908(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0,
                                            RJ_ARITH_ALU_SDWA | RJ_ARITH_DPP16, nullptr, nullptr,
                                            nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
