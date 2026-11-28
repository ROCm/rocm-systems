// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for conditionally scaled division FMA and FP-status expectations.

#include "fp/div_fmas_test.h"
#include "fp/div_omod_gfx11_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace {

// Literal exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
struct StatusCase {
  uint64_t a, b, c;
  uint32_t causes;
};

const StatusCase status_cases_f32[] = {
    {0x3f800000U, 0x3f800000U, 0x0U, 0},
    {0x0U, 0x7f800000U, 0x0U, 0},
    {0x7f7fffffU, 0x40000000U, 0x0U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f800001U, 0x3f800001U, 0x0U, RJ_ARITH_EXCEPTION_INEXACT},
    {0x1U, 0x3f000000U, 0x0U, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    // The inherited classifier tests tininess at normal precision, before
    // limiting the exponent. Rounding up to minimum normal only at subnormal
    // precision still reports UNDERFLOW; the closer product below does not.
    {0x00800000U, 0x3f7fffffU, 0, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x00800001U, 0x3f7ffffeU, 0, RJ_ARITH_EXCEPTION_INEXACT},
    {0x00800000U, 0x3f7fffffU, 0x80000001U,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x00800000U, 0x3f7ffffeU, 0, 0},
};

const StatusCase status_cases_f64[] = {
    {0x3ff0000000000000ULL, 0x3ff0000000000000ULL, 0x0ULL, 0},
    {0x0ULL, 0x7ff0000000000000ULL, 0x0ULL, 0},
    {0x7fefffffffffffffULL, 0x4000000000000000ULL, 0x0ULL,
     RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3ff0000000000001ULL, 0x3ff0000000000001ULL, 0x0ULL, RJ_ARITH_EXCEPTION_INEXACT},
    {0x1ULL, 0x3fe0000000000000ULL, 0x0ULL,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000000ULL, 0x3fefffffffffffffULL, 0,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000001ULL, 0x3feffffffffffffeULL, 0, RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000000ULL, 0x3fefffffffffffffULL, 0x8000000000000001ULL,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000000ULL, 0x3feffffffffffffeULL, 0, 0},
};

} // namespace

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 32>(rj_arith_v_div_fmas_f32_w32_gfx12, /*gfx12=*/true,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 32>(rj_arith_v_div_fmas_f32_w32, /*gfx12=*/false,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 64>(rj_arith_v_div_fmas_f32_w64_gfx12, /*gfx12=*/true,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 64>(rj_arith_v_div_fmas_f32_w64, /*gfx12=*/false,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 32>(rj_arith_v_div_fmas_f64_w32_gfx12, /*gfx12=*/true,
                                                status_cases_f64);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 32>(rj_arith_v_div_fmas_f64_w32, /*gfx12=*/false,
                                                status_cases_f64);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 64>(rj_arith_v_div_fmas_f64_w64_gfx12, /*gfx12=*/true,
                                                status_cases_f64);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 64>(rj_arith_v_div_fmas_f64_w64, /*gfx12=*/false,
                                                status_cases_f64);
}

// gfx1100/W7900 captures supplied in PR #13952: selected sNaNs quiet only
// with IEEE set (VCC=0).
template <int Width, int Lanes, typename Call> static void check_rdna3_nan(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr uint64_t one = Width == 32 ? 0x3f800000ULL : 0x3ff0000000000000ULL;
  constexpr uint64_t snan = Width == 32 ? 0x7f800001ULL : 0x7ff0000000000001ULL;
  constexpr uint64_t quiet = Width == 32 ? 0x00400000ULL : 0x0008000000000000ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        for (int operand : {0, 1, 2}) {
          Registers<8, Lanes> r;
          for (int i = 0; i < 3; ++i) {
            write_fp(r.v + 2 * i, Lanes - 1, Width, i == operand ? snan : one);
          }
          uint32_t status = 0x80000000U;
          ASSERT_EQ(call(cpu | semantics | ieee, Mask(1) << (Lanes - 1), 0, r.v + 6, r.v, r.v + 2,
                         r.v + 4, Mask(0), &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x80000000U);
          EXPECT_EQ(read_fp(r.v + 6, Lanes - 1, Width), snan | (ieee ? quiet : 0));
        }
      }
    }
  }
}

TEST(FpDivFmasGfx11, SignalingNan) {
  check_rdna3_nan<32, 32>(rj_arith_v_div_fmas_f32_w32);
  check_rdna3_nan<32, 64>(rj_arith_v_div_fmas_f32_w64);
  check_rdna3_nan<64, 32>(rj_arith_v_div_fmas_f64_w32);
  check_rdna3_nan<64, 64>(rj_arith_v_div_fmas_f64_w64);
}

TEST(FpDivFmasGfx950, SignalingNanAndGuards) {
  check_rdna3_nan<32, 32>(rj_arith_v_div_fmas_f32_w32_gfx950);
  check_rdna3_nan<32, 64>(rj_arith_v_div_fmas_f32_w64_gfx950);
  check_rdna3_nan<64, 32>(rj_arith_v_div_fmas_f64_w32_gfx950);
  check_rdna3_nan<64, 64>(rj_arith_v_div_fmas_f64_w64_gfx950);
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    Registers<8, 32> r;
    write_fp(r.v, 0, 32, 0x3f800000U);
    write_fp(r.v + 2, 0, 32, 0x3f800000U);
    write_fp(r.v + 4, 0, 32, 0U);
    EXPECT_EQ(rj_arith_v_div_fmas_f32_w32_gfx950(
                  cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                      RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                  1U, RJ_ARITH_ALU_OMOD_2, r.v + 6, r.v, r.v + 2, r.v + 4, 0U, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(rj_arith_v_div_fmas_f32_w32_gfx950(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 1U,
                                                 0, r.v + 6, r.v, r.v + 2, r.v + 4, 0U, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
  }
}

namespace {

// Smallest-normal boundary captures with VCC clear. Tininess is judged after rounding at normal
// precision: a value that reaches the smallest normal only on the subnormal grid still flushes
// when MODE flushes outputs. C is the negative subnormal next to the smallest normal; the negative
// product is 1.25 or 1.75 normal-precision ulps there, or far below one under round-down.
struct TinyCase {
  uint64_t a, b, c, flags, expected;
  bool post_scale = false; // VCC set: the result is scaled by 2^-64 (F32) before rounding.
};

constexpr uint64_t flush = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;

const TinyCase tiny_cases_f32[] = {
    {0x9a200000U, 0x1a000000U, 0x807fffffU, flush, 0x80000000U},
    {0x9a200000U, 0x1a000000U, 0x807fffffU, 0, 0x80800000U},
    {0x9a200000U, 0x1a000000U, 0x807fffffU, flush | RJ_ARITH_FP_ROUND_DOWN, 0x80800000U},
    {0x9a600000U, 0x1a000000U, 0x807fffffU, flush, 0x80800000U},
    {0x807fffffU, 0x00000001U, 0x807fffffU, flush, 0x80000000U},
    {0x807fffffU, 0x00000001U, 0x807fffffU, 0, 0x807fffffU},
    {0x807fffffU, 0x00000001U, 0x807fffffU, flush | RJ_ARITH_FP_ROUND_DOWN, 0x80000000U},
    {0x807fffffU, 0x00000001U, 0x807fffffU, RJ_ARITH_FP_ROUND_DOWN, 0x80800000U},
    // 2^-62 * (1 - 2^-24) scaled to 2^-126 - 2^-150: a tie that rounds up only on the subnormal
    // grid.
    {0x20800000U, 0x3f7fffffU, 0, flush, 0, true},
    {0x20800000U, 0x3f7fffffU, 0, 0, 0x00800000U, true},
};

const TinyCase tiny_cases_f64[] = {
    {0x9e54000000000000ULL, 0x1e60000000000000ULL, 0x800fffffffffffffULL, flush,
     0x8000000000000000ULL},
    {0x9e54000000000000ULL, 0x1e60000000000000ULL, 0x800fffffffffffffULL, 0, 0x8010000000000000ULL},
    {0x9e54000000000000ULL, 0x1e60000000000000ULL, 0x800fffffffffffffULL,
     flush | RJ_ARITH_FP_ROUND_DOWN, 0x8010000000000000ULL},
    {0x9e5c000000000000ULL, 0x1e60000000000000ULL, 0x800fffffffffffffULL, flush,
     0x8010000000000000ULL},
    {0x800fffffffffffffULL, 0x0010000000000000ULL, 0x800fffffffffffffULL, flush,
     0x8000000000000000ULL},
    {0x800fffffffffffffULL, 0x0010000000000000ULL, 0x800fffffffffffffULL, 0, 0x800fffffffffffffULL},
    {0x800fffffffffffffULL, 0x0010000000000000ULL, 0x800fffffffffffffULL,
     flush | RJ_ARITH_FP_ROUND_DOWN, 0x8000000000000000ULL},
    {0x800fffffffffffffULL, 0x0010000000000000ULL, 0x800fffffffffffffULL, RJ_ARITH_FP_ROUND_DOWN,
     0x8010000000000000ULL},
};

template <int Width, int Lanes, typename Call, typename Cases>
void check_tiny(Call call, const Cases &cases) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      SCOPED_TRACE(::testing::Message()
                   << std::hex << cpu << '/' << sample.a << '/' << sample.flags);
      Registers<8, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        write_fp(r.v, lane, Width, sample.a);
        write_fp(r.v + 2, lane, Width, sample.b);
        write_fp(r.v + 4, lane, Width, sample.c);
      }
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | sample.flags, ~Mask(0), 0, r.v + 6,
                     r.v, r.v + 2, r.v + 4, sample.post_scale ? ~Mask(0) : Mask(0), nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(read_fp(r.v + 6, lane, Width), sample.expected) << lane;
      }
    }
  }
}

} // namespace

// gfx1201 and gfx1100 give these results for both RDNA3 and gfx12 entry points.
TEST(FpDivFmas, TininessAfterRounding) {
  check_tiny<32, 32>(rj_arith_v_div_fmas_f32_w32_gfx12, tiny_cases_f32);
  check_tiny<32, 64>(rj_arith_v_div_fmas_f32_w64_gfx12, tiny_cases_f32);
  check_tiny<64, 32>(rj_arith_v_div_fmas_f64_w32_gfx12, tiny_cases_f64);
  check_tiny<64, 64>(rj_arith_v_div_fmas_f64_w64_gfx12, tiny_cases_f64);
  check_tiny<32, 32>(rj_arith_v_div_fmas_f32_w32, tiny_cases_f32);
  check_tiny<32, 64>(rj_arith_v_div_fmas_f32_w64, tiny_cases_f32);
  check_tiny<64, 32>(rj_arith_v_div_fmas_f64_w32, tiny_cases_f64);
  check_tiny<64, 64>(rj_arith_v_div_fmas_f64_w64, tiny_cases_f64);
}

// Compare backend policies on deterministic full-range operands and inherited gradual rounding.
template <int Width, int Lanes, bool Older, typename Call> static void check_cdna_fmas(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const auto input = make_fp_inputs<Width, Lanes>();
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int sample = 0; sample < 64; ++sample) {
      constexpr uint64_t semantics[] = {RJ_ARITH_SEMANTICS_LOOSE,
                                        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                        RJ_ARITH_SEMANTICS_COMPATIBILITY};
      uint64_t flags = semantics[sample % 3] | (uint64_t(sample & 3) << 24) |
                       (sample & 4 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                       (sample & 8 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                       (sample & 16 ? RJ_ARITH_FP_IEEE_MODE : 0);
      if (Older && (sample & 32)) {
        flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
      }
      const uint64_t mode = uint64_t((sample >> 2) & 3) * RJ_ARITH_ALU_OMOD_2 |
                            (sample & 32 ? RJ_ARITH_ALU_CLAMP : 0) |
                            (sample & 16 ? RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_C : 0);
      const Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL) | 1;
      const Mask condition = Mask(0xccccccccccccccccULL);
      const int destination = sample & 1 ? 0 : 6;
      auto expected = input, actual = input;
      uint32_t status = 0x55;
      ASSERT_EQ(call(flags, exec_mask, mode, expected.v + destination, expected.v, expected.v + 2,
                     expected.v + 4, condition, nullptr),
                0);
      ASSERT_EQ(call(flags | cpu, exec_mask, mode, actual.v + destination, actual.v, actual.v + 2,
                     actual.v + 4, condition, &status),
                0);
      EXPECT_EQ(status, 0x55U);
      actual.expect_preserved(input, destination, Width / 32);
      for (int reg = 0; reg < Width / 32; ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(actual.v[destination + reg][lane], expected.v[destination + reg][lane]);
        }
      }
    }
    // Cancellation and overflow expose the directed-rounding choices independently
    // of the pseudorandom scalar/SIMD comparisons above.
    constexpr uint64_t sign = 1ULL << (Width - 1);
    constexpr uint64_t one = Width == 32 ? 0x3f800000ULL : 0x3ff0000000000000ULL;
    constexpr uint64_t infinity = Width == 32 ? 0x7f800000ULL : 0x7ff0000000000000ULL;
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      for (int scenario = 0; scenario < 3; ++scenario) {
        Registers<8, Lanes> boundary;
        const uint64_t mode = scenario == 2 ? RJ_ARITH_ALU_OMOD_2 : 0;
        for (int lane = 0; lane < Lanes; ++lane) {
          const uint64_t negative = lane & 1 ? sign : 0;
          write_fp(boundary.v, lane, Width, (scenario ? infinity - 1 : one) | negative);
          write_fp(boundary.v + 2, lane, Width,
                   scenario == 1 ? one + (1ULL << (Width == 32 ? 23 : 52)) : one);
          write_fp(boundary.v + 4, lane, Width, scenario ? negative : one | (negative ^ sign));
        }
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                           (uint64_t(rounding) << 24),
                       Mask(~Mask(0)), mode, boundary.v + 6, boundary.v, boundary.v + 2,
                       boundary.v + 4, Mask(0), nullptr),
                  0);
        for (int lane = 0; lane < Lanes; ++lane) {
          const uint64_t negative = lane & 1 ? sign : 0;
          const bool to_infinity = rounding == 0 || rounding == (negative ? 2U : 1U);
          const uint64_t expected = scenario ? negative | (to_infinity ? infinity : infinity - 1)
                                    : rounding == 2 ? sign
                                                    : 0;
          EXPECT_EQ(read_fp(boundary.v + 6, lane, Width), expected);
        }
      }
    }
    // Half a minimum subnormal plus maximum subnormal rounds to minimum normal
    // before OMOD. Rounding at normal precision and flushing first would return zero.
    Registers<8, Lanes> boundary;
    constexpr uint64_t hidden = 1ULL << (Width == 32 ? 23 : 52);
    constexpr uint64_t half = Width == 32 ? 0x3f000000ULL : 0x3fe0000000000000ULL;
    write_fp(boundary.v, 0, Width, half);
    write_fp(boundary.v + 2, 0, Width, 1);
    write_fp(boundary.v + 4, 0, Width, hidden - 1);
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                   Mask(1), RJ_ARITH_ALU_OMOD_2, boundary.v + 6, boundary.v, boundary.v + 2,
                   boundary.v + 4, Mask(0), nullptr),
              0);
    EXPECT_EQ(read_fp(boundary.v + 6, 0, Width), 2 * hidden);
  }
  EXPECT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, Mask(0), 0, nullptr, nullptr, nullptr,
                 nullptr, Mask(0), nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
}

TEST(CdnaDivFmas, Arithmetic) {
  check_cdna_fmas<32, 32, true>(rj_arith_v_div_fmas_f32_w32_gfx942);
  check_cdna_fmas<32, 64, true>(rj_arith_v_div_fmas_f32_w64_gfx942);
  check_cdna_fmas<64, 32, true>(rj_arith_v_div_fmas_f64_w32_gfx942);
  check_cdna_fmas<64, 64, true>(rj_arith_v_div_fmas_f64_w64_gfx942);
  check_cdna_fmas<32, 32, false>(rj_arith_v_div_fmas_f32_w32_gfx1250);
  check_cdna_fmas<64, 32, false>(rj_arith_v_div_fmas_f64_w32_gfx1250);
}

namespace {

using rocjitsu::arith::test::DivOmodGfx11;

// W7900 (gfx1100, ROCm 7.2) wave32 lanes captured on 2026-11-18 with the omod3 probe: the hardware
// MODE word is written per lane (rounding, denormals, DX10_CLAMP, IEEE), VCC is set per wave.
// OMOD is active only with output denormals flushed and IEEE clear; the result rounds at normal
// precision, is scaled, and overflows per the guest rounding: infinity for nearest and the
// rounding toward the sign, max-finite otherwise. A denormal result becomes +0 before the scale,
// while the smallest normal halved keeps its sign. Signaling NaNs pass through; CLAMP applies last.
//
// The rows above the separator all hold an operand outside the exact AVX-512 backend's admitted
// exponent range, so they exercise the scalar model. The rows below it keep every operand inside
// that range, so with VCC clear they reach the backend this change newly selects: a product that
// rounds differently under each guest mode, the same product scaled up and down, an output
// modifier dropped because output denormals are preserved, and the zero-sign rule that makes an
// exact -0 come back as +0 once a modifier scales it.
const DivOmodGfx11 omod_lanes_f32[] = {
    {"mul2 overflow nearest", 0x100, 0x7f000000, 0x3f800000, 0, 0x7f800000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round up", 0x101, 0x7f000000, 0x3f800000, 0, 0x7f800000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round down", 0x102, 0x7f000000, 0x3f800000, 0, 0x7f7fffff, RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow toward zero", 0x103, 0x7f000000, 0x3f800000, 0, 0x7f7fffff,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 with input denormals allowed", 0x110, 0x7f000000, 0x3f800000, 0, 0x7f800000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with output denormals", 0x130, 0x7f000000, 0x3f800000, 0, 0x7f000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with IEEE", 0x300, 0x7f000000, 0x3f800000, 0, 0x7f000000, RJ_ARITH_ALU_OMOD_2},
    {"mul4 overflow nearest", 0x100, 0x7e800000, 0x3f800000, 0, 0x7f800000, RJ_ARITH_ALU_OMOD_4},
    {"mul4 overflow round down", 0x102, 0x7e800000, 0x3f800000, 0, 0x7f7fffff, RJ_ARITH_ALU_OMOD_4},
    {"div2 flushes the smallest normal", 0x100, 0x00800000, 0x3f800000, 0, 0x00000000,
     RJ_ARITH_ALU_OMOD_HALF},
    {"div2 keeps the flushed sign", 0x100, 0x80800000, 0x3f800000, 0, 0x80000000,
     RJ_ARITH_ALU_OMOD_HALF},
    {"mul2 of a tiny negative product is +0", 0x110, 0x80000001, 0x3f800000, 0, 0x00000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 round-down carry reaches the smallest normal", 0x102, 0x9a200000, 0x1a000000, 0x807fffff,
     0x81000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 nearest flushes the same product", 0x100, 0x9a200000, 0x1a000000, 0x807fffff, 0x00000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 toward zero flushes the same product", 0x103, 0x9a200000, 0x1a000000, 0x807fffff,
     0x00000000, RJ_ARITH_ALU_OMOD_2},
    // 2^-126 * (1 - 2^-24) is exact at normal precision and below the smallest normal, so OMOD
    // gives +0; rounding on the subnormal grid would reach the smallest normal and double it.
    {"mul2 of a tie below the smallest normal rounds at normal precision", 0x100, 0x00800000,
     0x3f7fffff, 0, 0x00000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 of the same tie round up", 0x101, 0x00800000, 0x3f7fffff, 0, 0x00000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 of the same tie after VCC scaling", 0x100, 0x20800000, 0x3f7fffff, 0, 0x00000000,
     RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 keeps a signaling NaN", 0x100, 0x7f800001, 0x3f800000, 0, 0x7f800001,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 after VCC scaling overflows nearest", 0x100, 0x7f000000, 0x3f800000, 0x40000000,
     0x7f800000, RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 after VCC scaling overflows toward zero", 0x103, 0x7f000000, 0x3f800000, 0x40000000,
     0x7f7fffff, RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 clamp after overflow", 0x100, 0x7f000000, 0x3f800000, 0, 0x3f800000,
     RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
    {"mul2 clamp of a negative overflow", 0x100, 0xff000000, 0x3f800000, 0, 0x00000000,
     RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
    {"mul2 clamp of a NaN with DX10_CLAMP", 0x100, 0x7f800001, 0x3f800000, 0, 0x00000000,
     RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
    {"mul2 clamp keeps a NaN without DX10_CLAMP", 0x000, 0x7f800001, 0x3f800000, 0, 0x7f800001,
     RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},

    // Operands inside the exact backend's admitted exponent range; VCC is clear in all of them.
    {"mul2 round down", 0x102, 0x436a3cd0, 0xb75bb10b, 0x42bff051, 0x433feebe, RJ_ARITH_ALU_OMOD_2},
    {"mul2 of exact -0", 0x100, 0x80000000, 0x3f800000, 0x80000000, 0x00000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 round up", 0x101, 0x3fc00001, 0x40200003, 0x3f000001, 0x41080004, RJ_ARITH_ALU_OMOD_2},
    {"div2 nearest", 0x100, 0x3fc00001, 0x40200003, 0x3f000001, 0x40080003, RJ_ARITH_ALU_OMOD_HALF},
    {"mul2 ignored with output denormals", 0x130, 0x3fc00001, 0x40200003, 0x3f000001, 0x40880003,
     RJ_ARITH_ALU_OMOD_2},
    {"mul4 round up", 0x101, 0x3f7fffff, 0x3f7fffff, 0x33800000, 0x40800000, RJ_ARITH_ALU_OMOD_4},
};

const DivOmodGfx11 omod_lanes_f64[] = {
    {"mul2 overflow nearest", 0x100, 0x7fe0000000000000, 0x3ff0000000000000, 0, 0x7ff0000000000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round up", 0x104, 0x7fe0000000000000, 0x3ff0000000000000, 0, 0x7ff0000000000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow round down", 0x108, 0x7fe0000000000000, 0x3ff0000000000000, 0,
     0x7fefffffffffffff, RJ_ARITH_ALU_OMOD_2},
    {"mul2 overflow toward zero", 0x10c, 0x7fe0000000000000, 0x3ff0000000000000, 0,
     0x7fefffffffffffff, RJ_ARITH_ALU_OMOD_2},
    {"mul2 with input denormals allowed", 0x140, 0x7fe0000000000000, 0x3ff0000000000000, 0,
     0x7ff0000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with output denormals", 0x180, 0x7fe0000000000000, 0x3ff0000000000000, 0,
     0x7fe0000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 ignored with IEEE", 0x300, 0x7fe0000000000000, 0x3ff0000000000000, 0, 0x7fe0000000000000,
     RJ_ARITH_ALU_OMOD_2},
    {"mul4 overflow nearest", 0x100, 0x7fd0000000000000, 0x3ff0000000000000, 0, 0x7ff0000000000000,
     RJ_ARITH_ALU_OMOD_4},
    {"mul4 overflow round down", 0x108, 0x7fd0000000000000, 0x3ff0000000000000, 0,
     0x7fefffffffffffff, RJ_ARITH_ALU_OMOD_4},
    {"div2 flushes the smallest normal", 0x100, 0x0010000000000000, 0x3ff0000000000000, 0,
     0x0000000000000000, RJ_ARITH_ALU_OMOD_HALF},
    {"div2 keeps the flushed sign", 0x100, 0x8010000000000000, 0x3ff0000000000000, 0,
     0x8000000000000000, RJ_ARITH_ALU_OMOD_HALF},
    {"mul2 of a tiny negative product is +0", 0x140, 0x8000000000000001, 0x3ff0000000000000, 0,
     0x0000000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 round-down carry reaches the smallest normal", 0x108, 0x9e54000000000000,
     0x1e60000000000000, 0x800fffffffffffff, 0x8020000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 nearest flushes the same product", 0x100, 0x9e54000000000000, 0x1e60000000000000,
     0x800fffffffffffff, 0x0000000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 toward zero flushes the same product", 0x10c, 0x9e54000000000000, 0x1e60000000000000,
     0x800fffffffffffff, 0x0000000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 of a tie below the smallest normal rounds at normal precision", 0x100,
     0x0010000000000000, 0x3fefffffffffffff, 0, 0x0000000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 of the same tie round up", 0x104, 0x0010000000000000, 0x3fefffffffffffff, 0,
     0x0000000000000000, RJ_ARITH_ALU_OMOD_2},
    {"mul2 keeps a signaling NaN", 0x100, 0x7ff0000000000001, 0x3ff0000000000000, 0,
     0x7ff0000000000001, RJ_ARITH_ALU_OMOD_2},
    {"mul2 after VCC scaling overflows nearest", 0x100, 0x7fe0000000000000, 0x3ff0000000000000,
     0x4000000000000000, 0x7ff0000000000000, RJ_ARITH_ALU_OMOD_2, true},
    {"mul2 after VCC scaling overflows toward zero", 0x10c, 0x7fe0000000000000, 0x3ff0000000000000,
     0x4000000000000000, 0x7fefffffffffffff, RJ_ARITH_ALU_OMOD_2, true},
    {"div2 after VCC scaling down", 0x100, 0x3810000000000000, 0x3fefffffffffffff, 0,
     0x2fffffffffffffff, RJ_ARITH_ALU_OMOD_HALF, true},
    {"mul2 clamp after overflow", 0x100, 0x7fe0000000000000, 0x3ff0000000000000, 0,
     0x3ff0000000000000, RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
    {"mul2 clamp of a negative overflow", 0x100, 0xffe0000000000000, 0x3ff0000000000000, 0,
     0x0000000000000000, RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
    {"mul2 clamp keeps a NaN without DX10_CLAMP", 0x000, 0x7ff0000000000001, 0x3ff0000000000000, 0,
     0x7ff0000000000001, RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
};

} // namespace

// Active RDNA3 OMOD is accepted in strict exact and reproduces the W7900 lanes at every CPU level.
TEST(FpDivFmasGfx11, CapturedOutputModifiers) {
  using rocjitsu::arith::test::check_div_fmas_omod_gfx11;
  check_div_fmas_omod_gfx11<32, 32>(rj_arith_v_div_fmas_f32_w32, omod_lanes_f32);
  check_div_fmas_omod_gfx11<64, 32>(rj_arith_v_div_fmas_f32_w64, omod_lanes_f32);
  check_div_fmas_omod_gfx11<32, 64>(rj_arith_v_div_fmas_f64_w32, omod_lanes_f64);
  check_div_fmas_omod_gfx11<64, 64>(rj_arith_v_div_fmas_f64_w64, omod_lanes_f64);
}

// W7900 wave32 capture: lane 19 alone holds the smallest normal, which div:2 flushes while every
// other lane halves 1.0, so the flushed lane sits in a later SIMD chunk. Wave64 repeats it in
// lane 51.
template <int Lanes, typename Call> static void check_omod_flushed_lane(Call call) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      Registers<8, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        r.v[0][lane] = lane % 32 == 19 ? 0x00800000U : 0x3f800000U;
        r.v[2][lane] = 0x3f800000U;
        r.v[4][lane] = 0;
      }
      ASSERT_EQ(call(cpu | semantics | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                         RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                     ~WaveMask<Lanes>(0), RJ_ARITH_ALU_OMOD_HALF, r.v + 6, r.v, r.v + 2, r.v + 4,
                     WaveMask<Lanes>(0), nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(r.v[6][lane], lane % 32 == 19 ? 0x00000000U : 0x3f000000U)
            << std::hex << cpu << '/' << semantics << " lane " << std::dec << lane;
      }
    }
  }
}

TEST(FpDivFmasGfx11, OmodFlushedLaneKeepsItsOperands) {
  check_omod_flushed_lane<32>(rj_arith_v_div_fmas_f32_w32);
  check_omod_flushed_lane<64>(rj_arith_v_div_fmas_f32_w64);
}
