// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP64 reciprocal, reciprocal square root, and square root.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/wave.h"
#include "fp/loose_unary_f64.h"
#include "fp/reference.h"
#include "fp/trans_f64_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

// Check FP64 RCP results and semantics/status support across CPU levels.
TEST(FpRcpF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<32, false>(rj_arith_v_rcp_f64_w32, 0);
}

// Check FP64 RCP results and semantics/status support across CPU levels.
TEST(FpRcpF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<32, true>(rj_arith_v_rcp_f64_w32_gfx12, 0);
}

// Check FP64 RCP results and semantics/status support across CPU levels.
TEST(FpRcpF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<64, false>(rj_arith_v_rcp_f64_w64, 0);
}

// Check FP64 RCP results and semantics/status support across CPU levels.
TEST(FpRcpF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<64, true>(rj_arith_v_rcp_f64_w64_gfx12, 0);
}

// Check FP64 RSQ results and semantics/status support across CPU levels.
TEST(FpRsqF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<32, false>(rj_arith_v_rsq_f64_w32, 1);
}

// Check FP64 RSQ results and semantics/status support across CPU levels.
TEST(FpRsqF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<32, true>(rj_arith_v_rsq_f64_w32_gfx12, 1);
}

// Check FP64 RSQ results and semantics/status support across CPU levels.
TEST(FpRsqF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<64, false>(rj_arith_v_rsq_f64_w64, 1);
}

// Check FP64 RSQ results and semantics/status support across CPU levels.
TEST(FpRsqF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<64, true>(rj_arith_v_rsq_f64_w64_gfx12, 1);
}

// Check FP64 SQRT results and semantics/status support across CPU levels.
TEST(FpSqrtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<32, false>(rj_arith_v_sqrt_f64_w32, 2);
}

// Check FP64 SQRT results and semantics/status support across CPU levels.
TEST(FpSqrtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<32, true>(rj_arith_v_sqrt_f64_w32_gfx12, 2);
}

// Check FP64 SQRT results and semantics/status support across CPU levels.
TEST(FpSqrtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<64, false>(rj_arith_v_sqrt_f64_w64, 2);
}

// Check FP64 SQRT results and semantics/status support across CPU levels.
TEST(FpSqrtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_trans_f64<64, true>(rj_arith_v_sqrt_f64_w64_gfx12, 2);
}

// Guest denormal controls affect unary arithmetic on both architectures.
TEST(FpRcpF64, OutputDenormalControls) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_rcp_f64_w32, rj_arith_v_rcp_f64_w32_gfx12}) {
      Registers<4, 32> r;
      write_fp(r.v, 0, 64, 0x7fe8000000000000ULL);
      uint32_t status = 0x80000000U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1, 0,
                     r.v + 2, r.v, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64), 0U);
      EXPECT_EQ(status, 0x80000000U);
    }
  }
}

// RCP/RSQ/SQRT ignore guest rounding in both the native and the modelled core;
// output scaling still uses that field.
TEST(FpTransF64, GuestRounding) {
  using namespace rocjitsu::arith::test;
  for (auto call :
       {rj_arith_v_rcp_f64_w32_gfx12, rj_arith_v_rsq_f64_w32_gfx12, rj_arith_v_sqrt_f64_w32_gfx12,
        rj_arith_v_rcp_f64_w32, rj_arith_v_rsq_f64_w32, rj_arith_v_sqrt_f64_w32}) {
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
        Registers<4, 32> r;
        for (int lane = 0; lane < 32; ++lane) {
          write_fp(r.v, lane, 64, 0x4008000000000000ULL); // 3, nonexact for each core.
        }
        ASSERT_EQ(call(cpu | semantics, ~0U, 0, r.v + 2, r.v, nullptr), RJ_ARITH_SUCCESS);
        const uint64_t nearest = read_fp(r.v + 2, 0, 64);
        for (uint64_t rounding :
             {RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
          ASSERT_EQ(call(cpu | semantics | rounding, ~0U, 0, r.v + 2, r.v, nullptr),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(read_fp(r.v + 2, 0, 64), nearest);
        }
      }
    }
  }
}

namespace {

using rocjitsu::arith::test::Registers;
using rocjitsu::arith::test::WaveMask;

enum class Trans { Rcp, Rsq, Sqrt };

// One captured lane. mode is the hardware MODE register: F64 rounding in bits
// 3:2, F64 denormals in bits 7:6 (bit 6 keeps inputs, bit 7 keeps outputs) and
// IEEE in bit 9. Modifiers use the rj-arith instruction flags.
struct Captured {
  const char *name;
  Trans op;
  uint64_t modifiers, input, expected;
  uint32_t mode;
};

// Translate a MODE value into rj-arith guest flags for an F64 result.
uint64_t guest_flags(uint32_t mode) {
  uint64_t flags = uint64_t((mode >> 2) & 3) << 24;
  if (!(mode & 0x40)) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  }
  if (!(mode & 0x80)) {
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
  if (mode & 0x200) {
    flags |= RJ_ARITH_FP_IEEE_MODE;
  }
  return flags;
}

// gfx1201 lanes, including the power-of-two neighbors, subnormal sources and
// results, signaling NaNs, and OMOD overflow under round-toward-zero.
const Captured gfx12_lanes[] = {
    {"RcpApproximation", Trans::Rcp, 0, 0x4004000000000000, 0x3fd999999c000000, 0x30},
    {"RcpOneAndHalf", Trans::Rcp, 0, 0x3ff8000000000000, 0x3fe5555554000000, 0x00},
    {"RcpAllFractionBits", Trans::Rcp, 0, 0x3fdfffffffffffff, 0x4000000000000001, 0xc0},
    {"RcpLowFractionBits", Trans::Rcp, 0, 0x3fe0000000000001, 0x3fffffffffffffff, 0x30},
    {"RcpSubnormalResultC0", Trans::Rcp, 0, 0x7fefffffffffffff, 0x0004000000000001, 0xc0},
    {"RcpSubnormalResult30", Trans::Rcp, 0, 0x7fefffffffffffff, 0, 0x30},
    {"RcpSubnormalSourceC0", Trans::Rcp, 0, 0x800fffffffffffff, 0xffd000000af80000, 0xc0},
    {"RcpSubnormalSource30", Trans::Rcp, 0, 0x800fffffffffffff, 0xfff0000000000000, 0x30},
    {"RcpSignalingNan", Trans::Rcp, 0, 0x7ff4000012345678, 0x7ffc000012345678, 0x30},
    {"RcpAbs", Trans::Rcp, RJ_ARITH_ALU_ABS_A, 0xffefffffffffffff, 0x0004000000000001, 0xf0},
    {"RcpMul4OverflowTowardZero", Trans::Rcp, RJ_ARITH_ALU_OMOD_4, 0x0010000000000000,
     0x7ff0000000000000, 0xff},
    {"RcpDiv2", Trans::Rcp, RJ_ARITH_ALU_OMOD_HALF, 0xd4e86ac4255b6895, 0xaae4f80d140c0000, 0xf0},
    {"RcpClampNan", Trans::Rcp, RJ_ARITH_ALU_CLAMP, 0xfff8000000000000, 0, 0x00},
    {"RsqApproximation", Trans::Rsq, 0, 0x4004000000000000, 0x3fe43d1364000000, 0x30},
    {"RsqLowFractionBits", Trans::Rsq, 0, 0x3ff0000000000001, 0x3fefffffffffffff, 0xc0},
    {"RsqNegative", Trans::Rsq, 0, 0xbff0000000000000, 0xfff8000000000000, 0x30},
    {"RsqSubnormalSourceC0", Trans::Rsq, 0, 0x0000000000000001, 0x6180000000000000, 0xc0},
    {"RsqNegativeSubnormalC0", Trans::Rsq, 0, 0x800fffffffffffff, 0xfff8000000000000, 0xc0},
    {"RsqNegativeSubnormal30", Trans::Rsq, 0, 0x800fffffffffffff, 0xfff0000000000000, 0x30},
    {"RsqNeg", Trans::Rsq, RJ_ARITH_ALU_NEG_A, 0xbff8000000000000, 0x3fea20bd74000000, 0xf0},
    {"RsqMul4", Trans::Rsq, RJ_ARITH_ALU_OMOD_4, 0x3ff0000000000001, 0x400fffffffffffff, 0xf0},
    {"SqrtOne", Trans::Sqrt, 0, 0x3ff0000000000000, 0x3ff0000004000000, 0x30},
    {"SqrtApproximation", Trans::Sqrt, 0, 0x3fdfffffffffffff, 0x3fe6a09e5fba0000, 0xc0},
    {"SqrtNegative", Trans::Sqrt, 0, 0xbff0000000000000, 0xfff8000000000000, 0xc0},
    {"SqrtNegativeSubnormalC0", Trans::Sqrt, 0, 0x800fffffffffffff, 0xfff8000000000000, 0xc0},
    {"SqrtNegativeSubnormal30", Trans::Sqrt, 0, 0x800fffffffffffff, 0x8000000000000000, 0x30},
    {"SqrtDiv2", Trans::Sqrt, RJ_ARITH_ALU_OMOD_HALF, 0x400921fb54442d18, 0x3fec5bf895180000, 0xf0},
    {"SqrtClampSubnormal", Trans::Sqrt, RJ_ARITH_ALU_CLAMP, 0x002d3fe0d72f3af5, 0x200e9807c4200000,
     0xf0},
};

// gfx1100 lanes with IEEE clear: signaling NaNs pass through unquieted, and
// kept output denormals disable OMOD.
const Captured gfx11_lanes[] = {
    {"RcpSignalingNan", Trans::Rcp, 0, 0x7ff4000012345678, 0x7ff4000012345678, 0x00},
    {"RcpLowFractionBits", Trans::Rcp, 0, 0x3ff0000000000001, 0x3fefffffffffffff, 0xf0},
    {"RcpApproximation", Trans::Rcp, 0, 0xc128975800000000, 0xbeb4d20a27080000, 0x00},
    {"RcpDiv2", Trans::Rcp, RJ_ARITH_ALU_OMOD_HALF, 0xd4e86ac4255b6895, 0xaae4f80d140c0000, 0x00},
    {"RcpDiv2KeptDenormals", Trans::Rcp, RJ_ARITH_ALU_OMOD_HALF, 0xd4e86ac4255b6895,
     0xaaf4f80d140c0000, 0xf0},
    {"RsqApproximation", Trans::Rsq, 0, 0x3e5eefe7dc2ef85f, 0x40b703488cf00000, 0x00},
    {"RsqMul4", Trans::Rsq, RJ_ARITH_ALU_OMOD_4, 0x41617baee1da94f1, 0x3f55a57930560000, 0x00},
    {"RsqMul4KeptDenormals", Trans::Rsq, RJ_ARITH_ALU_OMOD_4, 0x41617baee1da94f1,
     0x3f35a57930560000, 0xf0},
    {"SqrtSignalingNan", Trans::Sqrt, 0, 0x7ff4000012345678, 0x7ff4000012345678, 0xf0},
    {"SqrtApproximation", Trans::Sqrt, 0, 0x430fc0b20d0953d4, 0x417fe0494b5c0000, 0x00},
    {"SqrtClampNan", Trans::Sqrt, RJ_ARITH_ALU_CLAMP, 0x7ff0000000000001, 0, 0x00},
};

template <int Lanes>
using Call = int (*)(uint64_t, WaveMask<Lanes>, uint64_t, uint32_t *const *,
                     const uint32_t *const *, uint32_t *);

template <int Lanes> struct Calls {
  Call<Lanes> rcp, rsq, sqrt;
  Call<Lanes> operator[](Trans op) const {
    return op == Trans::Rcp ? rcp : op == Trans::Rsq ? rsq : sqrt;
  }
};

constexpr Calls<32> gfx12_w32{rj_arith_v_rcp_f64_w32_gfx12, rj_arith_v_rsq_f64_w32_gfx12,
                              rj_arith_v_sqrt_f64_w32_gfx12};
constexpr Calls<64> gfx12_w64{rj_arith_v_rcp_f64_w64_gfx12, rj_arith_v_rsq_f64_w64_gfx12,
                              rj_arith_v_sqrt_f64_w64_gfx12};
constexpr Calls<32> gfx11_w32{rj_arith_v_rcp_f64_w32, rj_arith_v_rsq_f64_w32,
                              rj_arith_v_sqrt_f64_w32};
constexpr Calls<64> gfx11_w64{rj_arith_v_rcp_f64_w64, rj_arith_v_rsq_f64_w64,
                              rj_arith_v_sqrt_f64_w64};

// Every exact semantics mode, and Compatibility, reproduces each captured lane
// in every wave lane at every CPU level.
template <int Lanes, size_t N>
void check_captured(const Calls<Lanes> &calls, const Captured (&lanes)[N]) {
  for (const Captured &c : lanes) {
    SCOPED_TRACE(c.name);
    for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                 RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
        Registers<4, Lanes> r;
        for (int lane = 0; lane < Lanes; ++lane) {
          rocjitsu::arith::test::write_fp(r.v, lane, 64, c.input);
        }
        ASSERT_EQ(calls[c.op](cpu | semantics | guest_flags(c.mode), ~WaveMask<Lanes>(0),
                              c.modifiers, r.v + 2, r.v, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < Lanes; ++lane) {
          ASSERT_EQ(rocjitsu::arith::test::read_fp(r.v + 2, lane, 64), c.expected)
              << cpu << '/' << semantics << '/' << lane;
        }
      }
    }
  }
}

} // namespace

// Exact and Compatibility semantics reproduce captured gfx1201 lanes.
TEST(FpTransF64Gfx12, CapturedLanes) {
  check_captured(gfx12_w32, gfx12_lanes);
  check_captured(gfx12_w64, gfx12_lanes);
}

// Exact and Compatibility semantics reproduce captured gfx1100 lanes.
TEST(FpTransF64Gfx11, CapturedLanes) {
  check_captured(gfx11_w32, gfx11_lanes);
  check_captured(gfx11_w64, gfx11_lanes);
}

TEST(FpNativeMathCdna, Arithmetic) {
  using namespace rocjitsu::arith::test;
  for (auto call : {rj_arith_v_rcp_f64_w32_gfx942, rj_arith_v_rcp_f64_w32_gfx1250}) {
    check_loose_unary_f64<32>(
        call, [](double x) { return 1.0 / x; }, call == rj_arith_v_rcp_f64_w32_gfx1250, false);
  }
  for (auto call : {rj_arith_v_rsq_f64_w32_gfx942, rj_arith_v_rsq_f64_w32_gfx1250}) {
    check_loose_unary_f64<32>(
        call, [](double x) { return 1.0 / std::sqrt(x); }, call == rj_arith_v_rsq_f64_w32_gfx1250,
        false);
  }
  for (auto call : {rj_arith_v_sqrt_f64_w32_gfx942, rj_arith_v_sqrt_f64_w32_gfx1250}) {
    check_loose_unary_f64<32>(
        call, [](double x) { return std::sqrt(x); }, call == rj_arith_v_sqrt_f64_w32_gfx1250,
        false);
  }
  check_loose_unary_f64<64>(
      rj_arith_v_rcp_f64_w64_gfx942, [](double x) { return 1.0 / x; }, false, false);
  check_loose_unary_f64<64>(
      rj_arith_v_rsq_f64_w64_gfx942, [](double x) { return 1.0 / std::sqrt(x); }, false, false);
  check_loose_unary_f64<64>(
      rj_arith_v_sqrt_f64_w64_gfx942, [](double x) { return std::sqrt(x); }, false, false);
  // OMOD uses the inherited scale-then-flush order, including a subnormal
  // reciprocal promoted to a normal value. Guest rounding does not alter this core.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_rcp_f64_w32_gfx942, rj_arith_v_rcp_f64_w32_gfx1250}) {
      for (double x : {0x1.8p1022, -0x1.8p1022, 0x1.8p1023, -0x1.8p1023, -0.0}) {
        Registers<4, 32> r;
        write_fp(r.v, 0, 64, fp_bits(x, 64));
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                               RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_DOWN;
        ASSERT_EQ(call(flags, 1, RJ_ARITH_ALU_OMOD_2, r.v + 2, r.v, nullptr), RJ_ARITH_SUCCESS);
        double expected = (1.0 / x) * 2.0;
        if (std::fpclassify(expected) == FP_SUBNORMAL || expected == 0) {
          expected = 0;
        }
        EXPECT_EQ(read_fp(r.v + 2, 0, 64), fp_bits(expected, 64));
      }
    }
  }
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (auto call : {rj_arith_v_rcp_f64_w32_gfx942, rj_arith_v_rcp_f64_w32_gfx1250,
                        rj_arith_v_rsq_f64_w32_gfx942, rj_arith_v_rsq_f64_w32_gfx1250,
                        rj_arith_v_sqrt_f64_w32_gfx942, rj_arith_v_sqrt_f64_w32_gfx1250}) {
        Registers<6, 32> r;
        write_fp(r.v, 0, 64, 0x7fe8000000000000ULL); // Reciprocal is subnormal.
        write_fp(r.v, 1, 64, 0x0008000000000000ULL); // Subnormal core input.
        uint32_t status = 0x80000000U;
        ASSERT_EQ(call(cpu | semantics, 3, 0, r.v + 2, r.v, &status), RJ_ARITH_SUCCESS);
        ASSERT_EQ(call(cpu | semantics | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                           RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                       3, 0, r.v + 4, r.v, &status),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 2; ++lane) {
          const bool reciprocal =
              call == rj_arith_v_rcp_f64_w32_gfx942 || call == rj_arith_v_rcp_f64_w32_gfx1250;
          uint64_t expected = read_fp(r.v + 2, lane, 64);
          if (lane == 1) {
            const bool root =
                call == rj_arith_v_sqrt_f64_w32_gfx942 || call == rj_arith_v_sqrt_f64_w32_gfx1250;
            expected = root ? 0 : 0x7ff0000000000000ULL;
          } else if (reciprocal && (expected & 0x7ff0000000000000ULL) == 0) {
            expected &= 0x8000000000000000ULL;
          }
          EXPECT_EQ(read_fp(r.v + 4, lane, 64), expected);
        }
        EXPECT_EQ(status, 0x80000000U);
      }
    }
  }
}

TEST(FpNativeMathCdna, CpuPaths) {
  using namespace rocjitsu::arith::test;
  const uint64_t inputs[] = {0x3ff8000000000000ULL, 1,
                             0x7fefffffffffffffULL, 0x8000000000000000ULL,
                             0x7ff0000000000000ULL, 0x7ff0000000001234ULL,
                             0xc010000000000000ULL};
  for (auto call : {rj_arith_v_rcp_f64_w32_gfx942, rj_arith_v_rcp_f64_w32_gfx1250,
                    rj_arith_v_rsq_f64_w32_gfx942, rj_arith_v_rsq_f64_w32_gfx1250,
                    rj_arith_v_sqrt_f64_w32_gfx942, rj_arith_v_sqrt_f64_w32_gfx1250}) {
    const uint64_t modes[] = {0, RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_2,
                              RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP};
    for (uint64_t mode : modes) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_LOOSE}) {
        for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
          const uint64_t flags = semantics | ieee | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                 RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_UP;
          Registers<6, 32> r;
          for (int lane = 0; lane < 32; ++lane) {
            write_fp(r.v, lane, 64, inputs[lane % 7]);
          }
          ASSERT_EQ(call(flags, ~0U, mode, r.v + 2, r.v, nullptr), RJ_ARITH_SUCCESS);
          for (uint64_t cpu : supported_cpu_levels()) {
            ASSERT_EQ(call(flags | cpu, ~0U, mode, r.v + 4, r.v, nullptr), RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              const uint64_t expected = read_fp(r.v + 2, lane, 64);
              const uint64_t actual = read_fp(r.v + 4, lane, 64);
              if (std::isnan(fp_value(expected, 64)) && lane % 7 != 5) {
                EXPECT_TRUE(std::isnan(fp_value(actual, 64)));
              } else {
                EXPECT_EQ(actual, expected) << cpu << ':' << mode << ':' << lane;
              }
            }
          }
        }
      }
    }
  }
}

// CDNA's native-double model honors reciprocal output flushing and SQRT's IEEE NaN rule.
TEST(FpNativeF64Cdna, OutputDenormalsAndSignalingNan) {
  using namespace rocjitsu::arith::test;
  for (auto cpu : supported_cpu_levels()) {
    for (bool ieee : {false, true}) {
      Registers<4, 32> r;
      r.v[0][0] = 1;
      r.v[1][0] = 0x7ff40000U;
      for (auto call : {rj_arith_v_rcp_f64_w32_gfx942, rj_arith_v_rsq_f64_w32_gfx942,
                        rj_arith_v_sqrt_f64_w32_gfx942, rj_arith_v_rcp_f64_w32_gfx1250,
                        rj_arith_v_rsq_f64_w32_gfx1250, rj_arith_v_sqrt_f64_w32_gfx1250}) {
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (ieee ? RJ_ARITH_FP_IEEE_MODE : 0),
                       1, 0, r.v + 2, r.v, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0], 1U);
        const bool gfx1250 = call == rj_arith_v_rcp_f64_w32_gfx1250 ||
                             call == rj_arith_v_rsq_f64_w32_gfx1250 ||
                             call == rj_arith_v_sqrt_f64_w32_gfx1250;
        EXPECT_EQ(r.v[3][0], ieee || gfx1250 ? 0x7ffc0000U : 0x7ff40000U);
      }
    }
    for (bool flush : {false, true}) {
      Registers<4, 32> r;
      r.v[0][0] = 0;
      r.v[1][0] = 0x7fe00000U; // 2^1023, whose reciprocal is subnormal.
      for (auto call : {rj_arith_v_rcp_f64_w32_gfx942, rj_arith_v_rcp_f64_w32_gfx1250}) {
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                           (flush ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0),
                       1, 0, r.v + 2, r.v, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0], 0U);
        EXPECT_EQ(r.v[3][0], flush ? 0U : 0x00080000U);
      }
    }
  }
}
