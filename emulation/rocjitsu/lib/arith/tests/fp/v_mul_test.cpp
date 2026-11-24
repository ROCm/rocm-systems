// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point multiplication.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/binary_arithmetic_test.h"
#include "fp/captured_gfx11_test.h"
#include "fp/elementwise.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpMulF16W32Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32_gfx10, [](double x, double y) { return x * y; },
      /*gfx12=*/false, /*fp_status_supported=*/false, /*compatibility_status=*/false,
      /*half_exact=*/false);
}

// RDNA2 keeps the RDNA3 model but refuses strict exact until its captures are reviewed.
TEST(FpMulF16W64Gfx10, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64_gfx10, [](double x, double y) { return x * y; },
      /*gfx12=*/false, /*fp_status_supported=*/false, /*compatibility_status=*/false,
      /*half_exact=*/false);
}

namespace {

// Selected gfx1201 default-mode FP exception contributions.
// Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
struct StatusCase {
  uint32_t a, b;
  uint32_t causes;
};

const StatusCase status_cases_f32[] = {
    {0x3f800000U, 0x40000000U, 0},
    {0x00ffffffU, 0x3f000000U, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f118e00U, 0x00e12000U, RJ_ARITH_EXCEPTION_INEXACT},
    {0x00000001U, 0x7fc00000U, 0},
    {0x00000001U, 0x7f800001U, RJ_ARITH_EXCEPTION_INVALID},
    {0x0U, 0x7f800000U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7f800001U, 0x3f800000U, RJ_ARITH_EXCEPTION_INVALID},
    {0x1U, 0x3f800000U, RJ_ARITH_EXCEPTION_INPUT_DENORM},
    {0x1U, 0x3f000000U,
     RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x7f7fffffU, 0x40000000U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f800001U, 0x3f800001U, RJ_ARITH_EXCEPTION_INEXACT},
};

// Check inherited MUL exception causes and accumulation, including empty EXEC.
template <int Lanes, typename Call> void check_status(Call call) {
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  using Registers = rocjitsu::arith::test::Registers<10, Lanes>;
  Registers input = rocjitsu::arith::test::make_fp_inputs<32, Lanes>();
  const auto cpu_levels = rocjitsu::arith::test::supported_cpu_levels();
  for (uint64_t cpu : cpu_levels) {
    for (const auto &test : status_cases_f32) {
      for (uint64_t instruction_flags : {uint64_t(0)}) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << instruction_flags << '/' << test.a << '/' << test.b);
        Registers r = input;
        const int lane = Lanes - 1;
        rocjitsu::arith::test::write_fp(r.v, lane, 32, test.a);
        rocjitsu::arith::test::write_fp(r.v + 2, lane, 32, test.b);
        Mask exec_mask = Mask(1) << lane;
        uint32_t fp_status = 0x80000000U;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, instruction_flags,
                       r.v + 6, r.v, r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        uint32_t causes = test.causes;
        EXPECT_EQ(fp_status, 0x80000000U | causes);
        // Preserve previously accumulated causes as well as unrelated bits.
        fp_status = 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, instruction_flags,
                       r.v + 6, r.v, r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
        Registers before = r;
        exec_mask = 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, instruction_flags,
                       r.v + 6, r.v, r.v + 2, &fp_status),
                  RJ_ARITH_SUCCESS);
        r.expect_preserved(before, 0, 0);
        EXPECT_EQ(fp_status, 0x40000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0 | causes);
      }
    }
  }
}

} // namespace

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_mul_f32_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true,
      /*fp_status_supported=*/true, /*compatibility_status=*/true);
  check_status<32>(rj_arith_v_mul_f32_w32_gfx12);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_mul_f32_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true,
      /*fp_status_supported=*/true, /*compatibility_status=*/true);
  check_status<64>(rj_arith_v_mul_f32_w64_gfx12);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_mul_f32_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false,
      /*fp_status_supported=*/false, /*compatibility_status=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_mul_f64_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_mul_f64_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_mul_f64_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpMulF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_mul_f64_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Reject unqualified status domains before touching outputs; on gfx12 Compatibility reports
// the shared hardware classifier even where explicit mode 3 is unqualified: CLAMP suppresses
// every cause, while an output modifier suppresses only UNDERFLOW and INEXACT and leaves the
// operand causes in place (gfx1201 captures).
TEST(FpMulF32, ReportingDomain) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_mul_f32_w32, rj_arith_v_mul_f32_w32_gfx12}) {
      for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS,
                          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
        for (uint32_t instruction_flags : {0U, RJ_ARITH_ALU_CLAMP, RJ_ARITH_ALU_OMOD_2}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x7f7fffffU;
          r.v[1][0] = 0x40000000U;
          r.v[2][0] = 0xdeadbeefU;
          uint32_t status = 0x80000000U;
          if (call != rj_arith_v_mul_f32_w32_gfx12 || fp || instruction_flags) {
            EXPECT_EQ(call(cpu | fp | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 1, instruction_flags,
                           nullptr, nullptr, nullptr, &status),
                      RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
          }
          ASSERT_EQ(call(cpu | fp | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, instruction_flags, r.v + 2,
                         r.v, r.v + 1, &status),
                    RJ_ARITH_SUCCESS);
          // The inherited RDNA3 classifier reads the raw operands, so neither CLAMP nor an
          // output modifier changes what this overflowing product reports there.
          const uint32_t causes = call != rj_arith_v_mul_f32_w32_gfx12 ? 0x28U
                                  : instruction_flags & RJ_ARITH_ALU_CLAMP
                                      ? 0U
                                      : ((instruction_flags & RJ_ARITH_ALU_OMOD_MASK) ? 8U : 0x28U);
          EXPECT_EQ(status, 0x80000000U | causes);
        }
      }
    }
  }
}

namespace {

// gfx1201 V_MUL_F32 captures (R9700, 2026-11-18, ~/scratch/trap-status/s1-valu/hw1201:
// EXCP_FLAG_USER cleared before and read after one lane's instruction). MODE words carry
// the F32 rounding field in bits [1:0] and the F32 denormal field in bits [5:4].
// The rows pin the shared classifier where a stored-result or native-nearest rule differs:
// the tininess midpoint under every rounding mode, the sign-dependent direction of guest
// rounding, flushed denormal inputs, NaN suppression of INPUT_DENORM, OVERFLOW before and
// after an output modifier, and CLAMP suppressing every cause.
struct MulCapture {
  uint32_t mode, a, b, modifiers, d, causes;
};

const MulCapture mul_f32_captures[] = {
    // (1 - 2^-24) * 2^-126 stores the minimum normal in nearest/up modes yet is tiny.
    {0xf0, 0x3f7fffff, 0x00800000, 0, 0x00800000, 0x30},
    {0xf5, 0x3f7fffff, 0x00800000, 0, 0x00800000, 0x30},
    {0xfa, 0x3f7fffff, 0x00800000, 0, 0x007fffff, 0x30},
    {0xff, 0x3f7fffff, 0x00800000, 0, 0x007fffff, 0x30},
    {0x00, 0x3f7fffff, 0x00800000, 0, 0x00000000, 0x30},
    {0xf0, 0x00ffffff, 0x3f000000, 0, 0x00800000, 0x30},
    // (1 + 2^-23)(1 - 2^-24) * 2^-126 rounds to the minimum normal from above: not tiny.
    {0xf0, 0x00800001, 0x3f7fffff, 0, 0x00800000, 0x20},
    {0xf0, 0x3f800000, 0x00800000, 0, 0x00800000, 0x00},
    // The signed product picks the direction: upward rounding of a negative tiny product
    // truncates below the minimum normal, of a positive one reaches it.
    {0xf5, 0x807fffff, 0x3f800001, 0, 0x807fffff, 0x32},
    {0x15, 0x807fffff, 0x3f800001, 0, 0x80000000, 0x32},
    {0xf5, 0x007fffff, 0x3f800001, 0, 0x00800000, 0x22},
    {0xf5, 0x007fffff, 0x3f800001, RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B, 0x807fffff, 0x32},
    {0xf0, 0x807fffff, 0x3f800001, 0, 0x80800000, 0x22},
    // Input flushing turns the denormal factor into zero; nothing is raised.
    {0xf0, 0x00000001, 0x3f000000, 0, 0x00000000, 0x32},
    {0x00, 0x00000001, 0x3f000000, 0, 0x00000000, 0x00},
    {0x50, 0x00000001, 0x3f000000, 0, 0x00000000, 0x32},
    // NaN operands suppress INPUT_DENORM; signaling NaNs and 0 * inf raise INVALID.
    {0xf0, 0x00000001, 0x7fc00000, 0, 0x7fc00000, 0x00},
    {0xf0, 0x7f800001, 0x00000001, 0, 0x7fc00001, 0x01},
    {0xf0, 0x00000000, 0x7f800000, 0, 0xffc00000, 0x01},
    // Overflow in every rounding mode, before or after an output modifier; OMOD drops
    // UNDERFLOW and INEXACT.
    {0xf0, 0x7f7fffff, 0x40000000, 0, 0x7f800000, 0x28},
    {0xff, 0x7f7fffff, 0x40000000, 0, 0x7f7fffff, 0x28},
    {0xf0, 0x7f000000, 0x3f800001, RJ_ARITH_ALU_OMOD_2, 0x7f800000, 0x08},
    {0xf0, 0x7f7fffff, 0x40000000, RJ_ARITH_ALU_OMOD_HALF, 0x7f800000, 0x08},
    {0xf0, 0x3f7fffff, 0x00800000, RJ_ARITH_ALU_OMOD_2, 0x00000000, 0x00},
    {0xf0, 0x00000001, 0x3f000000, RJ_ARITH_ALU_OMOD_2, 0x00000000, 0x02},
    // CLAMP suppresses every cause.
    {0xf0, 0x7f7fffff, 0x40000000, RJ_ARITH_ALU_CLAMP, 0x3f800000, 0x00},
    {0xf0, 0x00000000, 0x7f800000, RJ_ARITH_ALU_CLAMP, 0x00000000, 0x00},
    {0xf0, 0x7f800001, 0x00000001, RJ_ARITH_ALU_CLAMP, 0x00000000, 0x00},
};

// Translate a captured MODE word into the FP32 flags rocjitsu passes.
uint64_t mul_capture_flags(uint32_t mode) {
  uint64_t flags = uint64_t(mode & 3) << 24;
  if (!((mode >> 4) & 1)) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  }
  if (!((mode >> 4) & 2)) {
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
  return flags;
}

// Check captured values and causes under Compatibility on a lane in a later SIMD chunk, and
// under explicit mode 3 where it is qualified (default MODE, no OMOD/CLAMP), which must
// report the same bits. Only the gfx12 entry points carry this classifier: gfx11 has no
// status capture, because TRAPSTS.EXCP does not latch there with exceptions disabled.
template <int Lanes, typename Call> void check_mul_captures(Call call, bool gfx12) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr int lane = Lanes - 13;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &test : mul_f32_captures) {
      const uint64_t fp = mul_capture_flags(test.mode) | (gfx12 ? 0 : RJ_ARITH_FP_IEEE_MODE);
      SCOPED_TRACE(::testing::Message()
                   << std::hex << cpu << " mode 0x" << test.mode << ": 0x" << test.a << ", 0x"
                   << test.b << " mods 0x" << test.modifiers);
      const bool qualified = !mul_capture_flags(test.mode) &&
                             !(test.modifiers & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP));
      for (uint64_t semantics :
           {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
        if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL && (!gfx12 || !qualified)) {
          continue;
        }
        Registers<3, Lanes> r;
        r.v[0][lane] = test.a;
        r.v[1][lane] = test.b;
        r.v[2][lane] = 0xdeadbeefU;
        uint32_t status = 0x80000000U;
        ASSERT_EQ(call(cpu | semantics | fp, Mask(1) << lane, test.modifiers, r.v + 2, r.v, r.v + 1,
                       &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][lane], test.d);
        EXPECT_EQ(status, 0x80000000U | test.causes);
      }
    }
  }
}

} // namespace

TEST(FpMulF32Gfx12, Gfx1201StatusCaptures) {
  check_mul_captures<32>(rj_arith_v_mul_f32_w32_gfx12, /*gfx12=*/true);
  check_mul_captures<64>(rj_arith_v_mul_f32_w64_gfx12, /*gfx12=*/true);
}

namespace {

// RDNA3 Compatibility causes, which keep the inherited classifier until a gfx1100 capture
// establishes the gfx1201 rule there. Both rows differ from that rule, so unifying the two
// architectures changes what an RDNA3 guest with the matching trap enabled runs into:
// decoded MUL passes these causes to TRAPSTS and trap delivery.
struct Gfx11Cause {
  uint64_t fp;
  uint32_t a, b, modifiers, causes;
};

const Gfx11Cause gfx11_inherited_causes[] = {
    // A clamped multiply that overflows in IEEE mode still reports it; the gfx1201 rule
    // suppresses every cause under CLAMP.
    {RJ_ARITH_FP_IEEE_MODE, 0x7f7fffff, 0x40000000, RJ_ARITH_ALU_CLAMP,
     RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    // An inexact product scaled by an output modifier still reports INEXACT; the gfx1201
    // rule keeps only OVERFLOW there.
    {RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0x3f800001, 0x3f800001, RJ_ARITH_ALU_OMOD_2,
     RJ_ARITH_EXCEPTION_INEXACT},
};

} // namespace

// Pin the two causes above on the RDNA3 entry points, and check the gfx12 entry points give
// the requalified answer for the same inputs, so re-unifying the two is a test failure
// rather than a silent change in RDNA3 trap delivery.
TEST(FpMulF32Gfx11, KeepsInheritedCompatibilityCauses) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const Gfx11Cause &test : gfx11_inherited_causes) {
      SCOPED_TRACE(::testing::Message() << std::hex << cpu << ": 0x" << test.a << ", 0x" << test.b
                                        << " mods 0x" << test.modifiers);
      for (bool gfx12 : {false, true}) {
        Registers<3, 32> r;
        r.v[0][0] = test.a;
        r.v[1][0] = test.b;
        r.v[2][0] = 0xdeadbeefU;
        uint32_t status = 0x80000000U;
        auto call = gfx12 ? rj_arith_v_mul_f32_w32_gfx12 : rj_arith_v_mul_f32_w32;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | test.fp, 1, test.modifiers, r.v + 2,
                       r.v, r.v + 1, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(status, 0x80000000U | (gfx12 ? 0U : test.causes)) << "gfx12 " << gfx12;
      }
    }
  }
}

namespace {

// Inexact positive/negative products distinguish directed rounding without a
// computed FP reference. All lanes are finite so SIMD stays eligible.
template <int Lanes, typename Call> void check_mul_rounding(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (uint64_t rounding :
           {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
        for (bool negative : {false, true}) {
          SCOPED_TRACE(testing::Message()
                       << cpu << '/' << semantics << '/' << rounding << '/' << negative);
          Registers<3, Lanes> r;
          const uint32_t sign = negative ? 0x80000000U : 0;
          for (int lane = 0; lane < Lanes; ++lane) {
            r.v[0][lane] = sign | 0x3f800001U;
            r.v[1][lane] = 0x3f800001U;
          }
          uint32_t status = 0;
          ASSERT_EQ(call(cpu | semantics | rounding, ~Mask(0), 0, r.v + 2, r.v, r.v + 1, &status),
                    RJ_ARITH_SUCCESS);
          const bool away = rounding == (negative ? RJ_ARITH_FP_ROUND_DOWN : RJ_ARITH_FP_ROUND_UP);
          const uint32_t expected = sign | (away ? 0x3f800003U : 0x3f800002U);
          for (int lane = 0; lane < Lanes; ++lane) {
            EXPECT_EQ(r.v[2][lane], expected);
          }
        }
      }
    }
  }
}

} // namespace

TEST(FpMulF32, DirectedRounding) {
  check_mul_rounding<32>(rj_arith_v_mul_f32_w32);
  check_mul_rounding<32>(rj_arith_v_mul_f32_w32_gfx12);
  check_mul_rounding<64>(rj_arith_v_mul_f32_w64);
  check_mul_rounding<64>(rj_arith_v_mul_f32_w64_gfx12);
}

// Hardware rounds to FP16 before OMOD, so 65504 * 2 halves to infinity (or, rounding toward
// zero or down, to 65504 / 2); Loose can keep the finite wide product. Expected results are
// gfx1201 captures (rj-valu-audit-local hw-r9700-wide v_mul_f16__div2, 2026-11-18).
TEST(FpMulF16, OmodRounding) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_mul_f16_w32, rj_arith_v_mul_f16_w32_gfx12}) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
        for (uint64_t rounding :
             {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x7bffU;
          r.v[1][0] = 0x4000U;
          uint32_t status = 0;
          ASSERT_EQ(call(cpu | semantics | rounding | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1,
                         RJ_ARITH_ALU_OMOD_HALF, r.v + 2, r.v, r.v + 1, &status),
                    RJ_ARITH_SUCCESS);
          uint32_t expected =
              semantics == RJ_ARITH_SEMANTICS_LOOSE                                      ? 0x7bffU
              : rounding == RJ_ARITH_FP_ROUND_DOWN || rounding == RJ_ARITH_FP_ROUND_ZERO ? 0x77ffU
                                                                                         : 0x7c00U;
          EXPECT_EQ(r.v[2][0] & 0xffffU, expected);
          EXPECT_EQ(status, 0U);
        }
      }
    }
  }
}

// Tininess is judged after rounding. (1 - 2^-11) * 2^-14 is exact at FP16 precision, so it is
// tiny in every rounding mode although subnormal rounding can reach the smallest normal;
// (1 - 2^-20) * 2^-14 is not tiny unless rounded down. Flushing outputs and OMOD zero tiny
// results, OMOD even when MODE keeps denormals. Halving gives a signed zero only when the
// rounded result is below 2^-13. Expected results are gfx1201 captures in nearest, up, down
// and zero rounding (~0U: not captured); gfx1100 flushes the same products.
TEST(FpMulF16, TinyAfterRounding) {
  using namespace rocjitsu::arith::test;
  constexpr uint32_t not_captured = ~0U;
  const struct {
    uint32_t a, b;
    uint64_t fp_flags;
    uint32_t instruction_flags;
    uint32_t expected[4];
  } cases[] = {
      {0x3bff, 0x0400, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, {0, 0, 0, 0}},
      {0x3bff, 0x8400, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, {0x8000, 0x8000, 0x8000, 0x8000}},
      {0x3bff, 0x0400, 0, 0, {0x0400, 0x0400, 0x03ff, 0x03ff}},
      {0x3bfe, 0x0401, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, {0x0400, 0x0400, 0, 0}},
      {0x3bff, 0x0400, 0, RJ_ARITH_ALU_OMOD_2, {0, 0, 0, 0}},
      {0x3bfe, 0x0401, 0, RJ_ARITH_ALU_OMOD_2, {0x0800, 0x0800, 0, 0}},
      {0x3800, 0x87ff, 0, RJ_ARITH_ALU_OMOD_HALF, {0, 0, 0, 0}},
      {0x9873, 0x2b31, 0, RJ_ARITH_ALU_OMOD_HALF, {0x8400, not_captured, not_captured, 0x8000}},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (const auto &test : cases) {
        for (unsigned rounding = 0; rounding < 4; ++rounding) {
          if (test.expected[rounding] == not_captured) {
            continue;
          }
          const uint64_t flags = cpu | semantics | test.fp_flags | (uint64_t(rounding) << 24);
          // gfx11 applies OMOD only when output denormals are flushed.
          for (auto call : {rj_arith_v_mul_f16_w32, rj_arith_v_mul_f16_w32_gfx12}) {
            if (call == rj_arith_v_mul_f16_w32 && test.instruction_flags) {
              continue;
            }
            Registers<3, 32> r;
            r.v[0][0] = test.a;
            r.v[1][0] = test.b;
            r.v[2][0] = 0xa5a50000U;
            ASSERT_EQ(call(flags, 1, test.instruction_flags, r.v + 2, r.v, r.v + 1, nullptr),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(r.v[2][0], 0xa5a50000U | test.expected[rounding])
                << std::hex << test.a << " * " << test.b << " rounding " << rounding;
          }
        }
      }
    }
  }
}

TEST(FpMulF64, GuestRounding) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const struct {
    uint64_t a, b, expected[4];
  } cases[] = {{0x3ff0000000000001ULL,
                0x3ff0000000000001ULL,
                {0x3ff0000000000002ULL, 0x3ff0000000000003ULL, 0x3ff0000000000002ULL,
                 0x3ff0000000000002ULL}},
               {0xbff0000000000001ULL,
                0x3ff0000000000001ULL,
                {0xbff0000000000002ULL, 0xbff0000000000002ULL, 0xbff0000000000003ULL,
                 0xbff0000000000002ULL}},
               {0x8000000000000000ULL,
                0x3ff0000000000000ULL,
                {0x8000000000000000ULL, 0x8000000000000000ULL, 0x8000000000000000ULL,
                 0x8000000000000000ULL}},
               {0x7fefffffffffffffULL,
                0x4000000000000000ULL,
                {0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7fefffffffffffffULL,
                 0x7fefffffffffffffULL}}};
  for (const auto &sample : cases) {
    check_binary64_rounding<32>(rj_arith_v_mul_f64_w32, sample.a, sample.b, sample.expected, 0, 0,
                                false);
    check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, sample.a, sample.b, sample.expected);
    check_binary64_rounding<64>(rj_arith_v_mul_f64_w64, sample.a, sample.b, sample.expected, 0, 0,
                                false);
    check_binary64_rounding<64>(rj_arith_v_mul_f64_w64_gfx12, sample.a, sample.b, sample.expected);
  }
}

TEST(FpMulF64, OutputScalingAndFlushing) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const uint64_t overflow[] = {0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7fefffffffffffffULL,
                               0x7fefffffffffffffULL};
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, 0x7fefffffffffffffULL,
                              0x3ff0000000000000ULL, overflow, RJ_ARITH_ALU_OMOD_2);
}

// Tininess is judged after rounding. (1 - 2^-53) * 2^-1022 is exact at full precision, so it
// is tiny in every rounding mode although subnormal rounding can reach the smallest normal;
// (1 - 2^-52) * (2^-1022 + 2^-1074) is not tiny unless rounded down. Flushing outputs and OMOD
// zero tiny results, OMOD even when MODE keeps denormals. Expected results are gfx1201
// captures in nearest, up, down and zero rounding; gfx1100 flushes the same products.
TEST(FpMulF64, TinyAfterRounding) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const uint64_t tiny = 0x3fefffffffffffffULL, not_tiny = 0x3feffffffffffffeULL;
  const uint64_t zero[] = {0, 0, 0, 0};
  const uint64_t kept[] = {0x0010000000000000ULL, 0x0010000000000000ULL, 0x000fffffffffffffULL,
                           0x000fffffffffffffULL};
  const uint64_t not_tiny_flushed[] = {0x0010000000000000ULL, 0x0010000000000000ULL, 0, 0};
  const uint64_t not_tiny_doubled[] = {0x0020000000000000ULL, 0x0020000000000000ULL, 0, 0};
  for (bool gfx12 : {false, true}) {
    auto w32 = gfx12 ? rj_arith_v_mul_f64_w32_gfx12 : rj_arith_v_mul_f64_w32;
    auto w64 = gfx12 ? rj_arith_v_mul_f64_w64_gfx12 : rj_arith_v_mul_f64_w64;
    check_binary64_rounding<32>(w32, tiny, 0x0010000000000000ULL, zero, 0,
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, gfx12);
    check_binary64_rounding<32>(w32, not_tiny, 0x0010000000000001ULL, not_tiny_flushed, 0,
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, gfx12);
    check_binary64_rounding<64>(w64, tiny, 0x0010000000000000ULL, zero, 0,
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, gfx12);
  }
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, tiny, 0x0010000000000000ULL, kept);
  check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, tiny, 0x0010000000000000ULL, zero,
                              RJ_ARITH_ALU_OMOD_2);
  check_binary64_rounding<64>(rj_arith_v_mul_f64_w64_gfx12, not_tiny, 0x0010000000000001ULL,
                              not_tiny_doubled, RJ_ARITH_ALU_OMOD_2);
}

// gfx1201 captures: FP64 MUL returns the first NaN source, quieted, in operand order under every
// guest rounding mode.
TEST(FpMulF64, NanPriority) {
  using rocjitsu::arith::test::check_binary64_rounding;
  const uint64_t cases[][3] = {
      {0x7ff0000000000001ULL, 0x7ff8000000000000ULL, 0x7ff8000000000001ULL},
      {0xfff8000000000000ULL, 0x7ff8000000000000ULL, 0xfff8000000000000ULL},
      {0x7ff4000012345678ULL, 0x7ff8000000000000ULL, 0x7ffc000012345678ULL},
      {0x7ff8000000000000ULL, 0x7ff0000000000001ULL, 0x7ff8000000000000ULL},
      {0xfff8000000000000ULL, 0x7ff0000000000001ULL, 0xfff8000000000000ULL},
  };
  for (const auto &sample : cases) {
    const uint64_t expected[] = {sample[2], sample[2], sample[2], sample[2]};
    check_binary64_rounding<32>(rj_arith_v_mul_f64_w32_gfx12, sample[0], sample[1], expected);
    check_binary64_rounding<64>(rj_arith_v_mul_f64_w64_gfx12, sample[0], sample[1], expected);
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 lanes from the MODE.IEEE-clear corpus (hw1100-noieee): a signaling NaN source is
// returned unquieted, also under directed rounding and OMOD.
const CapturedGfx11 mul_f16_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x6b007c01, 0x4f160000, 0, 0xa5a57c01},
    {"E32RoundDown", 0, 0x15a, 0x323e0000, 0xd51d7c01, 0, 0xa5a57c01},
    {"NegB", RJ_ARITH_ALU_NEG_B, 0x1f0, 0xde7e0000, 0xa4cc7c01, 0, 0xa5a5fc01},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x32a07c01, 0xab200000, 0, 0xa5a57c01},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0x1f0,
     0xfdc50400, 0x5e2a0000, 0, 0xfdc5a5a5},
};

// gfx1100 (W7900) CLAMP lanes with a signaling-NaN source, also under OMOD (active with output
// flushing). DX10_CLAMP clear lets the NaN through CLAMP, unquieted with IEEE clear and quieted
// with IEEE set; DX10_CLAMP set clamps it to 0.
const CapturedGfx11 mul_f16_clamp_lanes[] = {
    {"Dx10ClampOff", RJ_ARITH_ALU_CLAMP, 0x0f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a57c01},
    {"Dx10ClampOn", RJ_ARITH_ALU_CLAMP, 0x1f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a50000},
    {"IeeeSet", RJ_ARITH_ALU_CLAMP, 0x2f0, 0xa5a57c01, 0xa5a53c00, 0, 0xa5a57e01},
    {"Mul2Dx10ClampOff", RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_2, 0x000, 0xa5a57c01, 0xa5a53c00, 0,
     0xa5a57c01},
    {"Mul2Dx10ClampOn", RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_2, 0x100, 0xa5a57c01, 0xa5a53c00, 0,
     0xa5a50000},
    {"Mul2IeeeSet", RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_2, 0x200, 0xa5a57c01, 0xa5a53c00, 0,
     0xa5a57e01},
};

// gfx1100 MUL_F64 returns the first NaN source with its modifiers applied, quieted only when
// MODE.IEEE is set. IEEE-set lanes are from the gfx1100 v2 capture (hw2-gfx1100).
const CapturedGfx11 mul_f64_lanes[] = {
    {"SignalingFirst", 0, 0x1f0, 0x7ff0000000000001, 0x7ff8000000000000, 0, 0x7ff0000000000001},
    {"NegativeQuietFirst", 0, 0x1f0, 0xfff8000000000000, 0x7ff8000000000000, 0, 0xfff8000000000000},
    {"SignalingSecondRoundZero", 0, 0x1ff, 0x0, 0x7ff0000000000001, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff8000000000000, 0x7ff8000000000000, 0,
     0xfff8000000000000},
    {"Mul2", RJ_ARITH_ALU_OMOD_2, 0x100, 0x7ff0000000000001, 0x0, 0, 0x7ff0000000000001},
    {"IeeeSetSignalingFirst", 0, 0x3f0, 0x7ff0000000000001, 0x7ff8000000000000, 0,
     0x7ff8000000000001},
    {"IeeeSetNegativeQuietFirst", 0, 0x3f0, 0xfff8000000000000, 0x7ff8000000000000, 0,
     0xfff8000000000000},
};

} // namespace

// Compatibility and strict exact reproduce captured gfx1100 FP16 MUL lanes.
TEST(FpMulF16Gfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 16>(rj_arith_v_mul_f16_w32, mul_f16_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_mul_f16_w64, mul_f16_lanes);
}

// Compatibility and strict exact reproduce captured gfx1100 clamped FP16 MUL NaN lanes.
TEST(FpMulF16Gfx11, CapturedClampedNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 16>(rj_arith_v_mul_f16_w32, mul_f16_clamp_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_mul_f16_w64, mul_f16_clamp_lanes);
}

namespace {

// gfx1100 (W7900, mkm2) Track C hw1100-wide corpus, 2026-11-18, dispatch MODE with DX10_CLAMP
// set and FP16 output denormals flushed (OMOD active on gfx11): 1.5 * 65504 overflows FP16
// before div:2 and halves to infinity. A kernel that halves the wide product first fails.
const CapturedGfx11 mul_f16_omod_lanes[] = {
    {"OverflowBeforeHalving", RJ_ARITH_ALU_OMOD_HALF, 0x150, 0x530d3e00, 0x848d7bff, 0, 0xa5a57c00},
};

// gfx1201 (RX 9070 XT, rj-valu-audit-local hw, 2026-11-05; directed-rounding and FP16_OVFL
// rows from the R9700 hw-r9700-wide corpus, 2026-11-18), FP16 denormals kept. The first NaN
// source wins and is quieted; 1.5 * 65504 overflows before div:2 except when rounding down,
// where the product rounds to 65504 and halves to 32752; FP16_OVFL saturates 65504 * 1 mul:2
// instead of producing infinity; high-half selectors keep the low destination half.
const CapturedGfx11 mul_f16_gfx12_lanes[] = {
    {"FirstSignalingNanQuieted", 0, 0xf0, 0xd6f47c01, 0xb2b87e00, 0, 0xa5a57e01},
    {"FirstNegativeNanWins", 0, 0xf0, 0x0badfe00, 0x8b8f7e00, 0, 0xa5a5fe00},
    {"SecondNan", 0, 0xf0, 0x42620400, 0x62bb7e00, 0, 0xa5a57e00},
    {"SecondSignalingNanQuieted", 0, 0xf0, 0x740b0400, 0xcc677c01, 0, 0xa5a57e01},
    {"OverflowBeforeHalving", RJ_ARITH_ALU_OMOD_HALF, 0xf0, 0x530d3e00, 0x848d7bff, 0, 0xa5a57c00},
    {"OverflowBeforeHalvingRoundUp", RJ_ARITH_ALU_OMOD_HALF, 0xf5, 0x530d3e00, 0x848d7bff, 0,
     0xa5a57c00},
    {"RoundedDownThenHalved", RJ_ARITH_ALU_OMOD_HALF, 0xfa, 0x530d3e00, 0x848d7bff, 0, 0xa5a577ff},
    {"Fp16OvflSaturatesMul2", RJ_ARITH_ALU_OMOD_2, 0x8000f0, 0x7f7c7bff, 0xa29a3c00, 0, 0xa5a57bff},
    {"HighHalves", RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, 0xf0,
     0x07010000, 0x57460000, 0, 0x225ea5a5},
};

} // namespace

// Compatibility and strict exact reproduce the captured gfx1100 OMOD overflow lane.
TEST(FpMulF16Gfx11, CapturedOmodOverflow) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 16>(rj_arith_v_mul_f16_w32, mul_f16_omod_lanes);
  check_captured_gfx11_exact<64, 16>(rj_arith_v_mul_f16_w64, mul_f16_omod_lanes);
}

// Compatibility and strict exact reproduce captured gfx1201 FP16 MUL lanes.
TEST(FpMulF16Gfx12, CapturedNanPriorityRoundingAndOmod) {
  using rocjitsu::arith::test::check_captured_gfx12_exact;
  check_captured_gfx12_exact<32, 16>(rj_arith_v_mul_f16_w32_gfx12, mul_f16_gfx12_lanes);
  check_captured_gfx12_exact<64, 16>(rj_arith_v_mul_f16_w64_gfx12, mul_f16_gfx12_lanes);
}

// Compatibility and strict exact reproduce captured gfx1100 FP64 MUL NaN lanes.
TEST(FpMulF64Gfx11, CapturedNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 64>(rj_arith_v_mul_f64_w32, mul_f64_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_mul_f64_w64, mul_f64_lanes);
}

namespace {

// gfx1100 (W7900, mkm2) lanes: hw1100-noieee (2026-09-25, MODE.IEEE clear) for the OMOD rows,
// hw2-gfx1100 (2026-10-06, MODE.IEEE set) for the IEEE row and the Track C hw1100-wide corpus
// (2026-11-18) for directed rounding. RDNA3 applies OMOD only with output denormals flushed and
// IEEE clear: mul:2 of 1.0 * min normal stays min normal with denormals kept (0x1f0) or IEEE
// set (0x300), while min normal squared under mul:2 with flushing is still flushed. Max * max
// toward -inf or zero stays finite.
const CapturedGfx11 mul_f64_rules[] = {
    {"Mul2MinNormalDenormalsKept", RJ_ARITH_ALU_OMOD_2, 0x1f0, 0x3ff0000000000000,
     0x0010000000000000, 0, 0x0010000000000000},
    {"Mul2NegMinNormalDenormalsKept", RJ_ARITH_ALU_OMOD_2, 0x1f0, 0xbff0000000000000,
     0x0010000000000000, 0, 0x8010000000000000},
    {"Mul2MinNormalIeeeFlush", RJ_ARITH_ALU_OMOD_2, 0x300, 0x3ff0000000000000, 0x0010000000000000,
     0, 0x0010000000000000},
    {"Mul2TinyProductFlushed", RJ_ARITH_ALU_OMOD_2, 0x100, 0x0010000000000000, 0x0010000000000000,
     0, 0},
    {"Mul2HalfTimesMinNormalFlushed", RJ_ARITH_ALU_OMOD_2, 0x100, 0x3fe0000000000000,
     0x0010000000000000, 0, 0},
    {"MaxSquaredRoundDown", 0, 0x1fa, 0x7fefffffffffffff, 0x7fefffffffffffff, 0,
     0x7fefffffffffffff},
    {"MaxSquaredRoundZero", 0, 0x1ff, 0x7fefffffffffffff, 0x7fefffffffffffff, 0,
     0x7fefffffffffffff},
    {"NegativeProductRoundUp", 0, 0x1f5, 0xbff8000000000000, 0x7fefffffffffffff, 0,
     0xffefffffffffffff},
    {"HalfBelowOneTimesMax", 0, 0x1f0, 0x3fdfffffffffffff, 0x7fefffffffffffff, 0,
     0x7fdffffffffffffe},
};

} // namespace

// Compatibility and strict exact reproduce the captured RDNA3 OMOD gate, tininess and rounding.
TEST(FpMulF64Gfx11, CapturedOmodTininessAndRounding) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 64>(rj_arith_v_mul_f64_w32, mul_f64_rules);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_mul_f64_w64, mul_f64_rules);
}

// Compatibility retains a normal FP16 product recovered by narrowing before OMOD.
TEST(FpMulF16Cdna, RoundedMinimumNormal) {
  using namespace rocjitsu::arith::test;
  for (auto cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_mul_f16_w32_gfx942, rj_arith_v_mul_f16_w32_gfx1250}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x3bff;
      r.v[1][0] = 0x0400;
      const uint64_t flags =
          cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
      ASSERT_EQ(call(flags, 1, RJ_ARITH_ALU_OMOD_2, r.v + 2, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0] & 0xffffU, 0x0800U);
    }
  }
}

// Wave64 DPP filters INVALID by writable lanes, but classifies MUL range causes over EXEC.
TEST(FpMulF32Gfx12, CompatibilityDppWave64Causes) {
  using namespace rocjitsu::arith::test;
  for (auto cpu : supported_cpu_levels()) {
    for (uint64_t rows : {4ULL, 8ULL}) {
      Registers<3, 64> r;
      for (int lane = 0; lane < 64; ++lane) {
        r.v[0][lane] = 0x3f800000U;
        r.v[1][lane] = 0x3f800000U;
      }
      r.v[0][0] = 0x7f7fffffU;
      r.v[1][0] = 0x40000000U;
      r.v[0][48] = 0x7f800001U;
      const auto before = r;
      const uint64_t exec_mask = 1ULL | (1ULL << 32) | (1ULL << 48);
      const uint64_t modifiers = RJ_ARITH_DPP16 | (0xe4ULL << RJ_ARITH_DPP_CTRL_SHIFT) |
                                 (rows << RJ_ARITH_DPP_ROW_SHIFT) | RJ_ARITH_DPP_BANK_MASK;
      uint32_t status = 0x40;
      ASSERT_EQ(rj_arith_v_mul_f32_w64_gfx12(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, exec_mask,
                                             modifiers, r.v + 2, r.v, r.v + 1, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(status, rows == 4 ? 0x68U : 0x69U);
      for (int lane = 0; lane < 64; ++lane) {
        const uint32_t expected = rows == 4 && lane == 32   ? 0x3f800000U
                                  : rows == 8 && lane == 48 ? 0x7fc00001U
                                                            : before.v[2][lane];
        EXPECT_EQ(r.v[2][lane], expected) << "lane=" << lane;
      }
      r.expect_preserved(before, 2, 1);
    }
  }
}

// Neither strict mode borrows another target's qualification.
TEST(FpMulF32Gfx1250, NumericAndFpStatusQualification) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    uint32_t a[64] = {0x7f800123}, b[64] = {0x3f800000}, out[64] = {};
    const uint32_t *x = a, *y = b;
    uint32_t *d = out;
    for (auto semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                           RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      const uint64_t flags = cpu | semantics;
      const int expected = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                               : RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
      for (bool wave64 : {false, true}) {
        uint32_t status = 0x80;
        out[0] = 0xdeadbeef;
        const int error = wave64 ? rj_arith_v_mul_f32_w64_gfx1250(flags, 1, 0, &d, &x, &y, &status)
                                 : rj_arith_v_mul_f32_w32_gfx1250(flags, 1, 0, &d, &x, &y, &status);
        EXPECT_EQ(error, expected);
        EXPECT_EQ(out[0], 0xdeadbeefU);
        EXPECT_EQ(status, 0x80U);
      }
    }
  }
}
