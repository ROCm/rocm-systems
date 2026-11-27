// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for legacy FP min/max and the shared selection policies.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <iterator>
#include <stdint.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(rj_arith_v_min_f16_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(rj_arith_v_min_f16_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(rj_arith_v_min_f32_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(rj_arith_v_min_f32_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(rj_arith_v_min_f64_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(rj_arith_v_min_f64_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(rj_arith_v_max_f16_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(rj_arith_v_max_f16_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(rj_arith_v_max_f32_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(rj_arith_v_max_f32_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(rj_arith_v_max_f64_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(rj_arith_v_max_f64_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Guest policy affects numeric-vs-NaN selection, clamping and input flushing,
// even though loose semantics does not fix NaN payloads or signed-zero ties.
TEST(FpSelection, GuestControls) {
  using namespace rocjitsu::arith::test;
  struct Case {
    decltype(&rj_arith_v_min_f32_w32) call;
    int width;
    uint64_t a, b, fp, mode, expected;
    bool nan;
  };
  const Case cases[] = {
      // Inherited selection policy preserves subnormal results under output-only flushing.
      {rj_arith_v_min_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, 1, false},
      {rj_arith_v_min_num_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, 1,
       false},
      {rj_arith_v_minimum_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, 1,
       false},
      {rj_arith_v_min_f64_w32, 64, 1, 0x3ff0000000000000ULL, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0,
       1, false},
      {rj_arith_v_min_f32_w32, 32, 0x7f800001, 0x3f800000, 0, 0, 0x3f800000, false},
      {rj_arith_v_min_f32_w32, 32, 0x7f800001, 0x3f800000, RJ_ARITH_FP_IEEE_MODE, 0, 0, true},
      {rj_arith_v_min_f32_w32, 32, 0x7fc00001, 0x7fc00002, RJ_ARITH_FP_DX10_CLAMP_OFF,
       RJ_ARITH_ALU_CLAMP, 0, true},
      {rj_arith_v_min_f32_w32, 32, 1, 0x3f800000, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0, 0, false},
      {rj_arith_v_min_f16_w32, 16, 1, 0x3c00, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0, 0, false},
      {rj_arith_v_min_f64_w32, 64, 1, 0x3ff0000000000000ULL, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0,
       0, false},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &test : cases) {
      SCOPED_TRACE(::testing::Message() << cpu << '/' << test.width << '/' << test.fp);
      Registers<6, 32> r;
      write_fp(r.v, 0, test.width, test.a);
      write_fp(r.v + 2, 0, test.width, test.b);
      ASSERT_EQ(test.call(cpu | RJ_ARITH_SEMANTICS_LOOSE | test.fp, 1, test.mode, r.v + 4, r.v,
                          r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      auto actual = read_fp(r.v + 4, 0, test.width);
      if (test.nan) {
        EXPECT_TRUE(std::isnan(fp_value(actual, test.width)));
      } else {
        EXPECT_EQ(actual, test.expected);
      }
    }
  }
}

TEST(FpSelection, CompatibilityOmodRounding) {
  using namespace rocjitsu::arith::test;
  check_selection_omod<16, 2>(rj_arith_v_min_f16_w32);
  check_selection_omod<16, 2>(rj_arith_v_min_num_f16_w32);
  check_selection_omod<32, 2>(rj_arith_v_min_f32_w32);
  check_selection_omod<32, 2>(rj_arith_v_min_num_f32_w32);
  check_selection_omod<64, 2>(rj_arith_v_min_f64_w32);
  check_selection_omod<64, 2>(rj_arith_v_min_num_f64_w32);
  check_selection_omod<16, 3>(rj_arith_v_min3_f16_w32);
  check_selection_omod<16, 3>(rj_arith_v_min3_num_f16_w32);
  check_selection_omod<32, 3>(rj_arith_v_min3_f32_w32);
  check_selection_omod<32, 3>(rj_arith_v_min3_num_f32_w32);
}

// F16C must not quiet a legacy sNaN before IEEE-sensitive selection.
TEST(FpSelection, HalfSignalingNan) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      Registers<4, 32> r;
      r.v[0][0] = 0x7c017c01;
      r.v[1][0] = 0x3c003c00;
      r.v[2][0] = 0x40004000;
      uint64_t flags = cpu | semantics | RJ_ARITH_FP_IEEE_MODE;
      ASSERT_EQ(rj_arith_v_min_f16_w32(flags, 1, 0, r.v + 3, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_TRUE(std::isnan(f16_value(uint16_t(r.v[3][0]))));
      ASSERT_EQ(rj_arith_v_min3_f16_w32(flags, 1, 0, r.v + 3, r.v + 1, r.v + 2, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_TRUE(std::isnan(f16_value(uint16_t(r.v[3][0]))));
      ASSERT_EQ(rj_arith_v_pk_min_f16_w32(flags, 1, 0, r.v + 3, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      for (int shift : {0, 16}) {
        EXPECT_TRUE(std::isnan(f16_value(uint16_t(r.v[3][0] >> shift))));
      }
      ASSERT_EQ(rj_arith_v_pk_min_num_f16_w32(flags, 1, 0, r.v + 3, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], 0x3c003c00U);
    }
  }
}

// Selected signaling NaNs survive OMOD in all affected scalar and SIMD families.
TEST(FpSelection, CompatibilityOmodNan) {
  using namespace rocjitsu::arith::test;
  check_selection_omod_nan<16, 2>(rj_arith_v_min_f16_w32);
  check_selection_omod_nan<16, 3>(rj_arith_v_min3_f16_w32);
  check_selection_omod_nan<32, 2>(rj_arith_v_min_f32_w32);
  check_selection_omod_nan<32, 3>(rj_arith_v_min3_f32_w32);
  check_selection_omod_nan<64, 2>(rj_arith_v_min_f64_w32);
}

// Directed gfx1100/gfx1201 Wave32 probes distinguish source priority, zero ties,
// and OMOD's treatment of negative tiny results. No broad hardware coverage is implied.
TEST(FpSelection, ExactDirectedBits) {
  using namespace rocjitsu::arith::test;
  using Call = decltype(&rj_arith_v_min_f32_w32);
  struct Case {
    Call call;
    int width;
    uint64_t a, b, expected, flags, instruction_flags;
  };
  const Case cases[] = {
      {rj_arith_v_min_f32_w32, 32, 0x7fc00001, 0x7fc00002, 0x7fc00001, 0, 0},
      {rj_arith_v_min_f32_w32, 32, 0x7f800001, 0x7f800002, 0x7f800001, 0, 0},
      {rj_arith_v_min_f32_w32, 32, 0x7f800001, 0x7f800002, 0x7fc00001, RJ_ARITH_FP_IEEE_MODE, 0},
      {rj_arith_v_min_f64_w32, 64, 0, 0x8000000000000000ULL, 0x8000000000000000ULL, 0, 0},
      {rj_arith_v_max_f64_w32, 64, 0x8000000000000000ULL, 0, 0, 0, 0},
      {rj_arith_v_min_num_f64_w32, 64, 0, 0x8000000000000000ULL, 0x8000000000000000ULL, 0, 0},
      {rj_arith_v_max_num_f64_w32, 64, 0x8000000000000000ULL, 0, 0, 0, 0},
      {rj_arith_v_min_f64_w32, 64, 0x8000000000000000ULL, 0, 0x8000000000000000ULL, 0, 0},
      {rj_arith_v_max_f64_w32, 64, 0, 0x8000000000000000ULL, 0, 0, 0},
      {rj_arith_v_min_num_f64_w32, 64, 0x8000000000000000ULL, 0, 0x8000000000000000ULL, 0, 0},
      {rj_arith_v_max_num_f64_w32, 64, 0, 0x8000000000000000ULL, 0, 0, 0},
      {rj_arith_v_min_f16_w32, 16, 0x8001, 0x3c00, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
       RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_min_num_f16_w32, 16, 0x8001, 0x3c00, 0, 0, RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_min_f64_w32, 64, 0x8000000000000001ULL, 0x3ff0000000000000ULL, 0,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_min_num_f64_w32, 64, 0x8000000000000001ULL, 0x3ff0000000000000ULL, 0, 0,
       RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_min_f32_w32, 32, 0x80000001, 0x3f800000, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
       RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_min_num_f32_w32, 32, 0x80000001, 0x3f800000, 0, 0, RJ_ARITH_ALU_OMOD_2}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &c : cases) {
        Registers<6, 32> r;
        write_fp(r.v, 0, c.width, c.a);
        write_fp(r.v + 2, 0, c.width, c.b);
        uint32_t status = 0;
        ASSERT_EQ(c.call(cpu | semantics | c.flags, 1, c.instruction_flags, r.v + 4, r.v, r.v + 2,
                         &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 4, 0, c.width), c.expected);
      }
    }
  }
}

TEST(FpSelection, HalfOverflowControl) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool saturate : {false, true}) {
      Registers<4, 32> r;
      r.v[0][0] = r.v[1][0] = r.v[2][0] = 0x7bff;
      ASSERT_EQ(rj_arith_v_min3_num_f16_w32(
                    cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (saturate ? RJ_ARITH_FP16_OVFL : 0), 1,
                    RJ_ARITH_ALU_OMOD_2, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], saturate ? 0xdead7bffU : 0xdead7c00U);
    }
  }
}

TEST(FpSelection, TernaryDppUpperRow) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int width : {16, 32}) {
      Registers<4, 64> r;
      for (int lane = 0; lane < 64; ++lane) {
        r.v[0][lane] = width == 16 ? 0x4000 : 0x40000000;
        r.v[1][lane] = r.v[2][lane] = width == 16 ? 0x4400 : 0x40800000;
        r.v[3][lane] = 0xdeadbeef;
      }
      r.v[0][63] = width == 16 ? 0x3c00 : 0x3f800000;
      auto call = width == 16 ? rj_arith_v_min3_num_f16_w64 : rj_arith_v_min3_num_f32_w64;
      const uint64_t instruction_flags = RJ_ARITH_DPP16 | RJ_ARITH_DPP_FI |
                                         (0x140ULL << RJ_ARITH_DPP_CTRL_SHIFT) |
                                         (8ULL << RJ_ARITH_DPP_ROW_SHIFT) | RJ_ARITH_DPP_BANK_MASK;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, (1ULL << 16) | (1ULL << 48),
                     instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][16], 0xdeadbeefU);
      EXPECT_EQ(r.v[3][48], width == 16 ? 0xdead3c00U : 0x3f800000U);
    }
  }
}

// Check native SIMD comparisons/min/max with bitwise NaN selection and quieting
// against the scalar model. NaN encoding checks enforce the Compatibility
// contract and exceed the Loose guarantees.
TEST(FpSelection, NativeNanSelection) {
  using namespace rocjitsu::arith::test;
  using Call = decltype(&rj_arith_v_min_num_f32_w32);
  struct Family {
    int width;
    uint64_t infinity, quiet, sign;
    Call calls[4];
  };
  const Family families[] = {
      {16,
       0x7c00,
       0x0200,
       0x8000,
       {rj_arith_v_min_num_f16_w32, rj_arith_v_max_num_f16_w32, rj_arith_v_minimum_f16_w32,
        rj_arith_v_maximum_f16_w32}},
      {32,
       0x7f800000,
       0x00400000,
       0x80000000,
       {rj_arith_v_min_num_f32_w32, rj_arith_v_max_num_f32_w32, rj_arith_v_minimum_f32_w32,
        rj_arith_v_maximum_f32_w32}},
      {64,
       0x7ff0000000000000ULL,
       0x0008000000000000ULL,
       0x8000000000000000ULL,
       {rj_arith_v_min_num_f64_w32, rj_arith_v_max_num_f64_w32, rj_arith_v_minimum_f64_w32,
        rj_arith_v_maximum_f64_w32}},
  };
  const auto levels = supported_cpu_levels();
  for (const auto &family : families) {
    // Pairs: signed zeros, smallest subnormals, largest finite values,
    // infinities, signaling NaNs, and quiet NaNs.
    const uint64_t values[] = {0,
                               family.sign,
                               1,
                               family.sign | 1,
                               family.infinity - 1,
                               family.sign | (family.infinity - 1),
                               family.infinity,
                               family.sign | family.infinity,
                               family.infinity | 1,
                               family.sign | family.infinity | 2,
                               family.infinity | family.quiet | 3,
                               family.sign | family.infinity | family.quiet | 4};
    constexpr int value_count = std::size(values);
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      SCOPED_TRACE(semantics);
      for (Call call : family.calls) {
        for (uint64_t a : values) {
          Registers<8, 32> r;
          for (int lane = 0; lane < 32; ++lane) {
            write_fp(r.v, lane, family.width, a);
            write_fp(r.v + 2, lane, family.width, values[lane % value_count]);
          }
          uint32_t status = 0;
          ASSERT_EQ(call(semantics, 0xffffffffU, 0, r.v + 4, r.v, r.v + 2, &status),
                    RJ_ARITH_SUCCESS);
          for (uint64_t cpu : levels) {
            if (cpu == RJ_ARITH_CPU_BASELINE) {
              continue;
            }
            ASSERT_EQ(call(cpu | semantics, 0xffffffffU, 0, r.v + 6, r.v, r.v + 2, &status),
                      RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              EXPECT_EQ(read_fp(r.v + 6, lane, family.width), read_fp(r.v + 4, lane, family.width))
                  << family.width << '/' << cpu << '/' << a << '/' << values[lane % value_count];
            }
          }
        }
      }
    }
  }
}

namespace {
// Check literal SDWA scaling results, destination overlap and masked selected-half stores.
template <int Bits, int Lanes, typename Call> void check_sdwa(Call call, bool maximum) {
  using namespace rocjitsu::arith::test;
  using Mask = WaveMask<Lanes>;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int aliasing_offset : {0, 1, 2}) {
        Registers<3, Lanes> input;
        for (int lane = 0; lane < Lanes; ++lane) {
          input.v[0][lane] = Bits == 16 ? 0x3e003e00U : 0x3fc00000U;
          input.v[1][lane] = Bits == 16 ? 0x41004100U : 0x40200000U;
        }
        Registers<3, Lanes> actual = input;
        const uint64_t flags = cpu | semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
        const uint64_t mode =
            RJ_ARITH_ALU_SDWA | RJ_ARITH_ALU_OMOD_2 |
            (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D : 0U);
        const Mask exec_mask = Mask(1) | (Mask(1) << (Lanes - 1));
        uint32_t status = 0x100;
        const int result = call(flags, exec_mask, mode, actual.v + aliasing_offset, actual.v,
                                actual.v + 1, &status);
        EXPECT_EQ(status, 0x100U);
        if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
          EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
          actual.expect_preserved(input, 0, 0);
          continue;
        }
        ASSERT_EQ(result, 0);
        const uint32_t expected =
            Bits == 16 ? (maximum ? 0x4500U : 0x4200U) : (maximum ? 0x40a00000U : 0x40400000U);
        for (int lane = 0; lane < Lanes; ++lane) {
          const uint32_t before = input.v[aliasing_offset][lane];
          const uint32_t output = Bits == 16 ? (expected << 16) | (before & 0xffffU) : expected;
          EXPECT_EQ(actual.v[aliasing_offset][lane], (exec_mask >> lane) & 1U ? output : before);
        }
        actual.expect_preserved(input, aliasing_offset, 1);
        EXPECT_EQ(call(flags, Mask(0), mode, nullptr, nullptr, nullptr, nullptr), 0);
        EXPECT_EQ(call(flags, Mask(0), mode | RJ_ARITH_DPP16, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
      }
    }
  }
}
} // namespace

// All legacy two-input precisions and wave widths share the same SDWA output stage.
TEST(FpLegacySelection, SdwaOutput) {
  using Call32 = decltype(&rj_arith_v_min_f16_w32);
  using Call64 = decltype(&rj_arith_v_min_f16_w64);
  struct Case {
    Call32 f16_wave32;
    Call64 f16_wave64;
    Call32 f32_wave32;
    Call64 f32_wave64;
    bool maximum;
  };
  const Case cases[] = {
      {rj_arith_v_min_f16_w32, rj_arith_v_min_f16_w64, rj_arith_v_min_f32_w32,
       rj_arith_v_min_f32_w64, false},
      {rj_arith_v_max_f16_w32, rj_arith_v_max_f16_w64, rj_arith_v_max_f32_w32,
       rj_arith_v_max_f32_w64, true},
      {rj_arith_v_min_f16_w32_gfx942, rj_arith_v_min_f16_w64_gfx942, rj_arith_v_min_f32_w32_gfx942,
       rj_arith_v_min_f32_w64_gfx942, false},
      {rj_arith_v_max_f16_w32_gfx942, rj_arith_v_max_f16_w64_gfx942, rj_arith_v_max_f32_w32_gfx942,
       rj_arith_v_max_f32_w64_gfx942, true},
  };
  for (const Case &sample : cases) {
    check_sdwa<16, 32>(sample.f16_wave32, sample.maximum);
    check_sdwa<16, 64>(sample.f16_wave64, sample.maximum);
    check_sdwa<32, 32>(sample.f32_wave32, sample.maximum);
    check_sdwa<32, 64>(sample.f32_wave64, sample.maximum);
  }
}

// Selected inherited SDWA rounding witnesses exercise the public API on every CPU path.
TEST(FpLegacySelection, SdwaRounding) {
  using namespace rocjitsu::arith::test;
  struct Case {
    uint32_t input, omod, expected[4];
  };
  constexpr Case cases[] = {
      {0x7f7fffffU, 1, {0x7f800000U, 0x7f800000U, 0x7f7fffffU, 0x7f7fffffU}},
      {0xff7fffffU, 2, {0xff800000U, 0xff7fffffU, 0xff800000U, 0xff7fffffU}},
      {0x00ffffffU, 3, {0x00800000U, 0x00800000U, 0, 0}},
      {0x80ffffffU, 3, {0x80800000U, 0, 0x80800000U, 0}},
      {0x00800000U, 3, {0, 0, 0, 0}},
      {0x00400000U, 1, {0x00800000U, 0x00800000U, 0x00800000U, 0x00800000U}},
      {1, 3, {0, 0, 0, 0}},
      {0x80000000U, 1, {0, 0, 0, 0}},
      {0xff800000U, 3, {0xff800000U, 0xff800000U, 0xff800000U, 0xff800000U}},
      {0x7f812345U, 1, {0x7fc12345U, 0x7fc12345U, 0x7fc12345U, 0x7fc12345U}},
      {0x3f800001U, 0, {0x3f800001U, 0x3f800001U, 0x3f800001U, 0x3f800001U}},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const Case &sample : cases) {
      for (int rounding = 0; rounding < 4; ++rounding) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << sample.input << '/' << rounding);
        Registers<1, 32> data;
        for (int lane = 0; lane < 32; ++lane) {
          data.v[0][lane] = sample.input;
        }
        uint32_t state = 0x80000000U;
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                               RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | (uint64_t(rounding) << 24);
        const uint64_t mode = RJ_ARITH_ALU_SDWA | (uint64_t(sample.omod) << 6);
        const uint32_t *source = data.v[0];
        ASSERT_EQ(
            rj_arith_v_min_f32_w32(flags, 0xaaaaaaaaU, mode, data.v, &source, &source, &state),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(state, 0x80000000U);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(data.v[0][lane], lane & 1 ? sample.expected[rounding] : sample.input);
        }
      }
    }
  }
}

// Selected output-stage literals retain tiny-value and modifier-order boundaries.
TEST(FpMinMax, OutputBoundaries) {
  using namespace rocjitsu::arith::test;
  const struct {
    decltype(&rj_arith_v_min_num_f32_w32) call;
    int width;
    uint64_t a, b, expected;
    uint32_t modifiers;
  } cases[] = {
      {rj_arith_v_max_num_f16_w32, 16, 1, 0, 0, RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_max_num_f32_w32, 32, 0x80000000, 0x80000000, 0, RJ_ARITH_ALU_OMOD_2},
      {rj_arith_v_max_num_f16_w32, 16, 0x8400, 0xbc00, 0x8000, RJ_ARITH_ALU_OMOD_HALF},
      {rj_arith_v_max_num_f32_w32, 32, 0x80800000, 0xbf800000, 0x80000000, RJ_ARITH_ALU_OMOD_HALF},
      {rj_arith_v_max_num_f64_w32, 64, 0x8010000000000000ULL, 0xbff0000000000000ULL,
       0x8000000000000000ULL, RJ_ARITH_ALU_OMOD_HALF},
      // ABS -> selection -> scaling -> clamp. Clamping before scaling would give 1.5.
      {rj_arith_v_min_num_f16_w32, 16, 0xba00, 0x4000, 0x3c00,
       RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_ALU_CLAMP},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      Registers<6, 32> r;
      write_fp(r.v, 0, sample.width, sample.a);
      write_fp(r.v + 2, 0, sample.width, sample.b);
      uint32_t status = 0x1234;
      ASSERT_EQ(sample.call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 1,
                            sample.modifiers, r.v + 4, r.v, r.v + 2, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 4, 0, sample.width), sample.expected)
          << "width=" << sample.width << " cpu=" << cpu;
      EXPECT_EQ(status, 0x1234U);
    }
  }
}
