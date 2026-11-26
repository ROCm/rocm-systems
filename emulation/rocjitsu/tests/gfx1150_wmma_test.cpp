// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1150 dense WMMA dispatch in both waves, including packed halves and integer formats.
#include "rdna_matrix_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::matrix_test {
namespace {
// Integral products isolate operand/half selection; existing Gfx11Dot2 fixtures cover rounding.
TEST(Gfx1150Wmma, FloatingFormatsAndPackedHalves) {
  ForceScalarGuard guard;
  for (bool scalar : {false, true}) {
    util::set_force_scalar_for_testing(scalar);
    for (auto semantics :
         {amdgpu::ArithmeticSemantics::Loose, amdgpu::ArithmeticSemantics::Compatibility}) {
      for (uint32_t width : {32, 64}) {
        for (bool bf16 : {false, true}) {
          const auto input =
              uniform_input({bf16 ? 0x40003f80U : 0x40003c00U, bf16 ? 0x3f803f80U : 0x3c003c00U,
                             0x40400000U, 0xdeadbeefU, 0});
          const auto opcode =
              bf16 ? rdna3_5::kVWmmaF3216x16x16Bf16Vop3p : rdna3_5::kVWmmaF3216x16x16F16Vop3p;
          const auto words = rdna3_5::build_vop3p(
              opcode, {.vdst = 24, .src0 = 256, .src1 = 264, .src2 = 272, .op_sel_hi = 3});
          // Eight (1+2) DOT2 products plus the FP32 accumulator 3 give 27.
          check(rocjitsu::IsaFamily::Rdna3_5, width, words, 0xf0, input,
                {{0, 0x41d80000U}, {255, 0x41d80000U}}, 256, semantics);
          for (bool high : {false, true}) {
            auto packed = input;
            for (int reg = 16; reg < 24; ++reg) {
              packed[reg].fill(bf16 ? 0x40804040U : 0x44004200U); // (3,4).
            }
            const auto packed_opcode =
                bf16 ? rdna3_5::kVWmmaBf1616x16x16Bf16Vop3p : rdna3_5::kVWmmaF1616x16x16F16Vop3p;
            const auto packed_words =
                rdna3_5::build_vop3p(packed_opcode, {.vdst = 24,
                                                     .op_sel = uint8_t(high ? 4 : 0),
                                                     .src0 = 256,
                                                     .src1 = 264,
                                                     .src2 = 272,
                                                     .op_sel_hi = 3});
            const uint32_t value = bf16 ? (high ? 0x41e0U : 0x41d8U) : (high ? 0x4f00U : 0x4ec0U);
            const uint32_t expected = high ? (value << 16) | 0xbeefU : 0xdead0000U | value;
            check(rocjitsu::IsaFamily::Rdna3_5, width, packed_words, 0xf0, packed,
                  {{0, expected}, {255, expected}}, 256, semantics);
          }
        }
      }
    }
  }
}

// Signed INT8/INT4 factors must use their own packing, at either wave width.
TEST(Gfx1150Wmma, IntegerFormats) {
  ForceScalarGuard guard;
  for (bool scalar : {false, true}) {
    util::set_force_scalar_for_testing(scalar);
    for (auto semantics :
         {amdgpu::ArithmeticSemantics::Loose, amdgpu::ArithmeticSemantics::Compatibility}) {
      for (uint32_t width : {32, 64}) {
        for (bool iu4 : {false, true}) {
          const auto input = uniform_input({iu4 ? 0xffffffffU : 0x81818181U,
                                            iu4 ? 0x22222222U : 0x02020202U, 0, 0xdeadbeefU, 0});
          const auto opcode =
              iu4 ? rdna3_5::kVWmmaI3216x16x16Iu4Vop3p : rdna3_5::kVWmmaI3216x16x16Iu8Vop3p;
          const auto words = rdna3_5::build_vop3p(
              opcode,
              {.vdst = 24, .src0 = 256, .src1 = 264, .src2 = 272, .op_sel_hi = 3, .neg = 3});
          const uint32_t expected = iu4 ? uint32_t(-32) : uint32_t(-4064);
          check(rocjitsu::IsaFamily::Rdna3_5, width, words, 0xf0, input,
                {{0, expected}, {255, expected}}, 256, semantics);
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::matrix_test
