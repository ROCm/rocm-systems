// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native FP8/BF8 accumulation and mixed-format regressions.
#include "fixtures/float_dot/rdna_f8_samples.h"
#include "rdna_matrix_test_util.h"
using namespace rocjitsu;
using namespace rocjitsu::matrix_test;

TEST(RdnaWmmaFloat, NativeF8Arithmetic) {
  // FP8_FP8, wave64: finite nonuniform inputs and D overlapping A.
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc464000u, 0x1c421100u}, 0x000000f0u, f8_input(64, false),
        kWmmaFp8);
  // BF8_BF8, wave32: finite nonuniform inputs and D overlapping B.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc494008u, 0x1c421100u}, 0x000000f0u, f8_input(32, true),
        kWmmaBf8);
  // FP8_BF8: the two operand formats must not be interchanged.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc474018u, 0x1c421100u}, 0x000000f0u,
        uniform_input({0x013840b8u, 0x407b3c01u, 0xbf000000u, 0x13579bdfu, 0xeeddc988u}),
        {{0, 0x486001e0u}, {255, 0x486001e0u}});
  // BF8_FP8: the two operand formats must not be interchanged.
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc484018u, 0x1c421100u}, 0x000000f0u,
        uniform_input({0x013c40bcu, 0x407e3801u, 0xbf000000u, 0x13579bdfu, 0xeeddc988u}),
        {{0, 0x44e0efc1u}, {255, 0x44e0efc1u}});
  // FP8 NaN factors in mixed-format WMMA use the factor-NaN payload.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc474018u, 0x1c421100u}, 0xf0,
        uniform_input({0x7f7f7f7fu, 0x3c3c3c3cu, 0, 0, 0}), {{0, 0xffc00a3du}, {255, 0xffc00a3du}});
  // BF8 infinity times zero takes precedence over the accumulator NaN.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc494018u, 0x1c421100u}, 0x000000f0u,
        uniform_input({0x7c7c7c7cu, 0x00000000u, 0xffc54321u, 0x13579bdfu, 0xeeddc988u}),
        {{0, 0xffc00a3du}, {255, 0xffc00a3du}});
  // Both infinity-cancellation branches use the factor-NaN payload.
  for (uint32_t width : {32, 64}) {
    check(ROCJITSU_CODE_ARCH_RDNA4, width, {0xcc494018u, 0x1c421100u}, 0xf0,
          uniform_input({0x7c7c7c7cu, 0x3cbc3cbcu, 0x3f800000u, 0, 0}),
          {{0, 0xffc00a3du}, {255, 0xffc00a3du}});
    check(ROCJITSU_CODE_ARCH_RDNA4, width, {0xcc494018u, 0x1c421100u}, 0xf0,
          uniform_input({0x7c7c7c7cu, 0x3c3c3c3cu, 0xff800000u, 0, 0}),
          {{0, 0xffc00a3du}, {255, 0xffc00a3du}});
  }
  // Native FP8/BF8 ignores C NEG and ABS, unlike F16/BF16 WMMA.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc464018u, 0x9c421100u}, 0xf0,
        uniform_input({0x38383838u, 0x38383838u, 0x3f800000u, 0, 0}),
        {{0, 0x41880000u}, {255, 0x41880000u}});
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc464418u, 0x1c421100u}, 0xf0,
        uniform_input({0x38383838u, 0x38383838u, 0xbf800000u, 0, 0}),
        {{0, 0x41700000u}, {255, 0x41700000u}});
}
