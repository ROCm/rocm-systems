// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense FP8/BF8 arithmetic, mixed formats, and FP16/BF16 packed accumulators.
#include "fixtures/float_dot/rdna_f16_samples.h"
#include "fixtures/float_dot/rdna_f8_samples.h"
#include "rdna_matrix_test_util.h"
using namespace rocjitsu;
using namespace rocjitsu::matrix_test;

TEST(RdnaWmmaFloat, NativeF8Arithmetic) {
  // FP8_FP8, wave64: finite nonuniform inputs and D overlapping A.
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc464000u, 0x1c421100u}, 0x000000f0u,
        float_input(64, 8, false, /*gfx12=*/true), kWmmaFp8);
  // BF8_BF8, wave32: finite nonuniform inputs and D overlapping B.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc494008u, 0x1c421100u}, 0x000000f0u,
        float_input(32, 8, true, /*gfx12=*/true), kWmmaBf8);
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

TEST(RdnaWmmaFloat, PackedHalvesAndAccumulator) {
  Registers input = float_input(64, 16, false, /*gfx12=*/false);
  check(ROCJITSU_CODE_ARCH_RDNA3, 64, {0xcc426018u, 0x1c421100u}, 0xf0, input, kWmmaF16Gfx11High);
  // RDNA3 FP16: write the low half and preserve the other destination half.
  check(ROCJITSU_CODE_ARCH_RDNA3, 32, {0xcc424018u, 0x1c421100u}, 0x000000f0u,
        uniform_input({0xbc003c00u, 0x42004000u, 0xc0003c00u, 0x13579bdfu, 0xeeddc988u}),
        {{0, 0x1357c700u}, {255, 0x1357c700u}});
  // RDNA3 FP16: write the high half and preserve the other destination half.
  check(ROCJITSU_CODE_ARCH_RDNA3, 32, {0xcc426018u, 0x1c421100u}, 0x000000f0u,
        uniform_input({0xbc003c00u, 0x42004000u, 0xc0003c00u, 0x13579bdfu, 0xeeddc988u}),
        {{0, 0xc9009bdfu}, {255, 0xc9009bdfu}});
  // RDNA3 BF16 flushes C on input, before adding one minimum-normal product.
  Registers tiny_c = uniform_input({0, 0, 0x007f007fu, 0x13579bdfu, 0});
  for (uint32_t lane = 0; lane < 32; ++lane) {
    tiny_c[0][lane] = 0x00000080u;
    tiny_c[8][lane] = 0x00003f80u;
  }
  check(ROCJITSU_CODE_ARCH_RDNA3, 32, {0xcc434018u, 0x1c421100u}, 0xf0, tiny_c,
        {{0, 0x13570080u}, {255, 0x13570080u}});
  // RDNA4 BF16: zero products preserve the subnormal packed accumulator.
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc434018u, 0x1c421100u}, 0x000000f0u,
        uniform_input({0x00000000u, 0x00000000u, 0x00010001u, 0x13579bdfu, 0xeeddc988u}),
        {{0, 0x00010001u}, {127, 0x00010001u}}, 128);
  // BF16 factors with FP32 accumulation use distinct raw encodings from FP16.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc414018u, 0x1c421100u}, 0x000000f0u,
        uniform_input({0xbf803f80u, 0x40404000u, 0xbf000000u, 0x13579bdfu, 0xeeddc988u}),
        {{0, 0xc1080000u}, {255, 0xc1080000u}});
}
