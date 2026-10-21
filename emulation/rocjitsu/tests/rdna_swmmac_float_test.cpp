// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse floating-point matrix indices, modifiers, and per-step narrowing.
#include "fixtures/float_dot/rdna_sparse_float_samples.h"
#include "rdna_matrix_test_util.h"
using namespace rocjitsu;
using namespace rocjitsu::matrix_test;

TEST(RdnaSwmmacFloat, SparseIndicesAndModifiers) {
  // Nonuniform FP16 factors and indices; D=v29 overlaps index v32.
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc50401du, 0x1c821100u}, 0xf0, float_input(64, 16, false),
        kSwmmacF32F16);
  // Wave64 key 2 selects the third 8-bit index set.
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc505018u, 0x1c821100u}, 0xf0, float_input(64, 16, false),
        {{0, 0xcc879066u}, {255, 0x47832038u}});
  // FP16 factors: key 1 and NEG_HI_B follow compressed product slots.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc504a18u, 0x1c821100u}, 0x000000f0u,
        uniform_input({0xbc003c00u, 0x42004000u, 0xbf000000u, 0xbf000000u, 0xeeddc988u}),
        {{0, 0x422e0000u}, {255, 0x422e0000u}});
  // Packed BF16: key 1, tied accumulator, and per-step narrowing.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc534818u, 0x1c821100u}, 0x000000f0u,
        uniform_input({0x3f003f00u, 0x3f003f00u, 0, 0x43804380u, 0x44444444u}),
        {{0, 0x43804380u}, {127, 0x43804380u}}, 128);
  // Mixed FP8/BF8 sparse factors use their native formats.
  check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc584018u, 0x1c821100u}, 0x000000f0u,
        uniform_input({0x013840b8u, 0x407b3c01u, 0xbf000000u, 0xbf000000u, 0xeeddc988u}),
        {{0, 0x48e01bb0u}, {255, 0x48e01bb0u}});
  // Wave64 selects the fourth index byte.
  check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc575818u, 0x1c821100u}, 0x000000f0u,
        uniform_input({0x013840b8u, 0x407e3801u, 0xbf000000u, 0xbf000000u, 0xeeddc988u}),
        {{0, 0x41784000u}, {255, 0x41784000u}});
  // gfx1201 sparse F16 supports C NEG, with ABS applied before NEG.
  for (uint32_t width : {32, 64}) {
    // Packed F16 with accumulator NEG, in both wave sizes.
    check(ROCJITSU_CODE_ARCH_RDNA4, width, {0xcc524818u, 0x9c821100u}, 0xf0,
          uniform_input({0x3c003c00u, 0x40004000u, 0, 0xb800b800u, 0xeeddc988u}),
          {{0, 0x50105010u}, {127, 0x50105010u}}, 128);
    Registers input = uniform_input({0x3c003c00u, 0x40004000u, 0, 0xbf000000u, 0xeeddc988u});
    check(ROCJITSU_CODE_ARCH_RDNA4, width, {0xcc504818u, 0x9c821100u}, 0xf0, input,
          {{0, 0x42020000u}, {255, 0x42020000u}});
    check(ROCJITSU_CODE_ARCH_RDNA4, width, {0xcc504c18u, 0x9c821100u}, 0xf0, input,
          {{0, 0x41fc0000u}, {255, 0x41fc0000u}});
  }
}
