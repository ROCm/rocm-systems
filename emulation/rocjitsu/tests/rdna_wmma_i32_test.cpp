// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Captured witnesses for saturation between integer product groups.
#include "fixtures/float_dot/rdna_integer_samples.h"
#include "rdna_matrix_test_util.h"
using namespace rocjitsu;
using namespace rocjitsu::matrix_test;

TEST(RdnaWmmaI32, IntermediateSaturation) {
  ForceScalarGuard guard;
  for (bool scalar : {false, true}) {
    SCOPED_TRACE(scalar);
    util::set_force_scalar_for_testing(scalar);
    // RDNA3 IU8: signed factors, CLAMP, C=v16, D=v24.
    check(ROCJITSU_CODE_ARCH_RDNA3, 32, {0xcc44c018u, 0x7c421100u}, 0xf0, integer_input(32, false),
          kWmmaIu8Gfx11);
    // RDNA4 IU8: signed factors, CLAMP, C=v16, D=v24.
    check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc44c018u, 0x7c421100u}, 0xf0, integer_input(64, true),
          {{0, 0x12335e33u},
           {31, 0x0000ba94u},
           {32, 0x7fffb643u},
           {63, 0xffffe324u},
           {128, 0x00001bbbu},
           {255, 0x80000000u}});
    // RDNA4 IU4_K32: signed factors, CLAMP, C=v16, D=v24.
    check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc4ac018u, 0x7c421100u}, 0xf0, integer_input(32, true),
          {{0, 0x8000009fu},
           {31, 0xffffffa3u},
           {32, 0x8000003du},
           {63, 0x7fffefcdu},
           {128, 0x80000f9du},
           {255, 0x7fffffffu}});
    // Wave64 IU4_K32 retains the identity product order under CLAMP.
    check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc4ac018u, 0x7c421100u}, 0xf0, integer_input(64, true),
          {{0, 0x1234560cu},
           {31, 0x0000010bu},
           {32, 0x7fffffcbu},
           {63, 0xffffffe1u},
           {128, 0x0000006fu},
           {255, 0x80000fe8u}});
    // RDNA3 IU4 saturates after each group of four products, not just at the end.
    Registers iu4{};
    for (uint32_t lane = 0; lane < 32; ++lane) {
      iu4[0][lane] = 0x88887777u;
      iu4[8][lane] = iu4[9][lane] = 0x77777777u;
      for (uint32_t reg = 16; reg < 24; ++reg)
        iu4[reg][lane] = 0x7ffffff5u;
    }
    check(ROCJITSU_CODE_ARCH_RDNA3, 32, {0xcc45c018u, 0x7c421100u}, 0xf0, iu4,
          {{0, 0x7fffff1fu}, {255, 0x7fffff1fu}});
    // Inline -1 must not be confused with the sentinel for reading C from a VGPR.
    Registers inline_c{};
    for (uint32_t reg = 24; reg < 32; ++reg)
      inline_c[reg].fill(0x55u);
    check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc444018u, 0x03061100u}, 0xf0, inline_c,
          {{0, 0xffffffffu}, {255, 0xffffffffu}});
  }
}
