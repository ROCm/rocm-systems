// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse integer index selection and intermediate saturation.
#include "fixtures/float_dot/rdna_integer_samples.h"
#include "rdna_matrix_test_util.h"
using namespace rocjitsu;
using namespace rocjitsu::matrix_test;

TEST(RdnaSwmmacI32, SaturationAndIndexOverlap) {
  ForceScalarGuard guard;
  for (bool scalar : {false, true}) {
    SCOPED_TRACE(scalar);
    util::set_force_scalar_for_testing(scalar);
    // IU8, key 3, signed factors and CLAMP; D=v29 overlaps index v32.
    check(ROCJITSU_CODE_ARCH_RDNA4, 64, {0xcc54d81du, 0x7c821100u}, 0xf0, integer_input(64, true),
          {{0, 0xffff1a17u},
           {31, 0x00005c69u},
           {32, 0xffffd507u},
           {63, 0x12346527u},
           {128, 0xffff9cf7u},
           {255, 0xecce839du}});
    // IU4_K64, key 0, signed factors and CLAMP; D=v24.
    check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc56c018u, 0x7c821100u}, 0xf0, integer_input(32, true),
          kSwmmacIu4K64);
    // IU4_K32, key 1 selects the upper 16-bit index set; C/D=v24.
    check(ROCJITSU_CODE_ARCH_RDNA4, 32, {0xcc55c818u, 0x7c821100u}, 0xf0, integer_input(32, true),
          {{0, 0xffffffcbu},
           {31, 0x0000000cu},
           {32, 0xffffff8eu},
           {63, 0x80000000u},
           {128, 0x7fffefe9u},
           {255, 0xfffffff6u}});
  }
}
