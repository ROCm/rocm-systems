// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for transfers between scalar values and individual VGPR lanes.

#include "register/helpers.h"

// Check transfers between scalar values and individual VGPR lanes.
TEST(v_readlane_b32_w32, Arithmetic) {
  rocjitsu::arith::test::Registers<1, 32> input;
  std::minstd_rand random;
  for (int lane = 0; lane < 32; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * 32; ++index) {
        auto actual = input;
        int selected = index % 32;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(rj_arith_v_readlane_b32_w32(cpu | semantics, 0, actual.v[0], actual.v,
                                              0xffffff00U | uint32_t(index)),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == 0 ? input.v[0][selected] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [](auto flags, auto mode) {
        return rj_arith_v_readlane_b32_w32(flags, mode, nullptr, nullptr, 0);
      },
      0);
}

// Check transfers between scalar values and individual VGPR lanes.
TEST(v_writelane_b32_w32, Arithmetic) {
  rocjitsu::arith::test::Registers<1, 32> input;
  std::minstd_rand random;
  for (int lane = 0; lane < 32; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * 32; ++index) {
        auto actual = input;
        int selected = index % 32;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(rj_arith_v_writelane_b32_w32(cpu | semantics, 0, actual.v, input.v[0][0],
                                               0xffffff00U | uint32_t(index)),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == selected ? input.v[0][0] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [](auto flags, auto mode) {
        return rj_arith_v_writelane_b32_w32(flags, mode, nullptr, 0, 0);
      },
      0);
}

// Check transfers between scalar values and individual VGPR lanes.
TEST(v_readfirstlane_b32_w32, Arithmetic) {
  rocjitsu::arith::test::Registers<1, 32> input;
  std::minstd_rand random;
  for (int lane = 0; lane < 32; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * 32; ++index) {
        auto actual = input;
        auto exec_mask = rocjitsu::arith::test::WaveMask<32>(index ? (1ULL << (index % 32)) : 0);
        int selected = index ? index % 32 : 0;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(
            rj_arith_v_readfirstlane_b32_w32(cpu | semantics, exec_mask, 0, actual.v[0], actual.v),
            RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == 0 ? input.v[0][selected] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [](auto flags, auto mode) {
        return rj_arith_v_readfirstlane_b32_w32(flags, 0, mode, nullptr, nullptr);
      },
      0);
}

// Check transfers between scalar values and individual VGPR lanes.
TEST(v_readlane_b32_w64, Arithmetic) {
  rocjitsu::arith::test::Registers<1, 64> input;
  std::minstd_rand random;
  for (int lane = 0; lane < 64; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * 64; ++index) {
        auto actual = input;
        int selected = index % 64;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(rj_arith_v_readlane_b32_w64(cpu | semantics, 0, actual.v[0], actual.v,
                                              0xffffff00U | uint32_t(index)),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == 0 ? input.v[0][selected] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [](auto flags, auto mode) {
        return rj_arith_v_readlane_b32_w64(flags, mode, nullptr, nullptr, 0);
      },
      0);
}

// Check transfers between scalar values and individual VGPR lanes.
TEST(v_writelane_b32_w64, Arithmetic) {
  rocjitsu::arith::test::Registers<1, 64> input;
  std::minstd_rand random;
  for (int lane = 0; lane < 64; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * 64; ++index) {
        auto actual = input;
        int selected = index % 64;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(rj_arith_v_writelane_b32_w64(cpu | semantics, 0, actual.v, input.v[0][0],
                                               0xffffff00U | uint32_t(index)),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == selected ? input.v[0][0] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [](auto flags, auto mode) {
        return rj_arith_v_writelane_b32_w64(flags, mode, nullptr, 0, 0);
      },
      0);
}

// Check transfers between scalar values and individual VGPR lanes.
TEST(v_readfirstlane_b32_w64, Arithmetic) {
  rocjitsu::arith::test::Registers<1, 64> input;
  std::minstd_rand random;
  for (int lane = 0; lane < 64; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * 64; ++index) {
        auto actual = input;
        auto exec_mask = rocjitsu::arith::test::WaveMask<64>(index ? (1ULL << (index % 64)) : 0);
        int selected = index ? index % 64 : 0;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(
            rj_arith_v_readfirstlane_b32_w64(cpu | semantics, exec_mask, 0, actual.v[0], actual.v),
            RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == 0 ? input.v[0][selected] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [](auto flags, auto mode) {
        return rj_arith_v_readfirstlane_b32_w64(flags, 0, mode, nullptr, nullptr);
      },
      0);
}
