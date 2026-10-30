// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for transfers between scalar values and individual VGPR lanes.

#include "register/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

// Check readlane selection, aliases and invalid controls for either wave size.
template <int Lanes, typename Call> void check_readlane(Call call) {
  rocjitsu::arith::test::Registers<1, Lanes> input;
  std::minstd_rand random;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * Lanes; ++index) {
        auto actual = input;
        int selected = index % Lanes;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(call(cpu | semantics, 0, actual.v[0], actual.v, 0xffffff00U | uint32_t(index)),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == 0 ? input.v[0][selected] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [=](auto flags, auto mode) { return call(flags, mode, nullptr, nullptr, 0); }, 0);
}

// Check writelane selection, aliases and invalid controls for either wave size.
template <int Lanes, typename Call> void check_writelane(Call call) {
  rocjitsu::arith::test::Registers<1, Lanes> input;
  std::minstd_rand random;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * Lanes; ++index) {
        auto actual = input;
        int selected = index % Lanes;
        // Writing the selected lane preserves every other lane.
        ASSERT_EQ(call(cpu | semantics, 0, actual.v, input.v[0][0], 0xffffff00U | uint32_t(index)),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == selected ? input.v[0][0] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [=](auto flags, auto mode) { return call(flags, mode, nullptr, 0, 0); }, 0);
}

// Check readfirstlane selection, aliases and invalid controls for either wave size.
template <int Lanes, typename Call> void check_readfirstlane(Call call) {
  rocjitsu::arith::test::Registers<1, Lanes> input;
  std::minstd_rand random;
  for (int lane = 0; lane < Lanes; ++lane) {
    input.v[0][lane] = rocjitsu::arith::test::random_word(random);
  }
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (int index = 0; index < 2 * Lanes; ++index) {
        auto actual = input;
        auto exec_mask =
            rocjitsu::arith::test::WaveMask<Lanes>(index ? (1ULL << (index % Lanes)) : 0);
        int selected = index ? index % Lanes : 0;
        // The scalar result overlaps a source lane; reads must precede stores.
        ASSERT_EQ(call(cpu | semantics, exec_mask, 0, actual.v[0], actual.v), RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(actual.v[0][lane], lane == 0 ? input.v[0][selected] : input.v[0][lane]);
        }
        actual.expect_preserved(input, 0, 1);
      }
    }
  }
  rocjitsu::arith::test::expect_invalid_flags(
      [=](auto flags, auto mode) { return call(flags, 0, mode, nullptr, nullptr); }, 0);
}

} // namespace

TEST(v_readlane_b32_w32, Arithmetic) { check_readlane<32>(rj_arith_v_readlane_b32_w32); }

TEST(v_writelane_b32_w32, Arithmetic) { check_writelane<32>(rj_arith_v_writelane_b32_w32); }

TEST(v_readfirstlane_b32_w32, Arithmetic) {
  check_readfirstlane<32>(rj_arith_v_readfirstlane_b32_w32);
}

TEST(v_readlane_b32_w64, Arithmetic) { check_readlane<64>(rj_arith_v_readlane_b32_w64); }

TEST(v_writelane_b32_w64, Arithmetic) { check_writelane<64>(rj_arith_v_writelane_b32_w64); }

TEST(v_readfirstlane_b32_w64, Arithmetic) {
  check_readfirstlane<64>(rj_arith_v_readfirstlane_b32_w64);
}
