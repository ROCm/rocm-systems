// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free unit tests for the NPM (Node Power Management) helpers in
// rocm_smi_npm.cc: set_npm_board_limit(), get_npm_node_power() and
// get_npm_board_max_limit().
//
// These functions take an arbitrary board_path string and operate on plain
// files underneath it (they do not require /sys or any specific sysfs
// layout), so we exercise them against a real temporary directory
// (FakeSysfsTree) instead of mocking /sys. This mirrors the existing
// (unexercised prior to this change) get_npm_board_limit()/
// get_npm_board_status() logic in the same translation unit, all of which
// share the same "board_path directory + single-value file" convention.
//
// Only the argument-validation guards are tested here: the public API rejects null and empty
// inputs with AMDSMI_STATUS_INVAL before it reaches these helpers, so only a direct call
// exercises them. The file-based behavior is tested through the public API in
// amdsmi_npm_limit_test.cc.

#include "rocm_smi/rocm_smi_npm.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>

#include "test_fixture_utils.h"

// ---------------------------------------------------------------------
// set_npm_board_limit()
// ---------------------------------------------------------------------

TEST(GpuUnit, SetNpmBoardLimitEmptyPathIsInvalidArgs) {
  EXPECT_EQ(amd::smi::set_npm_board_limit("", 100), RSMI_STATUS_INVALID_ARGS);
}

// ---------------------------------------------------------------------
// get_npm_node_power()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmNodePowerNullOutPtrIsInvalidArgs) {
  FakeSysfsTree board;
  board.WriteFile("node_power", "123");
  EXPECT_EQ(amd::smi::get_npm_node_power(board.root(), nullptr), RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, GetNpmNodePowerEmptyPathIsInvalidArgs) {
  uint64_t power = 0;
  EXPECT_EQ(amd::smi::get_npm_node_power("", &power), RSMI_STATUS_INVALID_ARGS);
}

// ---------------------------------------------------------------------
// get_npm_board_max_limit()
// ---------------------------------------------------------------------

TEST(GpuUnit, GetNpmBoardMaxLimitNullOutPtrIsInvalidArgs) {
  FakeSysfsTree board;
  board.WriteFile("max_node_power_limit", "6400");
  EXPECT_EQ(amd::smi::get_npm_board_max_limit(board.root(), nullptr), RSMI_STATUS_INVALID_ARGS);
}

TEST(GpuUnit, GetNpmBoardMaxLimitEmptyPathIsInvalidArgs) {
  uint64_t limit = 0;
  EXPECT_EQ(amd::smi::get_npm_board_max_limit("", &limit), RSMI_STATUS_INVALID_ARGS);
}
