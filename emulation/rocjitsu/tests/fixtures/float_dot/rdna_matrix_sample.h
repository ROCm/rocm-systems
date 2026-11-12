// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// One selected output word from a matrix capture, independent of its test harness.
#pragma once
#include <cstdint>
namespace rocjitsu::matrix_test {
// Destination word index is VGPR offset * wave width + lane.
struct Sample {
  uint32_t word, bits;
};
} // namespace rocjitsu::matrix_test
