// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

struct RdnaSwmmacIntegerCase {
  uint32_t width;
  uint32_t key;
  uint32_t dst;
  std::array<uint32_t, 256> expected;
};
