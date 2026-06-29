/*
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace rocr {
namespace core {

constexpr uint32_t kAsyncEventsPollNapFloorUs = 20;
constexpr uint32_t kAsyncEventsPollNapCeilingUs = 2000;

inline uint32_t NextAsyncEventsPollNapUs(uint32_t current) {
  if (current < kAsyncEventsPollNapFloorUs) {
    return kAsyncEventsPollNapFloorUs;
  }
  return std::min(current * 2, kAsyncEventsPollNapCeilingUs);
}

}  // namespace core
}  // namespace rocr
