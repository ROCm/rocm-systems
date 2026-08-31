// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace wsl {
namespace thunk {

inline uint64_t AvailableVramBudget(uint64_t budget, uint64_t current_usage, uint64_t total) {
  if (current_usage >= budget) return 0;

  const uint64_t available = budget - current_usage;
  return available < total ? available : total;
}

}  // namespace thunk
}  // namespace wsl
