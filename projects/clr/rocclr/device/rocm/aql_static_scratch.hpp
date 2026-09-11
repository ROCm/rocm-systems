// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>

namespace amd::roc::aql_resident {
// Reserve this fixed capacity at physical queue creation. Resident dispatches
// are admitted only when they fit; larger or dynamic-stack kernels retain the
// ordinary dispatch path. Reclamation must be disabled for this queue policy.
constexpr uint32_t kStaticScratchWave64LaneBytes = 256;
constexpr bool fitsStaticScratch(uint32_t privateBytes, bool wave32) {
  return privateBytes <= kStaticScratchWave64LaneBytes * (wave32 ? 2u : 1u);
}
}  // namespace amd::roc::aql_resident
