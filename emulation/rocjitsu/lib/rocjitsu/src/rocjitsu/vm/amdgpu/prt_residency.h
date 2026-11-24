// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu::amdgpu {

/// An explicitly reserved, nonresident partially-resident-texture interval.
struct PrtRange {
  uint64_t begin;
  uint64_t end;
};
using PrtRanges = std::vector<PrtRange>;

/// Sparse reservations are published as one immutable, ordered snapshot per
/// address space. The amount of storage depends on the number of reservations.
struct PrtResidency {
  std::atomic<std::shared_ptr<const PrtRanges>> nonresident;
};
} // namespace rocjitsu::amdgpu
