// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_abi.h"

#include <cstddef>
#include <cstdint>

namespace rocjitsu::consan {

/// Concrete, address-free byte geometry of one ConSan report allocation.
///
/// Capacities state how many ABI records each region can hold; offsets name
/// their aligned locations relative to the beginning of the allocation.
/// Unused regions alias the end of the allocation and have zero capacity. This
/// value owns no allocation, device address, generation, or lifetime.
struct ReportBufferLayout {
  bool valid = false;
  uint32_t watchpoint_capacity = 0;
  uint32_t causal_window_capacity = 0;
  uint32_t sync_metadata_capacity = 0;
  uint32_t pending_acquire_capacity = 0;
  size_t watchpoints_offset = sizeof(ReportHeader);
  size_t causal_windows_offset = sizeof(ReportHeader);
  size_t sync_metadata_offset = sizeof(ReportHeader);
  size_t pending_acquires_offset = sizeof(ReportHeader);
  size_t required_bytes = sizeof(ReportHeader);

  bool operator==(const ReportBufferLayout &) const = default;
};

static_assert(sizeof(ReportBufferLayout) <= 64);

} // namespace rocjitsu::consan
