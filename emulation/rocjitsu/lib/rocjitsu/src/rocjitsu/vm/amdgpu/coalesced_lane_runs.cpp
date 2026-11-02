// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/coalesced_lane_runs.h"

#include <algorithm>

namespace rocjitsu::amdgpu::detail {

RJ_NOINLINE void sort_lane_addresses(CoalescedLaneAddress *lanes, uint32_t count) {
  std::ranges::sort(lanes, lanes + count,
                    [](const CoalescedLaneAddress &a, const CoalescedLaneAddress &b) {
                      return a.address < b.address;
                    });
}

} // namespace rocjitsu::amdgpu::detail
