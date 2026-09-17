// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_identity_contracts.h"

namespace rocjitsu::consan::detail {
namespace {

[[nodiscard]] uint8_t
exact_entry_workgroup_capture_count(const OperatingPoint &point,
                                    const PersistentWorkgroupPrivateOffsets *private_offsets) {
  return static_cast<uint8_t>(point.persistent_sgprs.exact_workgroup.complete()) +
         static_cast<uint8_t>(point.exact_workgroup_vgprs.complete()) +
         static_cast<uint8_t>(private_offsets && private_offsets->complete());
}

} // namespace

bool has_exact_entry_workgroup_capture(const OperatingPoint &point,
                                       const PersistentWorkgroupPrivateOffsets *private_offsets) {
  return exact_entry_workgroup_capture_count(point, private_offsets) != 0u;
}

bool exact_entry_workgroup_capture_is_unambiguous(
    const OperatingPoint &point, const PersistentWorkgroupPrivateOffsets *private_offsets) {
  return exact_entry_workgroup_capture_count(point, private_offsets) <= 1u;
}

bool has_runtime_hardware_dispatch_id(const OperatingPoint &point) {
  return point.dispatch_sgpr.base().has_value();
}

} // namespace rocjitsu::consan::detail
