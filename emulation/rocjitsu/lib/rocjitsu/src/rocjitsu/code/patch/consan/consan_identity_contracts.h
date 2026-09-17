// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_instrumentation.h"
#include "rocjitsu/code/patch/consan/consan_report.h"

namespace rocjitsu::consan::detail {

[[nodiscard]] inline ReportBufferLayout
resolve_report_layout(const BoundRuntimeResources &resources, ReportBufferLayout fallback_layout) {
  return resources.report_layout
             ? revalidate_report_layout(*resources.report_layout, resources.report_buffer_size)
             : fallback_layout;
}

[[nodiscard]] bool has_exact_entry_workgroup_capture(
    const OperatingPoint &point,
    const PersistentWorkgroupPrivateOffsets *private_offsets = nullptr);

[[nodiscard]] bool exact_entry_workgroup_capture_is_unambiguous(
    const OperatingPoint &point,
    const PersistentWorkgroupPrivateOffsets *private_offsets = nullptr);

[[nodiscard]] bool has_runtime_hardware_dispatch_id(const OperatingPoint &point);

} // namespace rocjitsu::consan::detail
