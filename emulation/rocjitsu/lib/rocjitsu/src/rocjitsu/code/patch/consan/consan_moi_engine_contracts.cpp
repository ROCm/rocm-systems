// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"

namespace rocjitsu::consan_moi_detail {
namespace {

[[nodiscard]] uint8_t moi_exact_entry_workgroup_capture_count(
    const ConSanMoiOperatingPoint &point,
    const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets) {
  return static_cast<uint8_t>(point.moi_persistent_sgprs.exact_workgroup.complete()) +
         static_cast<uint8_t>(point.moi_exact_workgroup_vgprs.complete()) +
         static_cast<uint8_t>(private_offsets && private_offsets->complete());
}

} // namespace

bool record_replay_uses_automatic_banked_capture(const ConSanRequest &request,
                                                 uint32_t dispatch_token_capacity) {
  return request.moi_engine == ConSanMoiEngine::RecordReplay &&
         !request.moi_dynamic_access_records && dispatch_token_capacity != 0u;
}

bool record_replay_uses_automatic_banked_capture(const ConSanRequest &request,
                                                 const BoundRuntimeResources &resources) {
  return resources.moi_report_layout &&
         record_replay_uses_automatic_banked_capture(
             request, resources.moi_report_layout->record_replay_dispatch_token_capacity);
}

bool moi_has_exact_entry_workgroup_capture(
    const ConSanMoiOperatingPoint &point,
    const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets) {
  return moi_exact_entry_workgroup_capture_count(point, private_offsets) != 0u;
}

bool moi_exact_entry_workgroup_capture_is_unambiguous(
    const ConSanMoiOperatingPoint &point,
    const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets) {
  return moi_exact_entry_workgroup_capture_count(point, private_offsets) <= 1u;
}

bool moi_has_runtime_hardware_dispatch_id(const ConSanMoiOperatingPoint &point) {
  return point.moi_dispatch_identity.sgpr() || point.moi_dispatch_identity.vgpr();
}

} // namespace rocjitsu::consan_moi_detail
