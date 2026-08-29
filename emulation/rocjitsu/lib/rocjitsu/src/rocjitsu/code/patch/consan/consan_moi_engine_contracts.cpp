// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"

namespace rocjitsu::consan_moi_detail {
namespace {

[[nodiscard]] uint8_t
record_replay_entry_workgroup_capture_count(const ConSanMoiOperatingPoint &point) {
  return static_cast<uint8_t>(point.moi_persistent_sgprs.record_replay_workgroup.complete()) +
         static_cast<uint8_t>(point.moi_record_replay_workgroup_vgprs.complete()) +
         static_cast<uint8_t>(point.moi_record_replay_workgroup_private_offsets.complete());
}

} // namespace

bool record_replay_uses_automatic_banked_capture(const ConSanRequest &request,
                                                 uint32_t dispatch_token_capacity) {
  return request.moi_engine == ConSanMoiEngine::RecordReplay &&
         !request.moi_dynamic_access_records && dispatch_token_capacity != 0u;
}

ConSanMoiReportBufferLayout resolve_moi_report_layout(const ConSanRequest &request,
                                                      const BoundRuntimeResources &resources) {
  if (resources.moi_report_layout) {
    return revalidate_consan_moi_report_layout(*resources.moi_report_layout, request.moi_engine,
                                               resources.moi_report_buffer_size);
  }
  switch (request.moi_engine) {
  case ConSanMoiEngine::RecordReplay:
    return consan_moi_report_buffer_layout_for_bytes(
        resources.moi_report_buffer_size, request.moi_track_barriers, request.moi_track_atomics,
        request.moi_track_atomics);
  case ConSanMoiEngine::Sampled:
    return consan_moi_direct_sampled_report_buffer_layout_for_bytes(
        resources.moi_report_buffer_size);
  case ConSanMoiEngine::InlineShadow:
    return consan_moi_inline_shadow_report_buffer_layout_for_bytes(
        resources.moi_report_buffer_size);
  }
  return {};
}

bool record_replay_uses_automatic_banked_capture(const ConSanRequest &request,
                                                 const BoundRuntimeResources &resources) {
  return resources.moi_report_layout &&
         record_replay_uses_automatic_banked_capture(
             request, resources.moi_report_layout->record_replay_dispatch_token_capacity);
}

bool record_replay_requires_entry_workgroup_capture(ConSanMoiEngine engine) {
  return engine == ConSanMoiEngine::RecordReplay || engine == ConSanMoiEngine::Sampled;
}

bool record_replay_has_entry_workgroup_capture(const ConSanMoiOperatingPoint &point) {
  return record_replay_entry_workgroup_capture_count(point) != 0u;
}

bool record_replay_entry_workgroup_capture_is_unambiguous(const ConSanMoiOperatingPoint &point) {
  return record_replay_entry_workgroup_capture_count(point) <= 1u;
}

bool moi_has_runtime_hardware_dispatch_id(const ConSanMoiOperatingPoint &point) {
  return point.moi_dispatch_id_sgpr || point.moi_dispatch_id_vgpr;
}

void note_moi_persistent_vgpr_state(ConSanPatchInfo &patch, const ConSanMoiOperatingPoint &point,
                                    const ConSanMoiOperatingPoint &allocation) {
  patch.persistent_owner_vgpr = point.moi_owner_vgpr;
  patch.persistent_epoch_vgpr = point.moi_epoch_vgpr;
  patch.persistent_workgroup_key_vgpr = point.moi_workgroup_key_vgpr;
  patch.persistent_record_replay_workgroup_vgprs = point.moi_record_replay_workgroup_vgprs;
  patch.persistent_vgpr_state_owner_local = !allocation.owner_persistent_vgprs.empty();
  patch.persistent_vgpr_state_is_abi =
      point.moi_owner_vgpr && point.moi_epoch_vgpr && !point.moi_persistent_sgprs.complete();
}

} // namespace rocjitsu::consan_moi_detail
