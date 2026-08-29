// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

namespace rocjitsu::consan_moi_detail {

[[nodiscard]] ConSanMoiReportBufferLayout
resolve_moi_report_layout(const ConSanRequest &request, const BoundRuntimeResources &resources);

[[nodiscard]] bool
record_replay_uses_automatic_banked_capture(const ConSanRequest &request,
                                            const BoundRuntimeResources &resources);

[[nodiscard]] bool record_replay_uses_automatic_banked_capture(const ConSanRequest &request,
                                                               uint32_t dispatch_token_capacity);

[[nodiscard]] bool record_replay_requires_entry_workgroup_capture(ConSanMoiEngine engine);

[[nodiscard]] bool record_replay_has_entry_workgroup_capture(const ConSanMoiOperatingPoint &point);

[[nodiscard]] bool
record_replay_entry_workgroup_capture_is_unambiguous(const ConSanMoiOperatingPoint &point);

void note_moi_persistent_vgpr_state(ConSanPatchInfo &patch, const ConSanMoiOperatingPoint &point,
                                    const ConSanMoiOperatingPoint &allocation);

} // namespace rocjitsu::consan_moi_detail
