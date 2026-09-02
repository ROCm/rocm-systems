// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_pipeline.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sampled_sync.h"

#include <array>
#include <cstdint>
#include <vector>

namespace rocjitsu::consan_hook {

struct AutoMoiSampledEvidence {
  uint32_t index = 0;
  ConSanMoiSampledWatchpointEntry entry;
  ConSanMoiSampledSyncDecodeResult sync;
  uint64_t packed_watchpoint = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t cluster_workgroup_id = 0;
  const AutoMoiSampledStaticMapping *static_mapping = nullptr;
  bool sync_snapshot_usable = true;
};

enum class AutoMoiSampledEvidenceReason : uint8_t {
  MalformedWindow,
  EmptyWatchpoint,
  MalformedWatchpoint,
};

struct AutoMoiSampledEvidenceIssue {
  AutoMoiSampledEvidenceReason reason = AutoMoiSampledEvidenceReason::MalformedWindow;
  uint32_t index = 0;
  std::array<uint64_t, 10> words{};
};

struct AutoMoiSampledDecodedReport {
  uint64_t watchpoint_slots_examined = 0;
  uint64_t pending_release_slots_examined = 0;
  bool synchronization_evidence_complete = false;
  std::vector<AutoMoiSampledEvidence> evidence;
  std::vector<AutoMoiSampledEvidenceIssue> issues;
};

} // namespace rocjitsu::consan_hook
