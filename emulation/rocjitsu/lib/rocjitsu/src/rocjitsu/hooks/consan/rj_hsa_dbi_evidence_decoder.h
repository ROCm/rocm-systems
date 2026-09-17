// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_pipeline.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sync.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::consan::hook {

struct Evidence {
  uint32_t index = 0;
  WatchpointEntry entry;
  SyncDecodeResult sync;
  uint64_t packed_watchpoint = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t cluster_workgroup_id = 0;
  bool sync_snapshot_usable = true;
  const AccessStaticMapping *static_mapping = nullptr;
  uint64_t exact_lane_mask = 0;
};

static_assert(sizeof(Evidence) <= 128, "ConSan host evidence must fit within 128 bytes");

enum class EvidenceReason : uint8_t {
  MalformedWindow,
  EmptyWatchpoint,
  MalformedWatchpoint,
};

struct EvidenceIssue {
  EvidenceReason reason = EvidenceReason::MalformedWindow;
  uint32_t index = 0;
  std::array<uint64_t, 10> words{};
};

struct DecodedEvidence {
  uint64_t watchpoint_slots_examined = 0;
  uint64_t pending_release_slots_examined = 0;
  bool synchronization_evidence_complete = false;
  std::vector<Evidence> evidence;
  std::vector<EvidenceIssue> issues;
};

[[nodiscard]] DecodedEvidence decode_evidence(const ReportPipelineInput &input,
                                              const ReportHeader &header,
                                              std::span<const uint8_t> report_bytes,
                                              ReportSummary &summary);

} // namespace rocjitsu::consan::hook
