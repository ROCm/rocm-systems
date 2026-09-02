// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_pipeline.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sampled_sync.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::consan_hook {

enum class AutoMoiReportDecodeFailure : uint8_t {
  None,
  SnapshotTooSmall,
  InvalidHeader,
  LayoutMismatch,
};

enum class AutoMoiReportEvidenceReason : uint8_t {
  ExactMalformed,
  ReleasePublishing,
  SampledMalformedWindow,
  SampledEmptyWatchpoint,
  SampledMalformedWatchpoint,
  CompactDiagnosticTokenUnresolved,
};

/// Typed malformed/incomplete evidence. `words` preserve bounded raw detail
/// for stable rendering without allowing the renderer to reinterpret bytes.
struct AutoMoiReportEvidenceIssue {
  AutoMoiReportEvidenceReason reason = AutoMoiReportEvidenceReason::ExactMalformed;
  uint32_t index = 0;
  std::array<uint64_t, 12> words{};
};

struct AutoMoiExactShadowEvidence {
  uint32_t index = 0;
  ConSanMoiExactShadowEntry entry;
  ConSanMoiExactByteCellProvenance byte_provenance;
  uint64_t dispatch_id = 0;
  uint32_t version = 0;
};

struct AutoMoiInlineAtomicReleaseEvidence {
  uint32_t index = 0;
  ConSanMoiInlineAtomicReleaseSlot slot;
  ConSanMoiInlineCausalSnapshot snapshot;
};

struct AutoMoiInlineAcquiredTokenEvidence {
  uint32_t index = 0;
  ConSanMoiInlineAcquiredEpochTokenSlot token;
};

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

struct CompactRecordReplayAccessRecords {
  uint32_t committed_record_count = 0;
  std::vector<ConSanMoiAccessRecord> replay_records;
};

[[nodiscard]] CompactRecordReplayAccessRecords
compact_record_replay_access_records(std::span<const ConSanMoiAccessRecord> records,
                                     uint32_t publication_hint);

struct AutoMoiDecodedReport {
  AutoMoiReportDecodeFailure failure = AutoMoiReportDecodeFailure::None;
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  ConSanMoiReportHeader header;
  AutoMoiReportSummary summary;

  uint32_t deferred_token_qualified_diagnostic_count = 0;
  uint64_t sampled_watchpoint_slots_examined = 0;
  uint64_t sampled_pending_release_slots_examined = 0;
  bool sampled_synchronization_evidence_complete = false;

  std::vector<ConSanMoiAccessRecord> visible_access_slots;
  std::vector<ConSanMoiAccessRecord> replay_access_records;
  std::vector<ConSanMoiBarrierRecord> barrier_records;
  std::vector<ConSanMoiAtomicRecord> atomic_records;
  std::vector<ConSanMoiFenceRecord> fence_records;
  std::vector<ConSanMoiDiagnosticRecord> diagnostics;
  std::vector<AutoMoiExactShadowEvidence> exact_shadow;
  std::vector<AutoMoiInlineAtomicReleaseEvidence> inline_atomic_releases;
  std::vector<AutoMoiInlineAcquiredTokenEvidence> inline_acquired_tokens;
  std::vector<AutoMoiSampledEvidence> sampled;
  std::vector<AutoMoiReportEvidenceIssue> issues;

  [[nodiscard]] bool complete() const { return failure == AutoMoiReportDecodeFailure::None; }
};

[[nodiscard]] AutoMoiDecodedReport decode_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                                          const AutoMoiReportSnapshot &snapshot,
                                                          AutoMoiReportSummary initial_summary);

} // namespace rocjitsu::consan_hook
