// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu::consan_hook {

struct AutoMoiReportSnapshotRange {
  size_t offset = 0;
  size_t size = 0;
};

struct AutoMoiRecordReplaySnapshotPlan {
  std::array<AutoMoiReportSnapshotRange, 6> ranges{};
  size_t range_count = 0;
  size_t copied_bytes = 0;
};

/// Plans the cacheable host snapshot needed to summarize one quiescent
/// Record/Replay report. The open-addressed access table must be copied in
/// full because occupied slots may appear anywhere. Append-only event and
/// diagnostic arrays need only their published prefixes.
[[nodiscard]] std::optional<AutoMoiRecordReplaySnapshotPlan> plan_auto_moi_record_replay_snapshot(
    const ConSanMoiReportHeader &header, const ConSanMoiReportBufferLayout &layout,
    size_t allocation_size, uint32_t access_table_capacity, uint32_t visible_barriers,
    uint32_t visible_atomics, uint32_t visible_fences, uint32_t visible_diagnostics);

enum class AutoMoiReportSnapshotFailure : uint8_t {
  None,
  InvalidSource,
  CopyUnavailable,
  CopyFailed,
};

struct AutoMoiReportSnapshotRequest {
  const void *source = nullptr;
  size_t size = 0;
  ConSanMoiReportBufferLayout expected_layout;
  ConSanMoiEngine expected_engine = ConSanMoiEngine::RecordReplay;
  bool fine_grained = false;
};

/// Adapter used only for coarse-grained allocations. The lifecycle layer can
/// bind this contract to hsa_memory_copy; decoder and analyzer tests never
/// need an HSA API table.
using AutoMoiReportCopy = bool (*)(void *context, void *destination, const void *source,
                                   size_t size, int32_t *status);

struct AutoMoiReportSnapshot {
  std::vector<uint8_t> bytes;
  uint64_t copied_bytes = 0;
  AutoMoiReportSnapshotFailure failure = AutoMoiReportSnapshotFailure::None;
  int32_t copy_status = 0;

  [[nodiscard]] bool complete() const { return failure == AutoMoiReportSnapshotFailure::None; }
};

/// Captures one quiescent report into ordinary host memory. Invalid report
/// headers and layouts are copied in full so the evidence decoder, rather than
/// lifecycle code, owns their typed classification.
[[nodiscard]] AutoMoiReportSnapshot
capture_auto_moi_report_snapshot(const AutoMoiReportSnapshotRequest &request,
                                 AutoMoiReportCopy coarse_copy = nullptr,
                                 void *coarse_copy_context = nullptr);

} // namespace rocjitsu::consan_hook
