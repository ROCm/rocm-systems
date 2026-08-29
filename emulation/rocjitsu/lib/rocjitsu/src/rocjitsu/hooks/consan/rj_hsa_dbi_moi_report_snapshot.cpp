// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_snapshot.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rocjitsu::consan_hook {

std::optional<AutoMoiRecordReplaySnapshotPlan> plan_auto_moi_record_replay_snapshot(
    const ConSanMoiReportHeader &header, const ConSanMoiReportBufferLayout &layout,
    size_t allocation_size, uint32_t access_table_capacity, uint32_t visible_barriers,
    uint32_t visible_atomics, uint32_t visible_fences, uint32_t visible_diagnostics) {
  AutoMoiRecordReplaySnapshotPlan result;
  const auto append = [&](size_t offset, uint32_t count, size_t element_size) {
    if (count == 0)
      return true;
    if (offset > allocation_size ||
        static_cast<size_t>(count) > (allocation_size - offset) / element_size)
      return false;
    const size_t size = static_cast<size_t>(count) * element_size;
    if (result.range_count == result.ranges.size() ||
        result.copied_bytes > std::numeric_limits<size_t>::max() - size)
      return false;
    result.ranges[result.range_count++] = {.offset = offset, .size = size};
    result.copied_bytes += size;
    return true;
  };

  if (!append(0, 1, sizeof(ConSanMoiReportHeader)) ||
      !append(layout.access_records_offset, access_table_capacity, sizeof(ConSanMoiAccessRecord)) ||
      !append(layout.barrier_records_offset, visible_barriers, sizeof(ConSanMoiBarrierRecord)) ||
      !append(layout.atomic_records_offset, visible_atomics, sizeof(ConSanMoiAtomicRecord)) ||
      !append(layout.fence_records_offset, visible_fences, sizeof(ConSanMoiFenceRecord)) ||
      !append(layout.diagnostic_records_offset, visible_diagnostics,
              sizeof(ConSanMoiDiagnosticRecord))) {
    return std::nullopt;
  }
  if (access_table_capacity != header.access_record_capacity ||
      visible_barriers > header.barrier_record_capacity ||
      visible_atomics > header.atomic_record_capacity ||
      visible_fences > header.fence_record_capacity ||
      visible_diagnostics > header.diagnostic_capacity) {
    return std::nullopt;
  }
  return result;
}

AutoMoiReportSnapshot capture_auto_moi_report_snapshot(const AutoMoiReportSnapshotRequest &request,
                                                       AutoMoiReportCopy coarse_copy,
                                                       void *coarse_copy_context) {
  AutoMoiReportSnapshot result;
  if (request.source == nullptr || request.size < sizeof(ConSanMoiReportHeader)) {
    result.failure = AutoMoiReportSnapshotFailure::InvalidSource;
    return result;
  }

  result.bytes.resize(request.size);
  if (!request.fine_grained) {
    if (coarse_copy == nullptr) {
      result.failure = AutoMoiReportSnapshotFailure::CopyUnavailable;
      result.bytes.clear();
      return result;
    }
    if (!coarse_copy(coarse_copy_context, result.bytes.data(), request.source, request.size,
                     &result.copy_status)) {
      result.failure = AutoMoiReportSnapshotFailure::CopyFailed;
      result.bytes.clear();
      return result;
    }
    result.copied_bytes = request.size;
    return result;
  }

  std::memcpy(result.bytes.data(), request.source, sizeof(ConSanMoiReportHeader));
  const auto &header = *reinterpret_cast<const ConSanMoiReportHeader *>(result.bytes.data());
  if (request.expected_engine == ConSanMoiEngine::RecordReplay &&
      consan_moi_report_header_is_current(header) &&
      consan_moi_report_layout_matches_header(header, request.expected_layout,
                                              request.expected_engine, request.size)) {
    const uint32_t visible_barriers =
        std::min(header.barrier_record_count, header.barrier_record_capacity);
    const uint32_t visible_atomics =
        std::min(header.atomic_record_count, header.atomic_record_capacity);
    const uint32_t visible_fences =
        std::min(header.fence_record_count, header.fence_record_capacity);
    const uint32_t visible_diagnostics =
        std::min(header.diagnostic_count, header.diagnostic_capacity);
    const auto plan = plan_auto_moi_record_replay_snapshot(
        header, request.expected_layout, request.size,
        request.expected_layout.access_record_capacity, visible_barriers, visible_atomics,
        visible_fences, visible_diagnostics);
    if (plan) {
      const auto *source = static_cast<const uint8_t *>(request.source);
      for (size_t index = 0; index < plan->range_count; ++index) {
        const AutoMoiReportSnapshotRange &range = plan->ranges[index];
        if (range.offset != 0)
          std::memcpy(result.bytes.data() + range.offset, source + range.offset, range.size);
      }
      result.copied_bytes = plan->copied_bytes;
      return result;
    }
  }

  // A malformed header or sparse plan is decoder evidence, not a capture
  // failure. A full bounded copy preserves it without trusting any offsets or
  // counts read from the report itself.
  std::memcpy(result.bytes.data(), request.source, request.size);
  result.copied_bytes = request.size;
  return result;
}

} // namespace rocjitsu::consan_hook
