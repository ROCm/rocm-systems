// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_report_common_contract.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocjitsu::consan::hook {

enum class ReportSnapshotFailure : uint8_t {
  None,
  InvalidSource,
  CopyUnavailable,
  CopyFailed,
};

struct ReportSnapshotRequest {
  const void *source = nullptr;
  size_t size = 0;
  bool fine_grained = false;
};

/// Adapter used only for coarse-grained allocations. The lifecycle layer can
/// bind this contract to hsa_memory_copy; decoder and analyzer tests never
/// need an HSA API table.
using ReportCopy = bool (*)(void *context, void *destination, const void *source, size_t size,
                            int32_t *status);

struct ReportSnapshot {
  std::vector<uint8_t> bytes;
  uint64_t copied_bytes = 0;
  ReportSnapshotFailure failure = ReportSnapshotFailure::None;
  int32_t copy_status = 0;

  [[nodiscard]] bool complete() const { return failure == ReportSnapshotFailure::None; }
};

/// Captures one quiescent report into ordinary host memory. Invalid report
/// headers and layouts are copied in full so the evidence decoder, rather than
/// lifecycle code, owns their typed classification.
[[nodiscard]] ReportSnapshot capture_report_snapshot(const ReportSnapshotRequest &request,
                                                     ReportCopy coarse_copy = nullptr,
                                                     void *coarse_copy_context = nullptr);

} // namespace rocjitsu::consan::hook
