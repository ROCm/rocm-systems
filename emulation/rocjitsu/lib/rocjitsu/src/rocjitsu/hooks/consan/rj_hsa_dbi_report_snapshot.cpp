// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_snapshot.h"

#include <cstring>

namespace rocjitsu::consan::hook {

ReportSnapshot capture_report_snapshot(const ReportSnapshotRequest &request, ReportCopy coarse_copy,
                                       void *coarse_copy_context) {
  ReportSnapshot result;
  if (request.source == nullptr || request.size < sizeof(ReportHeader)) {
    result.failure = ReportSnapshotFailure::InvalidSource;
    return result;
  }

  result.bytes.resize(request.size);
  if (!request.fine_grained) {
    if (coarse_copy == nullptr) {
      result.failure = ReportSnapshotFailure::CopyUnavailable;
      result.bytes.clear();
      return result;
    }
    if (!coarse_copy(coarse_copy_context, result.bytes.data(), request.source, request.size,
                     &result.copy_status)) {
      result.failure = ReportSnapshotFailure::CopyFailed;
      result.bytes.clear();
      return result;
    }
    result.copied_bytes = request.size;
    return result;
  }

  // Copy the bounded allocation; the decoder validates its header and layout.
  std::memcpy(result.bytes.data(), request.source, request.size);
  result.copied_bytes = request.size;
  return result;
}

} // namespace rocjitsu::consan::hook
