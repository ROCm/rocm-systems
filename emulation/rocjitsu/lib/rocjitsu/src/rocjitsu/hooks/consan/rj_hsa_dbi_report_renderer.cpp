// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_renderer.h"

#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_rendering.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <utility>

namespace rocjitsu::consan::hook {

std::string format_report_text(const char *format, ...) {
  std::va_list args;
  va_start(args, format);
  std::va_list size_args;
  va_copy(size_args, args);
  const int size = std::vsnprintf(nullptr, 0, format, size_args);
  va_end(size_args);
  if (size < 0) {
    va_end(args);
    return {};
  }
  std::string result(static_cast<size_t>(size), '\0');
  std::vsnprintf(result.data(), result.size() + 1u, format, args);
  va_end(args);
  return result;
}

std::vector<ReportDiagnostic> render_report(const ReportRenderInput &render_input) {
  std::vector<ReportDiagnostic> rendered;
  const auto append = [&rendered](ReportDiagnosticKind kind, const char *format, auto... args) {
    rendered.push_back({kind, format_report_text(format, args...)});
  };
  constexpr ReportDiagnosticKind kFailure = ReportDiagnosticKind::Failure;
  constexpr ReportDiagnosticKind kSummary = ReportDiagnosticKind::Summary;

  const ReportPipelineInput &input = render_input.pipeline;
  const DecodedReport &decoded = render_input.decoded;
  const ReportSummary &summary = render_input.summary;
  if (!decoded.complete()) {
    const ReportHeader &invalid_header = decoded.header;
    if (decoded.failure == ReportDecodeFailure::InvalidHeader) {
      append(kFailure,
             "ConSan auto report reader=%llu has invalid header magic=0x%08x "
             "abi=%u header_size=%u",
             static_cast<unsigned long long>(input.reader), invalid_header.magic,
             invalid_header.abi_version, invalid_header.header_size);
    } else if (decoded.failure == ReportDecodeFailure::LayoutMismatch) {
      append(kFailure, "ConSan auto report reader=%llu has inconsistent ABI-v%u layout",
             static_cast<unsigned long long>(input.reader), kReportAbiVersion);
    } else {
      append(kFailure, "ConSan auto report reader=%llu has undersized host snapshot",
             static_cast<unsigned long long>(input.reader));
    }
    return rendered;
  }
  if (render_input.conflict_analysis == nullptr) {
    append(kFailure, "ConSan auto report reader=%llu has missing typed analysis",
           static_cast<unsigned long long>(input.reader));
    return rendered;
  }

  ConflictRendering conflict_rendering = render_conflicts(
      input, decoded.header, summary, decoded.records, *render_input.conflict_analysis);
  rendered.insert(rendered.end(), std::make_move_iterator(conflict_rendering.evidence.begin()),
                  std::make_move_iterator(conflict_rendering.evidence.end()));

  const ReportHeader &header = decoded.header;
  append(kSummary,
         "ConSan auto report reader=%llu addr=0x%llx bytes=%zu generation=%llu "
         "code_object=%s event_counter=%u diagnostics=%u%s fine_grained=%s",
         static_cast<unsigned long long>(input.reader),
         static_cast<unsigned long long>(input.source_address), input.size,
         static_cast<unsigned long long>(header.generation),
         input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
         header.event_counter, conflict_rendering.effective_diagnostic_count,
         conflict_rendering.summary_fields.c_str(), input.fine_grained ? "true" : "false");

  rendered.insert(rendered.end(), std::make_move_iterator(conflict_rendering.details.begin()),
                  std::make_move_iterator(conflict_rendering.details.end()));
  return rendered;
}

} // namespace rocjitsu::consan::hook
