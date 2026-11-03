// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file error_report.h
/// @brief Shared helper for the patch layer's optional-error-string convention:
///        functions that may fail take a `std::string *error_out` (often null)
///        and assign a diagnostic when it is non-null.

#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace rocjitsu {

/// @brief Write @p msg into @p out when @p out is non-null; no-op otherwise.
///
/// Replaces (does not append). Each function that takes a `std::string *`
/// out-param is contracted to report at most one failure per call. This is
/// so that the string holds "this call's single failure reason", as opposed to
/// a running log. The caller is in charge of Cross-call accumulation (e.g.
/// `std::vector<std::string>`).
inline void report(std::string *out, const char *msg) {
  if (out)
    *out = msg;
}

/// @brief Diagnostic sink that replaces an optional caller-owned error string.
/// @details The sink owns its optional prefix and prepends it on every replacement
/// of the output string; messages do not accumulate. Keep this callable alive as
/// long as a DiagnosticEmitter borrows it.
class OptionalStringDiagnosticSink {
public:
  explicit OptionalStringDiagnosticSink(std::string *out, std::string prefix = {})
      : out_(out), prefix_(std::move(prefix)) {}

  void operator()(std::string_view message) const {
    if (out_ != nullptr)
      *out_ = prefix_ + std::string(message);
  }

private:
  std::string *out_;
  std::string prefix_;
};

} // namespace rocjitsu
