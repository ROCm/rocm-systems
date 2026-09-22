// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file error_report.h
/// @brief Shared helper for the patch layer's optional-error-string convention:
///        functions that may fail take a `std::string *error_out` (often null)
///        and assign a diagnostic when it is non-null.

#pragma once

#include <string>

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

} // namespace rocjitsu
