// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_diagnostics.h
/// @brief Internal diagnostic adapter for gfx1250 B0-to-A0 translation.

#pragma once

#include "rocjitsu/code/dbt/gfx1250_b0_to_a0_library.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"

#include <vector>

namespace rocjitsu {

/// @brief Distinguish rejected input from an internal translation failure.
inline rj_status_t
gfx1250_b0_to_a0_translation_status(const std::vector<TranslationDiagnostic> &diagnostics,
                                    bool has_output, bool dispatchable) {
  if (has_diagnostic_kind(diagnostics, DiagnosticKind::InternalError))
    return ROCJITSU_STATUS_ERROR;
  if (!has_output || !dispatchable)
    return ROCJITSU_STATUS_INVALID_CODE_OBJECT;
  return ROCJITSU_STATUS_SUCCESS;
}

/// @details Given default visibility rather than left hidden because the
/// translator is now a shared library whose other symbols are all hidden -- that
/// is what makes its build id a meaningful cache identity. The in-tree library
/// test drives this fan-out directly and would otherwise be unable to reach it.
/// The version script names it explicitly, so the exported surface stays
/// deliberate rather than incidental.
RJ_API_EXPORT void
emit_gfx1250_b0_to_a0_diagnostics(rocjitsu::Gfx1250B0ToA0DiagnosticCallback callback,
                                  void *user_data,
                                  const std::vector<TranslationDiagnostic> &diagnostics) noexcept;

} // namespace rocjitsu
