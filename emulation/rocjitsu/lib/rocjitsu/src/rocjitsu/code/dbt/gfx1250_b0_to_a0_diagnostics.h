// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_diagnostics.h
/// @brief Internal C-view adapter for gfx1250 B0-to-A0 diagnostics.

#pragma once

#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <vector>

namespace rocjitsu {

/// @details Given default visibility rather than left hidden because the
/// translator is now a shared library whose other symbols are all hidden -- that
/// is what makes its build id a meaningful cache identity. The in-tree library
/// test drives this fan-out directly and would otherwise be unable to reach it.
/// The version script names it explicitly, so the exported surface stays
/// deliberate rather than incidental.
RJ_API_EXPORT void
emit_gfx1250_b0_to_a0_diagnostics(rj_gfx1250_b0_to_a0_diagnostic_callback_t callback,
                                  void *user_data,
                                  const std::vector<TranslationDiagnostic> &diagnostics) noexcept;

} // namespace rocjitsu
