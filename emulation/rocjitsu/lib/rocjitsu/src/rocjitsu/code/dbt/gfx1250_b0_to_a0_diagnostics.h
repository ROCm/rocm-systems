// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_diagnostics.h
/// @brief Internal C-view adapter for gfx1250 B0-to-A0 diagnostics.

#pragma once

#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <vector>

namespace rocjitsu {

void emit_gfx1250_b0_to_a0_diagnostics(
    rj_gfx1250_b0_to_a0_diagnostic_callback_t callback, void *user_data,
    const std::vector<TranslationDiagnostic> &diagnostics) noexcept;

} // namespace rocjitsu
