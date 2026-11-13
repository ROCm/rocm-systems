// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file json_fields.h
/// @brief Text-preserving field access shared by configuration parsing and launch overrides.
#pragma once

#include "rocjitsu/result.h"
#include "util/diagnostic.h"

#include <string>
#include <string_view>

namespace rocjitsu::config {

/// @brief Return a top-level field's original JSON value, or an empty string when absent.
[[nodiscard]] FailureOr<std::string>
json_top_level_field(std::string_view json, std::string_view field,
                     const util::DiagnosticEmitter &emit_error = {});

/// @brief Replace or insert a top-level field, preserving unrelated config text.
[[nodiscard]] FailureOr<std::string> set_top_level_field(std::string_view json,
                                                         std::string_view field,
                                                         std::string_view value,
                                                         const util::DiagnosticEmitter &emit_error);

} // namespace rocjitsu::config
