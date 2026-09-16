// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_translate_cli.h
/// @brief Command-line entry point shared by rj_dbt_translate and its tests.

#pragma once

#include "dbt_translate.h"

namespace rocjitsu::tools::detail {

using TranslateCodeObjectFn = ToolResult<TranslateOutput> (*)(const TranslateOptions &);

/// @brief Run the rj_dbt_translate command-line interface.
///
/// The translation callback keeps CLI result handling independently testable
/// without adding test-only command-line behavior to the production binary.
int run_dbt_translate_cli(int argc, char **argv, TranslateCodeObjectFn translate);

} // namespace rocjitsu::tools::detail
