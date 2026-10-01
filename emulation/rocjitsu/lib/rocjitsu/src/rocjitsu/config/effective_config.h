// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file effective_config.h
/// @brief Per-launch simulation-config overrides that leave the source file untouched.

#ifndef ROCJITSU_CONFIG_EFFECTIVE_CONFIG_H_
#define ROCJITSU_CONFIG_EFFECTIVE_CONFIG_H_

#include "rocjitsu/result.h"
#include "util/diagnostic.h"

#include <cstdint>
#include <string>
#include <string_view>

#include <sys/types.h>

namespace rocjitsu {
namespace config {

/// @brief Name of the effective config written inside an invocation runtime directory.
inline constexpr char kEffectiveConfigName[] = "effective_config.json";

/// @brief Return @p json with its top-level `cpu_thread_budget` set to @p budget.
///
/// @details Gives one launch its own ceiling without rewriting the input file or
/// setting process-wide state, so simulations built from other configs in the same
/// process keep the budgets their own configs ask for. The field is replaced where
/// it already appears and inserted otherwise; all other text is preserved verbatim.
/// Failure means @p json is not a simulation-config object, or the rewritten document
/// does not parse back with @p budget. The reason is emitted through @p emit_error.
[[nodiscard]] FailureOr<std::string>
json_with_cpu_thread_budget(std::string_view json, uint32_t budget,
                            const util::DiagnosticEmitter &emit_error = {});

/// @brief Write the effective config for one invocation and return its path.
///
/// @details The copy is placed in the invocation runtime directory next to the
/// config-path handoff an exec'd workload already reads, so it is reclaimed by the
/// same cleanup and never lands beside a config file the launcher does not own.
/// Failure means the source cannot be read, the budget cannot be applied, or the
/// copy cannot be written. The reason is emitted through @p emit_error.
[[nodiscard]] FailureOr<std::string>
write_effective_config(const std::string &source_path, uint32_t budget, pid_t pid,
                       const util::DiagnosticEmitter &emit_error = {});

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_EFFECTIVE_CONFIG_H_
