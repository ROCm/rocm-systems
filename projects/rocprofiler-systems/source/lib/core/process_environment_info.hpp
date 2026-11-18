// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <nlohmann/json_fwd.hpp>

#include <span>
#include <string>
#include <string_view>

namespace rocprofsys
{
/**
 * Keeps the allowlisted variables of an environment block and rewrites the user's home
 * directory in their values to "~".
 *
 * Only variables named by an explicit allowlist are kept, so nothing unlisted can reach
 * a shared database. Redaction is best effort: values such as PATH can still embed
 * other user names (e.g. /scratch/bobby), so the result must not be treated as
 * anonymised.
 *
 * @param entries "NAME=value" entries; entries without '=' or with an empty name are
 *                skipped.
 * @param home    Home directory to redact. Empty or "/" disables redaction.
 * @return A JSON object mapping variable names to redacted values; empty when nothing
 *         is allowlisted.
 */
[[nodiscard]] nlohmann::json
filter_environment(std::span<const char* const> entries, std::string_view home);

/**
 * Snapshots the calling process's environment with @ref filter_environment, using $HOME.
 */
[[nodiscard]] nlohmann::json
capture_process_environment();

/**
 * @return @p environment as compact JSON. Bytes that are not valid UTF-8 are replaced
 *         with U+FFFD; this never throws on them.
 */
[[nodiscard]] std::string
dump_environment(const nlohmann::json& environment);
}  // namespace rocprofsys
