// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace profiler_hub
{

// <cache_root>/<16 hex digits>. timeline/, catalog/, and summary.bin live under it.
// Empty when the root is empty, no sources are given, or a source is not a
// regular file. The hex digits are FNV-1a of each path, size, and last-write
// time, in the order of @p source_paths.
[[nodiscard]] std::string trace_cache_directory(
    std::string_view cache_root, const std::vector<std::string>& source_paths);

}  // namespace profiler_hub
