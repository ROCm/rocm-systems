// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace profiler_hub
{

// <cache_root>/<date>_<format>_<version>_<name>_<size>, for example
//   <cache_root>/20261106-110502_rocpd_v3.0.0_trace_70b_1024_32_1.5GB
// The parts are:
//   date     newest last-write time of the sources, local time, to the second
//            (YYYYMMDD-HHMMSS)
//   format   trace format from the probe, for example rocpd or unknown
//   version  schema version from the probe, for example v3.0.0
//   name     stem of the first source; "+N" is appended when N more sources follow
//   size     total size of the sources (KB, MB, or GB)
// timeline/, catalog/, and summary.bin live under the folder.
// Empty when the root is empty, no sources are given, or a source is not a
// regular file. The date pins the source contents for summary.bin, which does
// not check its sources, so the name must change when any source changes.
// The timeline and catalog caches also check each source's size, mtime, and path.
[[nodiscard]] std::string trace_cache_directory(
    std::string_view cache_root, const std::vector<std::string>& source_paths);

}  // namespace profiler_hub
