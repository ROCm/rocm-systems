// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/data_types.hpp"

#include <string>
#include <sys/types.h>

namespace rocprofsys::trace_cache::discovery
{

[[nodiscard]] data::directory_files_t
list_dir_files(const std::string& path);

[[nodiscard]] data::mapped_cache_files_t
find_cache_files(const pid_t& root_pid, const data::directory_files_t& dir_contents);

void
clear(const data::mapped_cache_files_t& cache_files);

void
merge_perfetto_files();

}  // namespace rocprofsys::trace_cache::discovery
