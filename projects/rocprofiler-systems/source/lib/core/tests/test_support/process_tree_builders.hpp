// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_metadata.hpp"
#include "core/output_file_registry.hpp"

#include <sys/types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::test_support
{

inline rocprofsys::output_file
make_row(const std::string& path, pid_t pid,
         std::optional<std::uintmax_t> size_bytes = std::nullopt)
{
    rocprofsys::output_file f{};
    f.path       = path;
    f.pid        = pid;
    f.size_bytes = size_bytes;
    return f;
}

inline rocprofsys::output::process_metadata
make_meta(pid_t pid, pid_t ppid, std::string command = "", std::vector<int> gpu_ids = {})
{
    rocprofsys::output::process_metadata m{};
    m.pid     = pid;
    m.ppid    = ppid;
    m.command = std::move(command);
    m.gpu_ids = std::move(gpu_ids);
    return m;
}

}  // namespace rocprofsys::test_support
