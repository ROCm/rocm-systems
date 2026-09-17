// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_metadata.hpp"

#include <sys/types.h>

#include <cstdint>
#include <numeric>
#include <span>
#include <string>

namespace rocprofsys::output
{

enum class output_format
{
    perfetto,
    rocpd,
    json,
    text
};

struct artifact
{
    std::string   path;
    pid_t         pid{ NO_PID };
    std::uint64_t size_bytes{ 0 };
    output_format format{ output_format::perfetto };
};

[[nodiscard]] inline std::uint64_t
sum_sizes(std::span<const artifact> items)
{
    return std::accumulate(
        items.begin(), items.end(), std::uint64_t{ 0 },
        [](std::uint64_t acc, const artifact& item) { return acc + item.size_bytes; });
}

}  // namespace rocprofsys::output
