// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>

namespace profiler_hub::common
{
using steady_clock_t = std::chrono::steady_clock;

struct file_closer
{
    void operator()(std::FILE* f) const
    {
        if(f != nullptr) std::fclose(f);
    }
};

inline std::uint64_t
elapsed_ms(steady_clock_t::time_point from)
{
    const auto elapsed = steady_clock_t::now() - from;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}
}  // namespace profiler_hub::common
