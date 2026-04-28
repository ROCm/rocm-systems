// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <bitset>
#include <cstdint>

namespace rocprofsys::sampling
{

// Plain data — replaces the timemory backtrace_metrics component for storage.
// 136 B layout matches the legacy bundle field-for-field (DEC-1 transcoder).
// TIMEMORY_PAPI_ARRAY_SIZE = 12 on this build (cmake/Packages.cmake:788).
struct backtrace_metrics_data
{
    std::bitset<6>            valid;
    int64_t                   cpu_ns      = 0;
    int64_t                   mem_peak_kb = 0;
    int64_t                   ctx_swch    = 0;
    int64_t                   page_flt    = 0;
    std::array<long long, 12> hw_counter  = {};
};

}  // namespace rocprofsys::sampling
