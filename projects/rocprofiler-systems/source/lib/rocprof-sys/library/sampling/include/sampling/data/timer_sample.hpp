// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/backtrace_metrics_data.hpp"
#include "sampling/data/stack_frame.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::sampling
{

struct timer_sample
{
    int64_t                  tid    = 0;
    uint64_t                 beg_ns = 0;
    uint64_t                 end_ns = 0;
    std::vector<stack_frame> stack;
    backtrace_metrics_data   metrics = {};

    bool operator<(timer_sample const& other) const noexcept
    {
        return beg_ns < other.beg_ns;
    }
};

}  // namespace rocprofsys::sampling
