// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/stack_frame.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::sampling
{

struct overflow_sample
{
    int64_t                  tid    = 0;
    uint64_t                 beg_ns = 0;
    uint64_t                 end_ns = 0;
    std::vector<stack_frame> stack;

    bool operator<(overflow_sample const& other) const noexcept
    {
        return beg_ns < other.beg_ns;
    }
};

}  // namespace rocprofsys::sampling
