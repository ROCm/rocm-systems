// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocprofsys::sampling
{

struct pause_interval
{
    uint64_t pause_ns  = 0;
    uint64_t resume_ns = 0;

    bool operator<(pause_interval const& o) const noexcept
    {
        return pause_ns < o.pause_ns;
    }
};

}  // namespace rocprofsys::sampling
