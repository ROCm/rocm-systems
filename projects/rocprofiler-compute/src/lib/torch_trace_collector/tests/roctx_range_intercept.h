// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace roctx_range_intercept
{

struct Recording
{
    std::vector<std::string> messages;
    std::size_t              pops = 0;
};

void      start_recording();
void      fail_next_push();
Recording stop_recording();

}  // namespace roctx_range_intercept
