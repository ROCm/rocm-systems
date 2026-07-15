// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>

namespace rocprofsys::control
{
using clock_duration = std::chrono::nanoseconds;
using clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock, clock_duration>;
}  // namespace rocprofsys::control
