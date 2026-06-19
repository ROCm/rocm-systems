// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <timemory/backends/hardware_counters.hpp>
#include <vector>

namespace rocprofsys
{
namespace avail
{
std::vector<tim::hardware_counters::info>
query_gpu_hw_counters();
}  // namespace avail
}  // namespace rocprofsys
