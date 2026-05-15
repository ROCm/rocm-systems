// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocm.hpp"
#include "library/rocprofiler-sdk.hpp"

#include <timemory/backends/cpu.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/utility/types.hpp>

#include <cstdlib>
#include <rocprofiler-sdk/rocprofiler.h>
#include <vector>

namespace rocprofsys
{
namespace rocm
{
std::vector<hardware_counter_info>
rocm_events()
{
    return rocprofiler_sdk::get_rocm_events_info();
}
}  // namespace rocm
}  // namespace rocprofsys
