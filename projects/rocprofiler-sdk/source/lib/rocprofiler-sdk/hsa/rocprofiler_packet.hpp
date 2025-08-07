// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/container/small_vector.hpp"

#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

namespace rocprofiler
{
namespace hsa
{
#define HSA_AMD_INTERFACE_VERSION                                                                  \
    ROCPROFILER_COMPUTE_VERSION(HSA_AMD_INTERFACE_VERSION_MAJOR, HSA_AMD_INTERFACE_VERSION_MINOR, 0)

#if HSA_AMD_INTERFACE_VERSION >= ROCPROFILER_COMPUTE_VERSION(1, 7, 0)
constexpr auto hsa_amd_memory_pool_executable_flag = HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG;
#else
constexpr auto hsa_amd_memory_pool_executable_flag = (1 << 2);
#endif

constexpr hsa_ext_amd_aql_pm4_packet_t null_amd_aql_pm4_packet = {
    .header            = 0,
    .pm4_command       = {0},
    .completion_signal = {.handle = 0}};

union rocprofiler_packet
{
    hsa_ext_amd_aql_pm4_packet_t ext_amd_aql_pm4;
    hsa_kernel_dispatch_packet_t kernel_dispatch;
    hsa_barrier_and_packet_t     barrier_and;
    hsa_barrier_or_packet_t      barrier_or;
    amd_aql_intercept_marker_t   marker;

    rocprofiler_packet()
    : ext_amd_aql_pm4{null_amd_aql_pm4_packet}
    {}

    rocprofiler_packet(hsa_ext_amd_aql_pm4_packet_t val)
    : ext_amd_aql_pm4{val}
    {}

    rocprofiler_packet(hsa_kernel_dispatch_packet_t val)
    : kernel_dispatch{val}
    {}

    rocprofiler_packet(hsa_barrier_and_packet_t val)
    : barrier_and{val}
    {}

    rocprofiler_packet(hsa_barrier_or_packet_t val)
    : barrier_or{val}
    {}

    rocprofiler_packet(amd_aql_intercept_marker_t val)
    : marker{val}
    {}

    ~rocprofiler_packet()                             = default;
    rocprofiler_packet(const rocprofiler_packet&)     = default;
    rocprofiler_packet(rocprofiler_packet&&) noexcept = default;

    rocprofiler_packet& operator=(const rocprofiler_packet&) = default;
    rocprofiler_packet& operator=(rocprofiler_packet&&) noexcept = default;
};
}  // namespace hsa
}  // namespace rocprofiler
