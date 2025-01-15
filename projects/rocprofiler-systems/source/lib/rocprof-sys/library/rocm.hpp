// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "core/defines.hpp"
#include "core/timemory.hpp"

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
#    include <rocprofiler-sdk/registration.h>
#    include <rocprofiler-sdk/rocprofiler.h>
#endif

#include <cstdint>
#include <mutex>
#include <vector>

namespace rocprofsys
{
namespace rocm
{
using hardware_counter_info = ::tim::hardware_counters::info;

std::vector<hardware_counter_info>
rocm_events();

#if !defined(ROCPROFSYS_USE_ROCM) || ROCPROFSYS_USE_ROCM == 0
inline std::vector<hardware_counter_info>
rocm_events()
{
    return std::vector<hardware_counter_info>();
}
#endif
}  // namespace rocm
}  // namespace rocprofsys

extern "C"
{
    struct rocprofiler_tool_configure_result_t;
    struct rocprofiler_client_id_t;

    using rocprofiler_configure_t =
        rocprofiler_tool_configure_result_t* (*) (uint32_t    version,
                                                  const char* runtime_version,
                                                  uint32_t    priority,
                                                  rocprofiler_client_id_t* client_id);

    rocprofiler_tool_configure_result_t* rocprofiler_configure(
        uint32_t version, const char* runtime_version, uint32_t priority,
        rocprofiler_client_id_t* client_id) ROCPROFSYS_PUBLIC_API;
}
