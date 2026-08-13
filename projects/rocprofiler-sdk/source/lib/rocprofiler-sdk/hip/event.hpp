// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace hip
{
namespace event
{
const char*
name_by_id(uint32_t id);

uint32_t
id_by_name(const char* name);

std::vector<const char*>
get_names();

std::vector<uint32_t>
get_ids();

using profiling_time = tracing::profiling_time;

void
barrier_complete(tracing::tracing_data&                               tracing_data_v,
                 rocprofiler_thread_id_t                              tid,
                 uint64_t                                             internal_corr_id,
                 uint64_t                                             ancestor_corr_id,
                 profiling_time                                       barrier_time,
                 rocprofiler_hip_event_operation_t                    operation,
                 rocprofiler_callback_tracing_hip_event_data_t callback_record);
}  // namespace event
}  // namespace hip
}  // namespace rocprofiler
