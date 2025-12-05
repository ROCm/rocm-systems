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

#include "generator.hpp"
#include "metadata.hpp"
#include "output_config.hpp"
#include "stream_info.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>

namespace rocprofiler
{
namespace tool
{

// Initialize perf user_events - registers the tracepoint early so external tools
// (like perf record) can subscribe before events are written.
// Call this at tool initialization, before profiling starts.
// Returns true if initialization succeeded, false otherwise.
bool
init_perf_user_events();

// Write kernel dispatch events to perf user_events.
// Must call init_perf_user_events() first.
void
write_perf_user_events(
    const output_config&                                               cfg,
    const metadata&                                                    tool_metadata,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>& kernel_dispatch_gen);

// Write HSA API events to perf user_events.
// Must call init_perf_user_events() first.
void
write_hsa_api_events(const output_config&                                          cfg,
                     const metadata&                                               tool_metadata,
                     const generator<rocprofiler_buffer_tracing_hsa_api_record_t>& hsa_api_gen);

// Write HIP API events to perf user_events.
// Must call init_perf_user_events() first.
void
write_hip_api_events(const output_config&                                       cfg,
                     const metadata&                                            tool_metadata,
                     const generator<tool_buffer_tracing_hip_api_ext_record_t>& hip_api_gen);

// Cleanup perf user_events resources.
// Call this at tool shutdown.
void
cleanup_perf_user_events();

}  // namespace tool
}  // namespace rocprofiler
