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

#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <hsa/hsa.h>

#include <cstdint>

namespace rocprofiler
{
namespace context
{
struct context;
struct correlation_id;
}  // namespace context

namespace kernel_dispatch
{
using context_t              = context::context;
using user_data_map_t        = tracing::external_correlation_id_map_t;
using external_corr_id_map_t = user_data_map_t;
using queue_info_session_t   = hsa::queue_info_session_t;
using packet_data_t          = hsa::packet_data_t;
using profiling_time         = tracing::profiling_time;

profiling_time
get_dispatch_time(const queue_info_session_t& session, packet_data_t& packet_data);

void
dispatch_complete(queue_info_session_t& session, packet_data_t& packet_data, profiling_time);

// Emit the KERNEL_DISPATCH_COMPLETE callback and buffered record from value data
// alone. Shared by dispatch_complete() (signal path) and the no-signal finalizer,
// which has no queue session -- its payload deliberately holds no `Queue&`.
void
emit_kernel_dispatch_record(tracing::tracing_data&                               tracing_data,
                            rocprofiler_callback_tracing_kernel_dispatch_data_t& callback_record,
                            context::correlation_id*                             correlation_id,
                            rocprofiler_thread_id_t                              tid,
                            uint64_t                                             start_timestamp,
                            uint64_t                                             end_timestamp);
}  // namespace kernel_dispatch
}  // namespace rocprofiler
