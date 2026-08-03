// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/fwd.h>

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace hip
{
namespace graph
{
/// Per-launch state held in TLS while inside hipGraphLaunch.
struct launch_state
{
    uint64_t                     graph_exec_id  = 0;
    uint64_t                     node_counter   = 0;
    rocprofiler_timestamp_t      start_ts       = {0};
    uint64_t                     dispatch_count = 0;
    rocprofiler_agent_id_t       agent_id       = {0};
    rocprofiler_queue_id_t       queue_id       = {0};
    rocprofiler_correlation_id_t correlation_id = {};
    rocprofiler_thread_id_t      thread_id      = 0;
};

/// Returns the currently-active launch state on this thread, or nullptr.
launch_state*
current_launch_state();

/// Internal table-wrapping installer (specialized in graph.cpp for the HIP dispatch table type).
template <typename TableT>
void
update_table(TableT* table);

/// Assigned at hipGraphInstantiate*; returns 0 if not tracked.
uint64_t
lookup_graph_exec_id(::hipGraphExec_t exec);

/// Printable name of a ::rocprofiler_hip_graph_operation_t id, or nullptr if out of range.
const char*
name_by_id(uint32_t id);

/// Returns the full set of valid ::rocprofiler_hip_graph_operation_t ids.
std::vector<uint32_t>
get_ids();

}  // namespace graph
}  // namespace hip
}  // namespace rocprofiler
