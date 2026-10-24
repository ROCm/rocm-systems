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
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace hip
{
namespace graph
{
/// The kernel nodes a stream capture created, in capture order, for one graph exec (graph.cpp).
struct capture_plan;

/// Per-launch state held in TLS while inside hipGraphLaunch.
struct launch_state
{
    uint64_t                            graph_exec_id  = 0;
    uint64_t                            node_counter   = 0;
    rocprofiler_timestamp_t             start_ts       = {0};
    uint64_t                            dispatch_count = 0;
    rocprofiler_agent_id_t              agent_id       = {0};
    rocprofiler_queue_id_t              queue_id       = {0};
    rocprofiler_correlation_id_t        correlation_id = {};
    rocprofiler_thread_id_t             thread_id      = 0;
    std::shared_ptr<const capture_plan> capture        = {};  // the exec's, if it was captured
    size_t                              capture_cursor = 0;   // next captured kernel node to match
};

/// Returns the currently-active launch state on this thread, or nullptr.
launch_state*
current_launch_state();

/// The graph identity of an operation a graph launch enqueues: its launch's graph_exec_id, the
/// operation's ordinal within the launch, and the internal correlation id of the HIP call whose
/// stream capture created its node. All are 0 outside a graph launch.
struct enqueue_identity
{
    rocprofiler_graph_exec_id_t graph_exec_id          = {0};
    rocprofiler_graph_node_id_t graph_node_id          = {0};
    uint64_t                    capture_correlation_id = 0;
};

/// Called once per kernel dispatch or memory copy as it is enqueued on this thread: the launch's
/// graph_exec_id and the next ordinal (launch_state::node_counter), so kernel dispatches and memory
/// copies share one numbering in enqueue order. For a kernel dispatch (kernel_id nonzero), also the
/// capture correlation id of the launch's next captured kernel node, if that node's kernel is this
/// one. Returns zeros outside a graph launch.
enqueue_identity
next_enqueue_identity(uint64_t kernel_id = 0);

/// True while a stream capture begun on this thread is active.
bool
capture_active();

/// Called by the HIP API wrapper when a traced HIP runtime call returns while capture_active(): the
/// graph nodes the call added to this thread's captures are attributed to its correlation id.
void
on_capture_api_exit(uint64_t internal_correlation_id);

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
