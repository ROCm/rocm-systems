// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/range_replay/queue_hooks.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/range_replay/executor.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

namespace rocprofiler
{
namespace range_replay
{
void
submission_hook(const hsa::Queue*                     queue,
                const hsa::rocprofiler_packet*        packets,
                uint64_t                              pkt_count,
                bool                                  graph_launch_active,
                hsa_amd_queue_intercept_packet_writer writer)
{
    if(!any_range_open() || this_thread_replaying()) return;

    const auto& _queue = *CHECK_NOTNULL(queue);

    if(auto* open_range = current_range(); open_range != nullptr)
    {
        note_submission(_queue, packets, pkt_count, graph_launch_active);
        ensure_entry_snapshot(*open_range, _queue, writer);
    }
    else if(const auto* rocp_agent = _queue.get_agent().get_rocp_agent(); rocp_agent != nullptr)
    {
        note_foreign_dispatch(rocp_agent->id.handle);
    }
}
}  // namespace range_replay
}  // namespace rocprofiler
